use clap::{Parser, Subcommand};
use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::chat::{ChatMessage, request::ChatMessageRequest};
use ollama_rs::generation::images::Image;
use base64::Engine;
use std::fs;
use std::path::PathBuf;
use log::{info, warn, error, debug};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;
use tokio::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "ollama-api-access")]
#[command(about = "A CLI tool to interact with Ollama API")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Query Ollama with a prompt
    Query {
        /// The prompt to send to Ollama
        #[arg(short, long, group = "prompt_input")]
        prompt: Option<String>,
        
        /// Read prompt from file
        #[arg(long, group = "prompt_input")]
        prompt_file: Option<PathBuf>,
        
        /// The model to use (default: llama3.2)
        #[arg(short, long, default_value = "llama3.2")]
        model: String,
    },
    /// Analyze an image with Ollama using a vision model
    AnalyzeImage {
        /// The prompt/question about the image
        #[arg(short, long, group = "prompt_input")]
        prompt: Option<String>,
        
        /// Read prompt from file
        #[arg(long, group = "prompt_input")]
        prompt_file: Option<PathBuf>,
        
        /// Path to the image file
        #[arg(short, long, group = "image_input")]
        image: Option<PathBuf>,
        
        /// Path to directory containing images to process
        #[arg(long, group = "image_input")]
        image_dir: Option<PathBuf>,
        
        /// URL to download image from
        #[arg(long, group = "image_input")]
        image_url: Option<String>,
        
        /// The vision model to use (default: llama3.2-vision)
        #[arg(short, long, default_value = "llama3.2-vision")]
        model: String,
    },
    /// Start an interactive chat session with Ollama
    Chat {
        /// The model to use for chat (default: llama3.2)
        #[arg(short, long, default_value = "llama3.2")]
        model: String,
    },
}

/// Represents different sources of images
#[derive(Debug, Clone)]
enum ImageSource {
    File(PathBuf),
    Url(String),
}

/// Download image data from URL
async fn download_image_data(url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    info!("📥 Downloading image from: {}", url);
    let response = reqwest::get(url).await?;
    
    if !response.status().is_success() {
        return Err(format!("Failed to download image: HTTP {}", response.status()).into());
    }
    
    let bytes = response.bytes().await?;
    Ok(bytes.to_vec())
}

/// Get image data from any source
async fn get_image_data(source: &ImageSource) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    match source {
        ImageSource::File(path) => {
            fs::read(path).map_err(|e| e.into())
        }
        ImageSource::Url(url) => {
            download_image_data(url).await
        }
    }
}

/// Get all image files from a directory
fn get_image_files_from_directory(dir: &PathBuf) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut image_files = Vec::new();
    
    if !dir.is_dir() {
        return Err(format!("Path is not a directory: {}", dir.display()).into());
    }
    
    let entries = fs::read_dir(dir)?;
    
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(extension) = path.extension() {
                let ext = extension.to_string_lossy().to_lowercase();
                match ext.as_str() {
                    "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff" | "tif" => {
                        image_files.push(path);
                    }
                    _ => {}
                }
            }
        }
    }
    
    // Sort the files for consistent ordering
    image_files.sort();
    
    Ok(image_files)
}

/// Process a single image with the given prompt and model
async fn process_single_image(
    image_data: &Vec<u8>,
    prompt_text: &str,
    model: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a fresh Ollama client for each image to ensure clean context
    let ollama = Ollama::default();
    

    // Encode image file
    let image_base64 = base64::prelude::BASE64_STANDARD.encode(&image_data);
    let image = Image::from_base64(&image_base64);
    
    // Create chat message with image
    let messages = vec![
        ChatMessage::user(prompt_text.to_string()).with_images(vec![image])
    ];
    
    let request = ChatMessageRequest::new(model.to_string(), messages);
    let response = ollama.send_chat_messages(request).await?;
    
    info!("📝 Analysis:");
    println!("{}", response.message.content);
    
    Ok(())
}

/// Represents a chat message in the conversation
#[derive(Debug, Clone)]
struct ChatMessageDisplay {
    content: String,
    is_user: bool,
    timestamp: Instant,
}

/// Application state for the chat interface
struct ChatApp {
    messages: Vec<ChatMessageDisplay>,
    input: String,
    should_quit: bool,
    model: String,
    ollama: Ollama,
    scroll_offset: usize,
}

impl ChatApp {
    fn new(model: String) -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            should_quit: false,
            model,
            ollama: Ollama::default(),
            scroll_offset: 0,
        }
    }

    fn add_message(&mut self, content: String, is_user: bool) {
        self.messages.push(ChatMessageDisplay {
            content,
            is_user,
            timestamp: Instant::now(),
        });
        // Auto-scroll to bottom
        if self.messages.len() > 10 {
            self.scroll_offset = self.messages.len() - 10;
        }
    }

    fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
    }

    fn scroll_down(&mut self) {
        let max_scroll = if self.messages.len() > 10 {
            self.messages.len() - 10
        } else {
            0
        };
        if self.scroll_offset < max_scroll {
            self.scroll_offset += 1;
        }
    }
}

/// Events that can occur in the chat application
#[derive(Debug)]
enum ChatEvent {
    Input(Event),
    OllamaResponse(String),
    OllamaError(String),
}

/// Run the chat interface
async fn run_chat(model: String) -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = ChatApp::new(model);
    
    // Add welcome message
    app.add_message(
        format!("Welcome to Ollama Chat! Using model: {}\nType your message and press Enter to send. Press Ctrl+C to quit.", app.model),
        false,
    );

    // Create channels for async communication
    let (tx, mut rx) = mpsc::unbounded_channel::<ChatEvent>();

    // Spawn input event handler
    let input_tx = tx.clone();
    tokio::spawn(async move {
        loop {
            if let Ok(event) = event::read() {
                if input_tx.send(ChatEvent::Input(event)).is_err() {
                    break;
                }
            }
        }
    });

    // Main event loop
    loop {
        // Draw UI
        terminal.draw(|f| ui(f, &app))?;

        // Handle events
        if let Ok(chat_event) = rx.try_recv() {
            match chat_event {
                ChatEvent::Input(event) => {
                    if let Event::Key(key) = event {
                        if key.kind == KeyEventKind::Press {
                            match key.code {
                                KeyCode::Char('c') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                                    app.should_quit = true;
                                }
                                KeyCode::Enter => {
                                    if !app.input.trim().is_empty() {
                                        let user_message = app.input.clone();
                                        app.add_message(user_message.clone(), true);
                                        app.input.clear();

                                        // Send to Ollama in background
                                        let ollama = app.ollama.clone();
                                        let model = app.model.clone();
                                        let tx_clone = tx.clone();
                                        
                                        tokio::spawn(async move {
                                            match send_to_ollama(&ollama, &model, &user_message).await {
                                                Ok(response) => {
                                                    let _ = tx_clone.send(ChatEvent::OllamaResponse(response));
                                                }
                                                Err(e) => {
                                                    let _ = tx_clone.send(ChatEvent::OllamaError(format!("Error: {}", e)));
                                                }
                                            }
                                        });
                                    }
                                }
                                KeyCode::Backspace => {
                                    app.input.pop();
                                }
                                KeyCode::Up => {
                                    app.scroll_up();
                                }
                                KeyCode::Down => {
                                    app.scroll_down();
                                }
                                KeyCode::Char(c) => {
                                    app.input.push(c);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                ChatEvent::OllamaResponse(response) => {
                    app.add_message(response, false);
                }
                ChatEvent::OllamaError(error) => {
                    app.add_message(error, false);
                }
            }
        }

        if app.should_quit {
            break;
        }

        // Small delay to prevent busy waiting
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

/// Send message to Ollama and get response
async fn send_to_ollama(
    ollama: &Ollama,
    model: &str,
    message: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let messages = vec![ChatMessage::user(message.to_string())];
    let request = ChatMessageRequest::new(model.to_string(), messages);
    let response = ollama.send_chat_messages(request).await?;
    Ok(response.message.content)
}

/// Draw the UI
fn ui(f: &mut Frame, app: &ChatApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(f.size());

    // Messages area
    let messages: Vec<ListItem> = app
        .messages
        .iter()
        .skip(app.scroll_offset)
        .take(chunks[0].height as usize - 2) // Account for borders
        .map(|msg| {
            let style = if msg.is_user {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Green)
            };
            
            let prefix = if msg.is_user { "You: " } else { "AI: " };
            let content = format!("{}{}", prefix, msg.content);
            
            ListItem::new(Text::from(Line::from(vec![
                Span::styled(content, style)
            ])))
        })
        .collect();

    let messages_block = Block::default()
        .borders(Borders::ALL)
        .title("Chat Messages (↑/↓ to scroll, Ctrl+C to quit)");
    
    let messages_list = List::new(messages).block(messages_block);
    f.render_widget(messages_list, chunks[0]);

    // Input area
    let input_block = Block::default()
        .borders(Borders::ALL)
        .title("Type your message (Enter to send)");
    
    let input_paragraph = Paragraph::new(app.input.as_str())
        .block(input_block)
        .wrap(Wrap { trim: true });
    
    f.render_widget(input_paragraph, chunks[1]);

    // Set cursor position
    f.set_cursor(
        chunks[1].x + app.input.len() as u16 + 1,
        chunks[1].y + 1,
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();
    
    let cli = Cli::parse();

    match cli.command {
        Commands::Query { prompt, prompt_file, model } => {
            let ollama = Ollama::default();
            
            // Get prompt from either --prompt or --prompt-file
            let prompt_text = match (prompt, prompt_file) {
                (Some(p), None) => p,
                (None, Some(file)) => fs::read_to_string(file)?,
                (Some(_), Some(_)) => return Err("Cannot specify both --prompt and --prompt-file".into()),
                (None, None) => return Err("Must specify either --prompt or --prompt-file".into()),
            };
            
            info!("🤖 Sending prompt to {} model...", model);
            
            let request = GenerationRequest::new(model, prompt_text);
            let response = ollama.generate(request).await?;
            
            info!("📝 Response:");
            println!("{}", response.response);
        }
        Commands::AnalyzeImage { prompt, prompt_file, image, image_dir, image_url, model } => {
            // Get prompt from either --prompt or --prompt-file
            let prompt_text = match (prompt, prompt_file) {
                (Some(p), None) => p,
                (None, Some(file)) => fs::read_to_string(file)?,
                (Some(_), Some(_)) => return Err("Cannot specify both --prompt and --prompt-file".into()),
                (None, None) => return Err("Must specify either --prompt or --prompt-file".into()),
            };
            
            // Create ImageSource instances based on command arguments
            let image_sources = match (image, image_dir, image_url) {
                (Some(img), None, None) => vec![ImageSource::File(img)],
                (None, Some(dir), None) => {
                    info!("🔍 Scanning directory: {}", dir.display());
                    let image_paths = get_image_files_from_directory(&dir)?;
                    image_paths.into_iter().map(ImageSource::File).collect()
                },
                (None, None, Some(url)) => vec![ImageSource::Url(url)],
                (Some(_), Some(_), None) => return Err("Cannot specify both --image and --image-dir".into()),
                (Some(_), None, Some(_)) => return Err("Cannot specify both --image and --image-url".into()),
                (None, Some(_), Some(_)) => return Err("Cannot specify both --image-dir and --image-url".into()),
                (Some(_), Some(_), Some(_)) => return Err("Cannot specify multiple image input options".into()),
                (None, None, None) => return Err("Must specify one of --image, --image-dir, or --image-url".into()),
            };
            
            if image_sources.is_empty() {
                return Err("No image sources found".into());
            }
            
            info!("🖼️  Analyzing {} image(s) with {} model...", image_sources.len(), model);
            
            let mut successful_count = 0;
            let mut failed_count = 0;
            
            // Process each image source
            for (index, image_source) in image_sources.iter().enumerate() {
                let total = image_sources.len();
                
                // Display appropriate message based on source type
                match image_source {
                    ImageSource::File(path) => {
                        info!("📁 Processing image {}/{}: {}", index + 1, total, path.display());
                    }
                    ImageSource::Url(url) => {
                        info!("🌐 Processing image {}/{}: {}", index + 1, total, url);
                    }
                }
                
                // Handle each image individually to avoid stopping on errors
                match get_image_data(image_source).await {
                    Ok(image_data) => {
                        match process_single_image(&image_data, &prompt_text, &model).await {
                            Ok(()) => {
                                successful_count += 1;
                            }
                            Err(e) => {
                                failed_count += 1;
                                match image_source {
                                    ImageSource::File(path) => {
                                        error!("❌ Error processing {}: {}", path.display(), e);
                                    }
                                    ImageSource::Url(url) => {
                                        error!("❌ Error processing {}: {}", url, e);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        failed_count += 1;
                        match image_source {
                            ImageSource::File(path) => {
                                error!("❌ Error reading file {}: {}", path.display(), e);
                            }
                            ImageSource::Url(url) => {
                                error!("❌ Error downloading {}: {}", url, e);
                            }
                        }
                    }
                }
                
                // Add separator between images if processing multiple
                if total > 1 && index < total - 1 {
                    debug!("{}", "=".repeat(50));
                }
            }
            
            // Print summary
            info!("📊 Processing Summary:");
            info!("✅ Successfully processed: {}", successful_count);
            if failed_count > 0 {
                warn!("❌ Failed to process: {}", failed_count);
            }
            info!("📈 Total images: {}", image_sources.len());
        }
        Commands::Chat { model } => {
            info!("🚀 Starting chat with {} model...", model);
            run_chat(model).await?;
        }
    }

    Ok(())
}

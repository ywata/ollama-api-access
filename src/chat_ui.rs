use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;
use tokio::sync::mpsc;
use std::time::{Duration, Instant};
use log::info;

use crate::{Config, ModelSpec, send_genai_query};

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
    model_spec: ModelSpec,
    client: genai::Client,
    config: Config,
    scroll_offset: u16,
    max_scroll: u16,
}

impl ChatApp {
    fn new(model: String, model_spec: ModelSpec, client: genai::Client, config: Config) -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            should_quit: false,
            model,
            model_spec,
            client,
            config,
            scroll_offset: 0,
            max_scroll: 0,
        }
    }

    fn add_message(&mut self, content: String, is_user: bool) {
        self.messages.push(ChatMessageDisplay {
            content,
            is_user,
            timestamp: Instant::now(),
        });
        // Keep current scroll position when new message arrives
    }

    fn scroll_up(&mut self) {
        if self.scroll_offset < self.max_scroll {
            self.scroll_offset = self.scroll_offset.saturating_add(1);
        }
    }

    fn scroll_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }

    /// Calculate total lines for messages up to (but not including) the given index
    fn count_lines_before(&self, message_index: usize) -> usize {
        let mut line_count = 0;
        for (i, msg) in self.messages.iter().enumerate() {
            if i >= message_index {
                break;
            }
            // Count lines in the message content
            line_count += msg.content.lines().count().max(1);
            // Add 1 for empty line between messages
            line_count += 1;
        }
        line_count
    }

    /// Scroll to show the last user message (query)
    fn scroll_to_last_user_message(&mut self) {
        // Find the last user message (should be the most recent query)
        if let Some(pos) = self.messages.iter().rposition(|msg| msg.is_user) {
            let lines_before = self.count_lines_before(pos);
            // Set scroll to show this message (with some context above if possible)
            self.scroll_offset = lines_before.saturating_sub(2) as u16;
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

/// Send message to AI and get response
async fn send_chat_message(
    client: &genai::Client,
    model_spec: &ModelSpec,
    config: &Config,
    message: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    send_genai_query(client, model_spec, config, message).await
}

/// Handle input events (keyboard)
fn handle_input_event(
    app: &mut ChatApp,
    event: Event,
    tx: &mpsc::UnboundedSender<ChatEvent>,
) -> Result<(), Box<dyn std::error::Error>> {
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

                        // Send to AI in background
                        let client = app.client.clone();
                        let model_spec = app.model_spec.clone();
                        let config = app.config.clone();
                        let tx_clone = tx.clone();
                        
                        tokio::spawn(async move {
                            match send_chat_message(&client, &model_spec, &config, &user_message).await {
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
    Ok(())
}

/// Handle backend chat events (responses, errors)
fn handle_chat_event(
    app: &mut ChatApp,
    event: ChatEvent,
    tx: &mpsc::UnboundedSender<ChatEvent>,
) -> Result<(), Box<dyn std::error::Error>> {
    match event {
        ChatEvent::Input(input_event) => {
            handle_input_event(app, input_event, tx)?;
        }
        ChatEvent::OllamaResponse(response) => {
            info!("📩 Received response from AI");
            app.add_message(response, false);
            // Scroll to show the query that this response is for
            app.scroll_to_last_user_message();
        }
        ChatEvent::OllamaError(error) => {
            app.add_message(error, false);
        }
    }
    Ok(())
}

/// Draw the UI
fn ui(f: &mut Frame, app: &mut ChatApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(f.size());

    // Messages area - Build formatted text with proper wrapping
    let mut text_lines = Vec::new();
    
    for msg in app.messages.iter() {
        let style = if msg.is_user {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Green)
        };
        
        let prefix = if msg.is_user { "You: " } else { "AI: " };
        
        // Split message by newlines and create styled lines
        for (i, line) in msg.content.lines().enumerate() {
            let content = if i == 0 {
                format!("{}{}", prefix, line)
            } else {
                format!("     {}", line) // Indent continuation lines
            };
            text_lines.push(Line::from(vec![Span::styled(content, style)]));
        }
        
        // Add empty line between messages for readability
        text_lines.push(Line::from(""));
    }

    // Calculate maximum scroll offset
    let visible_height = chunks[0].height.saturating_sub(2) as u16; // Account for borders
    let total_lines = text_lines.len() as u16;
    app.max_scroll = total_lines.saturating_sub(visible_height);

    let messages_block = Block::default()
        .borders(Borders::ALL)
        .title("Chat Messages (↑/↓ to scroll, Ctrl+C to quit)");
    
    let messages_paragraph = Paragraph::new(text_lines)
        .block(messages_block)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll_offset, 0));
    
    f.render_widget(messages_paragraph, chunks[0]);

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

/// Main event loop using tokio::select!
async fn chat_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut ChatApp,
    mut rx: mpsc::UnboundedReceiver<ChatEvent>,
    tx: mpsc::UnboundedSender<ChatEvent>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create interval for redrawing UI
    let mut redraw_interval = tokio::time::interval(Duration::from_millis(50));
    
    loop {
        tokio::select! {
            // Handle incoming chat events
            Some(chat_event) = rx.recv() => {
                handle_chat_event(app, chat_event, &tx)?;
                
                if app.should_quit {
                    break;
                }
            }
            // Periodic UI redraw
            _ = redraw_interval.tick() => {
                terminal.draw(|f| ui(f, app))?;
            }
        }
    }
    
    Ok(())
}

/// Run the chat interface
pub async fn run_chat(model: String, client: genai::Client, config: Config) -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let model_spec = ModelSpec::parse(&model);
    let display_model = model_spec.resolve_deployment(&config);
    let mut app = ChatApp::new(model, model_spec, client, config);
    
    // Add welcome message
    app.add_message(
        format!("Welcome to Multi-Provider AI Chat! Using model: {}\nType your message and press Enter to send. Press Ctrl+C to quit.", display_model),
        false,
    );

    // Create channels for async communication
    let (tx, rx) = mpsc::unbounded_channel::<ChatEvent>();

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

    // Run main event loop
    chat_loop(&mut terminal, &mut app, rx, tx).await?;

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

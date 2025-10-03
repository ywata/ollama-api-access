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
pub struct ChatMessageDisplay {
    pub content: String,
    pub is_user: bool,
    pub timestamp: Instant,
}

/// Application state for the chat interface
pub struct ChatApp {
    messages: Vec<ChatMessageDisplay>,
    input: String,
    should_quit: bool,
    model: String,
    model_spec: ModelSpec,
    client: genai::Client,
    config: Config,
    system_prompt: Option<String>,
    scroll_offset: u16,
    max_scroll: u16,
}

impl ChatApp {
    pub fn new(model: String, model_spec: ModelSpec, client: genai::Client, config: Config, system_prompt: Option<String>) -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            should_quit: false,
            model,
            model_spec,
            client,
            config,
            system_prompt,
            scroll_offset: 0,
            max_scroll: 0,
        }
    }

    pub fn add_message(&mut self, content: String, is_user: bool) {
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
    pub fn count_lines_before(&self, message_index: usize) -> usize {
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
    pub fn scroll_to_last_user_message(&mut self) {
        // Find the last user message (should be the most recent query)
        if let Some(pos) = self.messages.iter().rposition(|msg| msg.is_user) {
            let lines_before = self.count_lines_before(pos);
            // Set scroll to show this message (with some context above if possible)
            self.scroll_offset = lines_before.saturating_sub(2) as u16;
        }
    }

    // Accessor methods for testing
    #[cfg(test)]
    pub fn messages(&self) -> &[ChatMessageDisplay] {
        &self.messages
    }

    #[cfg(test)]
    pub fn scroll_offset(&self) -> u16 {
        self.scroll_offset
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
    system_prompt: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    send_genai_query(client, model_spec, config, message, system_prompt).await
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
                        let system_prompt = app.system_prompt.clone();
                        let tx_clone = tx.clone();
                        
                        tokio::spawn(async move {
                            match send_chat_message(&client, &model_spec, &config, &user_message, system_prompt.as_deref()).await {
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
pub async fn run_chat(model: String, client: genai::Client, config: Config, system_prompt: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let model_spec = ModelSpec::parse(&model);
    let display_model = model_spec.resolve_deployment(&config);
    let mut app = ChatApp::new(model, model_spec, client, config, system_prompt.clone());
    
    // Add welcome message
    let welcome_msg = if system_prompt.is_some() {
        format!("Welcome to Multi-Provider AI Chat! Using model: {}\nSystem prompt: Active\nType your message and press Enter to send. Press Ctrl+C to quit.", display_model)
    } else {
        format!("Welcome to Multi-Provider AI Chat! Using model: {}\nType your message and press Enter to send. Press Ctrl+C to quit.", display_model)
    };
    app.add_message(welcome_msg, false);

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

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a test ChatApp
    fn create_test_app() -> ChatApp {
        let config = Config::default();
        let model_spec = ModelSpec::parse("test:model");
        let client = genai::Client::default();
        ChatApp::new("test".to_string(), model_spec, client, config, None)
    }

    // Tests for count_lines_before
    #[test]
    fn test_count_lines_before_empty() {
        let app = create_test_app();
        assert_eq!(app.count_lines_before(0), 0);
    }

    #[test]
    fn test_count_lines_before_single_line_messages() {
        let mut app = create_test_app();
        app.add_message("Message 1".to_string(), true);
        app.add_message("Message 2".to_string(), false);
        app.add_message("Message 3".to_string(), true);

        // Before index 0: no messages
        assert_eq!(app.count_lines_before(0), 0);
        
        // Before index 1: Message 1 (1 line) + empty line (1) = 2
        assert_eq!(app.count_lines_before(1), 2);
        
        // Before index 2: Message 1 + Message 2 + 2 empty lines = 4
        assert_eq!(app.count_lines_before(2), 4);
        
        // Before index 3: all 3 messages + 3 empty lines = 6
        assert_eq!(app.count_lines_before(3), 6);
    }

    #[test]
    fn test_count_lines_before_multi_line_messages() {
        let mut app = create_test_app();
        app.add_message("Line 1\nLine 2\nLine 3".to_string(), true);
        app.add_message("Single line".to_string(), false);

        // Before index 0: nothing
        assert_eq!(app.count_lines_before(0), 0);
        
        // Before index 1: 3 lines from message + 1 empty line = 4
        assert_eq!(app.count_lines_before(1), 4);
        
        // Before index 2: 3 lines + 1 line + 2 empty lines = 6
        assert_eq!(app.count_lines_before(2), 6);
    }

    #[test]
    fn test_count_lines_before_empty_message() {
        let mut app = create_test_app();
        app.add_message("".to_string(), true);
        app.add_message("Message".to_string(), false);

        // Empty message should count as 1 line (max(1))
        assert_eq!(app.count_lines_before(1), 2); // 1 (message) + 1 (empty line)
        assert_eq!(app.count_lines_before(2), 4); // 2 + 1 (message) + 1 (empty line)
    }

    #[test]
    fn test_count_lines_before_beyond_message_count() {
        let mut app = create_test_app();
        app.add_message("Message 1".to_string(), true);
        app.add_message("Message 2".to_string(), false);

        // Should stop at message count
        assert_eq!(app.count_lines_before(10), 4); // Same as count_lines_before(2)
    }

    // Tests for scroll_to_last_user_message
    #[test]
    fn test_scroll_to_last_user_message_no_messages() {
        let mut app = create_test_app();
        let initial_offset = app.scroll_offset();
        
        app.scroll_to_last_user_message();
        
        // Should not change scroll offset
        assert_eq!(app.scroll_offset(), initial_offset);
    }

    #[test]
    fn test_scroll_to_last_user_message_no_user_messages() {
        let mut app = create_test_app();
        app.add_message("AI message".to_string(), false);
        
        let initial_offset = app.scroll_offset();
        app.scroll_to_last_user_message();
        
        // Should not change scroll offset if no user messages
        assert_eq!(app.scroll_offset(), initial_offset);
    }

    #[test]
    fn test_scroll_to_last_user_message_single_user_message() {
        let mut app = create_test_app();
        app.add_message("User query".to_string(), true);
        
        app.scroll_to_last_user_message();
        
        // Should scroll to 0 (saturating_sub(2) on 0 = 0)
        assert_eq!(app.scroll_offset(), 0);
    }

    #[test]
    fn test_scroll_to_last_user_message_multiple_exchanges() {
        let mut app = create_test_app();
        app.add_message("First user query".to_string(), true);
        app.add_message("First AI response".to_string(), false);
        app.add_message("Second user query".to_string(), true);
        app.add_message("Second AI response\nwith more\nlines".to_string(), false);
        
        app.scroll_to_last_user_message();
        
        // Last user message is at index 2
        // Before index 2: 1 + 1 + 2 empty lines = 4
        // 4 - 2 = 2
        assert_eq!(app.scroll_offset(), 2);
    }

    #[test]
    fn test_scroll_to_last_user_message_finds_last_not_first() {
        let mut app = create_test_app();
        app.add_message("First user message".to_string(), true);
        app.add_message("AI response".to_string(), false);
        app.add_message("Second user message".to_string(), true);
        
        app.scroll_to_last_user_message();
        
        // Should scroll to the LAST (second) user message, not the first
        // Last user is at index 2
        // Before index 2: 1 + 1 + 2 empty = 4, 4 - 2 = 2
        assert_eq!(app.scroll_offset(), 2);
    }

    #[test]
    fn test_add_message_preserves_scroll() {
        let mut app = create_test_app();
        app.add_message("First".to_string(), true);
        
        // Manually set scroll offset
        app.scroll_offset = 5;
        
        // Add new message
        app.add_message("Second".to_string(), false);
        
        // Scroll should be preserved (as per current implementation)
        assert_eq!(app.scroll_offset(), 5);
    }

    #[test]
    fn test_messages_accessor() {
        let mut app = create_test_app();
        app.add_message("Message 1".to_string(), true);
        app.add_message("Message 2".to_string(), false);
        
        let messages = app.messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "Message 1");
        assert!(messages[0].is_user);
        assert_eq!(messages[1].content, "Message 2");
        assert!(!messages[1].is_user);
    }
}

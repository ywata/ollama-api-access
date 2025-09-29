# Ollama API Access

A CLI tool to interact with Ollama API, featuring text queries, image analysis, and real-time chat.

## Features

- **Text Queries**: Send prompts to Ollama models
- **Image Analysis**: Analyze images with vision models
- **Real-time Chat**: Interactive terminal-based chat interface

## Installation

```bash
# Build the project
cargo build --release

# Or run directly
cargo run -- <command>
```

## Usage

### Text Query
```bash
# Query with prompt
cargo run -- query --prompt "Hello, how are you?" --model llama3.2

# Query from file
cargo run -- query --prompt-file prompt.txt --model llama3.2
```

### Image Analysis
```bash
# Analyze single image
cargo run -- analyze-image --prompt "What's in this image?" --image photo.jpg

# Analyze directory of images
cargo run -- analyze-image --prompt "Describe each image" --image-dir ./photos/

# Analyze image from URL
cargo run -- analyze-image --prompt "What do you see?" --image-url https://example.com/image.jpg
```

### Real-time Chat
```bash
# Start chat with default model (llama3.2)
cargo run -- chat

# Start chat with specific model
cargo run -- chat --model llama3.2-vision
```

#### Chat Controls
- **Enter**: Send message
- **↑/↓**: Scroll through message history
- **Ctrl+C**: Exit chat

## Dependencies

- **ollama-rs**: Ollama API client
- **clap**: Command-line argument parsing
- **tokio**: Async runtime
- **ratatui**: Terminal UI framework
- **crossterm**: Cross-platform terminal manipulation
- **reqwest**: HTTP client for image downloads
- **base64**: Image encoding

## Requirements

- Rust 1.70+
- Ollama server running locally
- Required models installed (llama3.2, llama3.2-vision, etc.)

## Chat Interface

The chat interface provides a modern terminal-based experience:

- **Message History**: Scrollable conversation view
- **Real-time Responses**: Non-blocking Ollama API calls
- **Cross-platform**: Works on Linux, Windows, and macOS
- **Clean UI**: Colored messages (cyan for user, green for AI)
- **Responsive**: Maintains UI responsiveness during API calls

## Examples

```bash
# Quick text query
cargo run -- query -p "Explain quantum computing" -m llama3.2

# Analyze an image
cargo run -- analyze-image -p "What's happening in this image?" -i screenshot.png -m llama3.2-vision

# Start interactive chat
cargo run -- chat -m llama3.2
```

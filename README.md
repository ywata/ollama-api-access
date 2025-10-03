# Multi-Provider AI Access

A CLI tool to interact with multiple AI providers (Ollama, OpenAI, Azure OpenAI), featuring text queries, image analysis, and real-time chat.

## Features

- **Multi-Provider Support**: Ollama (local), OpenAI, Azure OpenAI
- **Text Queries**: Send prompts to any AI model
- **Image Analysis**: Analyze images with vision models
- **Real-time Chat**: Interactive terminal-based chat interface
- **System Prompts**: Customize AI behavior with system instructions
- **Unified Interface**: Same commands work across all providers

## Configuration

Create a configuration file at `~/.config/ollama-api-access/config.toml`:

```toml
# Azure OpenAI Configuration
[providers.azure]
endpoint = "https://your-resource.openai.azure.com/"
api_key = "your-azure-openai-api-key"
api_version = "2024-02-15-preview"

# OpenAI Configuration  
[providers.openai]
api_key = "your-openai-api-key"

# Deployment Mappings (for Azure)
[deployments]
"azure:gpt-4" = "your-gpt4-deployment-name"
"azure:gpt-4o" = "your-gpt4o-deployment-name"
"azure:gpt-35-turbo" = "your-gpt35-turbo-deployment-name"
```

See `config.example.toml` for a complete example.

## Installation

```bash
# Build the project
cargo build --release

# Or run directly
cargo run -- <command>
```

## Model Specification Format

Use the `provider:model` format to specify which AI provider and model to use:

- `ollama:llama3.2` - Ollama (local)
- `openai:gpt-4` - OpenAI
- `azure:gpt-4` - Azure OpenAI (uses deployment mapping)
- `llama3.2` - Defaults to `ollama:llama3.2` (backward compatible)

## Usage

### Text Query
```bash
# Ollama (local) - default
cargo run -- query --prompt "Hello, how are you?" --model llama3.2

# OpenAI
cargo run -- query --prompt "Explain quantum computing" --model openai:gpt-4

# Azure OpenAI
cargo run -- query --prompt "Write a poem" --model azure:gpt-4

# Query from file
cargo run -- query --prompt-file prompt.txt --model openai:gpt-4o

# With system prompt
cargo run -- query --prompt "Hello" --system-file system_prompt.txt --model llama3.2
```

### Image Analysis
```bash
# Ollama vision model
cargo run -- analyze-image --prompt "What's in this image?" --image photo.jpg --model llama3.2-vision

# OpenAI GPT-4o Vision
cargo run -- analyze-image --prompt "Describe this image" --image photo.jpg --model openai:gpt-4o

# Azure OpenAI Vision
cargo run -- analyze-image --prompt "Analyze this" --image photo.jpg --model azure:gpt-4o

# Analyze directory of images
cargo run -- analyze-image --prompt "Describe each image" --image-dir ./photos/ --model openai:gpt-4o

# Analyze image from URL
cargo run -- analyze-image --prompt "What do you see?" --image-url https://example.com/image.jpg --model azure:gpt-4o

# With system prompt
cargo run -- analyze-image --prompt "What's here?" --image photo.jpg --system-file system_prompt.txt --model llama3.2-vision
```

### Real-time Chat
```bash
# Ollama (default)
cargo run -- chat --model llama3.2

# OpenAI
cargo run -- chat --model openai:gpt-4

# Azure OpenAI
cargo run -- chat --model azure:gpt-4

# Ollama vision model
cargo run -- chat --model ollama:llama3.2-vision

# With system prompt (applies to entire chat session)
cargo run -- chat --system-file system_prompt.txt --model llama3.2
```

#### Chat Controls
- **Enter**: Send message
- **↑/↓**: Scroll through message history
- **Ctrl+C**: Exit chat

## System Prompts

System prompts allow you to customize the AI's behavior and personality. Use the `--system-file` option to load instructions from a file:

```bash
# Create a system prompt file
cat > system_prompt.txt << EOF
You are a helpful technical assistant who provides concise, accurate answers.
Always include code examples when relevant.
EOF

# Use with any command
cargo run -- query --prompt "Explain recursion" --system-file system_prompt.txt
cargo run -- analyze-image --prompt "Describe" --image photo.jpg --system-file system_prompt.txt
cargo run -- chat --system-file system_prompt.txt
```

See `system_prompt.example.md` for a template.

## Dependencies

- **genai**: Multi-provider AI client library
- **clap**: Command-line argument parsing
- **tokio**: Async runtime
- **ratatui**: Terminal UI framework
- **crossterm**: Cross-platform terminal manipulation
- **reqwest**: HTTP client for image downloads
- **base64**: Image encoding
- **serde/toml**: Configuration file handling

## Requirements

- Rust 1.70+
- For Ollama: Ollama server running locally with required models installed
- For OpenAI: OpenAI API key
- For Azure OpenAI: Azure OpenAI endpoint, API key, and deployment names

## Provider-Specific Setup

### Ollama (Local)
```bash
# Install Ollama and pull models
ollama pull llama3.2
ollama pull llama3.2-vision
```

### OpenAI
Add to config file:
```toml
[providers.openai]
api_key = "sk-..."
```

### Azure OpenAI
Add to config file:
```toml
[providers.azure]
endpoint = "https://your-resource.openai.azure.com/"
api_key = "your-key"
api_version = "2024-02-15-preview"

[deployments]
"azure:gpt-4" = "your-deployment-name"
```

## Chat Interface

The chat interface provides a modern terminal-based experience:

- **Message History**: Scrollable conversation view
- **Real-time Responses**: Non-blocking AI API calls
- **Multi-Provider**: Switch providers by changing model parameter
- **Cross-platform**: Works on Linux, Windows, and macOS
- **Clean UI**: Colored messages (cyan for user, green for AI)
- **Responsive**: Maintains UI responsiveness during API calls

## Examples

```bash
# Quick text query with different providers
cargo run -- query -p "Explain quantum computing" -m llama3.2
cargo run -- query -p "Explain quantum computing" -m openai:gpt-4
cargo run -- query -p "Explain quantum computing" -m azure:gpt-4

# Analyze an image with vision models
cargo run -- analyze-image -p "What's in this image?" -i photo.jpg -m llama3.2-vision
cargo run -- analyze-image -p "What's in this image?" -i photo.jpg -m openai:gpt-4o
cargo run -- analyze-image -p "What's in this image?" -i photo.jpg -m azure:gpt-4o

# Start interactive chat with any provider
cargo run -- chat -m llama3.2
cargo run -- chat -m openai:gpt-4
cargo run -- chat -m azure:gpt-4
```

## Migration from Ollama-only

If you were using the previous version:
- Old: `--model llama3.2` → New: `--model llama3.2` (backward compatible!)
- The tool now supports multiple providers with the same interface
- Configuration file is auto-created on first run
- Add your API keys to enable cloud providers

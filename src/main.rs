use clap::{Parser, Subcommand};
use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::chat::{ChatMessage, request::ChatMessageRequest};
use ollama_rs::generation::images::Image;
use base64::Engine;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ollama-api-access")]
#[command(about = "A CLI tool to interact with Ollama API")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Chat with Ollama using a prompt
    Chat {
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
        
        /// The vision model to use (default: llama3.2-vision)
        #[arg(short, long, default_value = "llama3.2-vision")]
        model: String,
    },
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { prompt, prompt_file, model } => {
            let ollama = Ollama::default();
            
            // Get prompt from either --prompt or --prompt-file
            let prompt_text = match (prompt, prompt_file) {
                (Some(p), None) => p,
                (None, Some(file)) => fs::read_to_string(file)?,
                (Some(_), Some(_)) => return Err("Cannot specify both --prompt and --prompt-file".into()),
                (None, None) => return Err("Must specify either --prompt or --prompt-file".into()),
            };
            
            println!("🤖 Sending prompt to {} model...", model);
            
            let request = GenerationRequest::new(model, prompt_text);
            let response = ollama.generate(request).await?;
            
            println!("\n📝 Response:");
            println!("{}", response.response);
        }
        Commands::AnalyzeImage { prompt, prompt_file, image, image_dir, model } => {
            let ollama = Ollama::default();
            
            // Get prompt from either --prompt or --prompt-file
            let prompt_text = match (prompt, prompt_file) {
                (Some(p), None) => p,
                (None, Some(file)) => fs::read_to_string(file)?,
                (Some(_), Some(_)) => return Err("Cannot specify both --prompt and --prompt-file".into()),
                (None, None) => return Err("Must specify either --prompt or --prompt-file".into()),
            };
            
            // Get image paths from either --image or --image-dir
            let image_paths = match (image, image_dir) {
                (Some(img), None) => vec![img],
                (None, Some(dir)) => {
                    println!("🔍 Scanning directory: {}", dir.display());
                    get_image_files_from_directory(&dir)?
                },
                (Some(_), Some(_)) => return Err("Cannot specify both --image and --image-dir".into()),
                (None, None) => return Err("Must specify either --image or --image-dir".into()),
            };
            
            if image_paths.is_empty() {
                return Err("No image files found".into());
            }
            
            println!("🖼️  Analyzing {} image(s) with {} model...", image_paths.len(), model);
            
            // Process each image
            for (index, image_path) in image_paths.iter().enumerate() {
                println!("\n📁 Processing image {}/{}: {}", index + 1, image_paths.len(), image_path.display());
                
                // Read and encode image file
                let image_data = fs::read(&image_path)?;
                let image_base64 = base64::prelude::BASE64_STANDARD.encode(&image_data);
                
                // Create chat message with image
                let image = Image::from_base64(&image_base64);
                let messages = vec![
                    ChatMessage::user(prompt_text.clone()).with_images(vec![image])
                ];
                
                let request = ChatMessageRequest::new(model.clone(), messages);
                let response = ollama.send_chat_messages(request).await?;
                
                println!("📝 Analysis:");
                println!("{}", response.message.content);
                
                // Add separator between images if processing multiple
                if image_paths.len() > 1 && index < image_paths.len() - 1 {
                    println!("\n{}", "=".repeat(50));
                }
            }
        }
    }

    Ok(())
}

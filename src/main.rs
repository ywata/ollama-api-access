use clap::{Parser, Subcommand};
use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::generation::chat::{ChatMessage, request::ChatMessageRequest};
use ollama_rs::generation::images::Image;
use base64::Engine;
use std::fs;
use std::path::PathBuf;
use log::{info, warn, error, debug};

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
    }

    Ok(())
}

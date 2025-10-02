mod chat_ui;

use clap::{Parser, Subcommand};
use base64::Engine;
use std::fs;
use std::path::PathBuf;
use log::{info, warn, error, debug};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "ollama-api-access")]
#[command(about = "A CLI tool to interact with Ollama API")]
struct Cli {
    /// Path to config file (default: ~/.config/ollama-api-access/config.toml)
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,
    
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

/// Configuration for AI providers
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    #[serde(default)]
    providers: Providers,
    #[serde(default)]
    deployments: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
struct Providers {
    #[serde(default)]
    azure: Option<AzureConfig>,
    #[serde(default)]
    openai: Option<OpenAIConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct AzureConfig {
    endpoint: String,
    api_key: String,
    #[serde(default = "default_azure_api_version")]
    api_version: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OpenAIConfig {
    api_key: String,
}

fn default_azure_api_version() -> String {
    "2024-02-15-preview".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            providers: Providers::default(),
            deployments: HashMap::new(),
        }
    }
}

/// Parsed model specification
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub provider: String,
    pub model: String,
}

impl ModelSpec {
    /// Parse model string in format "provider:model" or just "model" (defaults to ollama)
    pub fn parse(model_str: &str) -> Self {
        if let Some((provider, model)) = model_str.split_once(':') {
            Self {
                provider: provider.to_string(),
                model: model.to_string(),
            }
        } else {
            Self {
                provider: "ollama".to_string(),
                model: model_str.to_string(),
            }
        }
    }

    /// Get the actual deployment name for Azure models
    pub fn resolve_deployment(&self, config: &Config) -> String {
        let key = format!("{}:{}", self.provider, self.model);
        config.deployments.get(&key)
            .cloned()
            .unwrap_or_else(|| self.model.clone())
    }
}

/// Load configuration from file
fn load_config(config_path: Option<PathBuf>) -> Result<Config, Box<dyn std::error::Error>> {
    let config_file = if let Some(ref path) = config_path {
        // Use the provided config path
        path.clone()
    } else {
        // Use default config directory
        let config_dir = dirs::config_dir()
            .ok_or("Could not determine config directory")?
            .join("ollama-api-access");
        config_dir.join("config.toml")
    };
    
    if config_file.exists() {
        let content = std::fs::read_to_string(&config_file)?;
        let config: Config = toml::from_str(&content)?;
        info!("Loaded config from: {}", config_file.display());
        Ok(config)
    } else {
        // Create default config file only if using default path
        if config_path.is_none() {
            let config_dir = config_file.parent()
                .ok_or("Could not determine config directory")?;
            std::fs::create_dir_all(config_dir)?;
            let default_config = Config::default();
            let content = toml::to_string_pretty(&default_config)?;
            std::fs::write(&config_file, content)?;
            
            info!("Created default config at: {}", config_file.display());
            info!("Please edit the config file to add your API keys and endpoints.");
            
            Ok(default_config)
        } else {
            Err(format!("Config file not found: {}", config_file.display()).into())
        }
    }
}

/// Create and configure genai client based on provider
fn create_genai_client(config: &Config) -> Result<genai::Client, Box<dyn std::error::Error>> {
    // For now, use environment variables for configuration
    // Set OPENAI_API_KEY, AZURE_OPENAI_API_KEY etc. from config if available
    
    unsafe {
        if let Some(openai_config) = &config.providers.openai {
            std::env::set_var("OPENAI_API_KEY", &openai_config.api_key);
        }
        
        if let Some(azure_config) = &config.providers.azure {
            std::env::set_var("AZURE_OPENAI_API_KEY", &azure_config.api_key);
            std::env::set_var("AZURE_OPENAI_ENDPOINT", &azure_config.endpoint);
            std::env::set_var("AZURE_OPENAI_API_VERSION", &azure_config.api_version);
        }
    }
    
    Ok(genai::Client::default())
}

/// Send a text query using genai
async fn send_genai_query(
    client: &genai::Client,
    model_spec: &ModelSpec,
    config: &Config,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    use genai::chat::{ChatMessage, ChatRequest};
    
    let model = model_spec.resolve_deployment(config);
    
    info!("🤖 Sending query to {}...", model);
    
    let chat_req = ChatRequest::new(vec![ChatMessage::user(prompt)]);
    
    let chat_res = client.exec_chat(&model, chat_req, None).await?;
    
    Ok(chat_res.content_text_as_str().unwrap_or("").to_string())
}

/// Send a vision query using genai
async fn send_genai_vision_query(
    client: &genai::Client,
    model_spec: &ModelSpec,
    config: &Config,
    prompt: &str,
    image_data: &[u8],
) -> Result<String, Box<dyn std::error::Error>> {
    use genai::chat::{ChatMessage, ChatRequest, ContentPart};
    
    let model = model_spec.resolve_deployment(config);
    
    info!("🖼️  Sending vision query to {}...", model);
    
    // Encode image to base64
    let image_base64 = base64::prelude::BASE64_STANDARD.encode(image_data);
    
    let chat_req = ChatRequest::new(vec![ChatMessage::user(vec![
        ContentPart::from_text(prompt),
        ContentPart::from_image_base64("image/jpeg", image_base64),
    ])]);
    
    let chat_res = client.exec_chat(&model, chat_req, None).await?;
    
    Ok(chat_res.content_text_as_str().unwrap_or("").to_string())
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
pub fn get_image_files_from_directory(dir: &PathBuf) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
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
    // Initialize logger
    env_logger::init();
    
    let cli = Cli::parse();

    // Load configuration
    let config = load_config(cli.config)?;
    let client = create_genai_client(&config)?;

    match cli.command {
        Commands::Query { prompt, prompt_file, model } => {
            // Get prompt from either --prompt or --prompt-file
            let prompt_text = match (prompt, prompt_file) {
                (Some(p), None) => p,
                (None, Some(file)) => fs::read_to_string(file)?,
                (Some(_), Some(_)) => return Err("Cannot specify both --prompt and --prompt-file".into()),
                (None, None) => return Err("Must specify either --prompt or --prompt-file".into()),
            };
            
            let model_spec = ModelSpec::parse(&model);
            let response = send_genai_query(&client, &model_spec, &config, &prompt_text).await?;
            
            info!("📝 Response:");
            println!("{}", response);
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
            
            let model_spec = ModelSpec::parse(&model);
            info!("🖼️  Analyzing {} image(s) with {}...", image_sources.len(), model_spec.resolve_deployment(&config));
            
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
                        match send_genai_vision_query(&client, &model_spec, &config, &prompt_text, &image_data).await {
                            Ok(response) => {
                                successful_count += 1;
                                info!("📝 Analysis:");
                                println!("{}", response);
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
            let model_spec = ModelSpec::parse(&model);
            info!("🚀 Starting chat with {}...", model_spec.resolve_deployment(&config));
            chat_ui::run_chat(model, client, config).await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for ModelSpec::parse
    #[test]
    fn test_model_spec_parse_with_provider() {
        let spec = ModelSpec::parse("azure:gpt-4");
        assert_eq!(spec.provider, "azure");
        assert_eq!(spec.model, "gpt-4");
    }

    #[test]
    fn test_model_spec_parse_without_provider() {
        let spec = ModelSpec::parse("llama3.2");
        assert_eq!(spec.provider, "ollama");
        assert_eq!(spec.model, "llama3.2");
    }

    #[test]
    fn test_model_spec_parse_with_multiple_colons() {
        let spec = ModelSpec::parse("openai:gpt-4:turbo");
        assert_eq!(spec.provider, "openai");
        assert_eq!(spec.model, "gpt-4:turbo");
    }

    #[test]
    fn test_model_spec_parse_empty_string() {
        let spec = ModelSpec::parse("");
        assert_eq!(spec.provider, "ollama");
        assert_eq!(spec.model, "");
    }

    #[test]
    fn test_model_spec_parse_only_colon() {
        let spec = ModelSpec::parse(":");
        assert_eq!(spec.provider, "");
        assert_eq!(spec.model, "");
    }

    #[test]
    fn test_model_spec_parse_provider_with_dash() {
        let spec = ModelSpec::parse("hugging-face:llama-2");
        assert_eq!(spec.provider, "hugging-face");
        assert_eq!(spec.model, "llama-2");
    }

    // Tests for ModelSpec::resolve_deployment
    #[test]
    fn test_resolve_deployment_with_mapping() {
        let mut config = Config::default();
        config.deployments.insert("azure:gpt-4".to_string(), "my-gpt4-deployment".to_string());
        
        let spec = ModelSpec {
            provider: "azure".to_string(),
            model: "gpt-4".to_string(),
        };
        
        assert_eq!(spec.resolve_deployment(&config), "my-gpt4-deployment");
    }

    #[test]
    fn test_resolve_deployment_without_mapping() {
        let config = Config::default();
        
        let spec = ModelSpec {
            provider: "ollama".to_string(),
            model: "llama3.2".to_string(),
        };
        
        assert_eq!(spec.resolve_deployment(&config), "llama3.2");
    }

    #[test]
    fn test_resolve_deployment_fallback_to_model_name() {
        let config = Config::default();
        
        let spec = ModelSpec {
            provider: "azure".to_string(),
            model: "gpt-4".to_string(),
        };
        
        // Should return model name when no deployment mapping exists
        assert_eq!(spec.resolve_deployment(&config), "gpt-4");
    }

    #[test]
    fn test_resolve_deployment_with_multiple_mappings() {
        let mut config = Config::default();
        config.deployments.insert("azure:gpt-4".to_string(), "deployment-1".to_string());
        config.deployments.insert("azure:gpt-3.5".to_string(), "deployment-2".to_string());
        config.deployments.insert("openai:gpt-4".to_string(), "deployment-3".to_string());
        
        let spec1 = ModelSpec::parse("azure:gpt-4");
        let spec2 = ModelSpec::parse("azure:gpt-3.5");
        let spec3 = ModelSpec::parse("openai:gpt-4");
        let spec4 = ModelSpec::parse("azure:gpt-4-turbo"); // Not mapped
        
        assert_eq!(spec1.resolve_deployment(&config), "deployment-1");
        assert_eq!(spec2.resolve_deployment(&config), "deployment-2");
        assert_eq!(spec3.resolve_deployment(&config), "deployment-3");
        assert_eq!(spec4.resolve_deployment(&config), "gpt-4-turbo");
    }

    // Tests for get_image_files_from_directory
    #[test]
    fn test_get_image_files_from_directory_with_images() {
        // Create a temporary directory with test files
        let temp_dir = std::env::temp_dir().join("ollama_test_images");
        let _ = std::fs::remove_dir_all(&temp_dir); // Clean up if exists
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        // Create test files
        let image_files = vec![
            temp_dir.join("test1.jpg"),
            temp_dir.join("test2.png"),
            temp_dir.join("test3.gif"),
            temp_dir.join("ignored.txt"),
            temp_dir.join("test4.JPEG"), // Test case sensitivity
        ];
        
        for file in &image_files {
            std::fs::write(file, b"test").unwrap();
        }
        
        let result = get_image_files_from_directory(&temp_dir).unwrap();
        
        // Should find 4 image files (excluding txt)
        assert_eq!(result.len(), 4);
        
        // Should be sorted
        assert!(result.windows(2).all(|w| w[0] <= w[1]));
        
        // Cleanup
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_get_image_files_from_directory_empty() {
        let temp_dir = std::env::temp_dir().join("ollama_test_empty");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        let result = get_image_files_from_directory(&temp_dir).unwrap();
        assert_eq!(result.len(), 0);
        
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_get_image_files_from_directory_not_a_directory() {
        let temp_file = std::env::temp_dir().join("ollama_test_file.txt");
        std::fs::write(&temp_file, b"test").unwrap();
        
        let result = get_image_files_from_directory(&temp_file);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a directory"));
        
        std::fs::remove_file(&temp_file).unwrap();
    }

    #[test]
    fn test_get_image_files_all_supported_extensions() {
        let temp_dir = std::env::temp_dir().join("ollama_test_extensions");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        let extensions = vec!["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif"];
        
        for (i, ext) in extensions.iter().enumerate() {
            let file = temp_dir.join(format!("test{}.{}", i, ext));
            std::fs::write(&file, b"test").unwrap();
        }
        
        let result = get_image_files_from_directory(&temp_dir).unwrap();
        assert_eq!(result.len(), 8);
        
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_get_image_files_mixed_case_extensions() {
        let temp_dir = std::env::temp_dir().join("ollama_test_case");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        // Create files with various case combinations
        let files = vec![
            temp_dir.join("test1.JPG"),
            temp_dir.join("test2.Png"),
            temp_dir.join("test3.GIF"),
            temp_dir.join("test4.TiFF"),
        ];
        
        for file in &files {
            std::fs::write(file, b"test").unwrap();
        }
        
        let result = get_image_files_from_directory(&temp_dir).unwrap();
        assert_eq!(result.len(), 4, "Should recognize case-insensitive extensions");
        
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_get_image_files_ignores_non_images() {
        let temp_dir = std::env::temp_dir().join("ollama_test_non_images");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        // Create a mix of image and non-image files
        let files = vec![
            ("test1.jpg", true),
            ("test2.txt", false),
            ("test3.pdf", false),
            ("test4.png", true),
            ("test5.doc", false),
            ("test6.gif", true),
        ];
        
        for (filename, _) in &files {
            let path = temp_dir.join(filename);
            std::fs::write(&path, b"test").unwrap();
        }
        
        let result = get_image_files_from_directory(&temp_dir).unwrap();
        assert_eq!(result.len(), 3, "Should only find image files");
        
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn test_get_image_files_nested_directories_ignored() {
        let temp_dir = std::env::temp_dir().join("ollama_test_nested");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        // Create file in root
        std::fs::write(temp_dir.join("root.jpg"), b"test").unwrap();
        
        // Create nested directory with file
        let nested = temp_dir.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("nested.jpg"), b"test").unwrap();
        
        let result = get_image_files_from_directory(&temp_dir).unwrap();
        // Should only find file in root directory, not in nested
        assert_eq!(result.len(), 1);
        assert!(result[0].file_name().unwrap() == "root.jpg");
        
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }
}

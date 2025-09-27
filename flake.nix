{
  description = "Rust project for Ollama API access";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rustfmt" "clippy" ];
        };

        nativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
          ollama
        ];

        buildInputs = with pkgs; [
          openssl
        ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
          pkgs.libiconv
          pkgs.darwin.apple_sdk.frameworks.Security
          pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          inherit buildInputs nativeBuildInputs;
          
          shellHook = ''
            echo "🦀 Rust development environment loaded!"
            echo "Rust version: $(rustc --version)"
            echo "Cargo version: $(cargo --version)"
            echo "Available tools: rustfmt, clippy, rust-analyzer"
          '';
        };

        # Uncomment when ready to build the package
        # packages.default = pkgs.rustPlatform.buildRustPackage {
        #   pname = "ollama-api-access";
        #   version = "0.1.0";
        #   
        #   src = ./.;
        #   
        #   cargoLock = {
        #     lockFile = ./Cargo.lock;
        #   };
        #   
        #   inherit nativeBuildInputs buildInputs;
        #   
        #   meta = with pkgs.lib; {
        #     description = "Rust application for Ollama API access";
        #     license = licenses.mit;
        #     maintainers = [ ];
        #   };
        # };
      });
}

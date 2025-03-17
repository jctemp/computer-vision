# NixOS and python
#   Hydra is not building packages that build against CUDA. Hence, scientific
#   computing is not really viable with Nix. Therefore, the templates provides
#   a basic shell.
{
  description = "Python flake";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachSystem ["x86_64-linux"] (
      system: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in {
        formatter = pkgs.alejandra;
        devShells.default =
          (pkgs.buildFHSUserEnv {
            name = "python hierarical environment";
            targetPkgs = pkgs: (let
              version = "311";
            in [
              pkgs.alejandra
              pkgs.uv

              pkgs."python${version}"
              pkgs."python${version}Packages".pip

              # pkgs.cudaPackages.cudatoolkit
              # (pkgs.cudaPackages.cudnn.override {autoAddDriverRunpath = pkgs.autoAddDriverRunpath;})

              (pkgs.vscode-with-extensions.override {
                vscodeExtensions = [
                  pkgs.vscode-extensions.ms-python.python
                  pkgs.vscode-extensions.ms-python.vscode-pylance
                  pkgs.vscode-extensions.charliermarsh.ruff
                ];
              })
            ]);
            runScript = "bash";
            profile = ''
              export PYTHONPATH="''${PYTHONPATH}:${inputs.self}"
            '';
          })
          .env;
      }
    );
}

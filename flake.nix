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

              # Essential standard library and headers
              pkgs.glibc
              pkgs.glibc.dev
              pkgs.glibc.static
              pkgs.binutils
              pkgs.binutils.bintools
              
              # Compiler toolchain - complete toolchain is important for crti.o
              pkgs.gcc
              pkgs.gcc-unwrapped
              pkgs.gcc-unwrapped.lib
              pkgs.gccStdenv
              pkgs.stdenv.cc.cc.lib
              pkgs.libgcc
              
              # C/C++ compiler tools
              pkgs.libclang
              pkgs.clang_multi
              pkgs.clang-tools
              pkgs.clang-manpages
              pkgs.clang-analyzer
              pkgs.gnumake
              pkgs.cmake
              
              # CUDA related
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.cuda_nvcc
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.cuda_cccl
              pkgs.cudaPackages.libcublas
              pkgs.cudaPackages.libcufft
              pkgs.cudaPackages.libcurand
              pkgs.cudaPackages.libcusolver
              pkgs.cudaPackages.libcusparse

              # Python
              pkgs."python${version}"
              pkgs."python${version}Packages".pip
              pkgs."python${version}Packages".setuptools
              pkgs."python${version}Packages".wheel

              (pkgs.vscode-with-extensions.override {
                vscodeExtensions = [
                  pkgs.vscode-extensions.ms-python.python
                  pkgs.vscode-extensions.ms-python.vscode-pylance
                  pkgs.vscode-extensions.charliermarsh.ruff

                  pkgs.vscode-extensions.ms-toolsai.jupyter
                  pkgs.vscode-extensions.ms-toolsai.vscode-jupyter-slideshow
                  pkgs.vscode-extensions.ms-toolsai.vscode-jupyter-cell-tags
                  pkgs.vscode-extensions.ms-toolsai.jupyter-renderers
                  pkgs.vscode-extensions.ms-toolsai.jupyter-keymap
                ];
              })
            ]);
            runScript = "bash";
            profile = ''
              export PYTHONPATH="''${PYTHONPATH}:${inputs.self}"
              
              # CUDA paths
              export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
              export CUDNN_HOME=${pkgs.cudaPackages.cudnn}
              export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
              export NVCC_PATH=${pkgs.cudaPackages.cuda_nvcc}/bin
              export PATH=$NVCC_PATH:${pkgs.cudaPackages.cudatoolkit}/bin:$PATH
              
              # Compiler paths
              export CC=${pkgs.gcc}/bin/gcc
              export CXX=${pkgs.gcc}/bin/g++
              
              # Library paths - critical for finding crti.o
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.cudaPackages.cuda_cccl}/lib:${pkgs.cudaPackages.libcublas}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libgcc}/lib:$LD_LIBRARY_PATH
              
              # Find crti.o and other startup files
              export LIBRARY_PATH=${pkgs.glibc}/lib:${pkgs.libgcc}/lib:${pkgs.gcc-unwrapped.lib}/lib:$LIBRARY_PATH
              
              # Include paths
              export C_INCLUDE_PATH=${pkgs.glibc.dev}/include:${pkgs.gcc-unwrapped}/include:$C_INCLUDE_PATH
              export CPLUS_INCLUDE_PATH=${pkgs.glibc.dev}/include:${pkgs.gcc-unwrapped}/include:$CPLUS_INCLUDE_PATH
              export CPATH=${pkgs.glibc.dev}/include:${pkgs.gcc-unwrapped}/include:$CPATH
              
              # CMAKE settings
              export CMAKE_PREFIX_PATH=${pkgs.cudaPackages.cudatoolkit}:$CMAKE_PREFIX_PATH
              
              # Help linker find the runtime files
              GCC_LIB_PATH=$(${pkgs.gcc}/bin/gcc -print-libgcc-file-name | xargs dirname)
              export LDFLAGS="-L$GCC_LIB_PATH $LDFLAGS"
              
              # This helps locate crti.o specifically
              echo "Using GCC library path: $GCC_LIB_PATH"
              export LIBRARY_PATH=$GCC_LIB_PATH:$LIBRARY_PATH
              
              # Verify crti.o exists and is accessible
              find $GCC_LIB_PATH -name "crti.o" || echo "WARNING: crti.o not found in GCC lib path"
              find ${pkgs.glibc}/lib -name "crti.o" || echo "WARNING: crti.o not found in glibc lib path"
            '';
          })
          .env;
      }
    );
}

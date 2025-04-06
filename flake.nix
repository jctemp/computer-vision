{
  description = "Minimal Python environment with CUDA in FHS";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = inputs: let
    systems = ["x86_64-linux"];
    eachSystem = systems: func: inputs.nixpkgs.lib.genAttrs systems (system: func system);
    eachDefaultSystem = eachSystem systems;
  in {
    formatter = eachDefaultSystem (system: inputs.nixpkgs.legacyPackages.${system}.alejandra);
    devShells = eachDefaultSystem (system: let
      pkgs = import inputs.nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      default = let
        python = pkgs.python312;
        cuda = pkgs.cudaPackages.cudatoolkit;
        cudnn = pkgs.cudaPackages.cudnn;
      in
        (pkgs.buildFHSUserEnv {
          name = "ml-fhs";
          targetPkgs = pkgs: (with pkgs; [
            # Basic tools
            bashInteractive
            git
            curl
            which

            # Build tools
            gcc
            binutils
            gnumake
            cmake
            ninja

            # Core libraries
            glib
            libGL
            stdenv.cc.cc.lib

            # Python and basic packages
            python
            python.pkgs.uv
            python.pkgs.pip

            # CUDA packages
            cuda
            cudnn
          ]);

          profile = ''
            # Set up compiler environment variables
            export CC=${pkgs.gcc}/bin/gcc
            export CXX=${pkgs.gcc}/bin/g++
            export CUDA_HOME=${cuda}
            export CUDNN_HOME=${cudnn}
            export CUDACXX=${cuda.cc}/bin/nvcc

            export LD_LIBRARY_PATH=/lib:/lib64:/usr/lib:/usr/lib64
            export LD_LIBRARY_PATH=${cuda}/lib:${cudnn}/lib:$LD_LIBRARY_PATH

            export PATH=$PATH:${cuda}/bin

            export UV_SYSTEM_PYTHON=${python}/bin/python
            export UV_PYTHONPATH=${python}/lib/python3.12/site-packages
            export PYTHONPATH=$PYTHONPATH:${python}/lib/python3.12/site-packages

            # Display environment info
            echo "ML FHS Environment activated"
            echo "Python: $(python --version)"
            echo "GCC: $(gcc --version | head -n 1)"
            echo "NVCC: $(nvcc --version | head -n 1)"
            echo "UV: $(uv --version)"
            echo "CUDA toolkit: $CUDA_HOME"

            # Helpful commands
            echo ""
            echo "UV Commands:"
            echo "  uv venv .venv              - Create virtual environment"
            echo "  uv pip install -r req.txt   - Install dependencies"
            echo "  uv pip install -e .         - Install package in dev mode"
            echo "  uv-init [project-name]      - Initialize ML project"
            echo "  cuda-test                   - Test CUDA availability"
          '';

          runScript = "bash";
        })
        .env;
    });
  };
}

{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        # CUDA only exists for x86_64 Linux (the GPU servers: spirit, oppy, karkinos).
        # On macOS (and aarch64-linux) we skip CUDA and let uv install CPU/MPS torch wheels.
        isLinux = nixpkgs.lib.hasSuffix "-linux" system;
        cudaSupport = system == "x86_64-linux";

        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = cudaSupport;
        };

        mpkgs = import inputs.nixpkgs_master {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = cudaSupport;
        };

        libList =
          [
            # Add needed packages here
            pkgs.stdenv.cc.cc
            pkgs.libGL
            pkgs.glib
            pkgs.zlib
          ]
          ++ pkgs.lib.optionals isLinux (
            with pkgs;
            [
              cudatoolkit

              # This is required for most app that uses graphics api
              # linuxPackages.nvidia_x11
            ]
          );
      in
      with pkgs;
      {
        devShells = {
          default =
            let
              # nixpkgs ships CUDA torch only on Linux; on Darwin python3.12-torch is
              # marked broken, so don't pull it from nix there - uv owns torch on Darwin.
              python_with_pkgs = pkgs.python312.withPackages (
                pp:
                pkgs.lib.optionals isLinux (
                  with pp;
                  [
                    # Add python pkgs here that you need from nix repos
                    torch-bin
                    torchvision-bin
                  ]
                )
              );
            in
            mkShell (
              {
                NIX_LD_LIBRARY_PATH = lib.makeLibraryPath libList;
                packages =
                  [
                    python_with_pkgs
                    python312Packages.venvShellHook
                    duckdb
                    uv
                  ]
                  ++ lib.optionals isLinux [ gcc ]
                  ++ libList;
                venvDir = "./.venv";
                postVenvCreation = ''
                  unset SOURCE_DATE_EPOCH
                '';
                postShellHook = ''
                  unset SOURCE_DATE_EPOCH
                '';
                shellHook = ''
                  export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
                  export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
                  ${lib.optionalString isLinux "export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}"}
                  runHook venvShellHook
                  export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
                  uv sync
                '';
              }
              // lib.optionalAttrs isLinux {
                # NIX_LD lets the dynamically-linked manylinux wheels uv installs find
                # their interpreter. It's an ELF/Linux concept; meaningless on Darwin.
                NIX_LD = runCommand "ld.so" { } ''
                  ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
                '';
              }
            );
        };
      }
    );
}

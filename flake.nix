{
  # pixi owns the Python environment (see pyproject.toml [tool.pixi]); there is
  # no uv here. This flake only provides a dev shell with `pixi` on PATH and,
  # on NixOS, the NVIDIA driver libs so conda's pytorch-gpu can load libcuda
  # (/run/opengl-driver/lib is not on the default search path there). On Ubuntu
  # (spirit) libcuda is already on the system path; on macOS none of this
  # applies. Pattern from shntnu/neusis templates/python-pixi.
  #
  #   GPU servers (Linux+CUDA):  pixi run snakemake -s train.smk ...
  #   macOS (Apple Silicon/MPS): pixi run -e osx snakemake -s train.smk ...
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell (
          {
            packages = [ pkgs.pixi ];
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            LD_LIBRARY_PATH = "/run/opengl-driver/lib";
          }
        );
      }
    );
}

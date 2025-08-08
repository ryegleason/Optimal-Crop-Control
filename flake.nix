{
  description = "Octave implementation of the soil CN model from Porporato et al.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
  in
  {
    devShells.x86_64-linux.default = pkgs.mkShell {
      packages = [
        (pkgs.octaveFull.withPackages(opkgs: with opkgs; [ statistics ]))
        pkgs.python312
        pkgs.python312Packages.jupyter
        pkgs.python312Packages.numpy
        pkgs.python312Packages.matplotlib
        pkgs.python312Packages.pandas
        pkgs.python312Packages.tqdm
      ];
    };
  };
}

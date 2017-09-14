with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "env";
  env = buildEnv { name = name; paths = buildInputs; };
  buildInputs = [
    python3Full
    python3Packages.check-manifest
    python3Packages.twine
    python3Packages.virtualenv
    python3Packages.wheel
  ];
}

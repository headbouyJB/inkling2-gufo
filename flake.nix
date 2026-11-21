{
  description = "Inkling 2 Mini extension on a pinned Gufo";
  # Pinned to Gufo v0.35.0 (0260e11). For a local Gufo checkout
  # use BUILD.md (the CMake route with GUFO_EXTENSION_DIRS), not this flake.
  inputs.gufo.url = "github:gufo-org/gufo/0260e1162f8713caee4d3f4ff77e017de7165f9c";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self ];
    };
  };
}

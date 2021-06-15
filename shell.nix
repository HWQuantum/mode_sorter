{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.hello
    (pkgs.python3.withPackages (ps: with ps; [ python-language-server pyls-mypy numpy scipy pytest ]))
    # keep this line if you use bash
    pkgs.bashInteractive
  ];
}

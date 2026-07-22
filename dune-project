(lang dune 3.15)

(generate_opam_files true)

(package
 (name autograd)
 (synopsis "autograd engine")
 (description
  "autograd in ocaml modelled after micrograd")
 (license GPL-3.0-or-later)
 (authors "autograd")
 (depends
  (ocaml (>= 5.0))
  dune))

let () =
  let open Autograd in
  let a = make_leaf 2.0 and b = make_leaf 3.0 and d = make_leaf 4.0 in
  let z = add (mul a b) (mul a d) in
  backward z;
  Printf.printf "autograd: d(a*b + a*d)/da = %g\n" a.grad

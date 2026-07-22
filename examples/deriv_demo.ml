(* Demo: z = a*b + a*d, print dz/da (should be b + d).  *)
let () =
  let open Autograd in
  let a = make_leaf 2.0 in
  let b = make_leaf 3.0 in
  let d = make_leaf 4.0 in
  let z = add (mul a b) (mul a d) in
  backward z;
  Printf.printf "z = a*b + a*d at a=2,b=3,d=4\n";
  Printf.printf "  z      = %g\n" z.vals;
  Printf.printf "  dz/da  = %g  (expected %g)\n" a.grad (b.vals +. d.vals);
  Printf.printf "  dz/db  = %g  (expected %g)\n" b.grad a.vals;
  Printf.printf "  dz/dd  = %g  (expected %g)\n" d.grad a.vals

(* Verify add/mul reverse-mode gradients against closed-form, and that
   the diamond case (shared parent) accumulates from both consumers.
   Run: dune exec test/test_value.exe *)
let eps = 1e-9

let approx a b =
  Float.abs (a -. b) <= eps *. (1.0 +. Float.max (Float.abs a) (Float.abs b))

let fails = ref 0

let check name got expected =
  if not (approx got expected) then begin
    incr fails;
    Printf.printf "FAIL %s: got %g, expected %g\n" name got expected
  end

open Autograd

let () =
  (* z = a*b + a*d  =>  dz/da = b + d, dz/db = a, dz/dd = a
     The diamond is the real test: a is shared by both addends, so
     its grad must accumulate from both before backward reaches it. *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 10.0) in
    let b = make_leaf (Random.float 10.0) in
    let d = make_leaf (Random.float 10.0) in
    let z = add (mul a b) (mul a d) in
    backward z;
    check "dz/da" a.grad (b.vals +. d.vals);
    check "dz/db" b.grad a.vals;
    check "dz/dd" d.grad a.vals;
    zero_grad z
  done;

  (* Chain rule depth: z = ((a*b)*a)*a = a^3 * b  =>  dz/da = 3*a^2*b *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 10.0) in
    let b = make_leaf (Random.float 10.0) in
    let z = mul (mul (mul a b) a) a in
    backward z;
    check "a^3*b dz/da" a.grad (3.0 *. a.vals *. a.vals *. b.vals);
    check "a^3*b dz/db" b.grad (a.vals *. a.vals *. a.vals);
    zero_grad z
  done;

  Printf.printf "done, %d failures\n" !fails;
  if !fails > 0 then exit 1

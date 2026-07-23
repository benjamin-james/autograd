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

  (* Check exp: z = (exp a) (exp b) *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 5.0) in
    let b = make_leaf (Random.float 5.0) in
    let z = add (exp a) (mul a b) in
    backward z;
    check "dz/da" a.grad ((Stdlib.exp a.vals) +. b.vals);
    check "dz/db" b.grad a.vals;
    zero_grad z
  done;
  (* Check exp: z=e^(e^a) for accumulation *)
  for _ = 1 to 100 do
    let a = make_leaf ((Random.float 4.0) -. 2.0) in
    let z = (exp (exp a)) in
    backward z;
    check "dz/da" a.grad (z.vals *. (Stdlib.exp a.vals));
    zero_grad z
  done;
  (* chain rule for exp, test multiple consumer on single var *)
  for _ = 1 to 100 do
    let a = make_leaf ((Random.float 4.0) -. 2.0) in
    let z = (mul a (exp a)) in
    backward z;
    check "dz/da" a.grad (z.vals +. (Stdlib.exp a.vals));
  done;
  (* test log a + log b*)
  for _ = 1 to 100 do
  let a = make_leaf ((Random.float 10.0) +. 1e-10) in
  let b = make_leaf ((Random.float 10.0) +. 1e-10) in
  let z = (add (log a) (log b)) in
  backward z;
  check "dz/da" a.grad (1.0 /. a.vals);
  check "dz/db" b.grad (1.0 /. b.vals);
  done;
  (* test relu *)
  for _ = 1 to 100 do
    let a = make_leaf ((Random.float 4.0) -. 2.0) in
    let z = (relu a) in
    backward z;
    check "dz/da" a.grad (if a.vals > 0. then 1. else 0.);
  done;
  (* subtraction, neg *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 10.0) in
    let b = make_leaf (Random.float 10.0) in
    let z = (sub (mul (neg a) a) b) in
    backward z;
    check "dz/da" a.grad (-2. *. a.vals);
    check "dz/db" b.grad (-1.)
  done;
  (* div: z = a / b  =>  dz/da = 1/b, dz/db = -a/b^2 *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 10.0) in
    let b = make_leaf (Random.float 10.0 +. 1.0) in
    let z = div a b in
    backward z;
    check "a/b dz/da" a.grad (1.0 /. b.vals);
    check "a/b dz/db" b.grad (-. a.vals /. (b.vals *. b.vals));
    zero_grad z
  done;
  (* div chain: z = (a / a) + (b / b) = 2  =>  grads should be 0 *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 10.0 +. 1.0) in
    let b = make_leaf (Random.float 10.0 +. 1.0) in
    let z = add (div a a) (div b b) in
    backward z;
    check "a/a dz/da" a.grad 0.0;
    check "b/b dz/db" b.grad 0.0;
    zero_grad z
  done;
  (* pow: z = a^3  =>  dz/da = 3 a^2 *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 5.0 +. 1.0) in
    let z = pow a 3 in
    backward z;
    check "a^3 dz/da" a.grad (3.0 *. a.vals *. a.vals);
    zero_grad z
  done;
  (* pow with shared base: z = a^2 * a^3 = a^5  =>  dz/da = 5 a^4 *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 5.0 +. 1.0) in
    let z = mul (pow a 2) (pow a 3) in
    backward z;
    check "a^2*a^3 dz/da" a.grad (5.0 *. a.vals ** 4.0);
    zero_grad z
  done;
  (* pow 0 and pow 1 edge cases *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 5.0 +. 1.0) in
    let z0 = pow a 0 in
    backward z0;
    check "a^0 dz/da" a.grad 0.0;
    zero_grad z0;
    let z1 = pow a 1 in
    backward z1;
    check "a^1 dz/da" a.grad 1.0;
    zero_grad z1
  done;
  (* sin: z = sin a + cos a  =>  dz/da = cos a - sin a *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 6.28) in
    let z = add (sin a) (cos a) in
    backward z;
    check "sin+cos dz/da" a.grad (Stdlib.cos a.vals -. Stdlib.sin a.vals);
    zero_grad z
  done;
  (* sin chain: z = sin(sin a)  =>  dz/da = cos(sin a) * cos a *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 6.28) in
    let z = sin (sin a) in
    backward z;
    check "sin(sin a) dz/da" a.grad (Stdlib.cos (Stdlib.sin a.vals) *. Stdlib.cos a.vals);
    zero_grad z
  done;
  (* cos chain: z = cos(a^2)  =>  dz/da = -sin(a^2) * 2a *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 3.0) in
    let z = cos (mul a a) in
    backward z;
    check "cos(a^2) dz/da" a.grad (-. Stdlib.sin (a.vals *. a.vals) *. (2.0 *. a.vals));
    zero_grad z
  done;
  (* tanh: z = tanh a  =>  dz/da = 1 - tanh^2 a *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 6.0 -. 3.0) in
    let z = tanh a in
    backward z;
    let t = Stdlib.tanh a.vals in
    check "tanh dz/da" a.grad (1.0 -. t *. t);
    zero_grad z
  done;
  (* tanh of sum: z = tanh(a + b)  =>  dz/da = dz/db = 1 - tanh^2(a+b) *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 4.0 -. 2.0) in
    let b = make_leaf (Random.float 4.0 -. 2.0) in
    let z = tanh (add a b) in
    backward z;
    let t = Stdlib.tanh (a.vals +. b.vals) in
    check "tanh(a+b) dz/da" a.grad (1.0 -. t *. t);
    check "tanh(a+b) dz/db" b.grad (1.0 -. t *. t);
    zero_grad z
  done;
  (* sigmoid: z = sigmoid a  =>  dz/da = s (1 - s) *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 8.0 -. 4.0) in
    let z = sigmoid a in
    backward z;
    let s = 1.0 /. (1.0 +. Stdlib.exp (-. a.vals)) in
    check "sigmoid dz/da" a.grad (s *. (1.0 -. s));
    zero_grad z
  done;
  (* sigmoid shared input: z = sigmoid a + a  =>  dz/da = s(1-s) + 1 *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 8.0 -. 4.0) in
    let z = add (sigmoid a) a in
    backward z;
    let s = 1.0 /. (1.0 +. Stdlib.exp (-. a.vals)) in
    check "sigmoid+a dz/da" a.grad (s *. (1.0 -. s) +. 1.0);
    zero_grad z
  done;
  (* sqrt: z = sqrt a  =>  dz/da = 1 / (2 sqrt a) *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 10.0 +. 1e-3) in
    let z = sqrt a in
    backward z;
    check "sqrt dz/da" a.grad (1.0 /. (2.0 *. Stdlib.sqrt a.vals));
    zero_grad z
  done;
  (* sqrt chain: z = sqrt(a*a) = |a|; for a>0 dz/da = 1 *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 10.0 +. 1e-3) in
    let z = sqrt (mul a a) in
    backward z;
    check "sqrt(a*a) dz/da" a.grad 1.0;
    zero_grad z
  done;
  (* mixed: z = sqrt(a^2 + b^2)  =>  dz/da = a / sqrt(a^2+b^2) *)
  for _ = 1 to 100 do
    let a = make_leaf (Random.float 5.0 +. 1e-3) in
    let b = make_leaf (Random.float 5.0 +. 1e-3) in
    let z = sqrt (add (pow a 2) (pow b 2)) in
    backward z;
    let r = Stdlib.sqrt (a.vals *. a.vals +. b.vals *. b.vals) in
    check "norm dz/da" a.grad (a.vals /. r);
    check "norm dz/db" b.grad (b.vals /. r);
    zero_grad z
  done;
  Printf.printf "done, %d failures\n" !fails;
  if !fails > 0 then exit 1

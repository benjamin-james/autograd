(* Central-difference gradient checker.
   Mutates each leaf in both +ve and -ve directions
   and compares each leaf's partial deriv by
   propogating the graph and reading the forward value
  *)
open Autograd

let h = 1e-6
let atol = 1e-6
let rtol = 1e-5
let passes = ref 0
let fails = ref 0

(* check builds the output, runs backward, and compares each
   leaf's analytic gradient against the central-difference estimate.
   A leaf passes when |analytic - fd| <= atol + rtol * max(|analytic|, |fd|). *)
let check ~name ~leaves f =
  let output = f leaves in
  backward output;
  let all_ok = ref true in
  List.iter
    (fun l ->
      let save = l.vals in
      l.vals <- save +. h;
      let out_plus = (f leaves).vals in
      l.vals <- save -. h;
      let out_minus = (f leaves).vals in
      l.vals <- save;
      let fd = (out_plus -. out_minus) /. (2.0 *. h) in
      let diff = Float.abs (l.grad -. fd) in
      let bound =
        atol +. (rtol *. Float.max (Float.abs l.grad) (Float.abs fd))
      in
      if diff > bound then begin
        incr fails;
        all_ok := false;
        Printf.printf
          "FAIL %s [leaf id=%d]: analytic=%g fd=%g |diff|=%g bound=%g\n" name
          l.id l.grad fd diff bound
      end
      else incr passes)
    leaves;
  !all_ok

(* argument expanders to ensure correct # is passed *)
let take1 = function [ a ] -> a | _ -> failwith "expected 1 leaf"
let take2 = function [ a; b ] -> (a, b) | _ -> failwith "expected 2 leaves"

let take3 = function
  | [ a; b; c ] -> (a, b, c)
  | _ -> failwith "expected 3 leaves"

type case = {
  name : string;
  arity : int;
  range : float * float;
  build : t list -> t;
}

let cases : case list =
  [
    {
      name = "a*b + a*d";
      arity = 3;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a, b, d = take3 xs in
          add (mul a b) (mul a d));
    };
    {
      name = "((a*b)*a)*a";
      arity = 2;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a, b = take2 xs in
          mul (mul (mul a b) a) a);
    };
    {
      name = "exp a + a*b";
      arity = 2;
      range = (-2.0, 2.0);
      build =
        (fun xs ->
          let a, b = take2 xs in
          add (exp a) (mul a b));
    };
    {
      name = "exp (exp a)";
      arity = 1;
      range = (-2.0, 2.0);
      build =
        (fun xs ->
          let a = take1 xs in
          exp (exp a));
    };
    {
      name = "a * exp a";
      arity = 1;
      range = (-2.0, 2.0);
      build =
        (fun xs ->
          let a = take1 xs in
          mul a (exp a));
    };
    {
      name = "log a + log b";
      arity = 2;
      range = (0.1, 5.0);
      build =
        (fun xs ->
          let a, b = take2 xs in
          add (log a) (log b));
    };
    {
      name = "(-a)*a - b";
      arity = 2;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a, b = take2 xs in
          sub (mul (neg a) a) b);
    };
    {
      name = "a / (exp b)";
      arity = 2;
      range = (-2.0, 2.0);
      build =
        (fun xs ->
          let a, b = take2 xs in
          div a (exp b));
    };
    {
      name = "a / a + b / b";
      arity = 2;
      range = (0.1, 5.0);
      build =
        (fun xs ->
          let a, b = take2 xs in
          add (div a a) (div b b));
    };
    {
      name = "a^3";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          pow a 3);
    };
    {
      name = "a^2 * a^3";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          mul (pow a 2) (pow a 3));
    };
    {
      name = "a^1";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          pow a 1);
    };
    {
      name = "a^0";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          pow a 0);
    };
    {
      name = "(relu a) / (exp a)";
      arity = 1;
      range = (-2.0, 2.0);
      build =
        (fun xs ->
          let a = take1 xs in
          div (relu a) (exp a));
    };
    {
      name = "sin a / a";
      arity = 1;
      range = (0.1, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          div (sin a) a);
    };
    {
      name = "sin a + cos a";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          add (sin a) (cos a));
    };
    {
      name = "sin (sin a)";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          sin (sin a));
    };
    {
      name = "cos(a*a)";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          cos (mul a a));
    };
    {
      name = "tanh a";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          tanh a);
    };
    {
      name = "tanh a+b";
      arity = 2;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a, b = take2 xs in
          tanh (add a b));
    };
    {
      name = "sigmoid a";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          sigmoid a);
    };
    {
      name = "sigmoid a + a";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          add (sigmoid a) a);
    };
    {
      name = "sqrt a";
      arity = 1;
      range = (0.1, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          sqrt a);
    };
    {
      name = "sqrt a*a";
      arity = 1;
      range = (0.1, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          sqrt (mul a a));
    };
    {
      name = "sqrt (exp a)";
      arity = 1;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a = take1 xs in
          sqrt (exp a));
    };
    {
      name = "sqrt(a^2 + b^2)";
      arity = 2;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a, b = take2 xs in
          sqrt (add (pow a 2) (pow b 2)));
    };
    {
      name = "tanh(a*b + sigmoid c) * sqrt(a^2+b^2)";
      arity = 3;
      range = (-5.0, 5.0);
      build =
        (fun xs ->
          let a, b, c = take3 xs in
          mul
            (tanh (add (mul a b) (sigmoid c)))
            (sqrt (add (pow a 2) (pow b 2))));
    };
  ]

let () =
  Random.self_init ();
  List.iter
    (fun c ->
      for _ = 1 to 50 do
        let leaves =
          List.init c.arity (fun _ ->
              let lo, hi = c.range in
              make_leaf (lo +. Random.float (hi -. lo)))
        in
        let name = c.name in
        ignore (check ~name ~leaves c.build)
      done)
    cases;
  Printf.printf "fd: %d passed, %d failed\n" !passes !fails;
  if !fails > 0 then exit 1

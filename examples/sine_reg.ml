(* Sine regression: fit y = sin(x) on [-pi, pi] with a 1->16->1 MLP.
   Harder than XOR (continuous target, more data points), but still
   small enough to train in a few thousand steps of scalar SGD. *)
open Autograd

let n_samples = 40
let n_epochs = 3000

let () =
  Random.self_init ();
  let rng () = Random.float 1.0 -. 0.5 in
  let net = Nn.make_mlp ~act:tanh ~rng [| 1; 16; 1 |] in
  let opt = Nn.make_sgd ~lr:0.05 ~momentum:0.9 net.params in
  (* dataset: y = sin(x), x in [-pi, pi]. Compute raw (no value) for now. *)
  let pi = 4.0 *. Stdlib.atan 1.0 in
  let xs =
    Array.init n_samples (fun i ->
        -.pi +. (float_of_int i *. (2.0 *. pi /. float_of_int (n_samples - 1))))
  in
  let ys = Array.map Stdlib.sin xs in
  let make_input x = [| make_leaf x |] in
  let predict x = (Nn.mlp_forward net (make_input x)).(0) in
  let loss_at () =
    let preds = Array.map predict xs in
    let targets = Array.map make_leaf ys in
    Nn.mse preds targets
  in
  let before = (loss_at ()).vals in
  for _ = 1 to n_epochs do
    let preds = Array.map predict xs in
    let targets = Array.map make_leaf ys in
    let loss = Nn.mse preds targets in
    backward loss;
    Nn.sgd_step opt
  done;
  let after = (loss_at ()).vals in
  Printf.printf "1-16-1 tanh MLP, sine regression (%d samples, %d epochs)\n"
    n_samples n_epochs;
  Printf.printf "  loss before = %g\n" before;
  Printf.printf "  loss after  = %g\n" after;
  (* print a few sample predictions vs targets *)
  Printf.printf "  sample predictions:\n";
  for i = 0 to n_samples - 1 do
    if i mod 8 = 0 then
      Printf.printf "    x=%6.3f  target=%6.3f  pred=%6.3f\n" xs.(i) ys.(i)
        (predict xs.(i)).vals
  done;
  if after < before then Printf.printf "  OK: loss decreased\n"
  else (
    Printf.printf "  FAIL: loss did not decrease\n";
    exit 1)

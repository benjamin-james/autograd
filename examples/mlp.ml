open Autograd

let () =
  Random.self_init ();
  let rng () = Random.float 1.0 -. 0.5 in
  let net = Nn.make_mlp ~act:gelu ~rng [| 2; 4; 1 |] in
  let opt = Nn.make_sgd ~lr:0.1 ~momentum:0.9 net.params in
  (* learn an XOR *)
  let xs =
    [| [| 0.0; 0.0 |]; [| 0.0; 1.0 |]; [| 1.0; 0.0 |]; [| 1.0; 1.0 |] |]
  in
  let ys = [| 0.0; 1.0; 1.0; 0.0 |] in
  let inputs = Array.map (Array.map make_leaf) xs in
  let targets = Array.map make_leaf ys in
  let predict_all () = Array.map (fun x -> (Nn.mlp_forward net x).(0)) inputs in
  let loss_at () = Nn.mse (predict_all ()) targets in
  let before = (loss_at ()).vals in
  for _ = 1 to 500 do
    let preds = predict_all () in
    let loss = Nn.mse preds targets in
    backward loss;
    Nn.sgd_step opt
  done;
  let after = (loss_at ()).vals in
  Printf.printf "2-4-1 XOR MLP\n";
  Printf.printf "  loss before = %g\n" before;
  Printf.printf "  loss after  = %g\n" after;
  if after < before then Printf.printf "  OK: loss decreased\n"
  else (
    Printf.printf "  FAIL: loss did not decrease\n";
    exit 1)

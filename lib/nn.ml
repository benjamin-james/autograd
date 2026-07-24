open Value

type param = t

(* neurons *)
type neuron = { weights : param array; bias : param; activation : t -> t }

let gelu x =
  mul (make_leaf 0.5)
    (mul x
       (add (make_leaf 1.0)
          (tanh
             (mul
                (sqrt (div (make_leaf 2.0) (make_leaf 3.14159265358)))
                (add x (mul (pow x 3) (make_leaf 0.044715)))))))

let silu x = mul x (sigmoid x)

let make_neuron ?(act = fun x -> x) ~(rng : unit -> float) n_in =
  {
    weights = Array.init n_in (fun _ -> make_leaf (rng ()));
    bias = make_leaf (rng ());
    activation = act;
  }

let neuron_forward n inputs =
  assert (Array.length inputs = Array.length n.weights);
  let acc = ref n.bias in
  for i = 0 to Array.length n.weights - 1 do
    acc := add !acc (mul n.weights.(i) inputs.(i))
  done;
  n.activation !acc

(* layers *)
type layer = { neurons : neuron array }

let make_layer ?act ~rng n_in n_out =
  { neurons = Array.init n_out (fun _ -> make_neuron ?act ~rng n_in) }

let layer_forward l inputs =
  Array.map (fun n -> neuron_forward n inputs) l.neurons

(* MLP *)
type mlp = { layers : layer array; params : param list (* cached *) }

let make_mlp ?act ~rng sizes =
  let layers =
    Array.init
      (Array.length sizes - 1)
      (fun i -> make_layer ?act ~rng sizes.(i) sizes.(i + 1))
  in
  let params =
    Array.to_list layers
    |> List.concat_map (fun l ->
        Array.to_list l.neurons
        |> List.concat_map (fun n -> n.bias :: Array.to_list n.weights))
  in
  { layers; params }

let mlp_forward mlp inputs =
  Array.fold_left (fun accum l -> layer_forward l accum) inputs mlp.layers

let mse preds targets =
  assert (Array.length preds = Array.length targets);
  let n = float_of_int (Array.length preds) in
  Array.fold_left
    (fun accum (p, t) ->
      let diff = sub p t in
      add accum (mul diff diff))
    (make_leaf 0.0)
    (Array.map2 (fun p t -> (p, t)) preds targets)
  |> fun sum -> mul sum (make_leaf (1. /. n))

let sgd_step ?(lr = 0.01) params =
  List.iter (fun p -> p.vals <- p.vals -. (lr *. p.grad)) params;
  List.iter (fun p -> p.grad <- 0.) params

open Value

type param = t

let gelu x =
  mul (make_leaf 0.5)
    (mul x
       (add (make_leaf 1.0)
          (tanh
             (mul
                (sqrt (div (make_leaf 2.0) (make_leaf 3.14159265358)))
                (add x (mul (pow x 3) (make_leaf 0.044715)))))))

let silu x = mul x (sigmoid x)

module type MODULE = sig
  type t

  val forward : t -> Value.t array -> Value.t array
  val parameters : t -> param list
end

(* neurons *)
type neuron = {
  weights : param array;
  bias : param;
  activation : Value.t -> Value.t;
}

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

let neuron_parameters n = n.bias :: Array.to_list n.weights

(* layers *)
type layer = { neurons : neuron array }

let make_layer ?act ~rng n_in n_out =
  { neurons = Array.init n_out (fun _ -> make_neuron ?act ~rng n_in) }

let layer_forward l inputs =
  Array.map (fun n -> neuron_forward n inputs) l.neurons

let layer_parameters l =
  Array.to_list l.neurons |> List.concat_map neuron_parameters

module Linear = struct
  type t = layer

  let forward = layer_forward
  let parameters = layer_parameters
end

(* Sequential: compose modules in order.
   Uses any list of moudles that shared Value.t array interface *)
module Sequential (M : MODULE) = struct
  type t = M.t array

  let forward modules inputs =
    Array.fold_left (fun acc m -> M.forward m acc) inputs modules

  let parameters modules = Array.to_list modules |> List.concat_map M.parameters
end

(* MLP *)
type mlp = { layers : layer array; params : param list (* cached *) }

let make_mlp ?act ~rng sizes =
  let layers =
    Array.init
      (Array.length sizes - 1)
      (fun i -> make_layer ?act ~rng sizes.(i) sizes.(i + 1))
  in
  let params = Array.to_list layers |> List.concat_map layer_parameters in
  { layers; params }

let mlp_forward mlp inputs =
  Array.fold_left (fun accum l -> layer_forward l accum) inputs mlp.layers

module MLP = struct
  type t = mlp

  let forward = mlp_forward
  let parameters m = m.params
end

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

(* SGD with momentum.
   Mutable velocity buffers, one per param,
   so it has to be reused across steps.
   Params ref. by identity (same Value.t refs in the network)
   so mutations are visible to the network immediately. 
   *)
type sgd = {
  lr : float;
  momentum : float;
  velocity : (param, float) Hashtbl.t;
  params : param list;
}

let make_sgd ?(lr = 0.01) ?(momentum = 0.0) params =
  { lr; momentum; velocity = Hashtbl.create (List.length params); params }

let sgd_step opt =
  List.iter
    (fun p ->
      let v =
        match Hashtbl.find_opt opt.velocity p with Some v -> v | None -> 0.0
      in
      let v' = (opt.momentum *. v) -. (opt.lr *. p.grad) in
      Hashtbl.replace opt.velocity p v';
      p.vals <- p.vals +. v')
    opt.params;
  List.iter (fun p -> p.grad <- 0.0) opt.params

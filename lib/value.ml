(* Minimal reverse-mode autograd: one float per node, add and mul only.

   Mirrors the first ~50 lines of karpathy micrograd. Each node records how it
   was built (ADD or MUL of two parents, or a leaf). [backward] walks
   the graph once in reverse-topological order, pushing gradients to
   parents via the chain rule. *)

type t = { id : int; mutable vals : float; mutable grad : float; op : op }
and op = ADD of t * t | MUL of t * t
  | EXP of t | LOG of t | SQRT of t
  | NEG of t | SUB of t * t
  | SIN of t | COS of t
  | TANH of t | SIGMOID of t
  | DIV of t * t
  | POW of t * int
  | RELU of t * bool | LEAF

let _id = ref 0

let make_leaf x =
  {
    id =
      (incr _id;
       !_id);
    vals = x;
    grad = 0.0;
    op = LEAF;
  }

let make_node x op =
  {
    id =
      (incr _id;
       !_id);
    vals = x;
    grad = 0.0;
    op;
  }

let add a b = make_node (a.vals +. b.vals) (ADD (a, b))
let mul a b = make_node (a.vals *. b.vals) (MUL (a, b))
let exp a = make_node (Stdlib.exp a.vals) (EXP (a))
let log a = make_node (Stdlib.log a.vals) (LOG (a))
let pow a n = make_node (a.vals ** float_of_int n) (POW (a, n))
let neg a = make_node (-. a.vals) (NEG (a))
let sub a b = make_node (a.vals -. b.vals) (SUB (a, b))
let div a b = make_node (a.vals /. b.vals) (DIV (a, b))
let sin a = make_node (Stdlib.sin a.vals) (SIN a)
let cos a = make_node (Stdlib.cos a.vals) (COS a)
let tanh a = make_node (Stdlib.tanh a.vals) (TANH a)
let sigmoid a = make_node (1. /. (1. +. Stdlib.exp (-. a.vals))) (SIGMOID a)
let sqrt a = make_node (Stdlib.sqrt a.vals) (SQRT a)

let relu a = make_node (if a.vals > 0.0 then a.vals else 0.0)
  (RELU (a, a.vals > 0.0))

let parents = function ADD (a, b) | MUL (a, b) | SUB (a, b) | DIV (a, b) -> [ a; b ]
  | EXP (a) | LOG (a) | NEG(a) | SIN(a) | COS(a) | TANH (a) | SIGMOID(a) | SQRT(a) | POW(a, _) | RELU(a, _) -> [a]
  | LEAF -> []

(* Post-order DFS: parents visited after all their consumers, so a
   shared parent's grad is fully accumulated before its own backward
   runs. Each node visited once. *)
let reverse_topo root =
  let seen = Hashtbl.create 64 in
  let acc = ref [] in
  let rec walk n =
    if not (Hashtbl.mem seen n.id) then begin
      Hashtbl.add seen n.id ();
      List.iter walk (parents n.op);
      acc := n :: !acc
    end
  in
  walk root;
  !acc

let backward root =
  root.grad <- 1.0;
  List.iter
    (fun n ->
      match n.op with
      | ADD (a, b) ->
          a.grad <- a.grad +. n.grad;
          b.grad <- b.grad +. n.grad
      | MUL (a, b) ->
          a.grad <- a.grad +. (b.vals *. n.grad);
          b.grad <- b.grad +. (a.vals *. n.grad)
      | SUB (a, b) ->
        a.grad <- a.grad +. n.grad;
        b.grad <- b.grad -. n.grad
      | DIV (a, b) ->
        a.grad <- a.grad +. (n.grad /. b.vals);
        b.grad <- b.grad -. (n.grad *. n.vals /. b.vals)
      | EXP (a) ->
        a.grad <- a.grad +. (n.vals *. n.grad)
      | LOG (a) ->
        a.grad <- a.grad +. (n.grad /. a.vals)
      | NEG (a) ->
        a.grad <- a.grad -. n.grad
      | SIN(a) ->
        a.grad <- a.grad +. n.grad *. (Stdlib.cos a.vals)
      | COS(a) ->
        a.grad <- a.grad -. n.grad *. (Stdlib.sin a.vals)
      | TANH(a) ->
        a.grad <- a.grad +. n.grad *. (1. -. n.vals *. n.vals)
      | SIGMOID(a) ->
        a.grad <- a.grad +. n.grad *. n.vals *. (1. -. n.vals)
      | POW (a, k) ->
        a.grad <- a.grad +. (if k <> 0 then (float_of_int k) *. (a.vals ** float_of_int (k-1)) *. n.grad else 0.0)
      | SQRT (a) ->
        a.grad <- a.grad +. n.grad /. (2. *. n.vals)
      | RELU (a, mask) -> a.grad <- a.grad +. (if mask then n.grad else 0.0)
      | LEAF -> ())
    (reverse_topo root)

let zero_grad root = List.iter (fun n -> n.grad <- 0.0) (reverse_topo root)

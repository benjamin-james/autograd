(* Minimal reverse-mode autograd: one float per node, add and mul only.

   Mirrors the first ~50 lines of karpathy micrograd. Each node records how it
   was built (ADD or MUL of two parents, or a leaf). [backward] walks
   the graph once in reverse-topological order, pushing gradients to
   parents via the chain rule. *)

type t = { id : int; mutable vals : float; mutable grad : float; op : op }
and op = ADD of t * t | MUL of t * t | EXP of t | LEAF

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

let parents = function ADD (a, b) | MUL (a, b) -> [ a; b ] | LEAF -> [] | EXP (a) -> [a]

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
      | EXP (a) ->
        a.grad <- a.grad +. (n.vals *. n.grad)
      | LEAF -> ())
    (reverse_topo root)

let zero_grad root = List.iter (fun n -> n.grad <- 0.0) (reverse_topo root)

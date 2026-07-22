(* Public facade: re-export [Value] so consumers can [open Autograd]
   and call [make_leaf], [add], [mul], [backward], ... unqualified.
 *)
include Value

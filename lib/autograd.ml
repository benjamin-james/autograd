(* Public facade: re-export [Value] so consumers can [open Autograd]
   and call [make_leaf], [add], [mul], [backward], ... unqualified.
 *)
include Value

let gelu = Nn.gelu
let silu = Nn.silu

module Nn = struct
  include Nn
end

module TensorOperationsCore

#===========================================================================================
    Backends
===========================================================================================#

export AbstractBackend

"""
    AbstractBackend

Supertype for selecting different implementation backends for this interface.
"""
abstract type AbstractBackend end

#===========================================================================================
    Interface
===========================================================================================#

using VectorInterface: scalartype
export tensoradd!, tensorcontract!, tensortrace!, tensorscalar, scalartype
export tensorstructure, tensoradd_structure, tensoradd_type, tensorcontract_structure,
       tensorcontract_type, tensoralloc, tensorfree!
export tensorcost, checkcontractible

include("interface.jl")

#===========================================================================================
    Index Operations
===========================================================================================#

export IndexTuple, Index2Tuple, linearize
export IndexError

include("indices.jl")

end

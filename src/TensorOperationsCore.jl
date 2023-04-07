module TensorOperationsCore

# ---------------------------------------------------------------------------------------- #
# Backends
# ---------------------------------------------------------------------------------------- #

export Backend, DefaultBackend
export contractbackend, contractbackend!
export addbackend, addbackend!
export tracebackend, tracebackend!
export allocatebackend, allocatebackend!, allocatetempbackend, allocatetempbackend!

include("backends.jl")

# ---------------------------------------------------------------------------------------- #
# Interface
# ---------------------------------------------------------------------------------------- #

using VectorInterface: scalartype
export tensoradd!, tensorcontract!, tensortrace!, tensorscalar, scalartype
export tensoralloc, tensoralloctemp, tensorfree!
export tensorcost, checkcontractible

include("interface.jl")

# ---------------------------------------------------------------------------------------- #
# Index operations
# ---------------------------------------------------------------------------------------- #

export IndexTuple, Index2Tuple, linearize
export IndexError

const IndexTuple{N} = NTuple{N,Int}
const Index2Tuple{N₁,N₂} = Tuple{IndexTuple{N₁},IndexTuple{N₂}}
linearize(p::Index2Tuple) = (p[1]..., p[2]...)

"""
    struct IndexError{<:AbstractString} <: Exception
    
exception type for reporting errors in the index specification.
"""
struct IndexError{S<:AbstractString} <: Exception
    msg::S
end

end

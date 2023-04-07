"""
    Backend

Supertype for selecting different implementation backends for this interface.
"""
abstract type Backend end

struct DefaultBackend <: Backend end

"""
    contractbackend(tensortype)

Get the default backend used for the contraction of two tensors.
"""
contractbackend(args...) = DefaultBackend()

"""
    contractbackend!(backend, tensortype)

Set the default backend used for the contraction of two tensors.
"""
function contractbackend!(nback::Backend, datatype=Any)
    @eval TensorOperationsCore TensorOperationsCore.contractbackend(::$datatype) = $nback
end

"""
    addbackend(tensortype)

Get the default backend used for the addition of two tensors.
"""
addbackend(args...) = DefaultBackend()

"""
    addbackend!(backend, tensortype)

Set the default backend used for the addition of two tensors.
"""
function addbackend!(nback::Backend, datatype=Any)
    @eval TensorOperationsCore TensorOperationsCore.addbackend(::$datatype) = $nback
end

"""
    tracebackend(tensortype)

Get the default backend used for the partial trace of a tensor
"""
tracebackend(args...) = DefaultBackend()

"""
    tracebackend!(backend, tensortype)

Set the default backend used for the partial trace of a tensor
"""
function tracebackend!(nback::Backend, datatype=Any)
    @eval TensorOperationsCore TensorOperationsCore.tracebackend(::$datatype) = $nback
end

"""
    allocatebackend(tensortype)

Get the default backend for the allocation of a tensor.
"""
allocatebackend(args...) = DefaultBackend()

"""
    allocatebackend!(backend, tensortype)

Set the default backend used for the allocation of a tensor.
"""
function allocatebackend!(nback::Backend, datatype=Any)
    @eval TensorOperationsCore TensorOperationsCore.allocatebackend(::$datatype) = $nback
end

"""
    allocatetempbackend(tensortype)

Get the default backend for the allocation of a temporary tensor.
"""
allocatetempbackend(args...) = DefaultBackend()

"""
    allocatetempbackend!(backend, tensortype)

Set the default backend used for the allocation of a temporary tensor.
"""
function allocatetempbackend!(nback::Backend, datatype=Any)
    @eval TensorOperationsCore TensorOperationsCore.allocatetempbackend(::$datatype) = $nback
end
#===========================================================================================
    Operations
===========================================================================================#

"""
    tensoradd!(backend, C, A, pA, conjA, α, β)

Implements `C = β * C + α * permutedims(opA(A), pA)` without creating the intermediate
temporary.  The operation `opA` acts as `conj` if `conjA` equals `:C` or as the identity if
`conjA` equals `:N`.
"""
function tensoradd! end

"""
    tensorcontract!(backend, C, pC, A, pA, conjA, B, pB, conjB, α, β)

Implements `C = β * C + α * permutedims(contract(opA(A), opB(B)), pC)` without creating the
intermediate temporary, where `A` and `B` are contracted such that the indices `pA[2]` of
`A` are contracted with indices `pB[1]` of `B`. The remaining indices `(pA[1]..., pB[2]...)`
are then permuted according to `pC`. The operation `opA` (`opB`) acts as `conj` if `conjA`
(`conjB`) equals `:C` or as the identity if `conjA` (`conjB`) equals `:N`.
"""
function tensorcontract! end

"""
    tensortrace!(backend, C, pC, A, pA, conjA, α, β)

Implements `C = β * C + α * permutedims(partialtrace(opA(A)), pC)` without creating the
intermediate temporary, where `A` is partially traced, such that indices in `pA[1]` are
contracted with indices in `pA[2]`, and the remaining indices are permuted according
to `pC`. The operation `opA` acts as `conj` if `conjA` equals `:C` or as the identity if
`conjA` equals `:N`.
"""
function tensortrace! end

"""
    tensorscalar(C)

Returns the single element of a tensor-like object with zero indices or dimensions.
"""
function tensorscalar end

#===========================================================================================
    Allocations
===========================================================================================#

"""
    tensorstructure(A, iA, conjA)

Obtain the information associated to indices `iA` of tensor `op(A)`, where `op` acts as
`conj` when `conjA` is `:C`, or as the identity if `conjA` is `:N`.
"""
function tensorstructure end

"""
    tensoralloc(backend, TC, pC, A, conjA)
    tensoralloc(backend, TC, pC, A, iA, conjA, B, iB, conjB)

Allocate memory for a tensor with indices `pC` and scalartype `TC` based on the indices of
`opA(A)`, or based on indices `iA` of `opA(A)` and `iB` of `opB(B)`. The operation `opA` 
(`opB`) acts as `conj` if `conjA` (`conjB`) equals `:C` or as the identity if `conjA`
(`conjB`) equals `:N`.
"""
function tensoralloc end

"""
    tensoralloctemp(backend, TC, pC, A, conjA)
    tensoralloctemp(backend, TC, pC, A, iA, conjA, B, iB, conjB)

Allocate memory for an intermediary tensor, with indices `pC` and scalartype `TC` based on
the indices of `opA(A)`, or based on indices `iA` of `opA(A)` and `iB` of `opB(B)`. The
operation `opA` (`opB`) acts as `conj` if `conjA` (`conjB`) equals `:C` or as the identity
if `conjA` (`conjB`) equals `:N`.
"""
function tensoralloctemp end

"""
    tensorfree!(backend, C)

Release the allocated memory of `C`.
"""
function tensorfree! end

#===========================================================================================
    Utility
===========================================================================================#

"""
    tensorcost(A, i)
    
Computes the contraction cost associated with the `i`th index of a tensor, such that the
total cost of a pairwise contraction is found as the product of the costs of all contracted
indices and all uncontracted indices.
"""
function tensorcost end

"""
    checkcontractible(A, iA, conjA, B, iB, conjB, label)
Verifies whether two tensors `A` and `B` are compatible for contraction, and throws an error
if not.
"""
function checkcontractible end

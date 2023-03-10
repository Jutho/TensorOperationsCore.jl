
"""
    tensoradd!(C, A, pA, conjA, α, β)

Implements `C = β * C + α * permutedims(opA(A), pA)` without creating the intermediate
temporary.  The operation `opA` acts as `conj` if `conjA` equals `:C` or as the identity if
`conjA` equals `:N`.
"""
function tensoradd! end
tensoradd!(C, A, pA, conjA, α, β) = tensoradd!(addbackend(typeof(C)), C, A, pA, conjA, α, β)

"""
    tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β)

Implements `C = β * C + α * permutedims(contract(opA(A), opB(B)), pC)` without creating the
intermediate temporary, where `A` and `B` are contracted such that the indices `pA[2]` of
`A` are contracted with indices `pB[1]` of `B`. The remaining indices `(pA[1]..., pB[2]...)`
are then permuted according to `pC`. The operation `opA` (`opB`) acts as `conj` if `conjA`
(`conjB`) equals `:C` or as the identity if `conjA` (`conjB`) equals `:N`.
"""
function tensorcontract! end
function tensorcontract!(C, pC, A, pA, conjA, B, pB, conjB, α, β)
    return tensorcontract!(contractbackend(typeof(C)), C, pC, A, pA, conjA, B, pB, conjB,
                           α, β)
end

"""
    tensortrace!(C, pC, A, pA, conjA, α, β)

Implements `C = β * C + α * permutedims(partialtrace(opA(A)), pC)` without creating the
intermediate temporary, where `A` is partially traced, such that indices in `pA[1]` are
contracted with indices in `pA[2]`, and the remaining indices are permuted according
to `pC`. The operation `opA` acts as `conj` if `conjA` equals `:C` or as the identity if
`conjA` equals `:N`.
"""
function tensortrace! end
function tensortrace!(C, pC, A, pA, conjA, α, β)
    return tensortrace!(tracebackend(typeof(C)), C, pC, A, pA, conjA, α, β)
end

"""
    tensorscalar(C)

Returns the single element of a tensor-like object with zero indices or dimensions.
"""
function tensorscalar end

"""
    tensoralloc(TC, pC, A, conjA)
    tensoralloc(TC, pC, A, iA, conjA, B, iB, conjB)

Allocate memory for a tensor with indices `pC` and scalartype `TC` based on the indices of
`opA(A)`, or based on indices `iA` of `opA(A)` and `iB` of `opB(B)`. The operation `opA` 
(`opB`) acts as `conj` if `conjA` (`conjB`) equals `:C` or as the identity if `conjA`
(`conjB`) equals `:N`.
"""
function tensoralloc end
tensoralloc(TC, pC, A, conjA) = tensoralloc(allocatebackend(typeof(A)), TC, pC, A, conjA)
function tensoralloc(TC, pC, A, iA, conjA, B, iB, conjB)
    return tensoralloc(allocatebackend(typeof(A)), TC, pC, A, iA, conjA, B, iB, conjB)
end

"""
    tensorfree(C)

Release the allocated memory of `C`.
"""
function tensorfree end
tensorfree(C) = tensorfree(allocatebackend(typeof(C)), C)

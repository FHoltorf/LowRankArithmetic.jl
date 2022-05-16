export TwoFactorRepresentation
"""
    X = U*Z'

    Two factor factorization of a low rank matrix into factors U ∈ ℝⁿˣʳ and Z ∈ ℝᵐˣʳ. 
    U should span the range of X while Z spans the co-range. The factorization is non-unique and
    U need not be orthogonal! However, U is often chosen orthogonal for cheap (pseudo)inversion. In that case,
    note that orthogonality of U is not guaranteed to and in fact will rarely be preserved under the operations that
    are supported (multiplication, addition, etc.). In order to reorthonormalize U, simply call `orthonormalize!(X, alg)`
    where alg refers to the algorithm used to compute the orthonormalization: 
        * GradientDescent() 
        * QRFact()
        * SVDFact()
"""
mutable struct TwoFactorRepresentation{uType, zType} <: AbstractLowRankRepresentation
    U::uType
    Z::zType                              
end

rank(LRA::TwoFactorRepresentation) = size(LRA.U, 2)
size(LRA::TwoFactorRepresentation) = (size(LRA.U,1), size(LRA.Z,1))
size(LRA::TwoFactorRepresentation, ::Val{1}) = size(LRA.U,1)
size(LRA::TwoFactorRepresentation, ::Val{2}) = size(LRA.Z,1)
size(LRA::TwoFactorRepresentation, i::Int) = size(LRA, Val(i))
Matrix(LRA::TwoFactorRepresentation) = LRA.U*LRA.Z'
getindex(LRA::TwoFactorRepresentation, i::Int, j::Int) = sum(LRA.U[i,k]*LRA.Z[j,k] for k in 1:rank(LRA)) # good enough for now
getindex(LRA::TwoFactorRepresentation, i, j::Int) = TwoFactorRepresentation(LRA.U[i,:], LRA.Z[[j],:])
getindex(LRA::TwoFactorRepresentation, i::Int, j) = TwoFactorRepresentation(LRA.U[[i],:], LRA.Z[j,:])
getindex(LRA::TwoFactorRepresentation, i, j) = TwoFactorRepresentation(LRA.U[i,:], LRA.Z[j,:])
getindex(LRA::TwoFactorRepresentation, ::Colon, j::AbstractVector) = TwoFactorRepresentation(LRA.U, LRA.Z[j,:])
getindex(LRA::TwoFactorRepresentation, i::AbstractVector, ::Colon) = TwoFactorRepresentation(LRA.U[i,:], LRA.Z)
getindex(LRA::TwoFactorRepresentation, ::Colon, j::Int) = TwoFactorRepresentation(LRA.U, LRA.Z[[j],:])
getindex(LRA::TwoFactorRepresentation, i::Int, ::Colon) = TwoFactorRepresentation(LRA.U[[i],:], LRA.Z)
hcat(A::TwoFactorRepresentation, B::TwoFactorRepresentation) = TwoFactorRepresentation(hcat(A.U, B.U), blockdiagonal(A.Z, B.Z))
vcat(A::TwoFactorRepresentation, B::TwoFactorRepresentation) = TwoFactorRepresentation(blockdiagonal(A.U, B.U), hcat(A.Z, B.Z))

# simple support of adjoints, probably not ideal though
adjoint(LRA::TwoFactorRepresentation) = TwoFactorRepresentation(conj(LRA.Z),conj(LRA.U)) 

# Is the following alternative better?
# *(A::SVDLikeRepresentation, B::SVDLikeRepresentation) = SVDLikeRepresentation(A.U, A.S*(A.V'*B.U)*B.S, B.V)
# it would preserve orthonormality of range/co-range factors but make core rectangular and increase the storage cost unnecessarily.


## *
*(A::AbstractMatrix, B::TwoFactorRepresentation) = TwoFactorRepresentation(A*B.U, B.Z)
*(A::TwoFactorRepresentation, B::AbstractMatrix) = TwoFactorRepresentation(A.U, B'*A.Z)
*(A::TwoFactorRepresentation, ::UniformScaling) = A
*(::UniformScaling, A::TwoFactorRepresentation) = A
function *(A::TwoFactorRepresentation,B::TwoFactorRepresentation)
    if rank(A) ≤ rank(B) # minimize upper bound on rank
        return TwoFactorRepresentation(A.U, B.Z*(B.U'*A.Z))
    else
        return TwoFactorRepresentation(A.U*(A.Z'*B.U), B.Z)
    end
end
*(A::TwoFactorRepresentation, α::Number) = TwoFactorRepresentation(A.U, α*A.Z) 
*(α::Number, A::TwoFactorRepresentation) = TwoFactorRepresentation(A.U, α*A.Z)
*(A::TwoFactorRepresentation, v::AbstractVector) = A.U*(A.Z'*v)
*(v::AbstractVector, A::TwoFactorRepresentation) = (v*A.U)*A.Z'

## +
+(A::TwoFactorRepresentation, B::TwoFactorRepresentation) = TwoFactorRepresentation(hcat(A.U, B.U), hcat(A.Z, B.Z))
+(A::AbstractMatrix, B::TwoFactorRepresentation) = A + Matrix(B)
+(A::TwoFactorRepresentation, B::AbstractMatrix) = Matrix(A) + B
-(A::TwoFactorRepresentation, B::TwoFactorRepresentation) = TwoFactorRepresentation(hcat(A.U, B.U), hcat(A.Z, -B.Z))
-(A::TwoFactorRepresentation, B::AbstractMatrix) = Matrix(A) - B
-(A::AbstractMatrix, B::TwoFactorRepresentation) = A - Matrix(B)

## .*
function hadamard(A::TwoFactorRepresentation, B::TwoFactorRepresentation) 
    @assert size(A) == size(B) "elementwise product is only defined between matrices of equal dimension"
    rA, rB = rank(A), rank(B)
    r_new = rA*rB
    U = ones(eltype(A.U), size(A,1), r_new)
    Z = ones(eltype(A.Z), size(A,2), r_new)
    AUcols = [@view A.U[:,i] for i in 1:rA]
    AZcols = [@view A.Z[:,i] for i in 1:rA]
    BUcols = [@view B.U[:,i] for i in 1:rB]
    BZcols = [@view B.Z[:,i] for i in 1:rB]
    k = 0
    for r1 in 1:rA, r2 in 1:rB
        k += 1
        U[:,k] = AUcols[r1] .* BUcols[r2]
        Z[:,k] = AZcols[r1] .* BZcols[r2]
    end
    return TwoFactorRepresentation(U,Z)
end

## .^
function elpow(A::TwoFactorRepresentation, d::Int)
    @assert d >= 1 "elementwise power operation 'elpow' only defined for positive powers"
    r = rank(A)
    r_new = binomial(r+d-1, d)
    Ucols = [@view A.U[:,i] for i in 1:r]
    Zcols = [@view A.Z[:,i] for i in 1:r]
    U = ones(eltype(A.U), size(A,1), r_new)
    Z = ones(eltype(A.Z), size(A,2), r_new)
    k = 0 
    for exps in multiexponents(r,d)
        k += 1
        Z[:, k] .*= multinomial(exps...) 
        for j in 1:r
            if exps[j] > 0
                U[:, k] .*= Ucols[j].^exps[j]
                Z[:, k] .*= Zcols[j].^exps[j]
            end
        end
    end
    return TwoFactorRepresentation(U, Z)
end

function elpow(A,d::Int)
    return A.^d
end

## special cases
function add_to_cols(LRA::TwoFactorRepresentation, v::AbstractVector)
    return LRA + TwoFactorRepresentation(v, ones(eltype(v), size(LRA, 2)))
end

function multiply_cols(LRA::TwoFactorRepresentation, v::AbstractVector)
    return hadamard(LRA, TwoFactorRepresentation(v, ones(eltype(v), size(LRA, 2))))
end

function add_to_cols(A, v::AbstractVector)
    return A .+ v
end

function multiply_cols(A, v::AbstractVector)
    return p .* A
end

function add_to_rows(LRA::TwoFactorRepresentation, v::AbstractVector)
    return LRA + TwoFactorRepresentation(ones(eltype(v), size(LRA, 1)), v)
end

function multiply_rows(LRA::TwoFactorRepresentation, v::AbstractVector)
    return hadamard(LRA, TwoFactorRepresentation(ones(eltype(v), size(LRA, 1)), v))
end

function add_to_rows(A, v)
    return A .+ v'
end

function multiply_rows(A, v)
    return A .* v'
end

function multiply_rows(A, v::Number)
    return v*A 
end

function add_scalar(LRA::TwoFactorRepresentation, α::Number)
    return LRA + TwoFactorRepresentation(ones(eltype(α), size(LRA, 1)), α*ones(eltype(α), size(LRA,2)))
end

function add_scalar(A, α::Number)
    return A .+ α
end

## rounding
function svd(A::TwoFactorRepresentation, alg=QRFact())
    orthonormalize!(A, alg)
    U_, S_, V_ = svd(A.Z)
    return SVDLikeRepresentation(A.U*U_, S_, V_)
end

function qr(A::TwoFactorRepresentation, alg=QRFact())
    orthonormalize!(A, alg)
    Q_, R_ = qr(A.Z')
    return TwoFactorRepresentation(A.U*Q_, R_')
end
## orthonormalization 
function orthonormalize!(LRA::TwoFactorRepresentation, alg)
    orthonormalize!(LRA.U, LRA.Z, alg)
end

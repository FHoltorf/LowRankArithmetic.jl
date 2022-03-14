export TwoFactorApproximation
"""
    X = U*Z'

    Two factor factorization of a low rank matrix into factors U ∈ ℝⁿˣʳ and Z ∈ ℝᵐˣʳ. 
    U should span the range of X while Z spans the co-range. The factorization is non-unique and
    U need not be orthogonal! However, U is often chosen orthogonal for cheap (pseudo)inversion. In that case,
    note that orthogonality of U is not guaranteed to and in fact will rarely be preserved under the operations that
    are supported (multiplication, addition, etc.). In order to reorthonormalize U, simply call `orthonormalize!(X, alg)`
    where alg refers to the algorithm used to compute the orthonormalization: 
        * GradientDescent() 
        * QR()
        * SVD()
"""
mutable struct TwoFactorApproximation{uType, zType} <: AbstractLowRankApproximation
    U::uType
    Z::zType                              
end

rank(LRA::TwoFactorApproximation) = size(LRA.U, 2)
size(LRA::TwoFactorApproximation) = (size(LRA.U,1), size(LRA.Z,1))
size(LRA::TwoFactorApproximation, ::Val{1}) = size(LRA.U,1)
size(LRA::TwoFactorApproximation, ::Val{2}) = size(LRA.Z,1)
size(LRA::TwoFactorApproximation, i::Int) = size(LRA, Val(i))
Matrix(LRA::TwoFactorApproximation) = LRA.U*LRA.Z'
getindex(LRA::TwoFactorApproximation, i::Int, j::Int) = sum(LRA.U[i,k]*LRA.Z[j,k] for k in 1:rank(LRA)) # good enough for now
getindex(LRA::TwoFactorApproximation, i, j::Int) = TwoFactorApproximation(LRA.U[i,:], LRA.Z[[j],:])
getindex(LRA::TwoFactorApproximation, i::Int, j) = TwoFactorApproximation(LRA.U[[i],:], LRA.Z[j,:])
getindex(LRA::TwoFactorApproximation, i, j) = TwoFactorApproximation(LRA.U[i,:], LRA.Z[j,:])
getindex(LRA::TwoFactorApproximation, ::Colon, j::AbstractVector) = TwoFactorApproximation(LRA.U, LRA.Z[j,:])
getindex(LRA::TwoFactorApproximation, i::AbstractVector, ::Colon) = TwoFactorApproximation(LRA.U[i,:], LRA.Z)
getindex(LRA::TwoFactorApproximation, ::Colon, j::Int) = TwoFactorApproximation(LRA.U, LRA.Z[[j],:])
getindex(LRA::TwoFactorApproximation, i::Int, ::Colon) = TwoFactorApproximation(LRA.U[[i],:], LRA.Z)
hcat(A::TwoFactorApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(hcat(A.U, B.U), blockdiagonal(A.Z, B.Z))
vcat(A::TwoFactorApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(blockdiagonal(A.U, B.U), hcat(A.Z, B.Z))

# simple support of adjoints, probably not ideal though
adjoint(LRA::TwoFactorApproximation) = TwoFactorApproximation(conj(LRA.Z),conj(LRA.U)) 

# Is the following alternative better?
# *(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(A.U, A.S*(A.V'*B.U)*B.S, B.V)
# it would preserve orthonormality of range/co-range factors but make core rectangular and increase the storage cost unnecessarily.


## *
*(A::AbstractMatrix, B::TwoFactorApproximation) = TwoFactorApproximation(A*B.U, B.Z)
*(A::TwoFactorApproximation, B::AbstractMatrix) = TwoFactorApproximation(A.U, B'*A.Z)
*(A::TwoFactorApproximation, ::UniformScaling) = A
*(::UniformScaling, A::TwoFactorApproximation) = A
function *(A::TwoFactorApproximation,B::TwoFactorApproximation)
    if rank(A) ≤ rank(B) # minimize upper bound on rank
        return TwoFactorApproximation(A.U, B.Z*(B.U'*A.Z))
    else
        return TwoFactorApproximation(A.U*(A.Z'*B.U), B.Z)
    end
end
*(A::TwoFactorApproximation, α::Number) = TwoFactorApproximation(A.U, α*A.Z) 
*(α::Number, A::TwoFactorApproximation) = TwoFactorApproximation(A.U, α*A.Z)
*(A::TwoFactorApproximation, v::AbstractVector) = A.U*(A.Z'*v)
*(v::AbstractVector, A::TwoFactorApproximation) = (v*A.U)*A.Z'

## +
+(A::TwoFactorApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(hcat(A.U, B.U), hcat(A.Z, B.Z))
+(A::AbstractMatrix, B::TwoFactorApproximation) = A + Matrix(B)
+(A::TwoFactorApproximation, B::AbstractMatrix) = Matrix(A) + B
-(A::TwoFactorApproximation, B::TwoFactorApproximation) = TwoFactorApproximation(hcat(A.U, B.U), hcat(A.Z, -B.Z))
-(A::TwoFactorApproximation, B::AbstractMatrix) = Matrix(A) - B
-(A::AbstractMatrix, B::TwoFactorApproximation) = A - Matrix(B)

## .*
function hadamard(A::TwoFactorApproximation, B::TwoFactorApproximation) 
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
    return TwoFactorApproximation(U,Z)
end

## .^
function elpow(A::TwoFactorApproximation, d::Int)
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
    return TwoFactorApproximation(U, Z)
end

function elpow(A,d::Int)
    return A.^d
end

## special cases
function add_to_cols(LRA::TwoFactorApproximation, v::AbstractVector)
    return LRA + TwoFactorApproximation(v, ones(eltype(v), size(LRA, 2)))
end

function multiply_cols(LRA::TwoFactorApproximation, v::AbstractVector)
    return hadamard(LRA, TwoFactorApproximation(v, ones(eltype(v), size(LRA, 2))))
end

function add_to_cols(A, v::AbstractVector)
    return A .+ v
end

function multiply_cols(A, v::AbstractVector)
    return p .* A
end

function add_to_rows(LRA::TwoFactorApproximation, v::AbstractVector)
    return LRA + TwoFactorApproximation(ones(eltype(v), size(LRA, 1)), v)
end

function multiply_rows(LRA::TwoFactorApproximation, v::AbstractVector)
    return hadamard(LRA, TwoFactorApproximation(ones(eltype(v), size(LRA, 1)), v))
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

function add_scalar(LRA::TwoFactorApproximation, α::Number)
    return LRA + TwoFactorApproximation(ones(eltype(α), size(LRA, 1)), α*ones(eltype(α), size(LRA,2)))
end

function add_scalar(A, α::Number)
    return A .+ α
end

## orthonormalization 
function orthonormalize!(LRA::TwoFactorApproximation, ::QR)
    Q, R = qr(LRA.U)
    LRA.U .= Matrix(Q) # ToDo: store Q in terms of householder reflections
    LRA.Z .= LRA.Z*R'
end

function orthonormalize!(LRA::TwoFactorApproximation, ::SVD)
    U, S, V = svd(LRA.U)
    LRA.U .= U 
    LRA.Z .= LRA.Z*V*Diagonal(S)
end

function orthonormalize!(LRA::TwoFactorApproximation, alg::GradientDescent)
    orthonormalize!(LRA.U, LRA.Z, alg)
end
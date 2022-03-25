export SVDLikeApproximation
"""
    X = U*S*V'

    SVD like factorization of a low rank matrix into factors U ∈ ℝⁿˣʳ, S ∈ ℝʳˣʳ, V ∈ ℝᵐˣʳ. 
    The columns of U span the range of X, the colomuns of V span the co-range of X and S describes the map between
    co-range and range. The factorization is non-unique and U and V need not be orthogonal! 
    However, often U and V are chosen orthogonal to allow for cheap (pseudo)inversion. In that case, note that
    orthogonality of U and V is not guaranteed to and in fact will rarely be preserved under the operations that
    are supported (multiplication, addition, etc.). In order to reorthonormalize U and V, simply call `orthonormalize!(X, alg)`
    where alg refers to the algorithm used to compute the orthonormalization: 
        * QR()
        * SVD()
        * GradientDescent()
"""
mutable struct SVDLikeApproximation{uType, sType, vType} <: AbstractLowRankApproximation
    U::uType
    S::sType
    V::vType
end 

rank(LRA::SVDLikeApproximation) = size(LRA.S,1)
size(LRA::SVDLikeApproximation) = (size(LRA.U,1), size(LRA.V,1))
size(LRA::SVDLikeApproximation, ::Val{1}) = size(LRA.U,1)
size(LRA::SVDLikeApproximation, ::Val{2}) = size(LRA.V,1)
size(LRA::SVDLikeApproximation, i::Int) = size(LRA, Val(i))
Matrix(LRA::SVDLikeApproximation) = LRA.U*LRA.S*LRA.V'
getindex(LRA::SVDLikeApproximation, i::Int, j::Int) = sum(LRA.U[i,k]*sum(LRA.S[k,s]*LRA.V[j,s] for s in 1:rank(LRA)) for k in 1:rank(LRA)) # good enough for now
getindex(LRA::SVDLikeApproximation, i, j::Int) = SVDLikeApproximation(LRA.U[i,:], LRA.S, LRA.V[[j],:])  # good enough for now
getindex(LRA::SVDLikeApproximation, i::Int, j) = SVDLikeApproximation(LRA.U[[i],:], LRA.S, LRA.V[j,:])  # good enough for now
getindex(LRA::SVDLikeApproximation, i, j) = SVDLikeApproximation(LRA.U[i,:], LRA.S, LRA.V[j,:])  # good enough for now
getindex(LRA::SVDLikeApproximation, ::Colon, j::AbstractVector) = SVDLikeApproximation(LRA.U, LRA.S, LRA.V[j,:])  # good enough for now
getindex(LRA::SVDLikeApproximation, ::Colon, j::Int) = SVDLikeApproximation(LRA.U, LRA.S, LRA.V[[j],:])  # good enough for now
getindex(LRA::SVDLikeApproximation, i::AbstractVector, ::Colon) = SVDLikeApproximation(LRA.U[i,:], LRA.S, LRA.V)  # good enough for now
getindex(LRA::SVDLikeApproximation, i::Int, ::Colon) = SVDLikeApproximation(LRA.U[[i],:], LRA.S, LRA.V)  # good enough for now
hcat(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(hcat(A.U, B.U), blockdiagonal(A.S, B.S), blockdiagonal(A.V, B.V))
vcat(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(blockdiagonal(A.U, B.U), blockdiagonal(A.S, B.S), hcat(A.V, B.V))

# simple support of adjoints, probably not ideal though
adjoint(LRA::SVDLikeApproximation) = TwoFactorRepresentation(conj(LRA.V),LRA.S',conj(LRA.U)) 

# converting between both representations
TwoFactorRepresentation(A::SVDLikeApproximation) = TwoFactorRepresentation(A.U, A.V*A.S')
function SVDLikeApproximation(A::TwoFactorRepresentation) 
    U, S, V = svd(A.Z)
    return SVDLikeApproximation(A.U*V, S', U)
end

## *
*(A::AbstractMatrix, B::SVDLikeApproximation) = SVDLikeApproximation(A*B.U, B.S, B.V)
*(A::SVDLikeApproximation, B::AbstractMatrix) = SVDLikeApproximation(A.U, A.S, B'*A.V)
*(A::SVDLikeApproximation, ::UniformScaling) = A
*(::UniformScaling, A::SVDLikeApproximation) = A
function *(A::SVDLikeApproximation, B::SVDLikeApproximation)
    if rank(A) ≤ rank(B)
        return SVDLikeApproximation(A.U, A.S, B.V*B.S'*(B.U'*A.V))
    else
        return SVDLikeApproximation(A.U*A.S*(A.V'*B.U), B.S, B.V)
    end
end
*(A::SVDLikeApproximation, α::Number) = SVDLikeApproximation(A.U, α*A.S, A.V) 
*(α::Number, A::SVDLikeApproximation) = SVDLikeApproximation(A.U, α*A.S, A.V)
*(A::SVDLikeApproximation, v::AbstractVector) = A.U*(A.S*(A.V'*v))
*(v::AbstractVector, A::SVDLikeApproximation) = ((v*A.U)*A.S)*A.V'

# default to TwoFactorRepresentation
*(A::TwoFactorRepresentation, B::SVDLikeApproximation) = A*TwoFactorRepresentation(B)
*(A::SVDLikeApproximation, B::TwoFactorRepresentation) = TwoFactorRepresentation(B)*A

## + 
+(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(hcat(A.U, B.U), blockdiagonal(A.S, B.S),hcat(A.V, B.V))
+(A::AbstractMatrix, B::SVDLikeApproximation) = A + Matrix(B)
+(A::SVDLikeApproximation, B::AbstractMatrix) = Matrix(A) + B
-(A::SVDLikeApproximation, B::SVDLikeApproximation) = SVDLikeApproximation(hcat(A.U, B.U), blockdiagonal(A.S, -B.S),hcat(A.V, B.V))
-(A::AbstractMatrix, B::SVDLikeApproximation) = A - Matrix(B)
-(A::SVDLikeApproximation, B::AbstractMatrix) = Matrix(A) - B

# default to TwoFactorRepresentation
+(A::TwoFactorRepresentation, B::SVDLikeApproximation) = A+TwoFactorRepresentation(B)
+(A::SVDLikeApproximation, B::TwoFactorRepresentation) = B+A
-(A::TwoFactorRepresentation, B::SVDLikeApproximation) = A-TwoFactorRepresentation(B)
-(A::SVDLikeApproximation, B::TwoFactorRepresentation) = TwoFactorRepresentation(A)-B

## .*
function hadamard(A::SVDLikeApproximation, B::SVDLikeApproximation) 
    @assert size(A) == size(B) "elementwise product is only defined between matrices of equal dimension"
    rA, rB = rank(A), rank(B)
    r_new = rA*rB
    U = ones(eltype(A.U), size(A,1), r_new)
    V = ones(eltype(A.V), size(A,2), r_new)
    S = ones(eltype(A.S), r_new, r_new)
    AUcols = [@view A.U[:,i] for i in 1:rA]
    AVcols = [@view A.V[:,i] for i in 1:rA]
    BUcols = [@view B.U[:,i] for i in 1:rB]
    BVcols = [@view B.V[:,i] for i in 1:rB]
    k = 0
    for r1 in 1:rA, r2 in 1:rB
        k += 1
        U[:,k] = AUcols[r1] .* BUcols[r2]
        V[:,k] = AVcols[r1] .* BVcols[r2]
        l = 0
        for k1 in 1:rA, k2 in 1:rB
            l += 1
            S[k,l] = A.S[r1,k1]*B.S[r2,k2]
        end
    end
    return SVDLikeApproximation(U,S,V)
end

# default to TwoFactorRepresentation
hadamard(A::TwoFactorRepresentation, B::SVDLikeApproximation) = hadamard(A, TwoFactorRepresentation(B))
hadamard(A::SVDLikeApproximation, B::TwoFactorRepresentation) = hadamard(TwoFactorRepresentation(A), B)

## .^2
# very suboptimal
function elpow(A::SVDLikeApproximation, d::Int)
    @assert d >= 1 "elementwise power operation 'elpow' only defined for positive powers"
    r = rank(A)
    r_new = binomial(r+d-1, d)

    # the following sequence is not ideal but cheap under the premise of low rank, need to be improved though
    U, S, V = svd(A.S)
    A.U .= A.U*U
    A.V .= A.V*V 
    A.S .= Matrix(Diagonal(S))

    Ucols = [@view A.U[:,i] for i in 1:r]
    Vcols = [@view A.V[:,i] for i in 1:r]
    U = ones(eltype(A.U), size(A,1), r_new)
    V = ones(eltype(A.V), size(A,2), r_new)
    S = zeros(eltype(A.S), r_new, r_new)
    k = 0 
    multi_exps = multiexponents(r,d) 
    for exps in multi_exps
        k += 1
        S[k,k] += multinomial(exps...)
        for j in 1:r
            if exps[j] != 0
                U[:, k] .*= Ucols[j].^exps[j]
                V[:, k] .*= Vcols[j].^exps[j]
                S[k,k] *= A.S[j,j]^exps[j]
            end
        end
    end
    return SVDLikeApproximation(U,S,V)
end

## special cases
function add_to_cols(LRA::SVDLikeApproximation, v::AbstractVector)
    return LRA + SVDLikeApproximation(v, ones(eltype(v), 1, 1), ones(eltype(v), size(LRA, 2)))
end

function multiply_cols(LRA::SVDLikeApproximation, v::AbstractVector)
    return hadamard(LRA, SVDLikeApproximation(v, ones(eltype(v), 1, 1), ones(eltype(v), size(LRA, 2))))
end

function add_to_rows(LRA::SVDLikeApproximation, v::AbstractVector)
    return LRA + SVDLikeApproximation(ones(eltype(v), size(LRA, 1)), ones(eltype(v), 1, 1), v)
end

function multiply_rows(LRA::SVDLikeApproximation, v::AbstractVector)
    return hadamard(LRA, SVDLikeApproximation(ones(eltype(v), size(LRA, 1)), ones(eltype(v), 1, 1), v))
end

function add_scalar(LRA::SVDLikeApproximation, α::Number)
    return LRA + SVDLikeApproximation(ones(eltype(α), size(LRA, 1)), [α], ones(eltype(α), size(LRA,2)))
end

## rounding
function svd(A::SVDLikeApproximation, alg=QR())
    orthonormalize!(A, alg)
    U_, S_, V_ = svd(A.S)
    return SVDLikeApproximation(A.U*U_, S_, A.V*V_)
end

function svd(A::TwoFactorRepresentation, alg=QR())
    orthonormalize!(A, alg)
    U_, S_, V_ = svd(A.Z)
    return SVDLikeApproximation(A.U*U_, S_, V_)
end

## orthonormalization 
function orthonormalize!(LRA::SVDLikeApproximation, alg)
    orthonormalize!(LRA.U, LRA.S', alg)
    orthonormalize!(LRA.V, LRA.S, alg)
end

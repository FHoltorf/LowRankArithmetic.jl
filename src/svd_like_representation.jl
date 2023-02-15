export SVDLikeRepresentation
"""
    X = U*S*V'

    SVD like factorization of a low rank matrix into factors U ∈ ℝⁿˣʳ, S ∈ ℝʳˣʳ, V ∈ ℝᵐˣʳ. 
    The columns of U span the range of X, the colomuns of V span the co-range of X and S describes the map between
    co-range and range. The factorization is non-unique and U and V need not be orthogonal! 
    However, often U and V are chosen orthogonal to allow for cheap (pseudo)inversion. In that case, note that
    orthogonality of U and V is not guaranteed to and in fact will rarely be preserved under the operations that
    are supported (multiplication, addition, etc.). In order to reorthonormalize U and V, simply call `orthonormalize!(X, alg)`
    where alg refers to the algorithm used to compute the orthonormalization: 
        * QRFact()
        * SVDFact()
        * GradientDescent()
"""
mutable struct SVDLikeRepresentation{uType, sType, vType} <: AbstractLowRankRepresentation
    U::uType
    S::sType
    V::vType
end 

rank(LRA::SVDLikeRepresentation) = size(LRA.S,1)
size(LRA::SVDLikeRepresentation) = (size(LRA.U,1), size(LRA.V,1))
size(LRA::SVDLikeRepresentation, ::Val{1}) = size(LRA.U,1)
size(LRA::SVDLikeRepresentation, ::Val{2}) = size(LRA.V,1)
size(LRA::SVDLikeRepresentation, i::Int) = size(LRA, Val(i))
Matrix(LRA::SVDLikeRepresentation) = LRA.U*LRA.S*LRA.V'
getindex(LRA::SVDLikeRepresentation, i::Int, j::Int) = sum(LRA.U[i,k]*sum(LRA.S[k,s]*LRA.V[j,s] for s in 1:rank(LRA)) for k in 1:rank(LRA))
getindex(LRA::SVDLikeRepresentation, i, j::Int) = SVDLikeRepresentation(LRA.U[i,:], LRA.S, LRA.V[[j],:])  
getindex(LRA::SVDLikeRepresentation, i::Int, j) = SVDLikeRepresentation(LRA.U[[i],:], LRA.S, LRA.V[j,:])  
getindex(LRA::SVDLikeRepresentation, i, j) = SVDLikeRepresentation(LRA.U[i,:], LRA.S, LRA.V[j,:])  
getindex(LRA::SVDLikeRepresentation, ::Colon, j::AbstractVector) = SVDLikeRepresentation(LRA.U, LRA.S, LRA.V[j,:])  
getindex(LRA::SVDLikeRepresentation, ::Colon, j::Int) = SVDLikeRepresentation(LRA.U, LRA.S, LRA.V[[j],:])  
getindex(LRA::SVDLikeRepresentation, i::AbstractVector, ::Colon) = SVDLikeRepresentation(LRA.U[i,:], LRA.S, LRA.V)  
getindex(LRA::SVDLikeRepresentation, i::Int, ::Colon) = SVDLikeRepresentation(LRA.U[[i],:], LRA.S, LRA.V)  
hcat(A::SVDLikeRepresentation, B::SVDLikeRepresentation) = SVDLikeRepresentation(hcat(A.U, B.U), blockdiagonal(A.S, B.S), blockdiagonal(A.V, B.V))
vcat(A::SVDLikeRepresentation, B::SVDLikeRepresentation) = SVDLikeRepresentation(blockdiagonal(A.U, B.U), blockdiagonal(A.S, B.S), hcat(A.V, B.V))

# simple support of adjoints, probably not ideal though
adjoint(LRA::SVDLikeRepresentation) = TwoFactorRepresentation(conj(LRA.V),LRA.S',conj(LRA.U)) 

# converting between representations
TwoFactorRepresentation(A::SVDLikeRepresentation) = TwoFactorRepresentation(A.U, A.V*A.S')
function SVDLikeRepresentation(A::TwoFactorRepresentation) 
    U, S, V = svd(A.Z)
    return SVDLikeRepresentation(A.U*V, S', U)
end

## *
*(A::AbstractMatrix, B::SVDLikeRepresentation) = SVDLikeRepresentation(A*B.U, B.S, B.V)
*(A::SVDLikeRepresentation, B::AbstractMatrix) = SVDLikeRepresentation(A.U, A.S, B'*A.V)
*(A::SVDLikeRepresentation, ::UniformScaling) = A
*(::UniformScaling, A::SVDLikeRepresentation) = A
function *(A::SVDLikeRepresentation, B::SVDLikeRepresentation)
    if rank(A) ≤ rank(B)
        return SVDLikeRepresentation(A.U, A.S, B.V*B.S'*(B.U'*A.V))
    else
        return SVDLikeRepresentation(A.U*A.S*(A.V'*B.U), B.S, B.V)
    end
end
*(A::SVDLikeRepresentation, α::Number) = SVDLikeRepresentation(A.U, α*A.S, A.V) 
*(α::Number, A::SVDLikeRepresentation) = SVDLikeRepresentation(A.U, α*A.S, A.V)
*(A::SVDLikeRepresentation, v::AbstractVector) = A.U*(A.S*(A.V'*v))
*(v::AbstractVector, A::SVDLikeRepresentation) = ((v*A.U)*A.S)*A.V'

# default to TwoFactorRepresentation
*(A::TwoFactorRepresentation, B::SVDLikeRepresentation) = A*TwoFactorRepresentation(B)
*(A::SVDLikeRepresentation, B::TwoFactorRepresentation) = TwoFactorRepresentation(B)*A

## + 
+(A::SVDLikeRepresentation, B::SVDLikeRepresentation) = SVDLikeRepresentation(hcat(A.U, B.U), blockdiagonal(A.S, B.S),hcat(A.V, B.V))
+(A::AbstractMatrix, B::SVDLikeRepresentation) = A + Matrix(B)
+(A::SVDLikeRepresentation, B::AbstractMatrix) = Matrix(A) + B
-(A::SVDLikeRepresentation, B::SVDLikeRepresentation) = SVDLikeRepresentation(hcat(A.U, B.U), blockdiagonal(A.S, -B.S), hcat(A.V, B.V))
-(A::AbstractMatrix, B::SVDLikeRepresentation) = A - Matrix(B)
-(A::SVDLikeRepresentation, B::AbstractMatrix) = Matrix(A) - B

# default to TwoFactorRepresentation
+(A::TwoFactorRepresentation, B::SVDLikeRepresentation) = A+TwoFactorRepresentation(B)
+(A::SVDLikeRepresentation, B::TwoFactorRepresentation) = B+A
-(A::TwoFactorRepresentation, B::SVDLikeRepresentation) = A-TwoFactorRepresentation(B)
-(A::SVDLikeRepresentation, B::TwoFactorRepresentation) = TwoFactorRepresentation(A)-B

## .*
function hadamard(A::SVDLikeRepresentation, B::SVDLikeRepresentation) 
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
    return SVDLikeRepresentation(U,S,V)
end

# default to TwoFactorRepresentation
hadamard(A::TwoFactorRepresentation, B::SVDLikeRepresentation) = hadamard(A, TwoFactorRepresentation(B))
hadamard(A::SVDLikeRepresentation, B::TwoFactorRepresentation) = hadamard(TwoFactorRepresentation(A), B)

## .^d
# very suboptimal
function elpow(A::SVDLikeRepresentation, d::Int)
    @assert d >= 1 "elementwise power operation 'elpow' only defined for positive powers"
    r = rank(A)
    r_new = binomial(r+d-1, d)

    # the following sequence is not ideal but cheap under the premise of low rank, need to be improved though
    U, S, V = svd(A.S)
    A.U .= A.U*U
    A.V .= A.V*V 
    A.S .= diagm(S)

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
    return SVDLikeRepresentation(U,S,V)
end

## special cases
function add_to_cols(LRA::SVDLikeRepresentation, v::AbstractVector)
    return LRA + SVDLikeRepresentation(v, ones(eltype(v), 1, 1), ones(eltype(v), size(LRA, 2)))
end

function multiply_cols(LRA::SVDLikeRepresentation, v::AbstractVector)
    return hadamard(LRA, SVDLikeRepresentation(v, ones(eltype(v), 1, 1), ones(eltype(v), size(LRA, 2))))
end

function add_to_rows(LRA::SVDLikeRepresentation, v::AbstractVector)
    return LRA + SVDLikeRepresentation(ones(eltype(v), size(LRA, 1)), ones(eltype(v), 1, 1), v)
end

function multiply_rows(LRA::SVDLikeRepresentation, v::AbstractVector)
    return hadamard(LRA, SVDLikeRepresentation(ones(eltype(v), size(LRA, 1)), ones(eltype(v), 1, 1), v))
end

function add_scalar(LRA::SVDLikeRepresentation, α::Number)
    return LRA + SVDLikeRepresentation(ones(eltype(α), size(LRA, 1)), [α], ones(eltype(α), size(LRA,2)))
end

## rounding
function svd(A::SVDLikeRepresentation; alg_orthonormalize=QRFact())
    orthonormalize!(A, alg_orthonormalize)
    U_, S_, V_ = svd(A.S)
    return SVDLikeRepresentation(A.U*U_, Diagonal(S_), A.V*V_)
end

function qr(A::SVDLikeRepresentation, alg=QRFact())
    orthonormalize!(A.U, A.S, alg)
    U_, R_ = qr(A.S*A.V')
    return TwoFactorRepresentation(A.U*U_, R_')
end

function round(A::SVDLikeRepresentation, alg = SVDFact(); tol = sqrt(eps(eltype(A.U))), rmin::Int = 1, rmax::Int = rank(A), alg_orthonormalize = QRFact())
    orthonormalize!(A, alg_orthonormalize)
    S_lr = truncated_svd(A.S, alg, tol = tol, rmin = rmin, rmax = rmax)
    return SVDLikeRepresentation(A.U*S_lr.U, S_lr.S, A.V*S_lr.V)
end

function round(A::SVDLikeRepresentation, rank::Int, alg = SVDFact(); alg_orthonormalize=QRFact())
    orthonormalize!(A, alg_orthonormalize)
    S_lr = truncated_svd(A.S, rank, alg)
    return SVDLikeRepresentation(A.U*S_lr.U, S_lr.S, A.V*S_lr.V)
end

function truncated_svd(A::SVDLikeRepresentation, ::TSVD; 
                       tol = 1e-8, rmin = 1, rmax = minimum(size(A)), 
                       alg_orthonormalize = QRFact()) 
    orthonormalize!(A, alg_orthonormalize)
    U_, S_, V_ = tsvd(A.S,rmax) 
    return SVDLikeRepresentation(A.U*U_, Diagonal(S_), A.V*V_)
end

## orthonormalization 
function orthonormalize!(LRA::SVDLikeRepresentation, alg)
    orthonormalize!(LRA.U, LRA.S', alg)
    orthonormalize!(LRA.V, LRA.S, alg)
end

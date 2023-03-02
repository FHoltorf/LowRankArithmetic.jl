export truncated_svd, TSVD

truncated_svd(A::AbstractMatrix, alg = SVDFact(); tol = sqrt(eps(eltype(A))), rmin = 1, rmax = minimum(size(A))) = truncated_svd(A, alg, tol, rmin, rmax)

function truncated_svd(A::AbstractMatrix, rank::Int, alg = SVDFact()) 
    @assert rank <= minimum(size(A)) "rank ≤ min(n,m)"
    return truncated_svd(A, alg; rmin = rank, rmax = rank)
end

function truncated_svd(A::AbstractMatrix, ::SVDFact, tol, rmin::Int, rmax::Int)
    U, S, V = svd(A)
    r = (rmax == rmin) ? rmax : max(min(truncate_to_tolerance(S, tol), rmax), rmin)
    return SVDLikeRepresentation(U[:,1:r], diagm(S[1:r]), V[:,1:r])
end

function truncated_svd(A::AbstractMatrix, ::TSVD, tol, rmin::Int, rmax::Int)
    U, S, V = tsvd(A, rmax)
    return SVDLikeRepresentation(U, diagm(S), V)
end

function truncated_svd(A::AbstractLowRankRepresentation, ::SVDFact=SVDFact(); 
                       tol = 1e-8, rmin = 1, rmax = minimum(size(A)), 
                       alg_orthonormalize = QRFact()) 
    Asvd = svd(A, alg_orthonormalize = alg_orthonormalize) 
    r = (rmax == rmin) ? rmax : max(min(truncate_to_tolerance(diag(Asvd.S), tol), rmax), rmin)
    return SVDLikeRepresentation(Asvd.U[:,1:r], Asvd.S[1:r,1:r], Asvd.V[:,1:r])
end

function truncated_svd(A::AbstractLowRankRepresentation, rank::Int, alg=SVDFact(); alg_orthonormalize = QRFact())
    return round(A, rank, alg, alg_orthonormalize = alg_orthonormalize)
end

function truncated_svd(A::TwoFactorRepresentation, rank::Int, alg=SVDFact(); alg_orthonormalize = QRFact())
    orthonormalize!(A, alg_orthonormalize)
    Z_lr = truncated_svd(A.Z, rank, alg)
    return SVDLikeRepresentation(A.U*Z_lr.V, Z_lr.S, Z_lr.U)
end

function truncate_to_tolerance(S, tol)
    s = 0
    r = length(S)
    for σ in reverse(S)
        s += σ^2
        if s > tol^2
            break
        end
        r -= 1
    end 
    return r
end 

function blockdiagonal(A::T1, B::T2) where {T1, T2 <: Union{AbstractMatrix, AbstractVector}}
    n1,m1 = size(A,1), size(A,2)
    n2,m2 = size(B,1), size(B,2)
    C = zeros(eltype(A), n1+n2, m1+m2)
    C[1:n1, 1:m1] .= A
    C[n1+1:end, m1+1:end] .= B
    return C
end
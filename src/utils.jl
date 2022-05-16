export truncated_svd, TSVD

struct TSVD end

truncated_svd(A::AbstractMatrix, alg = SVDFact(); tol = sqrt(eps(eltype(A))), rmax = minimum(size(A))) = truncated_svd(A, alg, tol, rmax)

function truncated_svd(A::AbstractMatrix, ::SVDFact, tol, rmax::Int)
    U, S, V = svd(A)
    r = min(truncate_to_tolerance(S, tol), rmax)
    return SVDLikeRepresentation(U[:,1:r], diagm(S[1:r]), V[:,1:r])
end

function truncated_svd(A::AbstractMatrix, ::TSVD, tol, rmax::Int)
    U, S, V = tsvd(A, rmax)
    return SVDLikeRepresentation(U, diagm(S), V)
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
export GradientDescent, SecondMomentMatching, GramSchmidt, QRFact, SVDFact, orthonormalize!
# QR, SVD not exported due to conflict with LinearAlgebra identifiers. 

struct GradientDescent
    maxiter::Int
    μ::Float64
    atol::Float64
    rtol::Float64
    function GradientDescent(;maxiter = 100, μ = 1.0, atol = 1e-8, rtol = 1e-8)
        return new(maxiter, μ, atol, rtol)
    end
end

struct QRFact end

struct SVDFact end

struct SecondMomentMatching end

struct GramSchmidt end

function orthonormalize!(U::AbstractMatrix, alg::GradientDescent)
    @unpack μ, atol, rtol, maxiter = alg 
    r = size(U,2)
    K = U'*U
    A = Matrix{eltype(K)}(I, r, r)
    dA = similar(A)
    iter = 0
    ϵ = norm(A'*K*A - I)^2
    while ϵ > atol && ϵ > r*rtol && iter < maxiter
        dA .= - K*A*(A'*K*A - I)
        A .+= μ*dA
        iter += 1
        ϵ = norm(A'*K*A - I)^2
    end
    if iter == maxiter
        @warn "Gradient flow orthonormalization did not converge. 
               Iterations exceeded maxiters = $maxiter. 
               Primal residual: $(norm(A'*K*A - I)^2)"
    end
    U .= U*A
end

function orthonormalize!(U::AbstractMatrix, Z::AbstractMatrix, alg::GradientDescent)
    @unpack μ, atol, rtol, maxiter = alg 
    K = U'*U
    r = size(U,2)
    A = Matrix{eltype(K)}(I, r, r)
    Ainv = Matrix{eltype(K)}(I, r, r)
    dA = similar(A)
    iter = 0
    ϵ = norm(A'*K*A - I)^2
    while ϵ > atol && ϵ > r*rtol && iter < maxiter
        dA .= - K*A*(A'*K*A - I)
        A .+= μ*dA
        Ainv .-= μ*Ainv*dA*Ainv
        iter += 1
        ϵ = norm(A'*K*A - I)^2
    end
    if iter == maxiter
        @warn "Gradient flow orthonormalization did not converge. 
               Iterations exceeded maxiters = $maxiter. 
               Primal residual: $(norm(A'*K*A - I)^2)"
    end
    U .= U*A
    Z .= Z*Ainv'
end

function orthonormalize!(U::AbstractMatrix, ::SecondMomentMatching)
    P = Symmetric(U'*U)
    Λ, V = eigen(P)
    U .= U*V*Diagonal(1 ./ sqrt.(Λ)) * V'
end

function orthonormalize!(U::AbstractMatrix, Z::AbstractMatrix, ::SecondMomentMatching)
    P = Symmetric(U'*U)
    Λ, V = eigen(P)
    U .= U*V*Diagonal(1 ./ sqrt.(Λ)) * V'
    Z .= Z*V*Diagonal(sqrt.(Λ)) * V'
end

function orthonormalize!(U::AbstractMatrix, ::QRFact)
    Q, _ = qr!(U)
    U .= Matrix(Q)
end

function orthonormalize!(U::AbstractMatrix, Z::AbstractMatrix, ::QRFact)
    Q, R = qr!(U)
    U .= Matrix(Q)
    Z .= Z*R'
end

function orthonormalize!(U::AbstractMatrix, ::SVDFact)
    Q, _, _ = svd!(U)
    U .= Q
end

function orthonormalize!(U::AbstractMatrix, Z::AbstractMatrix, ::SVDFact)
    Q, S, V = svd!(U)
    U .= Q
    Z .= Z*V*Diagonal(S)'
end

function orthonormalize!(U::AbstractMatrix, ::GramSchmidt)
    u = @view U[:,1]
    u ./= norm(u)
    for i in 2:size(U,2)
        u = @view U[:,i]
        subU = @view U[:,1:i-1]
        ips = subU'*u
        mul!(u,subU,ips,-1,1)
        u ./= norm(u)    
    end
end

function orthonormalize!(U::AbstractMatrix, Z::AbstractMatrix, ::GramSchmidt)
    u = @view U[:,1]
    z = @view Z[:,1]
    r = norm(u)
    u ./= r
    z .*= r
    for i in 2:size(U,2)
        u = @view U[:,i]
        z = @view Z[:,i]
        subU = @view U[:,1:i-1]
        subZ = @view Z[:,1:i-1]
        ips = subU'*u 
        mul!(u, subU, ips, -1, 1)
        mul!(subZ, z, ips', 1, 1) 
        r = norm(u)
        u ./= r
        z .*= r
    end
end


# this implementation could return in the inverse of the GramSchmidt procedure in R
# function orthonormalize!(U, Z, ::GramSchmidt)
#     v = @view U[:,1]
#     R = UpperDiagonal(zeros(size(U,2), size(U,2)))
#     R[1,1] = norm(v)
#     v ./= R[1,1]
#     for i in 2:size(U,2)
#         v = @view U[:,i]
#         subU = @view U[:,1:i-1]
#         u = subU'*v 
#         R[1:i-1,i] = u
#         R[i,i] = norm(v)
#         mul!(v,subU,u,-1,1)
#         v .*= R[i,i]  
#     end
#     Z .= Z*R'
# end

# ToDo: 
#  1. add general minimum second order moment matching orthonormalization:  
#         https://link.springer.com/content/pdf/10.1007/s00211-021-01178-8.pdf

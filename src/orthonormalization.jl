export QR, SVD, GradientDescent, SecondMomentMatching, orthonormalize!

struct GradientDescent
    maxiter::Int
    μ::Float64
    atol::Float64
    rtol::Float64
    function GradientDescent(;maxiter = 100, μ = 1.0, atol = 1e-8, rtol = 1e-8)
        return new(maxiter, μ, atol, rtol)
    end
end

struct QR end

struct SVD end

struct SecondMomentMatching end

function orthonormalize!(U, alg::GradientDescent)
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

function orthonormalize!(U, Z, alg::GradientDescent)
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

function orthonormalize!(U, ::SecondMomentMatching)
    P = Symmetric(U'*U)
    Λ, V = eigen(P)
    U .= U*V*Diagonal(1 ./ sqrt.(Λ)) * V'
end

function orthonormalize!(U, ::QR)
    Q, _ = qr(U)
    U .= Matrix(Q)
end

# ToDo: 
#  1. add Gram-Schmidt orthonormalization
#  2. add general minimum second order matching orthonormalization:  
#         https://link.springer.com/content/pdf/10.1007/s00211-021-01178-8.pdf

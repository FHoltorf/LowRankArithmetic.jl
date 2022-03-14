module LowRankArithmetic

using Combinatorics, LinearAlgebra, UnPack
import Base: +, -, *, size, Matrix, getindex, hcat, vcat, axes, broadcasted, BroadcastStyle
import LinearAlgebra: rank, adjoint, svd

abstract type AbstractLowRankApproximation end

# catch-all-cases
axes(LRA::AbstractLowRankApproximation) = map(Base.oneto, size(LRA))
hadamard(A::AbstractLowRankApproximation, B::AbstractMatrix) = Matrix(A) .* B
hadamard(A::AbstractMatrix, B::AbstractLowRankApproximation) = Matrix(A) .* B
hadamard(A,B) = A .* B

# standard linear algbera broadcasts 
broadcasted(::typeof(*), A::AbstractLowRankApproximation, B::AbstractLowRankApproximation) = hadamard(A,B)
broadcasted(::typeof(*), A::AbstractLowRankApproximation, b::AbstractVector) = multiply_cols(A,b)
broadcasted(::typeof(*), A::AbstractLowRankApproximation, b::Adjoint{<:Number, <:AbstractVector}) = multiply_rows(A, transpose(b))
broadcasted(::typeof(*), A::AbstractLowRankApproximation, b::Transpose{<:Number, <:AbstractVector}) = multiply_rows(A, transpose(b))
broadcasted(::typeof(*), A::AbstractLowRankApproximation, B::AbstractMatrix) = Matrix(A) .* B
broadcasted(::typeof(*), A::AbstractMatrix, B::AbstractLowRankApproximation) = A .* Matrix(B)

broadcasted(::typeof(+), A::AbstractLowRankApproximation, α::Number) = add_scalar(A, α)
broadcasted(::typeof(+), α::Number, A::AbstractLowRankApproximation) = add_scalar(A, α)
broadcasted(::typeof(+), A::AbstractLowRankApproximation, b::AbstractVector) = add_to_cols(A, b)
broadcasted(::typeof(+), A::AbstractLowRankApproximation, b::Adjoint{<:Number, <:AbstractVector}) = add_to_rows(A, transpose(b))
broadcasted(::typeof(+), A::AbstractLowRankApproximation, b::Transpose{<:Number, <:AbstractVector}) = add_to_rows(A, transpose(b))

broadcasted(::typeof(+), b::AbstractVector, A::AbstractLowRankApproximation) = add_to_cols(A, b)
broadcasted(::typeof(+), b::Adjoint{<:Number, <:AbstractVector}, A::AbstractLowRankApproximation) = add_to_rows(A, transpose(b))
broadcasted(::typeof(+), b::Transpose{<:Number, <:AbstractVector}, A::AbstractLowRankApproximation) = add_to_rows(A, transpose(b))

broadcasted(::typeof(^), A::AbstractLowRankApproximation, d::Int) = elpow(A, d)
broadcasted(::typeof(Base.literal_pow), ::typeof(^), A::AbstractLowRankApproximation, ::Val{d}) where d = elpow(A, d)

include("utils.jl")
include("orthonormalization.jl")
include("two_factor_approximation.jl")
include("svd_like_factorization.jl")
end

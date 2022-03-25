module LowRankArithmetic

using Combinatorics, LinearAlgebra, UnPack
import Base: +, -, *, size, Matrix, getindex, hcat, vcat, axes, broadcasted, BroadcastStyle
import LinearAlgebra: rank, adjoint, svd

abstract type AbstractLowRankRepresentation end

# catch-all-cases
axes(LRA::AbstractLowRankRepresentation) = map(Base.oneto, size(LRA))
hadamard(A::AbstractLowRankRepresentation, B::AbstractMatrix) = Matrix(A) .* B
hadamard(A::AbstractMatrix, B::AbstractLowRankRepresentation) = Matrix(A) .* B
hadamard(A,B) = A .* B

# standard linear algbera broadcasts 
broadcasted(::typeof(*), A::AbstractLowRankRepresentation, B::AbstractLowRankRepresentation) = hadamard(A,B)
broadcasted(::typeof(*), A::AbstractLowRankRepresentation, b::AbstractVector) = multiply_cols(A,b)
broadcasted(::typeof(*), A::AbstractLowRankRepresentation, b::Adjoint{<:Number, <:AbstractVector}) = multiply_rows(A, transpose(b))
broadcasted(::typeof(*), A::AbstractLowRankRepresentation, b::Transpose{<:Number, <:AbstractVector}) = multiply_rows(A, transpose(b))
broadcasted(::typeof(*), A::AbstractLowRankRepresentation, B::AbstractMatrix) = Matrix(A) .* B
broadcasted(::typeof(*), A::AbstractMatrix, B::AbstractLowRankRepresentation) = A .* Matrix(B)

broadcasted(::typeof(+), A::AbstractLowRankRepresentation, α::Number) = add_scalar(A, α)
broadcasted(::typeof(+), α::Number, A::AbstractLowRankRepresentation) = add_scalar(A, α)
broadcasted(::typeof(+), A::AbstractLowRankRepresentation, b::AbstractVector) = add_to_cols(A, b)
broadcasted(::typeof(+), A::AbstractLowRankRepresentation, b::Adjoint{<:Number, <:AbstractVector}) = add_to_rows(A, transpose(b))
broadcasted(::typeof(+), A::AbstractLowRankRepresentation, b::Transpose{<:Number, <:AbstractVector}) = add_to_rows(A, transpose(b))

broadcasted(::typeof(+), b::AbstractVector, A::AbstractLowRankRepresentation) = add_to_cols(A, b)
broadcasted(::typeof(+), b::Adjoint{<:Number, <:AbstractVector}, A::AbstractLowRankRepresentation) = add_to_rows(A, transpose(b))
broadcasted(::typeof(+), b::Transpose{<:Number, <:AbstractVector}, A::AbstractLowRankRepresentation) = add_to_rows(A, transpose(b))

broadcasted(::typeof(^), A::AbstractLowRankRepresentation, d::Int) = elpow(A, d)
broadcasted(::typeof(Base.literal_pow), ::typeof(^), A::AbstractLowRankRepresentation, ::Val{d}) where d = elpow(A, d)

include("utils.jl")
include("orthonormalization.jl")
include("two_factor_representation.jl")
include("svd_like_representation.jl")
end

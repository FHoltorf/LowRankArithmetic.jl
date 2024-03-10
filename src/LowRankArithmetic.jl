module LowRankArithmetic

using Combinatorics, LinearAlgebra, UnPack, TSVD
import Base: +, -, *, size, Matrix, getindex, hcat, vcat, axes, round, broadcasted, BroadcastStyle, eltype
import LinearAlgebra: rank, adjoint, svd, qr

struct TSVD end

abstract type AbstractLowRankRepresentation end

# catch-all-cases
axes(LRA::AbstractLowRankRepresentation) = map(Base.oneto, size(LRA))
hadamard(A::AbstractLowRankRepresentation, B::AbstractMatrix) = Matrix(A) .* B
hadamard(A::AbstractMatrix, B::AbstractLowRankRepresentation) = Matrix(A) .* B
hadamard(A,B) = A .* B

# standard linear algebra broadcasts 
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

include("orthonormalization.jl")
include("utils.jl")
include("two_factor_representation.jl")
include("svd_like_representation.jl")
end

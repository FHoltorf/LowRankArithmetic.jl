 
# <img align = center src="docs/assets/lowrankarithmetic_logo.png" alt="logo" width="150"/>  &emsp;LowRankArithmetic.jl 

[![Build Status](https://github.com/FHoltorf/LowRankArithmetic.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FHoltorf/LowRankArithmetic.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/FHoltorf/LowRankArithmetic.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/FHoltorf/LowRankArithmetic.jl)

LowRankArithmetic.jl facilitates the propagation of low rank factorizations through finite composition of a range of common linear algebra operations such as
* matrix-matrix/vector multiplication
* addition
* Hadamard products
* elementwise integer powers
* concatenation & slicing

Two types of low-rank representations are supported:

1. Two-factor representation:

$$
\mathbb{R}^{n\times m} \ni X = UZ^\top \text{ where }U\in \mathbb{R}^{n\times r}, Z\in \mathbb{R}^{m\times r}
$$
 
2.  SVD-like representation:

$$
\mathbb{R}^{n\times m} \ni X = USV^\top \text{ where } U\in \mathbb{R}^{n\times r}, S\in \mathbb{R}^{r\times r}, V\in \mathbb{R}^{m\times r}
$$

Note, however, that neither U and V need to be orthogonal, nor are S and Z required to be diagonal or lower triangular as may be familiar from the standard QR and SVD factorizations. In particular, these properties are not maintained when a QR or SVD factorization is propagated through the supported arithmetic operations. 

LowRankArithmetic.jl further supports efficient & robust svd-based rounding procedures to reduce the rank of a given low rank factorization. Also efficient Gram-Schmidt-, QR-, SVD-, and gradient flow-based reorthonormalization procedures for the U and V factors are available. 

## Acknowledgements
This work is supported by NSF Award PHY-2028125 "SWQU: Composable Next Generation Software Framework for Space Weather Data Assimilation and Uncertainty Quantification".
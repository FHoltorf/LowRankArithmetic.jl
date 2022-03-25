 
# LowRankArithmetic.jl &emsp;<img align = center src="docs/assets/lowrankarithmetic_logo.png" alt="logo" width="150"/>

[![Build Status](https://github.com/FHoltorf/LowRankArithmetic.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/FHoltorf/LowRankArithmetic.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/FHoltorf/LowRankArithmetic.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/FHoltorf/LowRankArithmetic.jl)

LowRankArithmetic.jl facilitates the propagation of low rank factorizations through the compositions of many common linear algebra operations such as
* matrix-matrix/vector multiplication
* addition
* Hadamard products
* elementwise integer powers
* concatenation & slicing

Two types of low-rank representations are supported:

1. Two-factor representation:

&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^{n\times m} \ni X = UZ^\top"> where 
<img src="https://render.githubusercontent.com/render/math?math=U\in \mathbb{R}^{n\times r}, Z\in \mathbb{R}^{m\times r}">

 
2.  SVD-like representation:

&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^{n\times m} \ni X = USV^\top"> where 
<img src="https://render.githubusercontent.com/render/math?math=U\in \mathbb{R}^{n\times r}, S\in \mathbb{R}^{r\times r}, V\in \mathbb{R}^{m\times r}">

Note, however, that $U$ and $V$ need not be orthogonal, nor are $S$ and $Z$ required to be diagonal or upper triangular as may be familiar from the QR or SVD factorizations. In particular, these properties are not maintained when a QR or SVD factorization is propagated through different arithmetic operations. 

LowRankArithmetic.jl further supports efficient & robust svd-based rounding procedures to reduce the rank of a given low rank factorization. Also efficient Gram-Schmidt-, QR-, SVD-, and gradient flow-based reorthonormalization procedures for the $U$ and $V$ factors are available. 


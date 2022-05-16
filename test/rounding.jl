@testset "Rounding SVDLikeRepresentation" begin
    Q = qr(randn(100,100)).Q
    D = Diagonal([2.0^(-j) for j in 1:100])
    A = exp(Matrix(Q))*D*exp(Matrix(transpose(Q)))
    LRA = truncated_svd(A, tol = 0.01)

    # svd based rounding 
    LRA_rounded = round(LRA + LRA)
    @test typeof(LRA_rounded) <: SVDLikeRepresentation
    @test rank(LRA_rounded) == rank(2*LRA)
    @test norm(Matrix(LRA_rounded) - Matrix(2*LRA)) <= 1e-8

    LRA_rounded = round(LRA .* LRA)
    @test rank(LRA_rounded) == rank(LRA.^2)
    @test norm(Matrix(LRA_rounded) - Matrix(LRA.^2)) <= 1e-8

    # tsvd based rounding
    LRA_tsvd_rounded = round(LRA .* LRA, TSVD(), rmax = 5)
    LRA_svd_rounded = round(LRA .* LRA, rmax = 5)
    @test typeof(LRA_tsvd_rounded) <: SVDLikeRepresentation
    @test norm(Matrix(LRA_tsvd_rounded) - Matrix(LRA_svd_rounded)) <= 1e-8
end

@testset "Rounding TwoFactorRepresentation" begin
    Q = qr(randn(100,100)).Q
    D = Diagonal([2.0^(-j) for j in 1:100])
    A = exp(Matrix(Q))*D*exp(Matrix(transpose(Q)))
    LRA = TwoFactorRepresentation(truncated_svd(A, tol = 0.01))

    LRA_rounded = round(LRA + LRA)
    @test typeof(LRA_rounded) <: TwoFactorRepresentation
    @test rank(LRA_rounded) == rank(2*LRA)
    @test norm(Matrix(LRA_rounded) - Matrix(2*LRA)) <= 1e-8

    LRA_rounded = round(LRA .* LRA)
    @test rank(LRA_rounded) == rank(LRA.^2)
    @test norm(Matrix(LRA_rounded) - Matrix(LRA.^2)) <= 1e-8

    # tsvd based rounding
    LRA_tsvd_rounded = round(LRA .* LRA, TSVD(), rmax = 5)
    LRA_svd_rounded = round(LRA .* LRA, rmax = 5)
    @test typeof(LRA_tsvd_rounded) <: TwoFactorRepresentation
    @test norm(Matrix(LRA_tsvd_rounded) - Matrix(LRA_svd_rounded)) <= 1e-8
end
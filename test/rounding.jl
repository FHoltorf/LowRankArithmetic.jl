@testset "Truncated SVD" begin
    Q = diagm([2.0^(-j) for j in 1:100])
    tol = sum(2.0^(-j) for j in 27:100)

    # via SVD
    Q_red = truncated_svd(Q, 12)
    @test rank(Q_red) == 12
    @test norm(Matrix(Q_red) - diagm([j <= 12 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9

    Q_red = truncated_svd(Q, tol = tol)
    @test rank(Q_red) == 26
    @test norm(Matrix(Q_red) - diagm([j <= 26 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9

    Q_red = truncated_svd(Q, rmin = 30)
    @test rank(Q_red) == 30
    @test norm(Matrix(Q_red) - diagm([j <= 30 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9

    Q_red = truncated_svd(Q, rmax = 12)
    @test rank(Q_red) == 12
    @test norm(Matrix(Q_red) - diagm([j <= 12 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9

    #via TSVD
    Q_red = truncated_svd(Q, TSVD(), rmax = 12)
    @test rank(Q_red) == 12
    @test norm(Matrix(Q_red) - diagm([j <= 12 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9

    Q_red = truncated_svd(Q, 12, TSVD())
    @test rank(Q_red) == 12 
    @test norm(Matrix(Q_red) - diagm([j <= 12 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9

    # test truncated SVD for low rank matrices
    Q_lr_svd = truncated_svd(Q, 50)
    Q_lr_tf = TwoFactorRepresentation(Q_lr_svd)
    Qs = [Q_lr_svd, Q_lr_tf]
    algs = [SVDFact(), TSVD()]
    
    # via SVD
    for Q_lr in Qs
        Q_red = truncated_svd(Q_lr, 12)
        @test rank(Q_red) == 12
        @test norm(Matrix(Q_red) - diagm([j <= 12 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9

        Q_red = truncated_svd(Q_lr; rmax = 17, rmin = 7)
        @test rank(Q_red) == 17
        @test norm(Matrix(Q_red) - diagm([j <= 17 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9

        tol = sum(2.0^(-j) for j in 27:50)
        Q_red = truncated_svd(Q_lr, tol = tol)
        @test rank(Q_red) == 26
        @test norm(Matrix(Q_red) - diagm([j <= 26 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9
    end

    # via TSVD
    for Q_lr in Qs
        Q_red = truncated_svd(Q_lr, 12, TSVD())
        @test rank(Q_red) == 12
        @test norm(Matrix(Q_red) - diagm([j <= 12 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9

        Q_red = truncated_svd(Q_lr, TSVD(); rmax = 17, rmin = 7)
        @test rank(Q_red) == 17
        @test norm(Matrix(Q_red) - diagm([j <= 17 ? 2.0^(-j) : 0 for j in 1:100])) < 1e-9
    end
end

@testset "Rounding SVDLikeRepresentation" begin
    Q = qr(randn(100,100)).Q
    D = Diagonal([2.0^(-j) for j in 1:100])
    A = exp(Matrix(Q))*D*exp(Matrix(Q'))
    LRA = truncated_svd(A, tol = 0.01)

    # svd based rounding 
    LRA_rounded = round(LRA + LRA)
    LRA_truncated = truncated_svd(LRA + LRA)
    @test typeof(LRA_rounded) <: SVDLikeRepresentation
    @test rank(LRA_rounded) == rank(2*LRA)
    @test rank(LRA_truncated) == rank(2*LRA)
    @test norm(Matrix(LRA_rounded) - Matrix(2*LRA)) <= 1e-8
    @test norm(Matrix(LRA_truncated) - Matrix(2*LRA)) <= 1e-8

    LRA_rounded = round(LRA .* LRA)
    @test rank(LRA_rounded) == rank(LRA.^2)
    @test norm(Matrix(LRA_rounded) - Matrix(LRA.^2)) <= 1e-8

    # tsvd based rounding
    LRA_tsvd_rounded = round(LRA .* LRA, TSVD(), rmax = 5)
    LRA_svd_rounded = round(LRA .* LRA, rmax = 5)
    LRA_tsvd_truncated = truncated_svd(LRA .* LRA, 5, TSVD())
    @test typeof(LRA_tsvd_rounded) <: SVDLikeRepresentation
    @test norm(Matrix(LRA_tsvd_rounded) - Matrix(LRA_svd_rounded)) <= 1e-8
end

@testset "Rounding TwoFactorRepresentation" begin
    Q = qr(randn(100,100)).Q
    D = Diagonal([2.0^(-j) for j in 1:100])
    A = exp(Matrix(Q))*D*exp(Matrix(transpose(Q)))
    LRA = TwoFactorRepresentation(truncated_svd(A, tol = 0.01))

    # svd based rounding
    LRA_rounded = round(LRA + LRA)
    LRA_truncated = truncated_svd(LRA + LRA)
    @test typeof(LRA_rounded) <: TwoFactorRepresentation
    @test rank(LRA_rounded) == rank(2*LRA)
    @test rank(LRA_truncated) == rank(2*LRA)
    @test norm(Matrix(LRA_rounded) - Matrix(2*LRA)) <= 1e-8
    @test norm(Matrix(LRA_truncated) - Matrix(2*LRA)) <= 1e-8

    # tsvd based rounding
    LRA_tsvd_rounded = round(LRA .* LRA, TSVD(), rmax = 5)
    LRA_svd_rounded = round(LRA .* LRA, rmax = 5)
    LRA_tsvd_truncated = truncated_svd(LRA .* LRA, 5, TSVD())
    @test typeof(LRA_tsvd_rounded) <: TwoFactorRepresentation
    @test norm(Matrix(LRA_tsvd_rounded) - Matrix(LRA_svd_rounded)) <= 1e-8
    @test norm(Matrix(LRA_tsvd_truncated) - Matrix(LRA_svd_rounded)) <= 1e-8
end

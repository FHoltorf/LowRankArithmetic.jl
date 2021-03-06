@testset "Matrix orthonormalization tests" begin
    algs = [QRFact(),
            SVDFact(), 
            SecondMomentMatching(),
            GramSchmidt()]
    for alg in algs
        U = randn(1000, 10)
        Z = randn(1000, 10)
        X = U*Z'
        orthonormalize!(U, Z, alg)
        @test maximum(abs.(U'*U - I)) <= 1e-8
        @test maximum(abs.(U*Z' - X)) <= 1e-8

        U = randn(1000, 10)
        orthonormalize!(U, alg)
        @test maximum(abs.(U'*U - I)) <= 1e-8
    end
end

@testset "LRA orthonormalization tests" begin
    algs = [QRFact(),
            SVDFact(), 
            SecondMomentMatching(),
            GramSchmidt()]
    for alg in algs
        U = randn(1000, 10)
        Z = randn(1000, 10)
        X = U*Z'
        LRA = TwoFactorRepresentation(U,Z)
        orthonormalize!(LRA, alg)
        @test maximum(abs.(LRA.U'*LRA.U - I)) <= 1e-8
        @test maximum(abs.(LRA - X)) <= 1e-8

        U = randn(1000, 10)
        S = randn(10, 10)
        V = randn(1000, 10)
        X = U*S*V'
        LRA = SVDLikeRepresentation(U,S,V)
        orthonormalize!(LRA, alg)
        @test maximum(abs.(LRA.U'*LRA.U-I)) + maximum(abs.(LRA.V'*LRA.V-I)) <= 1e-8
        @test maximum(abs.(LRA - X)) <= 1e-8
    end
end
using LinearAlgebra

@testset "TwoFactorRepresentation - Basic Arithmetic" begin
    U1, Z1 = randn(100, 4), randn(1000, 4)

    U2, Z2 = randn(100, 2), randn(1000, 2)

    A = U1*Z1'
    lr_A = TwoFactorRepresentation(U1,Z1)

    B = U2*Z2'
    lr_B = TwoFactorRepresentation(U2,Z2)

    # testing addition
    @test norm(Matrix(lr_A + lr_B) - (A + B)) < 1e-10

    U3, Z3 = randn(1000, 7), randn(128, 7)
    C = U3*Z3'
    lr_C = TwoFactorRepresentation(U3, Z3)
    full_factor = randn(size(A))'

    # testing product
    @test norm(Matrix(lr_A*lr_C) - A*C) < 1e-10
    @test norm(Matrix(lr_A*full_factor) - A*full_factor) < 1e-10
    @test norm(Matrix(full_factor*lr_A) - full_factor*A) < 1e-10

    # testing elementwise product
    U4, Z4 = randn(100, 2), randn(1000,2)
    D = U4*Z4'
    lr_D = TwoFactorRepresentation(U4,Z4)
    @test norm(Matrix(lr_A .* lr_D) - A .* D) < 1e-10

    # testing elementwise power
    for d in 1:5
        @test norm(Matrix(lr_A .^ d) - A.^d) < 1e-8
    end

    # test sum 
    lr_matrices = [TwoFactorRepresentation(randn(100, 5), randn(100, 5)) for i in 1:5]
    full_representations = [Matrix(lr_matrix) for lr_matrix in lr_matrices]
    @test norm(Matrix(sum(lr_matrices)) - sum(full_representations)) < 1e-8

    # test prod
    lr_matrices = [TwoFactorRepresentation(randn(100, 5), randn(100, 5)) for i in 1:5]
    full_representations = [Matrix(lr_matrix) for lr_matrix in lr_matrices]    
    @test norm(Matrix(prod(lr_matrices)) -  prod(full_representations)) < 1e-5

    # indexing tests
    A = TwoFactorRepresentation([i + j for i in 1:10, j in 1:3], [i + j for i in 1:5, j in 1:3])
    # individual indexing
    for i in 1:10, j in 1:5
        @test A[i, j] == Matrix(A)[i,j]
    end
    # slicing
    for i in 1:10
        @test all(Matrix(A[i, :]) .== Matrix(A)[[i],:])
    end
    for j in 1:5
        @test all(Matrix(A[:, j]) .== Matrix(A)[:, [j]])
    end
    # more indexing
    @test all(Matrix(A[[1, 3, 7], [5,1]]) .== Matrix(A)[[1,3,7], [5,1]])
    @test all(Matrix(A[[1, 3, 7], 2]) .== Matrix(A)[[1,3,7], [2]])
    @test all(Matrix(A[2, [3,2]]) .== Matrix(A)[[2], [3,2]])
    
    # concatenation tests
    B = TwoFactorRepresentation([i + j for i in 1:10, j in 1:3], [i + j for i in 1:5, j in 1:3])
    @test all(Matrix(hcat(A,B)) .== hcat(Matrix(A), Matrix(B)))
    @test all(Matrix(vcat(A,B)) .== vcat(Matrix(A), Matrix(B)))
end

@testset "SVDLikeRepresentation - Basic Arithmetic" begin
    U1, S1, V1 = randn(100, 4), randn(4,4), randn(1000, 4)

    U2, S2, V2 = randn(100, 2), randn(2,2), randn(1000, 2)

    A = U1*S1*V1'
    lr_A = SVDLikeRepresentation(U1,S1,V1)

    B = U2*S2*V2'
    lr_B = SVDLikeRepresentation(U2,S2,V2)

    # testing addition
    @test norm(Matrix(lr_A + lr_B) - (A + B)) < 1e-10

    U3, S3, V3 = randn(1000, 7), randn(7,7), randn(128, 7)
    C = U3*S3*V3'
    lr_C = SVDLikeRepresentation(U3, S3, V3)
    full_factor = randn(size(A))'

    # testing product
    @test norm(Matrix(lr_A*lr_C) - A*C) < 1e-8
    @test norm(Matrix(lr_A*full_factor) - A*full_factor) < 1e-8
    @test norm(Matrix(full_factor*lr_A) - full_factor*A) < 1e-8

    # testing elementwise product
    U4, S4, V4 = randn(100, 2), randn(2,2), randn(1000,2)
    D = U4*S4*V4'
    lr_D = SVDLikeRepresentation(U4,S4,V4)
    @test norm(Matrix(lr_A .* lr_D) - A .* D) < 1e-8

    # testing elementwise power
    # very lenient test due to poor implementation
    for d in 1:5
        @test norm(Matrix(lr_A .^ d) - A.^d) < 1e-3
    end

    # test sum 
    lr_matrices = [SVDLikeRepresentation(randn(100, 5), randn(5,5), randn(100, 5)) for i in 1:5]
    full_representations = [Matrix(lr_matrix) for lr_matrix in lr_matrices]
    @test norm(Matrix(sum(lr_matrices)) - sum(full_representations)) < 1e-8

    # test prod
    @test norm(Matrix(prod(lr_matrices)) -  prod(full_representations)) < 1e-5

    # indexing tests
    A = SVDLikeRepresentation([i + j for i in 1:10, j in 1:3], ones(3,3), [i + j for i in 1:5, j in 1:3])
    # individual indexing
    for i in 1:10, j in 1:5
        @test A[i, j] == Matrix(A)[i,j]
    end
    # slicing
    for i in 1:10
        @test all(Matrix(A[i, :]) .== Matrix(A)[[i],:])
    end
    for j in 1:5
        @test all(Matrix(A[:, j]) .== Matrix(A)[:, [j]])
    end
    # more indexing
    @test all(Matrix(A[[1, 3, 7], [5,1]]) .== Matrix(A)[[1,3,7], [5,1]])
    @test all(Matrix(A[[1, 3, 7], 2]) .== Matrix(A)[[1,3,7], [2]])
    @test all(Matrix(A[2, [3,2]]) .== Matrix(A)[[2], [3,2]])
    
    # concatenation tests
    B = SVDLikeRepresentation([i + j for i in 1:10, j in 1:3], ones(3,3), [i + j for i in 1:5, j in 1:3])
    @test all(Matrix(hcat(A,B)) .== hcat(Matrix(A), Matrix(B)))
    @test all(Matrix(vcat(A,B)) .== vcat(Matrix(A), Matrix(B)))
end

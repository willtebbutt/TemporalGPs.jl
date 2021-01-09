@testset "model_test_utils" begin
    storages = [
        (name="dense storage", val=ArrayStorage(Float64)),
        (name="static storage", val=SArrayStorage(Float64)),
    ]
    rng = MersenneTwister(123456)
    @testset "storage = $(storage.name)" for storage in storages
        @testset "random_vector" begin
            a = random_vector(rng, 3, storage.val)
            @test is_of_storage_type(a, storage.val)
            @test length(a) == 3
        end
        @testset "random_matrix" begin
            A = random_matrix(rng, 4, 3, storage.val)
            @test is_of_storage_type(A, storage.val)
            @test size(A) == (4, 3)
        end
        @testset "random_nice_psd_matrix" begin
            Σ = random_nice_psd_matrix(rng, 11, storage.val)
            @test all(eigvals(Σ) .> 0)
            @test all(eigvals(Σ) .< 1)
            @test is_of_storage_type(Σ, storage.val)
        end
        @testset "random_gaussian" begin
            x = random_gaussian(rng, 3, storage.val)
            @test is_of_storage_type(x, storage.val)
            @test length(x.m) == 3
            @test size(x.P) == (3, 3)
            @test all(eigvals(x.P) .> 0) 
        end
        @testset "GaussMarkovModel" begin
            @testset "time-varying" begin
                x = random_tv_gmm(rng, Forward(), 3, 11, storage.val)
                @test length(x) == 11
                @test dim(x) == 3
            end
            @testset "time-invariant" begin
                x = random_ti_gmm(rng, Forward(), 2, 12, storage.val)
                @test length(x) == 12
                @test dim(x) == 2
            end
        end
        # @testset "LGSSM" begin
        #     @testset "SmallOutput -- time-varying" begin
        #         x = random_tv_lgssm(rng, Forward(), 3, 2, 11, storage.val)
        #         @test length(x) == 11

        #         # Just run `rand_tangent` without checking correctness.
        #         rand_tangent(rng, x)
        #     end
        #     @testset "SmallOutput -- time-invariant" begin
        #         x = random_ti_lgssm(rng, Forward(), 3, 2, 11, storage.val)
        #         @test length(x) == 11
        #         @test x.transitions.As isa Fill
        #         @test x.transitions.as isa Fill
        #         @test x.transitions.Qs isa Fill
        #         @test x.emissions.A isa Fill
        #         @test x.emissions.a isa Fill
        #         @test x.emissions.Q isa Fill

        #         # Just run `rand_tangent` without checking correctness.
        #         rand_tangent(rng, x)
        #     end
        #     @testset "LargeOutput -- time-varying" begin
        #     end
        # end
    end
end

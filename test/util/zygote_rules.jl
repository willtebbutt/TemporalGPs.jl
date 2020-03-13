using StaticArrays
using TemporalGPs: time_exp

@testset "zygote_rules" begin
    @testset "SVector" begin
        rng = MersenneTwister(123456)
        N = 5
        x = randn(rng, N)
        ȳ = SVector{N}(randn(rng, N))
        adjoint_test(SVector{N}, ȳ, x)
    end
    @testset "SMatrix" begin
        rng = MersenneTwister(123456)
        P, Q = 5, 4
        X = randn(rng, P, Q)
        Ȳ = SMatrix{P, Q}(randn(rng, P, Q))
        adjoint_test(SMatrix{P, Q}, Ȳ, X)
    end
    # @testset "SArray" begin
    #     rng = MersenneTwister(123456)
    #     P, Q = 4, 5
    #     X = randn(rng, P, Q)
    #     Ȳ = SArray{Tuple{P, Q}}(randn(rng, P, Q))
    #     adjoint_test(SArray{Tuple{P, Q}}, Ȳ, X)
    # end
    @testset "time_exp" begin
        rng = MersenneTwister(123456)
        A = randn(rng, 3, 3)
        t = 0.1
        ΔB = randn(rng, 3, 3)
        adjoint_test(t->time_exp(A, t), ΔB, t)
    end
    @testset "collect(::Fill)" begin
        rng = MersenneTwister(123456)
        P = 11
        Q = 3
        xs = [
            randn(rng),
            randn(rng, 1, 2),
            SMatrix{1, 2}(randn(rng, 1, 2)),
        ]
        Δs = [
            (randn(rng, P), randn(rng, P, Q)),
            ([randn(rng, 1, 2) for _ in 1:P], [randn(rng, 1, 2) for _ in 1:P, _ in 1:Q]),
            (
                [SMatrix{1, 2}(randn(rng, 1, 2)) for _ in 1:P],
                [SMatrix{1, 2}(randn(rng, 1, 2)) for _ in 1:P, _ in 1:Q],
            ),
        ]
        @testset "$(typeof(x)) element" for (x, Δ) in zip(xs, Δs)
            adjoint_test(x->collect(Fill(x, P)), first(Δ), x)
            adjoint_test(x->collect(Fill(x, P, Q)), last(Δ), x)
        end
    end
    @testset "reinterpret" begin
        rng = MersenneTwister(123456)
        P = 11
        y = randn(rng, P)
        Δy = randn(rng, P)
        T = SVector{1, Float64}
        α = T.(randn(rng, P))
        Δα = T.(randn(rng, P))
        adjoint_test(y->reinterpret(T, y), Δα, y)
        adjoint_test(α->reinterpret(Float64, α), Δy, α)
    end
end

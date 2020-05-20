using TemporalGPs:
    Immutable,
    correlate,
    decorrelate,
    correlate_pullback,
    decorrelate_pullback,
    copy_first

@testset "checkpointed_immutable_pullbacks" begin

    # Construct an LGSSM.
    rng = MersenneTwister(123456)
    Dlat = 3
    Dobs = 2
    N = 10
    storage = ArrayStorage(Float64)
    model = random_ti_lgssm(rng, Dlat, Dobs, N, storage)

    model_checkpointed = checkpointed(model)

    # Generate data for correlate / decorrelate operations.
    y = rand(model)
    _, α = TemporalGPs.decorrelate(model, y)

    @testset "$(name)" for (name, foo_pullback) in [
        ("correlate", correlate_pullback),
        ("decorrelate", decorrelate_pullback),
    ]

        # Perform filtering / gradient propagation with no checkpointing.
        (lml_naive, ys_naive), pb_naive = foo_pullback(Immutable(), model, α, copy_first)

        # Perform filtering / gradient propagation with checkpointing.
        (lml_checkpoint, ys_checkpoint), pb_checkpoint = foo_pullback(
            Immutable(),
            model_checkpointed,
            α,
            copy_first,
        )

        @test lml_naive ≈ lml_checkpoint
        @test ys_naive == ys_checkpoint

        Δlml = randn()
        Δys = [randn(Dobs) for _ in 1:N]

        _, Δmodel_naive, Δαs_naive, _ = pb_naive((Δlml, Δys))
        _, Δmodel_checkpoint, Δαs_checkpoint = pb_checkpoint((Δlml, Δys))

        @test Δmodel_naive == Δmodel_checkpoint.model
        @test Δαs_naive == Δαs_checkpoint

    end
    @testset "logpdf" begin
        @test logpdf(model, y) ≈ logpdf(model_checkpointed, y)

        lml_naive, pb_naive = Zygote.pullback(logpdf, model, y)
        lml_check, pb_check = Zygote.pullback(logpdf, model_checkpointed, y)

        @test lml_naive ≈ lml_check

        Δmodel_naive, Δy_naive = pb_naive(1.0)
        Δmodel_check, Δy_check = pb_check(1.0)

        @test Δmodel_naive == Δmodel_check.model
        @test Δy_naive == Δy_check
    end
end
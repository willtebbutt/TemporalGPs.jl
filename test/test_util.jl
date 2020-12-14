using ChainRulesCore: backing
using TemporalGPs: Gaussian, copy_first, pick_last, correlate, decorrelate

create_psd_matrix(A::AbstractMatrix) = A * A' + I

function create_psd_stable_matrix(A::AbstractMatrix)
    B = create_psd_matrix(A)
    λ, U = eigen(B)
    λ .+= 1
    λ ./= (maximum(λ) + 1e-1 * maximum(λ))
    return Matrix(Symmetric(U * Diagonal(λ) * U'))
end



# Make FiniteDifferences work with some of the types in this package. Shame this isn't
# automated...

import FiniteDifferences: to_vec

function to_vec(x::Fill)
    x_vec, back_vec = to_vec(FillArrays.getindex_value(x))
    function Fill_from_vec(x_vec)
        return Fill(back_vec(x_vec), length(x))
    end
    return x_vec, Fill_from_vec
end

function to_vec(x::Union{Zeros, Ones})
    return Vector{eltype(x)}(undef, 0), _ -> x
end

function to_vec(x::Base.ReinterpretArray)
    return to_vec(collect(x))
end

function to_vec(x::T) where {T<:NamedTuple}
    isempty(fieldnames(T)) && throw(error("Expected some fields. None found."))
    vecs_and_backs = map(name->to_vec(getfield(x, name)), fieldnames(T))
    vecs, backs = first.(vecs_and_backs), last.(vecs_and_backs)
    x_vec, back = to_vec(vecs)
    function namedtuple_to_vec(x′_vec)
        vecs′ = back(x′_vec)
        x′s = map((back, vec)->back(vec), backs, vecs′)
        return (; zip(fieldnames(T), x′s)...)
    end
    return x_vec, namedtuple_to_vec
end

function to_vec(x::T) where {T<:StaticArray}
    x_dense = collect(x)
    x_vec, back_vec = to_vec(x_dense)
    function StaticArray_to_vec(x_vec)
        return T(back_vec(x_vec))
    end
    return x_vec, StaticArray_to_vec
end

function to_vec(x::TemporalGPs.Gaussian)
    m_vec, m_from_vec = to_vec(x.m)
    P_vec, P_from_vec = to_vec(x.P)

    x_vec, x_back = to_vec((m_vec, P_vec))

    function Gaussian_from_vec(x_vec)
        mP_vec = x_back(x_vec)

        m = m_from_vec(mP_vec[1])
        P = P_from_vec(mP_vec[2])

        return TemporalGPs.Gaussian(m, P)
    end

    return x_vec, Gaussian_from_vec
end

function to_vec(gmm::TemporalGPs.GaussMarkovModel)
    A_vec, A_back = to_vec(gmm.A)
    a_vec, a_back = to_vec(gmm.a)
    Q_vec, Q_back = to_vec(gmm.Q)
    H_vec, H_back = to_vec(gmm.H)
    h_vec, h_back = to_vec(gmm.h)
    x0_vec, x0_back = to_vec(gmm.x0)

    gmm_vec, gmm_back = to_vec((A_vec, a_vec, Q_vec, H_vec, h_vec, x0_vec))

    function GaussMarkovModel_from_vec(gmm_vec)
        vecs = gmm_back(gmm_vec)
        A = A_back(vecs[1])
        a = a_back(vecs[2])
        Q = Q_back(vecs[3])
        H = H_back(vecs[4])
        h = h_back(vecs[5])
        x0 = x0_back(vecs[6])
        return TemporalGPs.GaussMarkovModel(A, a, Q, H, h, x0)
    end

    return gmm_vec, GaussMarkovModel_from_vec
end

function to_vec(model::TemporalGPs.LGSSM)
    gmm_vec, gmm_from_vec = to_vec(model.gmm)
    Σ_vec, Σ_from_vec = to_vec(model.Σ)

    model_vec, back = to_vec((gmm_vec, Σ_vec))

    function LGSSM_from_vec(model_vec)
        tmp = back(model_vec)
        gmm = gmm_from_vec(tmp[1])
        Σ = Σ_from_vec(tmp[2])
        return TemporalGPs.LGSSM(gmm, Σ)
    end

    return model_vec, LGSSM_from_vec
end

function to_vec(X::BlockDiagonal)
    Xs = blocks(X)
    Xs_vec, Xs_from_vec = to_vec(Xs)

    function BlockDiagonal_from_vec(Xs_vec)
        Xs = Xs_from_vec(Xs_vec)
        return BlockDiagonal(Xs)
    end

    return Xs_vec, BlockDiagonal_from_vec
end

function to_vec(X::KroneckerProduct)
    A, B = getmatrices(X)
    A_vec, A_from_vec = to_vec(A)
    B_vec, B_from_vec = to_vec(B)
    X_vec, back = to_vec((A_vec, B_vec))

    function KroneckerProduct_from_vec(X_vec)
        (A_vec, B_vec) = back(X_vec)
        A = A_from_vec(A_vec)
        B = B_from_vec(B_vec)
        return A ⊗ B
    end

    return X_vec, KroneckerProduct_from_vec
end

to_vec(::typeof(copy_first)) = Bool[], _ -> copy_first
to_vec(::typeof(pick_last)) = Bool[], _ -> pick_last
to_vec(::Nothing) = Bool[], _ -> nothing

# Ensure that to_vec works for the types that we care about in this package.
@testset "custom FiniteDifferences stuff" begin
    @testset "NamedTuple" begin
        a, b = 5.0, randn(2)
        t = (a=a, b=b)
        nt_vec, back = to_vec(t)
        @test nt_vec isa Vector{Float64}
        @test back(nt_vec) == t
    end
    @testset "Fill" begin
        @testset "$(typeof(val))" for val in [5.0, randn(3)]
            x = Fill(val, 5)
            x_vec, back = to_vec(x)
            @test x_vec isa Vector{Float64}
            @test back(x_vec) == x
        end
    end
    @testset "Zeros{T}" for T in [Float32, Float64]
        x = Zeros{T}(4)
        x_vec, back = to_vec(x)
        @test x_vec isa Vector{eltype(x)}
        @test back(x_vec) == x
    end
    @testset "gaussian" begin
        @testset "Gaussian" begin
            x = TemporalGPs.Gaussian(randn(3), randn(3, 3))
            x_vec, back = to_vec(x)
            @test back(x_vec) == x
        end
    end
    @testset "to_vec(::GaussMarkovModel)" begin
        N = 11
        A = [randn(2, 2) for _ in 1:N]
        a = [randn(2) for _ in 1:N]
        Q = [randn(2, 2) for _ in 1:N]
        H = [randn(3, 2) for _ in 1:N]
        h = [randn(3) for _ in 1:N]
        x0 = TemporalGPs.Gaussian(randn(2), randn(2, 2))
        gmm = TemporalGPs.GaussMarkovModel(A, a, Q, H, h, x0)

        gmm_vec, gmm_from_vec = to_vec(gmm)
        @test gmm_vec isa Vector{<:Real}
        @test gmm_from_vec(gmm_vec) == gmm
    end
    @testset "to_vec(::LGSSM)" begin
        N = 11

        A = [randn(2, 2) for _ in 1:N]
        a = [randn(2) for _ in 1:N]
        Q = [randn(2, 2) for _ in 1:N]
        H = [randn(3, 2) for _ in 1:N]
        h = [randn(3) for _ in 1:N]
        x0 = TemporalGPs.Gaussian(randn(2), randn(2, 2))
        gmm = TemporalGPs.GaussMarkovModel(A, a, Q, H, h, x0)

        Σ = [randn(3, 3) for _ in 1:N]

        model = TemporalGPs.LGSSM(gmm, Σ)

        model_vec, model_from_vec = to_vec(model)
        @test model_from_vec(model_vec) == model
    end
    @testset "to_vec(::BlockDiagonal)" begin
        Ns = [3, 5, 1]
        Xs = map(N -> randn(N, N), Ns)
        X = BlockDiagonal(Xs)

        X_vec, X_from_vec = to_vec(X)
        @test X_vec isa Vector{<:Real}
        @test X_from_vec(X_vec) == X
    end
    @testset "to_vec(::KroneckerProduct" begin
        A = randn(4, 5)
        B = randn(6, 7)
        X = A ⊗ B

        X_vec, X_from_vec = to_vec(X)
        @test X_vec isa Vector{<:Real}
        @test X_from_vec(X_vec) == X
    end
end

my_zero(x) = zero(x)
my_zero(x::AbstractArray{<:Real}) = zero(x)
my_zero(x::AbstractArray) = map(my_zero, x)
my_zero(x::Tuple) = map(my_zero, x)

# My version of isapprox
function fd_isapprox(x_ad::Nothing, x_fd, rtol, atol)
    return fd_isapprox(x_fd, my_zero(x_fd), rtol, atol)
end
function fd_isapprox(x_ad::AbstractArray, x_fd::AbstractArray, rtol, atol)
    return all(fd_isapprox.(x_ad, x_fd, rtol, atol))
end
function fd_isapprox(x_ad::Real, x_fd::Real, rtol, atol)
    return isapprox(x_ad, x_fd; rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::NamedTuple, x_fd, rtol, atol)
    f = (x_ad, x_fd)->fd_isapprox(x_ad, x_fd, rtol, atol)
    return all([f(getfield(x_ad, key), getfield(x_fd, key)) for key in keys(x_ad)])
end
function fd_isapprox(x_ad::Tuple, x_fd::Tuple, rtol, atol)
    return all(map((x, x′)->fd_isapprox(x, x′, rtol, atol), x_ad, x_fd))
end
function fd_isapprox(x_ad::Dict, x_fd::Dict, rtol, atol)
    return all([fd_isapprox(get(()->nothing, x_ad, key), x_fd[key], rtol, atol) for
        key in keys(x_fd)])
end
function fd_isapprox(x::Gaussian, y::Gaussian, rtol, atol)
    return isapprox(x.m, y.m; rtol=rtol, atol=atol) &&
        isapprox(x.P, y.P; rtol=rtol, atol=atol)
end

# function adjoint_test(
#     f, ȳ, x...;
#     rtol=1e-9,
#     atol=1e-9,
#     fdm=FiniteDifferences.central_fdm(5, 1),
#     print_results=false,
#     test=true,
#     check_infers=true,
# )
#     # Compute forwards-pass and j′vp.
#     adj_fd = j′vp(fdm, f, ȳ, x...)
#     y, back = Zygote.pullback(f, x...)
#     adj_ad = (back(ȳ))

#     if check_infers
#         @inferred Zygote.pullback(f, x...)
#         @inferred back(ȳ)
#     end

#     # Check that forwards-pass agrees with plain forwards-pass.
#     test && @test fd_isapprox(y, f(x...), rtol, atol)

#     # Check that ad and fd adjoints (approximately) agree.
#     print_results && print_adjoints(adj_ad, adj_fd, rtol, atol)
#     test && @test fd_isapprox(adj_ad, adj_fd, rtol, atol)

#     return adj_ad, adj_fd
# end

function LinearAlgebra.dot(x̄::NamedTuple, ẋ)
    return sum(n -> dot(getproperty(x̄, n), getproperty(ẋ, n)), propertynames(x̄))
end

function adjoint_test(
    f, ȳ, x::Tuple, ẋ::Tuple;
    rtol=1e-9,
    atol=1e-9,
    fdm=central_fdm(5, 1),
    test=true,
    check_infers=true,
    context=Zygote.Context(),
)
    # Compute <Jᵀ ȳ, ẋ> = <x̄, ẋ> using Zygote.
    y, pb = Zygote._pullback(context, f, x...)
    x̄ = pb(ȳ)[2:end]
    # @show typeof(x̄[1])
    # @show typeof(backing(backing(ẋ)[1]))
    # @show map(length ∘ first ∘ to_vec, x̄[1].gmm)
    # @show map(length ∘ first ∘ to_vec, backing(backing(backing(ẋ)[1]).gmm))
    inner_ad = dot(x̄, ẋ)
    # inner_ad = dot(first(to_vec(x̄)), first(to_vec(ẋ)))

    # Approximate <ȳ, J ẋ> = <ȳ, ẏ> using FiniteDifferences.
    ẏ = jvp(fdm, f, zip(x, ẋ)...)
    inner_fd = dot(ȳ, ẏ)
    # inner_fd = dot(first(to_vec(ȳ)), first(to_vec(ẏ)))

    # Check that Zygote didn't modify the forwards-pass.
    test && @test fd_isapprox(y, f(x...), rtol, atol)

    # Check for approximate agreement in "inner-products".
    test && @test fd_isapprox(inner_ad, inner_fd, rtol, atol)

    # Check type inference if requested.
    if check_infers
        @inferred Zygote._pullback(context, f, x...)
        @inferred pb(ȳ)
    end

    return x̄
end


function print_adjoints(adjoint_ad, adjoint_fd, rtol, atol)
    @show typeof(adjoint_ad), typeof(adjoint_fd)

    # println("ad")
    # display(adjoint_ad)
    # println()

    # println("fd")
    # display(adjoint_fd)
    # println()

    adjoint_ad, adjoint_fd = to_vec(adjoint_ad)[1], to_vec(adjoint_fd)[1]
    println("atol is $atol, rtol is $rtol")
    println("ad, fd, abs, rel")
    abs_err = abs.(adjoint_ad .- adjoint_fd)
    rel_err = abs_err ./ adjoint_ad
    display([adjoint_ad adjoint_fd abs_err rel_err])
    println()
end

using BenchmarkTools

function check_adjoint_allocations(
    f, Δoutput, input...;
    context=Zygote.Context(),
    max_forward_allocs=0,
    max_backward_allocs=0,
)
    _, pb = _pullback(context, f, input...)
    @test allocs(
        @benchmark(_pullback($context, $f, $input...); samples=1, evals=1),
    ) <= max_forward_allocs
    @test allocs(@benchmark $pb($Δoutput) samples=1 evals=1) <= max_backward_allocs
end

function benchmark_adjoint(f, ȳ, args...; disp=false)
    disp && println("primal")
    primal = @benchmark $f($args...)
    if disp
        display(primal)
        println()
    end

    disp && println("pullback generation")
    forward_pass = @benchmark Zygote.pullback($f, $args...)
    if disp
        display(forward_pass)
        println()
    end

    y, back = Zygote.pullback(f, args...)

    disp && println("pullback evaluation")
    reverse_pass = @benchmark $back($ȳ)
    if disp
        display(reverse_pass)
        println()
    end

    return primal, forward_pass, reverse_pass
end

"""
    ssm_interface_tests(
        rng::AbstractRNG, build_ssm, θ...;
        check_infers=true, check_rmad=false, rtol=1e-9, atol=1e-9,
    )

Basic consistency tests that any ssm should be able to satisfy. The purpose of these tests
is not to ensure correctness of any given implementation, only to ensure that it is self-
consistent and implements the required interface.

`build_ssm` should be a unary function that, when called on `θ`, should return an
`AbstractSSM`.
"""
function ssm_interface_tests(
    rng::AbstractRNG, build_ssm, θ...;
    check_infers=true, check_rmad=false, rtol=1e-9, atol=1e-9, check_ad=true,
)
    ssm = build_ssm(θ...)

    y = rand(rng, ssm)
    lml_decor, α = decorrelate(ssm, y, copy_first)

    @testset "basics" begin
        @inferred storage_type(ssm)
        @inferred dim_latent(ssm)
        (@inferred dim_obs(ssm)) == length(first(y))
        @test length(ssm) == length(y)
        @test eltype(ssm) == eltype(first(y))
        @test is_of_storage_type(ssm, storage_type(ssm))
    end

    @testset "decorrelate" begin
        @test decorrelate(ssm, y, copy_first) == decorrelate(ssm, y)
        @test lml_decor == logpdf(ssm, y)
        @test α == whiten(ssm, y)
        @test decorrelate(ssm, y, pick_last) == _filter(ssm, y)

        if check_infers
            @inferred decorrelate(ssm, y, copy_first)
            @inferred decorrelate(ssm, y)
            @inferred logpdf(ssm, y)
            @inferred whiten(ssm, y)
            @inferred _filter(ssm, y)
        end

        if check_ad
            @testset "decorrelate AD" begin
                Δ = rand_zygote_tangent(decorrelate(ssm, y, copy_first))
                adjoint_test(
                    (y, θ) -> decorrelate(build_ssm(θ...), y, copy_first), Δ, y, θ;
                    atol=atol, rtol=rtol, check_infers=false,
                )
                @code_warntype _pullback(Context(), decorrelate, ssm, y, copy_first)
                # @inferred _pullback(Context(), decorrelate, ssm, y, copy_first)
                out, pb = _pullback(Context(), decorrelate, ssm, y, copy_first)
                @code_warntype pb(Δ)
                # @inferred pb(Δ)
            end

            # @testset "logpdf AD" begin
            #     Δ = rand_zygote_tangent(lml_decor)
            #     adjoint_test(
            #         (y, θ) -> logpdf(build_ssm(θ...), y), Δ, y, θ;
            #         atol=atol, rtol=rtol,
            #     )
            #     @inferred _pullback(Context(), logpdf, ssm, y)
            #     out, pb = _pullback(Context(), logpdf, ssm, y)
            #     @inferred pb(Δ)
            # end

            # @testset "whiten" begin
            #     Δ = rand_zygote_tangent(α)
            #     adjoint_test(
            #         (y, θ) -> whiten(build_ssm(θ...), y), Δ, y, θ;
            #         atol=atol, rtol=rtol,
            #     )
            #     @inferred _pullback(Context(), whiten, ssm, y)
            #     out, pb = _pullback(Context(), whiten, ssm, y)
            #     @inferred pb(Δ)
            # end

            lml, ds = _filter(ssm, y)
            Δfilter = (
                rand_tangent(lml_decor),
                map(d -> (m=rand_tangent(mean(d)), P=rand_tangent(cov(d))), ds),
            )
            adjoint_test(
                (y, θ) -> _filter(build_ssm(θ...), y), Δfilter, y, θ; check_infers=false,
            )
        end
    end

    @testset "correlate" begin
        lml, y_cor = correlate(ssm, α, copy_first)
        @test (lml, y_cor) == correlate(ssm, α)
        @test lml ≈ logpdf(ssm, y)
        @test y_cor ≈ y
        @test y_cor == unwhiten(ssm, α)

        _lml, _y = logpdf_and_rand(rng, ssm)
        @test _lml ≈ logpdf(ssm, _y)
        @test length(_y) == length(y)

        if check_infers
            @inferred correlate(ssm, α, copy_first)
            @inferred correlate(ssm, α)
            @inferred unwhiten(ssm, α)
            @inferred logpdf_and_rand(rng, ssm)
        end

        # if check_ad
        #     @testset "correlate" begin
        #         Δ = rand_zygote_tangent(correlate(ssm, α, copy_first))
        #         adjoint_test(
        #             (α, θ) -> correlate(build_ssm(θ...), α, copy_first), Δ, α, θ;
        #             atol=atol, rtol=rtol,
        #         )
        #         @inferred _pullback(Context(), correlate, ssm, α, copy_first)
        #         out, pb = _pullback(Context(), correlate, ssm, α, copy_first)
        #         @inferred pb(Δ)
        #     end
        #     @testset "unwhiten" begin
        #         Δ = rand_zygote_tangent(y)
        #         adjoint_test(
        #             (α, θ) -> unwhiten(build_ssm(θ...), α), Δ, α, θ;
        #             atol=atol, rtol=rtol
        #         )
        #         @inferred _pullback(Context(), unwhiten, ssm, α)
        #         out, pb = _pullback(Context(), unwhiten, ssm, α)
        #         @inferred pb(Δ)
        #     end
        #     adjoint_test(
        #         θ -> logpdf_and_rand(deepcopy(rng), build_ssm(θ...)),
        #         rand_zygote_tangent((_lml, _y)), θ,
        #     )
        # end
    end

    @testset "statistics" begin
        ds = marginals(ssm)
        @test vcat(mean.(ds)...) ≈ mean(ssm)
        @test vcat(diag.(cov.(ds))...) ≈ diag(cov(ssm))

        if check_infers
            @inferred marginals(ssm)
        end

        if check_ad
            # adjoint_test(θ -> marginals(build_ssm(θ...)), rand_zygote_tangent(ds), θ)
        end
    end
end

function FiniteDifferences.rand_tangent(rng::AbstractRNG, A::StaticArray)
    return map(x -> rand_tangent(rng, x), A)
end

# Hacks to make rand_tangent play nicely with Zygote.
rand_zygote_tangent(A) = Zygote.wrap_chainrules_output(FiniteDifferences.rand_tangent(A))

Zygote.wrap_chainrules_output(x::Array) = map(Zygote.wrap_chainrules_output, x)

# function rand_zygote_tangent(A::Union{Tuple, NamedTuple})
#     t = FiniteDifferences.rand_tangent(A)
#     return ChainRulesCore.backing(t)
# end

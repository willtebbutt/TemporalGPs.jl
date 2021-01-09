"""
    LTISDE

A lightweight wrapper around a `GP` `f` that tells this package to handle inference in `f`.
Can be constructed via the `to_sde` function.
"""
struct LTISDE{Tf<:GP{<:Stheno.ZeroMean}, Tstorage<:StorageType} <: AbstractGP
    f::Tf
    storage::Tstorage
end

function to_sde(f::GP{<:Stheno.ZeroMean}, storage_type=ArrayStorage(Float64))
    return LTISDE(f, storage_type)
end

storage_type(f::LTISDE) = f.storage



"""
    const FiniteLTISDE = FiniteGP{<:LTISDE}

A `FiniteLTISDE` is just a regular `FiniteGP` that happens to contain an `LTISDE`, as
opposed to any other `AbstractGP`.
"""
const FiniteLTISDE = FiniteGP{<:LTISDE}

# Deal with a bug in Stheno.
function FiniteGP(f::LTISDE, x::AbstractVector{<:Real})
    return FiniteGP(f, x, convert(eltype(storage_type(f)), 1e-12))
end

# Implement Stheno's version of the FiniteGP API. This will eventually become AbstractGPs
# API, but Stheno is still on a slightly different API because I've yet to update it.

Stheno.mean(ft::FiniteLTISDE) = mean.(marginals(build_lgssm(ft)))

Stheno.cov(ft::FiniteLTISDE) = cov(FiniteGP(ft.f.f, ft.x, ft.Σy))

Stheno.marginals(ft::FiniteLTISDE) = vcat(map(marginals, marginals(build_lgssm(ft)))...)

Stheno.rand(rng::AbstractRNG, ft::FiniteLTISDE) = rand(rng, build_lgssm(ft))
Stheno.rand(ft::FiniteLTISDE) = rand(Random.GLOBAL_RNG, ft)
function Stheno.rand(rng::AbstractRNG, ft::FiniteLTISDE, N::Int)
    return hcat([rand(rng, ft) for _ in 1:N]...)
end
Stheno.rand(ft::FiniteLTISDE, N::Int) = rand(Random.GLOBAL_RNG, ft, N)

function Stheno.logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Real})
    model = build_lgssm(ft)
    return logpdf(model, restructure(y, model.emissions))
end

restructure(y::AbstractVector{<:Real}, ::StructArray{<:ScalarOutputLGC}) = y



# Converting GPs into LGSSMs.

using Stheno: MeanFunction, ConstMean, ZeroMean, BaseKernel, Sum, Stretched, Scaled

function build_lgssm(ft::FiniteLTISDE)
    As, as, Qs, Hs, hs, x0 = lgssm_components(ft.f.f.k, ft.x, ft.f.storage)
    return LGSSM(
        GaussMarkovModel(Forward(), As, as, Qs, x0),
        build_emissions(map(adjoint, Hs), hs, build_Σs(ft)),
    )
end

build_Σs(ft::FiniteLTISDE) = build_Σs(ft.x, ft.Σy)

build_Σs(::AbstractVector{<:Real}, Σ::Diagonal{<:Real}) = Σ.diag

function build_emissions(Hs::AbstractVector, hs::AbstractVector, Σs::AbstractVector)
    return StructArray{get_type(Hs, hs, Σs)}((Hs, hs, Σs))
end

function get_type(Hs_prime, hs::AbstractVector{<:Real}, Σs)
    THs = eltype(Hs_prime)
    Ths = eltype(hs)
    TΣs = eltype(Σs)
    T = ScalarOutputLGC{THs, Ths, TΣs}
    return T
end

function get_type(Hs_prime, hs::AbstractVector{<:AbstractVector}, Σs)
    THs = eltype(Hs_prime)
    Ths = eltype(hs)
    TΣs = eltype(Σs)
    T = SmallOutputLGC{THs, Ths, TΣs}
    return T
end

@inline function Zygote.wrap_chainrules_output(x::NamedTuple)
    return map(Zygote.wrap_chainrules_output, x)
end



# Generic constructors for base kernels.

function lgssm_components(
    k::BaseKernel, t::AbstractVector, storage::StorageType{T},
) where {T<:Real}

    # Compute stationary distribution and sde.
    x0 = stationary_distribution(k, storage)
    P = x0.P
    F, q, H = to_sde(k, storage)

    # Use stationary distribution + sde to compute finite-dimensional Gauss-Markov model.
    t = vcat([first(t) - 1], t)
    As = map(Δt -> time_exp(F, T(Δt)), diff(t))
    as = Fill(Zeros{T}(size(first(As), 1)), length(As))
    Qs = map(A -> P - A * P * A', As)
    Hs = Fill(H, length(As))
    hs = Fill(zero(T), length(As))

    return As, as, Qs, Hs, hs, x0
end

function lgssm_components(
    k::BaseKernel, t::Union{StepRangeLen, RegularSpacing}, storage_type::StorageType{T},
) where {T<:Real}

    # Compute stationary distribution and sde.
    x0 = stationary_distribution(k, storage_type)
    P = x0.P
    F, q, H = to_sde(k, storage_type)

    # Use stationary distribution + sde to compute finite-dimensional Gauss-Markov model.
    A = time_exp(F, T(step(t)))
    As = Fill(A, length(t))
    as = Fill(Zeros{T}(size(F, 1)), length(t))
    Q = P - A * P * A'
    Qs = Fill(Q, length(t))
    Hs = Fill(H, length(t))
    hs = Fill(zero(T), length(As))

    return As, as, Qs, Hs, hs, x0
end

# Fallback definitions for most base kernels.
function to_sde(k::BaseKernel, ::ArrayStorage{T}) where {T<:Real}
    F, q, H = to_sde(k, SArrayStorage(T))
    return collect(F), q, collect(H)
end

function stationary_distribution(k::BaseKernel, ::ArrayStorage{T}) where {T<:Real}
    x = stationary_distribution(k, SArrayStorage(T))
    return Gaussian(collect(x.m), collect(x.P))
end



# Matern-1/2

function to_sde(k::Matern12, s::SArrayStorage{T}) where {T<:Real}
    F = SMatrix{1, 1, T}(-1)
    q = convert(T, 2)
    H = SVector{1, T}(1)
    return F, q, H
end

function stationary_distribution(k::Matern12, s::SArrayStorage{T}) where {T<:Real}
    return Gaussian(
        SVector{1, T}(0),
        SMatrix{1, 1, T}(1),
    )
end

Zygote.@adjoint function to_sde(k::Matern12, storage_type)
    return to_sde(k, storage_type), Δ->(nothing, nothing)
end

Zygote.@adjoint function stationary_distribution(k::Matern12, storage_type)
    return stationary_distribution(k, storage_type), Δ->(nothing, nothing)
end



# Matern - 3/2

function to_sde(k::Matern32, ::SArrayStorage{T}) where {T<:Real}
    λ = sqrt(3)
    F = SMatrix{2, 2, T}(0, -3, 1, -2λ)
    q = convert(T, 4 * λ^3)
    H = SVector{2, T}(1, 0)
    return F, q, H
end

function stationary_distribution(k::Matern32, ::SArrayStorage{T}) where {T<:Real}
    return Gaussian(
        SVector{2, T}(0, 0),
        SMatrix{2, 2, T}(1, 0, 0, 3),
    )
end

Zygote.@adjoint function to_sde(k::Matern32, storage_type)
    return to_sde(k, storage_type), Δ->(nothing, nothing)
end

Zygote.@adjoint function stationary_distribution(k::Matern32, storage_type)
    return stationary_distribution(k, storage_type), Δ->(nothing, nothing)
end



# Matern - 5/2

function to_sde(k::Matern52, ::SArrayStorage{T}) where {T<:Real}
    λ = sqrt(5)
    F = SMatrix{3, 3, T}(0, 0, -λ^3, 1, 0, -3λ^2, 0, 1, -3λ)
    q = convert(T, 8 * λ^5 / 3)
    H = SVector{3, T}(1, 0, 0)
    return F, q, H
end

function stationary_distribution(k::Matern52, ::SArrayStorage{T}) where {T<:Real}
    κ = 5 / 3
    m = SVector{3, T}(0, 0, 0)
    P = SMatrix{3, 3, T}(1, 0, -κ, 0, κ, 0, -κ, 0, 25)
    return Gaussian(m, P)
end

Zygote.@adjoint function to_sde(k::Matern52, storage_type)
    return to_sde(k, storage_type), Δ->(nothing, nothing)
end

Zygote.@adjoint function stationary_distribution(k::Matern52, storage_type)
    return stationary_distribution(k, storage_type), Δ->(nothing, nothing)
end



# Scaled

function lgssm_components(k::Scaled, ts::AbstractVector, storage_type::StorageType)
    As, as, Qs, Hs, hs, x0 = lgssm_components(k.k, ts, storage_type)
    σ = sqrt(convert(eltype(storage_type), only(k.σ²)))
    Hs = map(H->σ * H, Hs)
    hs = map(h->σ * h, hs)
    return As, as, Qs, Hs, hs, x0
end



# Stretched

function lgssm_components(k::Stretched, ts::AbstractVector, storage_type::StorageType)
    return lgssm_components(k.k, apply_stretch(only(k.a), ts), storage_type)
end

apply_stretch(a, ts::AbstractVector{<:Real}) = a * ts

apply_stretch(a, ts::StepRangeLen) = a * ts

apply_stretch(a, ts::RegularSpacing) = RegularSpacing(a * ts.t0, a * ts.Δt, ts.N)



# Sum

function lgssm_components(k::Sum,ts::AbstractVector, storage_type::StorageType)
    As_l, as_l, Qs_l, Hs_l, hs_l, x0_l = lgssm_components(k.kl, ts, storage_type)
    As_r, as_r, Qs_r, Hs_r, hs_r, x0_r = lgssm_components(k.kr, ts, storage_type)

    As = map(blk_diag, As_l, As_r)
    as = map(vcat, as_l, as_r)
    Qs = map(blk_diag, Qs_l, Qs_r)
    Hs = map(vcat, Hs_l, Hs_r)
    hs = hs_l + hs_r
    x0 = Gaussian(vcat(x0_l.m, x0_r.m), blk_diag(x0_l.P, x0_r.P))

    return As, as, Qs, Hs, hs, x0
end

Base.vcat(x::Zeros{T, 1}, y::Zeros{T, 1}) where {T} = Zeros{T}(length(x) + length(y))

function blk_diag(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    return hvcat(
        (2, 2),
        A, zeros(T, size(A, 1), size(B, 2)), zeros(T, size(B, 1), size(A, 2)), B,
    )
end

function blk_diag(A::SMatrix{DA, DA, T}, B::SMatrix{DB, DB, T}) where {DA, DB, T}
    zero_AB = zeros(SMatrix{DA, DB, T})
    zero_BA = zeros(SMatrix{DB, DA, T})
    return [[A zero_AB]; [zero_BA B]]
end

Zygote.@adjoint function blk_diag(A, B)
    function blk_diag_adjoint(Δ)
        ΔA = Δ[1:size(A, 1), 1:size(A, 2)]
        ΔB = Δ[size(A, 1)+1:end, size(A, 2)+1:end]
        return (ΔA, ΔB)
    end
    return blk_diag(A, B), blk_diag_adjoint
end

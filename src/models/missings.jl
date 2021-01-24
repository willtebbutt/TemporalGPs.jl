# Several strategies for missing data handling were attempted.
# 1. Use `missing`s as expected. This turned out to be problematic for type-stability.
# 2. Sentinel values (NaNs). Also problematic for type-stability because Zygote.
# 3. (The adopted strategy) - replace missings with arbitrary observations and _large_
#   observation noises. While not optimal, type-stability is preserved inside the
#   performance-sensitive code.
#
# In an ideal world, strategy 1 would work. Unfortunately Zygote isn't up to it yet.

function Stheno.logpdf(model::LGSSM, y::AbstractVector{Union{Missing, T}}) where {T}
    model_with_missings, y_filled_in = transform_model_and_obs(model, y)
    return logpdf(model_with_missings, y_filled_in) + _logpdf_volume_compensation(y, model)
end

function _filter(model::LGSSM, y::AbstractVector{Union{Missing, T}}) where {T}
    model_with_missings, y_filled_in = transform_model_and_obs(model, y)
    return _filter(model_with_missings, y_filled_in)
end

function posterior(model::LGSSM, y::AbstractVector{Union{Missing, T}}) where {T}
    model_with_missings, y_filled_in = transform_model_and_obs(model, y)
    return posterior(model_with_missings, y_filled_in)
end

function transform_model_and_obs(model::LGSSM, y::AbstractVector)
    Σs_filled_in, y_filled_in = fill_in_missings(model.emissions.Q, y)
    model_with_missings = replace_observation_noise_cov(model, Σs_filled_in)
    return model_with_missings, y_filled_in
end

function replace_observation_noise_cov(model::LGSSM, Σs_new::AbstractVector)
    return LGSSM(model.transitions, replace_noise_cov(model.emissions, Σs_new))
end

function replace_noise_cov(emissions::StructArray{T}, Qs_new) where {T<:AbstractLGC}
    return StructArray{T}((emissions.A, emissions.a, Qs_new))
end

_large_var_const() = 1e15

function _logpdf_volume_compensation(y, model)
    emissions = model.emissions
    y_obs_count = sum(n -> y[n] === missing ? dim_out(emissions[n]) : 0, eachindex(y))
    return y_obs_count * log(2π * _large_var_const()) / 2
end


Zygote.@nograd _logpdf_volume_compensation

function fill_in_missings(
    Σs::Vector, y::AbstractVector{Union{Missing, T}},
) where {T}

    # Fill in observation covariance matrices with very large values.
    Σs_filled_in = map(eachindex(y)) do n
        (y[n] === missing) ? build_large_var(Σs[n]) : Σs[n]
    end

    # Fill in missing y's with zero vectors (any other choice would also have been fine).
    y_filled_in = Vector{T}(undef, length(y))
    map!(
        n -> y[n] === missing ? get_zero(size(Σs[n], 1), T) : y[n],
        y_filled_in,
        eachindex(y),
    )
    return Σs_filled_in, y_filled_in
end

# We need to densify anyway, might as well do it here and save having to implement the
# pullback twice.
function fill_in_missings(Σs::Fill, y::AbstractVector{Union{Missing, T}}) where {T}
    return fill_in_missings(collect(Σs), y)
end

function Zygote._pullback(
    ::AContext,
    ::typeof(fill_in_missings),
    Σs::Vector,
    y::AbstractVector{Union{T, Missing}},
) where {T}
    function pullback_fill_in_missings(Δ)
        ΔΣs_filled_in = Δ[1]
        Δy_filled_in = Δ[2]

        # The cotangent of a `Missing` doesn't make sense, so should be a `DoesNotExist`.
        Δy = Vector{Union{eltype(Δy_filled_in), DoesNotExist}}(undef, length(y))
        map!(
            n -> y[n] === missing ? DoesNotExist() : Δy_filled_in[n],
            Δy, eachindex(y),
        )

        # Fill in missing locations with zeros. Opting for type-stability to keep things
        # simple.
        ΔΣs = map(
            n -> y[n] === missing ? zero(Σs[n]) : ΔΣs_filled_in[n],
            eachindex(y),
        )

        # return nothing, ΔΣs, Δy
        return nothing, ΔΣs, Δy
    end
    return fill_in_missings(Σs, y), pullback_fill_in_missings
end

get_zero(D::Int, ::Type{Vector{T}}) where {T} = zeros(T, D)

get_zero(::Int, ::Type{T}) where {T<:SVector} = zeros(T)

get_zero(::Int, ::Type{T}) where {T<:Real} = zero(T)

build_large_var(S::T) where {T<:Matrix} = T(_large_var_const() * I, size(S))

build_large_var(::T) where {T<:SMatrix} = T(_large_var_const() * I)

build_large_var(::T) where {T<:Real} = T(_large_var_const())

ChainRulesCore.@non_differentiable build_large_var(::Any)
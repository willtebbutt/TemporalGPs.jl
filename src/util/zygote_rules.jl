using Zygote: @adjoint, accum

# getfield with a fallback to nothing.
@generated function maybegetfield(x, name::Val{s}) where {s}
    if hasfield(x, s)
        return :(getfield(x, s))
    else
        return nothing
    end
end

my_I(N::Int) = Diagonal(Ones(N))
Zygote.@nograd my_I

@adjoint function SVector{D}(x::AbstractVector) where {D}
    return SVector{D}(x), Δ::AbstractVector -> (convert(typeof(x), Δ),)
end

@adjoint function SMatrix{D1, D2}(X::AbstractMatrix) where {D1, D2}
    return SMatrix{D1, D2}(X), Δ::AbstractMatrix -> (convert(typeof(X), Δ),)
end

@adjoint function SMatrix{1, 1}(a)
    return SMatrix{1, 1}(a), Δ::AbstractMatrix -> (first(Δ),)
end

# Implementation of the matrix exponential that assumes one doesn't require access to the
# gradient w.r.t. `A`, only `t`. The former is a bit compute-intensive to get at, while the
# latter is very cheap.

time_exp(A, t) = exp(A * t)
ZygoteRules.@adjoint function time_exp(A, t)
    B = exp(A * t)
    return B, Δ->(nothing, sum(Δ .*  (A * B)))
end

# THIS IS A TEMPORARY FIX WHILE I WAIT FOR #445 IN ZYGOTE TO BE MERGED.
# FOR SOME REASON THIS REALLY HELPS...
@adjoint function (::Type{T})(x, sz) where {T <: Fill}
    back(Δ::AbstractArray) = (sum(Δ), nothing)
    back(Δ::NamedTuple) = (Δ.value, nothing)
    return Fill(x, sz), back
end

@adjoint function collect(x::Fill)
    function collect_Fill_back(Δ)
        return ((value=reduce(accum, Δ), axes=nothing),)
    end
    return collect(x), collect_Fill_back
end

# Implement a restrictive-as-possible implementation of the adjoint because this is a
# dangerous operation that causes segfaults (etc) if its done wrong.
@adjoint function reinterpret(T::Type{<:SVector{1, V}}, x::Vector{V}) where {V<:Real}
    function reinterpret_back(Δ::Vector{<:SVector{1, V}})
        return (nothing, reinterpret(V, Δ))
    end
    return reinterpret(T, x), reinterpret_back
end

@adjoint function reinterpret(T::Type{V}, x::Vector{<:SVector{1, V}}) where {V<:Real}
    function reinterpret_back(Δ::Vector{V})
        return (nothing, reinterpret(SVector{1, V}, Δ))
    end
    return reinterpret(V, x), reinterpret_back
end

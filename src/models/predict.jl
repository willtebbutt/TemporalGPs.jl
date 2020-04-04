#
# Generic implementation. Good for StaticArrays.
#

@inline function predict(mf::AV, Pf::AM, A::AM, a::AV, Q::AM)
    return A * mf + a, (A * Pf) * A' + Q
end

@adjoint function predict(m::AV, P::AM, A::AM, a::AV, Q::AM)
    return predict_pullback(m, P, A, a, Q)
end

function predict_pullback(m::AV, P::AM, A::AM, a::AV, Q::AM)
    mp = A * m + a # 1
    T = A * P # 2
    Pp = T * A' + Q # 3
    return (mp, Pp), function(Δ)
        Δmp = Δ[1]
        ΔPp = Δ[2]

        # 3
        ΔQ = ΔPp
        ΔA = ΔPp' * T
        ΔT = ΔPp * A

        # 2
        ΔA += ΔT * P'
        ΔP = A'ΔT

        # 1
        ΔA += Δmp * m'
        Δm = A'Δmp
        Δa = Δmp

        return Δm, ΔP, ΔA, Δa, ΔQ
    end
end



#
# `A <: Matrix{<:Real}`.
#

function predict(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::AM{T},
    a::Vector{T},
    Q::Matrix{T},
) where {T<:Real}
    mp = Vector{T}(undef, size(mf))
    Pp = Matrix{T}(undef, size(Pf))
    return predict!(mp, Pp, mf, Pf, A, a, Q)
end

function predict!(
    mp::Union{Vector{T}, SubArray{T, 1}},
    Pp::Union{Matrix{T}, SubArray{T, 2}},
    mf::Union{Vector{T}, SubArray{T, 1}},
    Pf::Symmetric{T, <:Union{Matrix{T}, SubArray{T, 2}}},
    A::Matrix{T},
    a::Union{Vector{T}, SubArray{T, 1}},
    Q::Matrix{T},
) where {T<:Real}

    # Compute predictive mean.
    mp .= a
    mp = mul!(mp, A, mf, one(T), one(T))

    # Compute predictive covariance.
    APf = mul!(Matrix{T}(undef, size(Pf)), A, Pf, one(T), zero(T))

    Pp .= Q
    Pp = mul!(Pp, APf, A', one(T), one(T))

    return mp, Pp
end

function predict_pullback(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::Matrix{T},
    a::Vector{T},
    Q::Matrix{T},
) where {T<:Real}

    # Pre-allocate for output.
    mp = Vector{T}(undef, size(mf))
    Pp = Matrix{T}(undef, size(Pf))

    # 1: Compute predictive mean.
    mp = mul!(copy!(mp, a), A, mf, one(T), one(T))

    # 2: compute A * Pf
    APf = mul!(Matrix{T}(undef, size(Pf)), A, Pf, one(T), zero(T))

    # 3: compute APf * A' + Q
    Pp = mul!(copy!(Pp, Q), APf, A', one(T), one(T))

    return (mp, Pp), function(Δ)
        Δmp = Δ[1]
        ΔPp = Δ[2]

        # Pre-allocate for cotangents.
        Δmf = fill(zero(T), size(mf))
        ΔPf = fill(zero(T), size(Pf))
        ΔA = fill(zero(T), size(A))
        Δa = fill(zero(T), size(a))
        ΔQ = fill(zero(T), size(Q))

        return predict_pullback_accum!(Δmp, ΔPp, Δmf, ΔPf, ΔA, Δa, ΔQ, mf, Pf, A, a, Q)
    end
end

function predict_pullback_accum!(
    Δmp::Union{Vector{T}, SubArray{T, 1}},
    ΔPp::Union{Matrix{T}, SubArray{T, 2}},
    Δmf::Union{Vector{T}, SubArray{T, 1}},
    ΔPf::Union{Matrix{T}, SubArray{T, 2}},
    ΔA::Union{Matrix{T}, SubArray{T, 2}},
    Δa::Union{Vector{T}, SubArray{T, 1}},
    ΔQ::Union{Matrix{T}, SubArray{T, 2}},
    mf::Union{Vector{T}, SubArray{T, 1}},
    Pf::Symmetric{T, <:Union{Matrix{T}, SubArray{T, 2}}},
    A::Union{Matrix{T}, SubArray{T, 2}},
    a::Union{Vector{T}, SubArray{T, 1}},
    Q::Union{Matrix{T}, SubArray{T, 2}},
) where {T<:Real}

    # Re-compute A * Pf
    APf = mul!(Matrix{T}(undef, size(Pf)), A, Pf, one(T), zero(T))

    # Pre-allocate for ΔAPf.
    ΔAPf = Matrix{T}(undef, size(APf))

    # 3
    ΔQ .+= ΔPp
    ΔA = mul!(ΔA, ΔPp', APf, one(T), one(T))
    ΔAPf = mul!(ΔAPf, ΔPp, A)

    # 2
    ΔA = mul!(ΔA, ΔAPf, Pf', one(T), one(T))
    ΔPf = mul!(ΔPf, A', ΔAPf, one(T), one(T))

    # 1
    ΔA = mul!(ΔA, Δmp, mf', one(T), one(T))
    Δmf = mul!(Δmf, A', Δmp, one(T), one(T))
    Δa .+= Δmp

    return Δmf, ΔPf, ΔA, Δa, ΔQ
end



#
# A <: BlockDiagonal{<:Real}
#

function predict(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::BlockDiagonal{T, TM},
    a::Vector{T},
    Q::BlockDiagonal{T, TM},
) where {T<:Real, TM<:AbstractMatrix{T}}
    mp = fill(zero(T), size(mf))
    Pp = fill(zero(T), size(Pf))
    return predict!(mp, Pp, mf, Pf, A, a, Q)
end

function predict!(
    mp::Vector{T},
    Pp::Matrix{T},
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::BlockDiagonal{T, TM},
    a::Vector{T},
    Q::BlockDiagonal{T, TM},
) where {T<:Real, TM<:AbstractMatrix{T}}

    # Compute predictive covariance. Only works with the upper triangle.
    row_lb = 1
    @views for n in 1:nblocks(A)

        # Determine rows to consider.
        (δ_r, δ_c) = blocksize(A, n)
        @assert δ_r === δ_c
        row_ids = row_lb:(row_lb + δ_r - 1)

        # Update diagonal element of Pp.
        predict!(
            mp[row_ids],
            Pp[row_ids, row_ids],
            mf[row_ids],
            Symmetric(Pf.data[row_ids, row_ids]),
            getblock(A, n),
            a[row_ids],
            getblock(Q, n),
        )

        # Update elements above the diagonal.
        col_lb = row_lb + δ_r
        for m in (n + 1):nblocks(A)
            col_ids = col_lb:(col_lb + δ_r - 1)
            APf = getblock(A, n) * Pf.data[row_ids, col_ids]
            mul!(Pp[row_ids, col_ids], APf, getblock(A, m)')
            col_lb += δ_r
        end

        # Shift the rows considered.
        row_lb += δ_r
    end
    return mp, Pp
end

function predict_pullback(
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::BlockDiagonal{T, TM},
    a::Vector{T},
    Q::BlockDiagonal{T, TM},
) where {T<:Real, TM<:AbstractMatrix{T}}
    return predict(mf, Pf, A, a, Q), function(Δ)
        Δmp = Δ[1]
        ΔPp = Δ[2]
        Δmf = fill(zero(T), size(mf))
        ΔPf = fill(zero(T), size(Pf))
        ΔA = get_tangent_storage(A, zero(T))
        Δa = fill(zero(T), size(a))
        ΔQ = get_tangent_storage(Q, zero(T))
        return predict_pullback_accum!(Δmp, ΔPp, Δmf, ΔPf, ΔA, Δa, ΔQ, mf, Pf, A, a, Q)
    end
end

function predict_pullback_accum!(
    Δmp::Vector{T},
    ΔPp::Matrix{T},
    Δmf::Vector{T},
    ΔPf::Matrix{T},
    ΔA::NamedTuple{(:blocks,)},
    Δa::Vector{T},
    ΔQ::NamedTuple{(:blocks,)},
    mf::Vector{T},
    Pf::Symmetric{T, Matrix{T}},
    A::BlockDiagonal{T, TM},
    a::Vector{T},
    Q::BlockDiagonal{T, TM},
) where {T<:Real, TM<:AbstractMatrix{T}}

    # Compute predictive covariance. Only works with the upper triangle.
    row_lb = 1
    @views for n in 1:nblocks(A)

        # Determine rows to consider.
        (δ_r, δ_c) = blocksize(A, n)
        @assert δ_r === δ_c
        row_ids = row_lb:(row_lb + δ_r - 1)

        # Update diagonal element of Pp.
        predict_pullback_accum!(
            Δmp[row_ids],
            ΔPp[row_ids, row_ids],
            Δmf[row_ids],
            ΔPf[row_ids, row_ids],
            ΔA.blocks[n],
            Δa[row_ids],
            ΔQ.blocks[n],
            mf[row_ids],
            Symmetric(Pf.data[row_ids, row_ids]),
            getblock(A, n),
            a[row_ids],
            getblock(Q, n),
        )

        # Update elements above the diagonal.
        col_lb = row_lb + δ_r
        for m in (n + 1):nblocks(A)
            col_ids = col_lb:(col_lb + δ_r - 1)

            # Recompute APf.
            APf = getblock(A, n) * Pf.data[row_ids, col_ids]

            # Compute ΔA of mth block, and ΔAnPf.
            mul!(ΔA.blocks[m], ΔPp[row_ids, col_ids]', APf, one(T), one(T))
            ΔAPf = ΔPp[row_ids, col_ids] * getblock(A, m)

            # Compute ΔA of nth block and ΔPf.
            mul!(ΔA.blocks[n], ΔAPf, Pf.data[row_ids, col_ids]', one(T), one(T))
            mul!(ΔPf[row_ids, col_ids], getblock(A, n)', ΔAPf, one(T), one(T))

            col_lb += δ_r
        end

        # Shift the rows considered.
        row_lb += δ_r
    end
    return Δmf, ΔPf, ΔA, Δa, ΔQ
end

get_tangent_storage(A::Matrix{T}, val::T) where {T<:Real} = fill(val, size(A))
function get_tangent_storage(A::BlockDiagonal{T}, val::T) where {T<:Real}
    return (blocks=map(block -> get_tangent_storage(block, val), A.blocks), )
end
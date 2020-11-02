"""
    trace_term(k::Separable, x::AbstractVector, y::AbstractVector, z::AbstractVector)

Compute the trace-term in the collapsed ELBO. This current interface is a hack while I'm
still trying things out and checking the maths -- it will be replaced by something less
insane once I'm confident that everything works.
"""
function trace_term(
    k::Separable,
    x::SpaceTimeGrid,
    y::AbstractVector{<:AbstractVector{<:Real}},
    z::AbstractVector,
    S::AbstractVector,
    storage::StorageType,
)
    # Construct temporal model.
    ts = x.xr
    time_kernel = k.r
    gmm_time = TemporalGPs.GaussMarkovModel(time_kernel, ts, storage)
    time_marginals = TemporalGPs.time_marginal(gmm_time)

    # Compute spatial covariance between inducing inputs, and inducing points + obs. points.
    # Since we've constrained the observations to occur at the same locations at each point
    # in time, these are constant through time.
    space_kernel = k.l
    x_space = x.xl
    K_space_z_chol = cholesky(Symmetric(pw(space_kernel, z)))
    K_space_zx = pw(space_kernel, z, x_space)
    K_space_xx = Stheno.ew(space_kernel, x_space)
    K_space_cond_diag = K_space_xx - Stheno.diag_Xt_invA_X(K_space_z_chol, K_space_zx)

    # Compute the trace term by summing over everything. I'm assuming that each `S[t]` is
    # `Diagonal`.
    return -0.5 * sum(1:length(gmm_time)) do t
        H = gmm_time.H[t]
        return sum(eachindex(K_space_xx)) do n
            tr(H' * inv(S[t][n, n]) * H * kron(K_space_cond_diag[n], time_marginals[t].P))
        end
    end
end

function trace_term(
    k::Separable,
    x::RegularInTime,
    y::AbstractVector{<:AbstractVector{<:Real}},
    z::AbstractVector,
    S::AbstractVector,
    storage::StorageType,
)
    # Construct temporal model.
    ts = x.ts
    time_kernel = k.r
    gmm_time = TemporalGPs.GaussMarkovModel(time_kernel, ts, storage)
    time_marginals = TemporalGPs.time_marginal(gmm_time)

    # Compute spatial covariance between inducing inputs, and inducing points + obs. points.
    # Since we've constrained the observations to occur at the same locations at each point
    # in time, these are constant through time.
    space_kernel = k.l
    x_space = x.vs
    K_space_z_chol = cholesky(Symmetric(pw(space_kernel, z)))

    # Compute the trace term by summing over everything. I'm assuming that each `S[t]` is
    # `Diagonal`.
    return -0.5 * sum(1:length(gmm_time)) do t
        H = gmm_time.H[t]
        K_space_zx = pw(space_kernel, z, x_space[t])
        K_space_xx = Stheno.ew(space_kernel, x_space[t])
        K_space_cond_diag = K_space_xx - Stheno.diag_Xt_invA_X(K_space_z_chol, K_space_zx)
        return sum(eachindex(K_space_xx)) do n
            tr(H' * inv(S[t][n, n]) * H * kron(K_space_cond_diag[n], time_marginals[t].P))
        end
    end
end

function time_marginal(gmm::GaussMarkovModel)
    m = gmm.x0.m
    P = gmm.x0.P
    marginals = Vector{typeof(gmm.x0)}(undef, length(gmm))
    m1, P1 = predict(m, P, gmm.A[1], gmm.a[1], gmm.Q[1])
    marginals[1] = Gaussian(m1, P1)
    for t in 2:length(gmm)
        mt, Pt = predict(
            marginals[t-1].m, marginals[t-1].P, gmm.A[t], gmm.a[t], gmm.Q[t],
        )
        marginals[t] = Gaussian(mt, Pt)
    end
    return marginals
end

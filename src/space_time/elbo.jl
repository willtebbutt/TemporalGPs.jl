"""
    elbo(
        k::Separable,
        x::AbstractVector,
        y::AbstractVector,
        z::AbstractVector,
        S::AbstractVector,
    )

This is a bodge while I improve the API generally.
"""
function Stheno.elbo(
    k::Separable,
    x::AbstractVector,
    y::AbstractVector,
    z::AbstractVector,
    S::AbstractVector,
)
    f = to_sde(GP(DTCSeparable(z, k), GPC()))
    return logpdf(f(x, S), y) + trace_term(k, x, y, z, S, ArrayStorage(Float64))
end

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

"""
    optimal_approximate_posterior_marginals(
        k::Separable,
        x::AbstractVector,
        y::AbstractVector,
        z::AbstractVector,
        S::AbstractVector,
        r_pred::AbstractVector,
    )

This is very ugly and assumes that you want to make predictions at the same locations in
space at each point in time.
"""
function optimal_approximate_posterior_marginals(
    k::Separable,
    x::AbstractVector,
    y::AbstractVector,
    z::AbstractVector,
    S::AbstractVector,
    r_pred::AbstractVector,
)
    # Construct low-rank DTC model.
    f = to_sde(GP(DTCSeparable(z, k), GPC()))

    # Compute approximate posterior marginals over pseudo-points.
    ū_smooth = smooth_latents(f(x, S), y)

    # Compute cross-covariance between target points and pseudo-points.
    space_kernel = k.l
    C_fp_u = Stheno.pairwise(space_kernel, r_pred, z)
    C_u = cholesky(Symmetric(Stheno.pairwise(space_kernel, z)))
    C_fp_u_Λ_u = C_fp_u / C_u
    Cr_rpred_diag = Stheno.elementwise(space_kernel, r_pred)

    # Obtain the projections required to get u from ū.
    time_kernel = k.r
    ts = get_times(x)
    gmm = GaussMarkovModel(time_kernel, ts, ArrayStorage(Float64))
    ident = Matrix{Float64}(I, length(z), length(z))

    # Compute the approximate posterior marginals.
    return map(eachindex(ū_smooth)) do n
        Hu = kron(ident, gmm.H[n])
        C_fp_u_Λ_u_Hu = C_fp_u_Λ_u * Hu

        # Compute approx. post mean.
        m = C_fp_u_Λ_u_Hu * ū_smooth[n].m

        # Compute approx. post variance.
        ct = only(Stheno.ew(time_kernel, [ts[n]]))

        conditioning_term = (Cr_rpred_diag - Stheno.diag_Xt_invA_X(C_u, C_fp_u')) * ct
        inflation_term = Stheno.diag_Xt_A_X(
            cholesky(Symmetric(ū_smooth[n].P)), C_fp_u_Λ_u_Hu',
        )
        s = conditioning_term + inflation_term
        return Normal.(m, sqrt.(s))
    end
end

function smooth_latents(model::LGSSM, ys::AbstractVector)

    lml, x_filter = filter(model, ys)
    ε = convert(eltype(model), 1e-12)

    # Smooth
    x_smooth = Vector{typeof(last(x_filter))}(undef, length(ys))
    x_smooth[end] = x_filter[end]
    for k in reverse(1:length(x_filter) - 1)
        x = x_filter[k]
        x′ = predict(model[k + 1], x)

        U = cholesky(Symmetric(x′.P + ε * I)).U
        Gt = U \ (U' \ (model.gmm.A[k + 1] * x.P))
        x_smooth[k] = Gaussian(
            _compute_ms(x.m, Gt, x_smooth[k + 1].m, x′.m),
            _compute_Ps(x.P, Gt, x_smooth[k + 1].P, x′.P),
        )
    end

    return x_smooth
end

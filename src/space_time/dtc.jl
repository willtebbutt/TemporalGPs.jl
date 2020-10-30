"""
    DTCSeparable{Tz<:AbstractVector, Tk<:SeparableKernel} <: Kernel

Kernel used to specify a particular class of low-rank GP. This is a bit of a hack and may
well be superceded by something else at a later date.
"""
struct DTCSeparable{Tz<:AbstractVector, Tk<:Separable} <: Kernel
    z::Tz
    k::Tk
end

function GaussMarkovModel(k_dtc::DTCSeparable, x::SpaceTimeGrid, storage)

    # Construct temporal model.
    k = k_dtc.k
    ts = x.xr
    time_kernel = k.r
    gmm_time = GaussMarkovModel(time_kernel, ts, storage)

    # Compute spatial covariance between inducing inputs, and ioducing points + obs. points.
    space_kernel = k.l
    x_space = x.xl
    z_space = k_dtc.z
    K_space_z = pw(space_kernel, z_space)
    K_space_xz = pw(space_kernel, x_space, z_space)

    # Get some size info.
    M = length(z_space)
    N = length(x_space)
    ident_M = my_I(eltype(storage), M)
    ident_N = my_I(eltype(storage), N)
    ident_D = my_I(eltype(storage), dim_latent(gmm_time))

    # G is the time-invariant component of the H-matrices. It is only time-invariant because
    # we have the same obsevation locations at each point in time.
    G = kron(K_space_xz / cholesky(Symmetric(K_space_z + 1e-9I)), ident_D)

    # Construct approximately low-rank model spatio-temporal LGSSM.
    A = map(A -> kron(ident_M, A), gmm_time.A)
    a = map(a -> repeat(a, M), gmm_time.a)
    Q = map(Q -> kron(K_space_z, Q), gmm_time.Q)
    H = map(H -> kron(ident_N, H) * G, gmm_time.H) # This is currently O(N^2).
    h = map(h -> repeat(h, N), gmm_time.h) # This should currently be zero.
    x = Gaussian(
        repeat(gmm_time.x0.m, M),
        kron(K_space_z, gmm_time.x0.P),
    )
    return GaussMarkovModel(A, a, Q, H, h, x)
end

function GaussMarkovModel(k_dtc::DTCSeparable, x::RegularInTime, storage)

    # Construct temporal model.
    k = k_dtc.k
    ts = x.ts
    time_kernel = k.r
    gmm_time = GaussMarkovModel(time_kernel, ts, storage)

    # Compute spatial covariance between inducing inputs, and ioducing points + obs. points.
    space_kernel = k.l
    z_space = k_dtc.z
    K_space_z = pw(space_kernel, z_space)
    K_space_z_chol = cholesky(Symmetric(K_space_z + 1e-9I))

    # Get some size info.
    M = length(z_space)
    ident_M = my_I(eltype(storage), M)
    ident_D = my_I(eltype(storage), dim_latent(gmm_time))

    # Construct approximately low-rank model spatio-temporal LGSSM.
    A = map(A -> kron(ident_M, A), gmm_time.A)
    a = map(a -> repeat(a, M), gmm_time.a)
    Q = map(Q -> kron(K_space_z, Q), gmm_time.Q)
    H = map((H, v) -> kron(pw(space_kernel, v, z_space) / K_space_z_chol, H), gmm_time.H, x.vs)# Do some stuff about timegmm_time.H) # This is currently O(N^2). Don't worry though.
    h = map((h, v) -> repeat(h, length(v)), gmm_time.h, x.vs) # This should currently be zero.
    x = Gaussian(
        repeat(gmm_time.x0.m, M),
        kron(K_space_z, gmm_time.x0.P),
    )
    return GaussMarkovModel(A, a, Q, H, h, x)
end

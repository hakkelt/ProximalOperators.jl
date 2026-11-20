# Hinge loss function

export SqrHingeLoss

"""
    SqrHingeLoss(y, μ=1; threaded=true)

Return the squared Hinge loss
```math
f(x) = μ⋅∑_i \\max\\{0, 1 - y_i ⋅ x_i\\}^2,
```
where `y` is an array and `μ` is a positive parameter.

`threaded = false` forbids this operator from using more than one thread; see
[`is_threaded`](@ref).
"""
struct SqrHingeLoss{R, T, Th}
    y::T
    mu::R
    function SqrHingeLoss{R, T, Th}(y::T, mu::R) where {R, T, Th}
        if mu <= 0
            error("parameter mu must be positive")
        else
            new(y, mu)
        end
    end
end

is_separable(f::Type{<:SqrHingeLoss}) = true
is_convex(f::Type{<:SqrHingeLoss}) = true
is_smooth(f::Type{<:SqrHingeLoss}) = true

@threadable SqrHingeLoss{<:Any, <:Any, Th} Arithmetic

SqrHingeLoss(b::T, mu::R=1; threaded::Bool=true) where {R, T} =
    SqrHingeLoss{R, T, threaded}(b, mu)

function (f::SqrHingeLoss)(x)
    R = eltype(x)
    return f.mu * sum(max.(R(0), (R(1) .- f.y .* x)).^2)
end

function gradient!(g, f::SqrHingeLoss, x)
    R = eltype(x)
    fy, mu = f.y, f.mu
    # Inline `@elementwise_loop` on host storage, for the reason given above `prox!` below.
    # Device storage goes through `map_reduce_prox_idx!`, whose `_map_reduce_prox_idx!` has a
    # GPU-native override in `GpuExt`; the inline loop would bypass that override and
    # scalar-index the device array instead.
    if is_cpu_storage(x)
        acc = R(0)
        @elementwise_loop execution_strategy(f, x) reduction = ((+, acc),) for i in eachindex(x, g)
            xi = x[i]
            zz = 1 - fy[i] * xi
            g[i] = ifelse(zz > 0, -2 * mu * fy[i] * zz, R(0))
            acc += max(R(0), 1 - fy[i] * xi)^2
        end
        return f.mu * acc
    end
    acc = map_reduce_prox_idx!(
        f, g,
        function (i, xi)
            zz = 1 - fy[i] * xi
            zz > 0 ? -2 * mu * fy[i] * zz : zero(zz)
        end,
        (i, _) -> max(zero(mu), 1 - fy[i] * x[i])^2,
        x,
    )
    return f.mu * acc
end

# Inline `@elementwise_loop` on host storage: through `map_reduce_prox_idx!`'s `h`/`g`
# closures the bounds checks stay outside `@inbounds` (it does not cross a call boundary) and
# the division-heavy body does not vectorize. Inline, with `ifelse` in place of branches, the
# body vectorizes in every chunk of the threaded loop (see `_threaded_loop` in
# `execution.jl`): at 2^20 elements over 4 threads, 0.2-0.3 ms against about 2 ms through the
# closures. Device storage keeps going through `map_reduce_prox_idx!`, whose
# GPU-native override in `GpuExt` the inline loop would otherwise bypass, scalar-indexing the
# device array instead.
function prox!(z, f::SqrHingeLoss, x, gamma)
    R = eltype(x)
    fy, mu = f.y, f.mu
    if is_cpu_storage(x)
        v = R(0)
        @elementwise_loop execution_strategy(f, x) reduction = ((+, v),) for k in eachindex(x, z)
            yk = fy[k]
            xk = x[k]
            inactive = yk * xk >= 1
            zk = ifelse(inactive, xk, (xk + 2 * mu * gamma * yk) / (1 + 2 * mu * gamma * yk^2))
            z[k] = zk
            r = 1 - yk * zk
            v += ifelse(inactive, R(0), r * r)
        end
        return f.mu * v
    end
    v = map_reduce_prox_idx!(
        f, z,
        function (k, xk)
            yk = fy[k]
            yk * xk >= 1 ? xk : (xk + 2 * mu * gamma * yk) / (1 + 2 * mu * gamma * yk^2)
        end,
        function (k, zk)
            yk = fy[k]
            inactive = yk * x[k] >= 1
            r = 1 - yk * zk
            inactive ? R(0) : r * r
        end,
        x,
    )
    return f.mu * v
end

function prox_naive(f::SqrHingeLoss, x, gamma)
    flag = f.y .* x .<= 1
    z = copy(x)
    z[flag] = (x[flag] .+ 2 .* f.mu .* gamma .* f.y[flag]) ./ (1 + 2 .* f.mu .* gamma .* f.y[flag].^2)
    return z, f.mu * sum(max.(0, 1 .- f.y .* z).^2)
end

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

function gradient!(y, f::SqrHingeLoss, x)
    R = eltype(x)
    fy, mu = f.y, f.mu
    acc = map_reduce_prox_idx!(
        f, y,
        function (i, xi)
            zz = 1 - fy[i] * xi
            zz > 0 ? -2 * mu * fy[i] * zz : R(0)
        end,
        (i, _) -> max(R(0), 1 - fy[i] * x[i])^2,
        x,
    )
    return f.mu * acc
end

function prox!(z, f::SqrHingeLoss, x, gamma)
    R = eltype(x)
    fy, mu = f.y, f.mu
    # the `if`/`else` became a select so the body has a single exit: only the accumulation is
    # really conditional, and it contributes zero on the inactive side. The contribution
    # depends on the *input* at that index as well as the output, hence the index-taking
    # helper rather than the plain one.
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

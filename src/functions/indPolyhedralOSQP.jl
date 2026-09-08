# IndPolyhedral: OSQP implementation
#
# The solver-backed constructors and `prox!` live in the package extension
# `ext/ProximalOperatorsOSQPExt.jl`, which is only loaded once the OSQP package
# is available (`using OSQP`). Without OSQP loaded, constructing an
# `IndPolyhedralOSQP` (directly or via `IndPolyhedral(...; solver=:osqp)`)
# raises an informative error.

struct IndPolyhedralOSQP{R, M} <: IndPolyhedral
    l::AbstractVector{R}
    A::AbstractMatrix{R}
    u::AbstractVector{R}
    mod::M
    # Explicit inner constructor: suppresses the auto-generated 4-positional-arg
    # outer constructor, which would otherwise shadow the `(l, A, xmin, xmax)`
    # constructor added by ProximalOperatorsOSQPExt.
    IndPolyhedralOSQP{R, M}(l, A, u, mod) where {R, M} = new{R, M}(l, A, u, mod)
end

# properties

is_proximable(::Type{<:IndPolyhedralOSQP}) = false

# constructors
#
# The real implementations are added to this function by
# ProximalOperatorsOSQPExt; this fallback only fires when OSQP is not loaded.

function IndPolyhedralOSQP(args...; kwargs...)
    error(
        "IndPolyhedralOSQP requires the OSQP package: run `using OSQP` before " *
        "constructing IndPolyhedralOSQP(...) or IndPolyhedral(...; solver=:osqp)."
    )
end

# function evaluation

function (f::IndPolyhedralOSQP)(x)
    R = eltype(x)
    Ax = f.A * x
    return all(f.l .<= Ax .<= f.u) ? R(0) : Inf
end

# naive prox

# we want to compute the projection p of a point x
#
# primal problem is: minimize_p (1/2)||p-x||^2 + g(Ap)
# where g is the indicator of the box [l, u]
#
# dual problem is: minimize_y (1/2)||-A'y||^2 - x'A'y + g*(y)
# can solve with (fast) dual proximal gradient method

function prox_naive(f::IndPolyhedralOSQP, x, gamma)
    R = eltype(x)
    y = zeros(R, size(f.A, 1)) # dual vector
    y1 = y
    g = IndBox(f.l, f.u)
    gstar = Conjugate(g)
    gstar_y = R(0)
    stepsize = R(1)/opnorm(Matrix(f.A*f.A'))
    for it = 1:1e6
        w = y + (it-1)/(it+2)*(y - y1)
        y1 = y
        z = w - stepsize * (f.A * (f.A'*w - x))
        y, = prox(gstar, z, stepsize)
        if norm(y-w)/(1+norm(w)) <= 1e-12 break end
    end
    p = -f.A'*y + x
    return p, R(0)
end

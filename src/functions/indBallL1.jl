# indicator of the L1 norm ball with given radius

export IndBallL1

"""
    IndBallL1(r=1.0)

Return the indicator function of the ``L_1`` norm ball
```math
S = \\left\\{ x : \\sum_i |x_i| \\leq r \\right\\}.
```
Parameter `r` must be positive.
"""
struct IndBallL1{R, B}
    r::R
    buf::B
    function IndBallL1{R, B}(r::R, buf::B) where {R, B}
        if r <= 0
            error("parameter r must be positive")
        else
            new(r, buf)
        end
    end
end

is_convex(f::Type{<:IndBallL1}) = true
is_set_indicator(f::Type{<:IndBallL1}) = true
is_proximable(f::Type{<:IndBallL1}) = false

IndBallL1(r::R=1.0; buf=nothing) where R = IndBallL1{R, typeof(buf)}(r, buf)
IndBallL1{R}(r::R) where R = IndBallL1{R, Nothing}(r, nothing)

function preallocate(f::IndBallL1, x::AbstractArray{<:Real})
    abs_x = similar(x)
    return IndBallL1(f.r; buf = (
        sig = input_signature(x),
        abs_x = abs_x,
        condat_buffers(abs_x)...,
    ))
end

function preallocate(f::IndBallL1, x::AbstractArray{<:Complex})
    R = real(eltype(x))
    abs_x = similar(x, R)
    return IndBallL1(f.r; buf = (
        sig = input_signature(x),
        abs_x = abs_x,
        y_temp = similar(x, R),
        condat_buffers(abs_x)...,
    ))
end

function (f::IndBallL1)(x)
    R = real(eltype(x))
    if norm(x, 1) - f.r > f.r*eps(R)
        return R(Inf)
    end
    return R(0)
end

function prox!(y, f::IndBallL1, x::AbstractArray{<:Real}, gamma)
    R = eltype(x)
    check_input(f, x)
    if norm(x, 1) <= f.r
        y .= x
        return R(0)
    else # do a projection of abs(x) onto simplex then recover signs
        b = get_buffers(f, x)
        w = condat_work(f, x)
        b.abs_x .= abs.(x)
        simplex_proj_condat!(y, f.r, b.abs_x, w.v, w.v_tilde)
        y .*= sign.(x)
        return R(0)
    end
end

function prox!(y, f::IndBallL1, x::AbstractArray{<:Complex}, gamma)
    R = real(eltype(x))
    check_input(f, x)
    if norm(x, 1) <= f.r
        y .= x
        return R(0)
    else # do a projection of abs(x) onto simplex then recover signs
        b = get_buffers(f, x)
        w = condat_work(f, x)
        b.abs_x .= real.(abs.(x))
        simplex_proj_condat!(b.y_temp, f.r, b.abs_x, w.v, w.v_tilde)
        y .= b.y_temp .* sign.(x)
        return R(0)
    end
end

function prox_naive(f::IndBallL1, x, gamma)
    R = real(eltype(x))
    # do a simple bisection (aka binary search) on λ
    L = R(0)
    U = maximum(abs, x)
    λ = L
    v = R(0)
    maxit = 120
    for _ in 1:maxit
        λ = (L + U) / 2
        v = sum(max.(abs.(x) .- λ, R(0)))
        # modify lower or upper bound
        (v < f.r) ? U = λ : L = λ
        # exit condition
        if abs(L - U) < (1 + abs(U))*eps(R)
            break
        end
    end
    return sign.(x) .* max.(R(0), abs.(x) .- λ), R(0)
end

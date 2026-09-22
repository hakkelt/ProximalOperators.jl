# indicator of a simplex

export IndSimplex

"""
    IndSimplex(a=1.0)

Return the indicator of the simplex
```math
S = \\left\\{ x : x \\geq 0, \\sum_i x_i = a \\right\\}.
```

By default `a=1`, therefore ``S`` is the probability simplex.
"""
struct IndSimplex{R, B}
    a::R
    buf::B
    function IndSimplex{R, B}(a::R, buf::B) where {R, B}
        if a <= 0
            error("parameter a must be positive")
        else
            new(a, buf)
        end
    end
end

is_convex(f::Type{<:IndSimplex}) = true
is_set_indicator(f::Type{<:IndSimplex}) = true

IndSimplex(a::R=1; buf=nothing) where R = IndSimplex{R, typeof(buf)}(a, buf)
IndSimplex{R}(a::R) where R = IndSimplex{R, Nothing}(a, nothing)

preallocate(f::IndSimplex, x::AbstractArray{<:Real}) = IndSimplex(
    f.a; buf = (sig = input_signature(x), condat_buffers(x)...)
)

function (f::IndSimplex)(x)
    R = eltype(x)
    if all(x .>= 0) && sum(x) ≈ f.a
        return R(0)
    end
    return R(Inf)
end

# Work vectors for `simplex_proj_condat!`: two stacks of at most `length(x)`
# elements each. `condat_buffers` allocates them with capacity already reserved,
# so that repeated calls push and pop without reallocating.
function condat_buffers(x)
    R = eltype(x)
    N = length(x)
    v = Vector{R}(undef, 0)
    v_tilde = Vector{R}(undef, 0)
    sizehint!(v, N)
    sizehint!(v_tilde, N)
    return (v = v, v_tilde = v_tilde)
end

# Like `get_buffers`, but the non-preallocated path hands out empty vectors that
# grow on demand, exactly as the original implementation did: reserving capacity
# up front would allocate more than the current code does.
@inline condat_work(f, x) = _condat_work(f, f.buf, x)
@inline _condat_work(f, ::Nothing, x) =
    (v = real(eltype(x))[], v_tilde = real(eltype(x))[])
@inline _condat_work(f, buf, x) = (check_signature(buf.sig, x, f); buf)

function simplex_proj_condat!(y, a, x)
    R = eltype(x)
    return simplex_proj_condat!(y, a, x, R[], R[])
end

function simplex_proj_condat!(y, a, x, v, v_tilde)
    # Implements algorithm proposed in:
    # Condat, L. "Fast projection onto the simplex and the l1 ball",
    # Mathematical Programming, 158:575–585, 2016.
    R = eltype(x)
    empty!(v)
    empty!(v_tilde)
    push!(v, x[1])
    rho = x[1] - a
    N = length(x)
    for k in 2:N
        if x[k] > rho
            rho += (x[k] - rho) / (length(v) + 1)
            if rho > x[k] - a
                push!(v, x[k])
            else
                append!(v_tilde, v)
                empty!(v)
                push!(v, x[k])
                rho = x[k] - a
            end
        end
    end
    for z in v_tilde
        if z > rho
            push!(v, z)
            rho += (z - rho) / length(v)
        end
    end
    v_changed = true
    while v_changed == true
        v_changed = false
        k = 1
        while k <= length(v)
            z = v[k]
            if z <= rho
                deleteat!(v, k)
                v_changed = true
                rho += (rho - z) / length(v)
            else
                k = k + 1
            end
        end
    end
    y .= max.(x .- rho, R(0))
end

function prox!(y, f::IndSimplex, x, gamma)
    b = condat_work(f, x)
    simplex_proj_condat!(y, f.a, x, b.v, b.v_tilde)
    return eltype(x)(0)
end

function prox_naive(f::IndSimplex, x, gamma)
    R = eltype(x)
    low = minimum(x)
    upp = maximum(x)
    v = x
    s = Inf
    for i = 1:100
        if abs(s)/f.a ≈ 0
            break
        end
        alpha = (low+upp)/2
        v = max.(x .- alpha, R(0))
        s = sum(v) - f.a
        if s <= 0
            upp = alpha
        else
            low = alpha
        end
    end
    return v, R(0)
end

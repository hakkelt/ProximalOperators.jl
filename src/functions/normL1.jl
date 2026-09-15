# L1 norm (times a constant, or weighted)

export NormL1

"""
    NormL1(λ=1)

With a nonnegative scalar parameter λ, return the ``L_1`` norm
```math
f(x) = λ\\cdot∑_i|x_i|.
```
With a nonnegative array parameter λ, return the weighted ``L_1`` norm
```math
f(x) = ∑_i λ_i|x_i|.
```
"""
struct NormL1{T, B}
    lambda::T
    buf::B
    function NormL1{T, B}(lambda::T, buf::B) where {T, B}
        if !(eltype(lambda) <: Real)
            error("λ must be real")
        end
        if any(lambda .< 0)
            error("λ must be nonnegative")
        else
            new(lambda, buf)
        end
    end
end

is_separable(f::Type{<:NormL1}) = true
is_convex(f::Type{<:NormL1}) = true
is_positively_homogeneous(f::Type{<:NormL1}) = true

NormL1(lambda::R=1; buf=nothing) where R = NormL1{R, typeof(buf)}(lambda, buf)
NormL1{R}(lambda::R) where R = NormL1{R, Nothing}(lambda, nothing)

# only the weighted variant needs scratch space, for the ∑_i λ_i|y_i| it returns
preallocate(f::NormL1{<:AbstractArray}, x::AbstractArray) = NormL1(
    f.lambda; buf = (sig = input_signature(x), lam_abs_y = similar(x, real(eltype(x))))
)

(f::NormL1)(x) = f.lambda * norm(x, 1)

(f::NormL1{<:AbstractArray})(x) = norm(f.lambda .* x, 1)

# ∑_i λ_i |y_i|, computed through the buffer so that the summation order (and
# therefore the result, bit for bit) is the same as `sum(lambda .* abs.(y))`
@inline function weighted_l1(f::NormL1{<:AbstractArray}, x, y)
    t = get_buffers(f, x).lam_abs_y
    t .= f.lambda .* abs.(y)
    return sum(t)
end

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Real}, gamma)
    @assert length(y) == length(x) == length(f.lambda)
    @inbounds @simd for i in eachindex(x)
        gl = gamma * f.lambda[i]
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    end
    return weighted_l1(f, x, y)
end

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Complex}, gamma)
    @assert length(y) == length(x) == length(f.lambda)
    @inbounds @simd for i in eachindex(x)
        gl = gamma * f.lambda[i]
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    end
    return weighted_l1(f, x, y)
end

function prox!(y, f::NormL1, x::AbstractArray{<:Real}, gamma)
    @assert length(y) == length(x)
    n1y = eltype(x)(0)
    gl = gamma * f.lambda
    @inbounds @simd for i in eachindex(x)
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
        n1y += y[i] > 0 ? y[i] : -y[i]
    end
    return f.lambda * n1y
end

function prox!(y, f::NormL1, x::AbstractArray{<:Complex}, gamma)
    @assert length(y) == length(x)
    gl = gamma * f.lambda
    n1y = real(eltype(x))(0)
    @inbounds @simd for i in eachindex(x)
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
        n1y += abs(y[i])
    end
    return f.lambda * n1y
end

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Real}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(f.lambda) == length(gamma)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda[i]
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
    end
    return weighted_l1(f, x, y)
end

function prox!(y, f::NormL1{<:AbstractArray}, x::AbstractArray{<:Complex}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(f.lambda) == length(gamma)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda[i]
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
    end
    return weighted_l1(f, x, y)
end

function prox!(y, f::NormL1, x::AbstractArray{<:Real}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(gamma)
    n1y = eltype(x)(0)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda
        y[i] = x[i] + (x[i] <= -gl ? gl : (x[i] >= gl ? -gl : -x[i]))
        n1y += y[i] > 0 ? y[i] : -y[i]
    end
    return f.lambda * n1y
end

function prox!(y, f::NormL1, x::AbstractArray{<:Complex}, gamma::AbstractArray)
    @assert length(y) == length(x) == length(gamma)
    n1y = real(eltype(x))(0)
    @inbounds @simd for i in eachindex(x)
        gl = gamma[i] * f.lambda
        y[i] = sign(x[i]) * (abs(x[i]) <= gl ? 0 : abs(x[i]) - gl)
        n1y += abs(y[i])
    end
    return f.lambda * n1y
end

function gradient!(y, f::NormL1, x)
    y .= f.lambda .* sign.(x)
    return f(x)
end

function prox_naive(f::NormL1, x, gamma)
    y = sign.(x).*max.(0, abs.(x) .- gamma .* f.lambda)
    return y, norm(f.lambda .* y,1)
end

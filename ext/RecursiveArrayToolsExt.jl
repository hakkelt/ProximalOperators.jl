module RecursiveArrayToolsExt
using RecursiveArrayTools
using ProximalOperators
import ProximalCore: prox, prox!, gradient, gradient!

(f::PrecomposedSlicedSeparableSum)(x::ArrayPartition) = f(x.x)
function prox!(y::ArrayPartition, f::PrecomposedSlicedSeparableSum, x::ArrayPartition, gamma)
    _, fy = prox!(y.x, f, x.x, gamma)
    return y, fy
end

(g::SeparableSum)(xs::ArrayPartition) = g(xs.x)
function prox!(ys::ArrayPartition, g::SeparableSum, xs::ArrayPartition, gamma::Number)
    _, fy = prox!(ys.x, g, xs.x, gamma)
    return ys, fy
end
function prox!(ys::ArrayPartition, g::SeparableSum, xs::ArrayPartition, gammas::Tuple)
    _, fy = prox!(ys.x, g, xs.x, gammas)
    return ys, fy
end
function prox(g::SeparableSum, xs::ArrayPartition, gamma=1)
    y, fy = prox(g, xs.x, gamma)
    return ArrayPartition(y), fy
end
function gradient!(grads::ArrayPartition, g::SeparableSum, xs::ArrayPartition)
    _, f_val = gradient!(grads.x, g, xs.x)
    return grads, f_val
end
function gradient(g::SeparableSum, xs::ArrayPartition)
    grad, f_val = gradient(g, xs.x)
    return ArrayPartition(grad), f_val
end

end # module RecursiveArrayToolsExt

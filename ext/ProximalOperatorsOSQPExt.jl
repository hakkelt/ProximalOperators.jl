module ProximalOperatorsOSQPExt

using LinearAlgebra
using SparseArrays
using OSQP

using ProximalOperators
using ProximalOperators: IndPolyhedralOSQP
import ProximalCore: prox!

# constructors

function ProximalOperators.IndPolyhedralOSQP(
    l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R}
) where R
    m, n = size(A)
    if !all(l .<= u)
        error("function is improper (are some bounds inverted?)")
    end
    mod = OSQP.Model()
    OSQP.setup!(mod; P=SparseMatrixCSC{R}(I, n, n), l=l, A=sparse(A), u=u, verbose=false,
        eps_abs=eps(R), eps_rel=eps(R),
        eps_prim_inf=eps(R), eps_dual_inf=eps(R))
    return IndPolyhedralOSQP{R, typeof(mod)}(l, A, u, mod)
end

ProximalOperators.IndPolyhedralOSQP(
    l::AbstractVector{R}, A::AbstractMatrix{R}, u::AbstractVector{R},
    xmin::AbstractVector{R}, xmax::AbstractVector{R}
) where R =
    IndPolyhedralOSQP([l; xmin], [A; I], [u; xmax])

ProximalOperators.IndPolyhedralOSQP(
    l::AbstractVector{R}, A::AbstractMatrix{R}, args...
) where R =
    IndPolyhedralOSQP(
        l, SparseMatrixCSC(A), R(Inf).*ones(R, size(A, 1)), args...
    )

ProximalOperators.IndPolyhedralOSQP(
    A::AbstractMatrix{R}, u::AbstractVector{R}, args...
) where R =
    IndPolyhedralOSQP(
        R(-Inf).*ones(R, size(A, 1)), SparseMatrixCSC(A), u, args...
    )

# prox

function prox!(y, f::IndPolyhedralOSQP, x, gamma)
    R = eltype(x)
    OSQP.update!(f.mod; q=-x)
    results = OSQP.solve!(f.mod)
    y .= results.x
    return R(0)
end

end # module ProximalOperatorsOSQPExt

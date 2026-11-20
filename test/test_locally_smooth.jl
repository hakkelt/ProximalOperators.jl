# A function whose gradient jumps somewhere (a kink at zero, an edge of a set) is not locally
# smooth: a forward-backward method would take a gradient step on it instead of its prox.
@testset "is_locally_smooth" begin
    for f in (
            NormL1(), NormL1([1.0, 2.0]), NormL2(), NormLinf(), ElasticNet(), SumPositive(),
            DistL2(IndBallL2()),
        )
        @test !is_locally_smooth(f)
    end
    @test is_locally_smooth(LogBarrier())
    @test is_locally_smooth(SqrNormL2())
end

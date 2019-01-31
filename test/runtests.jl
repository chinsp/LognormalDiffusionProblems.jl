using GaussianRandomFields, LognormalDiffusionProblems, MultilevelEstimators, Random, SimpleMultigrid, Test

Random.seed!(100)

#
# Test GRF options
#
@testset "GRF options              " begin
    estimator = init_lognormal(ML(), MC(), length_scale=0.01, smoothness=0.5)	
    @test estimator isa Estimator{<:ML, <:MC}
    estimator = init_lognormal(ML(), MC(), covariance_function=Exponential(0.1))	
    @test estimator isa Estimator{<:ML, <:MC}
    estimator = init_lognormal(ML(), MC(), covariance_function=Exponential(0.3), grf_generator=KarhunenLoeve(101))	
    @test estimator isa Estimator{<:ML, <:MC}
    estimator = init_lognormal(ML(), MC(), covariance_function=Exponential(0.5), minpadding=i->8*2^i[1])	
    @test estimator isa Estimator{<:ML, <:MC}
    estimator = init_lognormal(ZC(2), MC(), covariance_function=Exponential(0.5), minpadding=i->8*2^maximum(i.I))	
    @test estimator isa Estimator{<:ZC, <:MC}
end

#
# MLMC, Qoi1, no reuse
#
@testset "MLMC, Qoi1               " begin
    for solver in [MGSolver(W(3, 3)), MSGSolver(W(3, 3)), DirectSolver()]
        estimator = init_lognormal(ML(), MC(), solver=solver, max_index_set_param=5)
        @test estimator isa Estimator{<:ML, <:MC}
        level = Level(0)
        dQ, Q = estimator.sample_function(level, randn(estimator.options[:nb_of_uncertainties](level)))
        @test dQ isa Float64
        @test Q isa Float64
        @test dQ == Q
        for level in get_index_set(estimator, estimator.options[:max_index_set_param])
            dQ, Q = estimator.sample_function(level, randn(estimator.options[:nb_of_uncertainties](level)))
            @test dQ isa Float64
            @test Q isa Float64
        end
    end
end

#
# TDMC, Qoi2, no reuse
#
@testset "TDMC, Qoi2               " begin
    for solver in [MGSolver(W(3, 3)), MSGSolver(W(3, 3)), DirectSolver()]
        estimator = init_lognormal(TD(2), MC(), solver=solver)
        @test estimator isa Estimator{<:TD{2}, <:MC}
        index = Index(0, 0)
        dQ, Q = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
        @test dQ isa Float64
        @test Q isa Float64
        @test dQ == Q
        for index in get_index_set(estimator, estimator.options[:max_index_set_param])
            dQ, Q = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
            @test dQ isa Float64
            @test Q isa Float64
        end
    end
end

#
# MLMC, Qoi3, no reuse
#
@testset "MLMC, Qoi3               " begin
    for solver in [MGSolver(W(3, 3)), MSGSolver(W(3, 3))]
        estimator = init_lognormal(ML(), MC(), qoi=Qoi3(), max_index_set_param=3, nb_of_coarse_dofs=4, solver=solver)
        @test estimator isa Estimator{<:ML, <:MC}
        level = Level(0)
        dQ, Q = estimator.sample_function(level, randn(estimator.options[:nb_of_uncertainties](level)))
        @test dQ isa Vector{<:Float64}
        @test length(dQ) == 16
        @test Q isa Vector{<:Float64}
        @test length(Q) == 16
        @test all(dQ .== Q)
        for level in get_index_set(estimator, estimator.options[:max_index_set_param])
            dQ, Q = estimator.sample_function(level, randn(estimator.options[:nb_of_uncertainties](level)))
            @test dQ isa Vector{<:Float64}
            @test length(dQ) == 16
            @test Q isa Vector{<:Float64}
            @test length(Q) == 16
        end
    end
end

#
# ADMC, Qoi4, no reuse
#
@testset "ADMC, Qoi4               " begin
    for solver in [MGSolver(W(3, 3)), MSGSolver(W(3, 3))]
        estimator = init_lognormal(AD(2), MC(), solver=solver, qoi=Qoi4())
        @test estimator isa Estimator{<:AD{2}, <:MC}
        index = Index(0, 0)
        dQ, Q = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
        @test dQ isa Float64
        @test Q isa Float64
        @test dQ == Q
        for index in get_index_set(estimator.options[:max_search_space], estimator.options[:max_index_set_param])
            dQ, Q = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
            @test dQ isa Float64
            @test Q isa Float64
        end
    end
end

#
# MG W-cycle analyse
#
@testset "ML, MG, W-cycle analysis " begin
    estimator = init_lognormal(ML(), MC(), solver=MGSolver(W(4, 3)), max_index_set_param=5, analyse=AnalyseV())
    @test estimator isa Estimator{<:ML, <:MC}
    for level in get_index_set(estimator, estimator.options[:max_index_set_param])
        for i in 1:10
            resnorms = estimator.sample_function(level, randn(estimator.options[:nb_of_uncertainties](level)))
            @test resnorms isa Vector{<:Float64}
            @test resnorms[end] < 1e-10
        end
    end
end

#
# TD MSG W-cycle analyse
#
@testset "TD, MSG, W-cycle analysis" begin
    estimator = init_lognormal(TD(2), MC(), solver=MSGSolver(W(3, 3)), max_index_set_param=6, analyse=AnalyseV())
    @test estimator isa Estimator{<:TD{2}, <:MC}
    for index in get_index_set(estimator, estimator.options[:max_index_set_param])
        for i in 1:10
            resnorms = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
            @test resnorms isa Vector{<:Float64}
            @test resnorms[end] < 1e-10
        end
    end
end

#
# MG FMG analyse
#
@testset "ML, MG, FMG analysis     " begin
    estimator = init_lognormal(ML(), MC(), solver=MGSolver(W(3, 3)), max_index_set_param=5, analyse=AnalyseFMG())
    @test estimator isa Estimator{<:ML, <:MC}
    for level in get_index_set(estimator, estimator.options[:max_index_set_param])
        for i in 1:10
            iters = estimator.sample_function(level, randn(estimator.options[:nb_of_uncertainties](level)))
            @test iters isa Vector{<:Int64}
            for j in 1:length(iters)
                @test iters[j] < 20
            end
        end
    end
end

#
# MSG FMG analyse
#
@testset "TD, MSG, FMG analysis    " begin
    estimator = init_lognormal(TD(2), MC(), solver=MSGSolver(W(4, 3)), analyse=AnalyseFMG(), nb_of_coarse_dofs=8)
    @test estimator isa Estimator{<:TD{2}, <:MC}
    for index in get_index_set(estimator, estimator.options[:max_index_set_param])
        for i in 1:10
            iters = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
            @test iters isa Matrix{<:Int64}
            for j in 1:length(iters)
                @test iters[j] < 20
            end
        end
    end
end

#
# U{1}, MC, Qoi1, reuse
#
@testset "U{1}, MC, Qoi1           " begin
    estimator = init_lognormal(U(1), MC())
    @test estimator isa Estimator{<:U{1}, <:MC}
    index = Level(0)
    dQ, Q = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
    @test dQ isa Vector{<:Float64}
    @test Q isa Vector{<:Float64}
    @test all(dQ .== Q)
    for index in get_index_set(estimator.options[:max_search_space], estimator.options[:max_index_set_param])
        dQ, Q = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
        @test dQ isa Vector{<:Float64}
        @test Q isa Vector{<:Float64}
        @test dQ[1] == Q[1]
    end
end

#
# U{2}, MC, Qoi1, reuse
#
@testset "U{2}, MC, Qoi1           " begin
    estimator = init_lognormal(U(2), MC())
    @test estimator isa Estimator{<:U{2}, <:MC}
    index = Index(0, 0)
    dQ, Q = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
    @test dQ isa Matrix{<:Float64}
    @test Q isa Matrix{<:Float64}
    @test all(dQ .== Q)
    for index in get_index_set(estimator.options[:max_search_space], estimator.options[:max_index_set_param])
        dQ, Q = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
        @test dQ isa Matrix{<:Float64}
        @test Q isa Matrix{<:Float64}
        @test dQ[1] == Q[1]
    end
end

#
# U{2}, MC, Qoi3, reuse
#
@testset "U{2}, MC, Qoi3           " begin
    estimator = init_lognormal(U(2), MC(), qoi=Qoi3())
    @test estimator isa Estimator{<:U{2}, <:MC}
    index = Index(0, 0)
    dQ, Q = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
    @test dQ isa Matrix{<:Vector{<:Float64}}
    @test Q isa Matrix{<:Vector{<:Float64}}
    @test all(dQ[1] .== Q[1])
    for index in get_index_set(estimator.options[:max_search_space], estimator.options[:max_index_set_param])
        dQ, Q = estimator.sample_function(index, randn(estimator.options[:nb_of_uncertainties](index)))
        @test dQ isa Matrix{<:Vector{<:Float64}}
        @test Q isa Matrix{<:Vector{<:Float64}}
        @test all(dQ[1] .== Q[1])
    end
end

module LognormalDiffusionProblems

## dependencies ##

using Distributed, FFTW, GaussianRandomFields, Interpolations, PaddedViews, Printf, Retry, Statistics

using MultilevelEstimators, NotSoSimpleMultigrid, SimpleMultigrid

## import statements ##

import GaussianRandomFields: GaussianRandomFieldGenerator

import SimpleMultigrid: MultigridIterable, norm_of_residu, iterate, μ_cycle!, Grid, P, Cubic, residu

import NotSoSimpleMultigrid: grids_at_level, parent_iter, high_freq_mode, P̃, μ_cycle!

## export statements ##

export init_lognormal, sample_lognormal, compute_grf, Qoi1, Qoi2, Qoi3, Qoi4, MGSolver, MSGSolver

## include statements ##

include("init.jl")

include("solvers.jl")

include("sample.jl")

end # module

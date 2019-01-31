# LognormalDiffusionProblems.jl

| **Documentation** | **Build Status** | **Coverage** |
|-------------------|------------------|--------------|
| [![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://PieterjanRobbe.github.io/MultilevelEstimators.jl/stable) [![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://PieterjanRobbe.github.io/MultilevelEstimators.jl/dev) | [![Build Status](https://travis-ci.org/PieterjanRobbe/LognormalDiffusionProblems.jl.png)](https://travis-ci.org/PieterjanRobbe/LognormalDiffusionProblems.jl) [![Build status](https://ci.appveyor.com/api/projects/status/90tupeaq36rdiopp?svg=true)](https://ci.appveyor.com/project/PieterjanRobbe/lognormaldiffusionproblems-jl) | [![Coverage](https://codecov.io/gh/PieterjanRobbe/LognormalDiffusionProblems.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/PieterjanRobbe/LognormalDiffusionProblems.jl) [![Coverage Status](https://coveralls.io/repos/github/PieterjanRobbe/LognormalDiffusionProblems.jl/badge.svg)](https://coveralls.io/github/PieterjanRobbe/MLognormalDiffusionProblems.jl) |

LognormalDiffusionProblems is a package that contains an example on how to use [MultilevelEstimators.jl](https://github.com/PieterjanRobbe/MultilevelEstimators.jl). The package provides code for sampling from and solving a 2d lognormal diffusion problem. This package is mainly used for testing MultilevelEstimators.

## Installation

This package depends on [SimpleMultgrid.jl](https://github.com/PieterjanRobbe/SimpleMultigrid.jl) and [NotSoSimpleMultigrid.jl](https://github.com/PieterjanRobbe/NotSoSimpleMultigrid.jl), two custom packages for solving elliptic problems with a simple Multigrid method that are not (yet) registered in `METADATA.jl`.

Read the [instructions](https://PieterjanRobbe.github.io/MultilevelEstimators.jl/dev/#Installation-1) in the documentation for more details on how to install LognormalDiffusionProblems and its dependencies.

## Documentation

Documentation for MultilevelEstimators is available [here](https://PieterjanRobbe.github.io/MultilevelEstimators.jl/dev).

## Related Packages

- [**MultilevelEstimators.jl**](https://github.com/PieterjanRobbe/MultilevelEstimators.jl) &mdash; base package that contains the multilevel algorithms for solving the problems in this package
- [**Reporter.jl**](https://github.com/PieterjanRobbe/Reporter.jl) &mdash; automatic generation of diagnostic information and reports for a MultilevelEstimators simulation
#
# AbstractQoi
#
abstract type AbstractQoi end

struct Qoi1 <:AbstractQoi end

struct Qoi2 <:AbstractQoi end

struct Qoi3 <:AbstractQoi end

struct Qoi4 <:AbstractQoi end

#
# AbstractSolver
#
abstract type AbstractSolver end

abstract type AbstractMGSolver <: AbstractSolver end

struct DirectSolver <: AbstractSolver end

struct MGSolver{C} <: AbstractMGSolver
	cycle::C
end

struct MSGSolver{C} <: AbstractMGSolver
	cycle::C
end

#
# AbstractReuse
#
abstract type AbstractReuse end

struct Reuse <: AbstractReuse end

struct NoReuse <: AbstractReuse end

#
# AbstractAnalyse
#
abstract type AbstractAnalyse end

struct NoAnalyse <: AbstractAnalyse end

struct AnalyseV <: AbstractAnalyse end

struct AnalyseFMG <: AbstractAnalyse end

#
# problem parameters
#
macro get_arg(key_name, default_value)
    @eval get_arg(args::Dict{Symbol, Any}, ::Val{$key_name}) = haskey(args, $key_name) ? args[$key_name] : $default_value
end

get_arg(args::Dict{Symbol,Any}, arg::Symbol) = get_arg(args, Val(arg))

get_arg(args::Dict{Symbol,Any}, arg::Val{T}) where T = throw(ArgumentError(string("in init_lognormal, invalid key ", T, " found")))

@get_arg :nb_of_coarse_dofs args[:index_set] isa SL ? 4*2^get_arg(args, :max_index_set_param) : 4

@get_arg :covariance_function Matern(get_arg(args, :length_scale), get_arg(args, :smoothness))

@get_arg :length_scale 0.1

@get_arg :smoothness 1

@get_arg :max_index_set_param 6

@get_arg :grf_generator CirculantEmbedding()

@get_arg :minpadding index->0

@get_arg :qoi Qoi1()

@get_arg :solver args[:index_set] isa U && ndims(args[:index_set]) > 1 ? MSGSolver(W(3, 3)) : MGSolver(W(3, 3))

@get_arg :analyse NoAnalyse()

@get_arg :max_search_space ndims(args[:index_set]) == 1 ? ML() : TD(2)

forbidden_keys() = [:nb_of_coarse_dofs, :covariance_function, :length_scale, :smoothness, :grf_generator, :minpadding, :index_set, :qoi, :damping, :solver, :analyse]

#
# init_lognormal
#
function init_lognormal(index_set::AbstractIndexSet, sample_method::AbstractSampleMethod; kwargs...)

    # read optional arguments
    args = Dict{Symbol,Any}(kwargs)
    args[:index_set] = index_set

	# required keys
	args[:max_index_set_param] = get_arg(args, :max_index_set_param)
	if index_set isa Union{AD, U}
		args[:max_search_space] = get_arg(args, :max_search_space)
	end

    # compute Gaussian random fields
    cov_fun = CovarianceFunction(2, get_arg(args, :covariance_function))
    indices = get_max_index_set(index_set, args)
    m0 = get_arg(args, :nb_of_coarse_dofs)
    minpadding = get_arg(args, :minpadding)
	p = minpadding isa Function ? minpadding : i -> minpadding
    grf_generator = get_arg(args, :grf_generator)
	grfs = Dict(index => compute_grf(cov_fun, grf_generator, m0, index, p(index)) for index in indices)

    # sample function
    qoi = get_arg(args, :qoi)
    solver = get_arg(args, :solver)
	reuse = index_set isa U ? Reuse() : NoReuse()
	analyse = get_arg(args, :analyse)
	if analyse != NoAnalyse() && get_arg(args, :solver) == DirectSolver()
		throw(ArgumentError("no analyse available for DirectSolver."))
	end
    sample_function = (index, x) -> sample_lognormal(index, x, grfs[index], qoi, solver, reuse, analyse)

    # distributions
    s = maximum(randdim.(collect(values(grfs))))
    distributions = [Normal() for i in 1:s]

	# set nb_of_uncertainties for more efficient sampling
	args[:nb_of_uncertainties] = index -> randdim(grfs[index])

	# set nb of qoi
	if get_arg(args, :qoi) == Qoi3
		args[:nb_of_qoi] = 16
	end

    # estimator
	for key in forbidden_keys() 
		delete!(args, key)
	end
    Estimator(index_set, sample_method, sample_function, distributions; args...)

end

get_max_index_set(index_set, args) = get_index_set(index_set, get_arg(args, :max_index_set_param))

get_max_index_set(::SL, args) = [Level(0)]

get_max_index_set(::Union{AD, U}, args) = get_index_set(get_arg(args, :max_search_space), get_arg(args, :max_index_set_param))

#
# grf computations
#
grid_size(index::Index) = 2 .^index.I
grid_size(level::Level) = (2^level[1], 2^level[1])

function compute_grf(cov_fun, grf_generator, m0, index, p)
    n = m0 .* grid_size(index)
    pts = broadcast(i -> range(1/i, stop=1-1/i, length=i-1), n)
    compute_grf(cov_fun, grf_generator, pts, p)
end

compute_grf(cov_fun, grf_generator::GaussianRandomFieldGenerator, pts, p) = GaussianRandomField(cov_fun, grf_generator, pts...)

compute_grf(cov_fun, grf_generator::CirculantEmbedding, pts, p) = GaussianRandomField(cov_fun, grf_generator, pts..., minpadding=p, measure=false)

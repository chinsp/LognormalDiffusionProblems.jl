#
# sample
#
function sample_lognormal(index::Index, x::Vector{<:AbstractFloat}, grf::GaussianRandomField, qoi::AbstractQoi, solver::AbstractSolver, reuse::AbstractReuse, analyse::AbstractAnalyse)

	# wrap the sample code in a try-catch  
	ntries = 3
	max_badness = 10

	@repeat ntries try

		Z = my_grf_sample(grf, view(x, 1:randdim(grf)))
		k = exp.(Z)
		sz = size(k) .+ 1

		# direct-discretization function
		f(n, m) = begin
			step = div.(sz, (n, m))
			range = StepRange.(step, step, size(k))
			view(k, range...)
		end
		g(n, m) = elliptic2d(f(n, m))

		if analyse isa AnalyseV
			return μ_cycle_solve(g, sz, solver, 50)
		else
			# solve
			xfs, szs, iters = FMG_solve(g, sz, index, solver, reuse)
			if analyse isa AnalyseFMG 
				return iters
			else
				Qf = apply_qoi(xfs, szs, index, reuse, qoi)

				# compute difference
				dQ = copy(Qf)
				if reuse isa NoReuse
					for (key, val) in diff(index)
						szc = div.(sz, max.(1, (index - key).I .* 2))
						xcs, szcs = FMG_solve(g, szc, key, solver, reuse)
						Qc = apply_qoi(xcs, szcs, key, reuse, qoi)
						dQ += val*Qc
					end
				else
					for i in CartesianIndices(dQ)
						index_ = Index(i - one(i))
						for (key, val) in diff(index_)
							dQ[i] += val*Qf[key + one(key)]
						end
					end
				end

				# return results, if badness is ok
				badness = maximum(abs.(mean.(dQ)))
				if badness < max_badness
					return dQ, Qf
				else
					@warn string("invalid sample detected, badness ", @sprintf("%7.5e", badness))
					throw(ErrorException("something went wrong computing this sample, rethrowing error after $ntries tries"))
				end
			end
		end

	catch e

		@retry if true
			randn!(x) # sample x again and retry
		end

	end
end

#
# custom GRF sampling
#
# TODO: FFT plans cannot deal with pmap (unique C pointer cannot be serialized; afaik)
my_grf_sample(grf::GaussianRandomField, x::AbstractVector) = sample(grf, xi=x)

function my_grf_sample(grf::GaussianRandomField{CirculantEmbedding}, x::AbstractVector)
	v = grf.data[1]

	# compute multiplication with square root of circulant embedding via FFT
	y = v .* reshape(x, size(v))
	w = fft!(complex(y)) # this is slower than using the plan, but works in parallel

	# extract realization of random field
	z = Array{eltype(grf.cov)}(undef, length.(grf.pts))
	@inbounds for i in CartesianIndices(z)
		wi = w[i]
		z[i] = real(wi) + imag(wi)
	end
	z
end

#
# apply quantity of interest
#
apply_qoi(xfs, szs, index, ::NoReuse, qoi) = apply_qoi(reshape(xfs[1], szs[1] .- 1), qoi)

apply_qoi(xfs, szs, index, ::Reuse, qoi) = map(i->apply_qoi(reshape(xfs[i], szs[i] .- 1), qoi), Base.Iterators.reverse(eachindex(xfs)))

"point evaluation of solution at [0.5, 0.5]"
@inline function apply_qoi(x, ::Qoi1)
	sz = size(x) .+ 1
	x[div.(sz, 2)...]
end

"average value of solution over [0.25:0.5, 0.25:0.5]"
@inline function apply_qoi(x, ::Qoi2)
	sz = size(x) .+ 1
	i_end = div.(sz, 2)
	i_start = div.(i_end, 2)
	16*trapz(trapz(view(x, UnitRange.(i_start, i_end)...), 1), 2)[1]
end

"point evaluation of solution at 16 points along middle line"
@inline function apply_qoi(x, ::Qoi3)
	xp = PaddedView(0, x, size(x).+2, (2,2))
	itp = interpolate(xp, BSpline(Linear()))
	sz = size(x) .+ 1
	itp(div(sz[1], 2), range(1, stop=size(xp,2), length=16))
end

"flux through right-most side"
@inline function apply_qoi(x, ::Qoi4)
	px = PaddedView(zero(eltype(x)), x, size(x).+2, (2,2))
	n, m = size(px)
	trapz((m-1)*view(px, :, m-1), 1)[1]
end

#
# trapezoidal rule for computation of integrals in quantity of interest 
#
function trapz(A, dim)
	sz = size(A)
	Rpre = CartesianIndices(sz[1:dim-1])
	Rpost = CartesianIndices(sz[dim+1:end])
	szs = [sz...]
	n = szs[dim]
	szs[dim] = 1
	B = Array{eltype(A)}(undef, szs...)
	trapz!(B, A, Rpre, Rpost, n)
end

@noinline function trapz!(B, A, Rpre, Rpost, n)
	fill!(B, zero(eltype(B)))
	for Ipost in Rpost
		for Ipre in Rpre
			for i = 2:n
				B[Ipre, 1, Ipost] += A[Ipre, i, Ipost] + A[Ipre, i-1, Ipost]
			end
		end
	end
	B./(2(n-1))
end

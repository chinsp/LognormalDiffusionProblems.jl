#
# μ-cycle analysis
#

"return the Multigrid residual norm after `iters` multigrid μ-cycles"
function μ_cycle_solve(f::Function, sz::Dims, solver::AbstractSolver, iters::Integer)
	mg = MG(f, sz, solver)
	fill!(mg.grids[1].b, 1)
	for iter in Iterators.take(mg, iters) end
    mg.resnorm
end

MG(f, sz, solver::MGSolver) = SimpleMultigrid.MultigridMethod(f, sz, solver.cycle, smoother=RedBlackGaussSeidel())
MG(f, sz, solver::MSGSolver) = NotSoSimpleMultigrid.MultigridMethod(f, sz, solver.cycle, smoother=RedBlackGaussSeidel())

#
# FMG cycle
#

"use FMG to solve the PDE, and optionally return the number of μ-cycle iterations on each grid"
function FMG_solve(f::Function, sz::Dims, index::Index, solver::AbstractSolver, reuse::AbstractReuse)
	mg = MG(f, sz, solver)
	fill!(mg.grids[1].b, 1)
    sol, iters = FMG!(mg, 1)
	R = selectrange(index, reuse)
    view(sol, R), view(size.(mg.grids), R), iters
end

selectrange(index::Index, ::Reuse) = CartesianIndices(index .+ one(index)) 

selectrange(index::Index, ::NoReuse) = CartesianIndex(1) 

#
# custom FMG implementation for MG that returns the coarse solutions
#
function FMG!(mg::MultigridIterable{<:Any, <:AbstractVector}, grid_ptr::Integer)
    grids = mg.grids
    if grid_ptr == length(grids)
        fill!(grids[grid_ptr].x, zero(eltype(grids[grid_ptr].x))) 
        sol = Vector{Vector{eltype(grids[1].x)}}(undef, length(grids))
        iters = Vector{typeof(grid_ptr)}(undef, length(grids))
    else
		copyto!(grids[grid_ptr + 1].b, grids[grid_ptr].R*grids[grid_ptr].b)
        sol, ν₀s = FMG!(mg, grid_ptr + 1)
		copyto!(grids[grid_ptr].x, P(Cubic(), grids[grid_ptr + 1].sz...) * grids[grid_ptr + 1].x)
    end
    iter = 0
    while !converged(grids, grid_ptr) && iter < 20
		μ_cycle!(grids, μ(mg.cycle_type), mg.cycle_type.ν₁, mg.cycle_type.ν₂, grid_ptr, mg.smoother)
        iter += 1
    end
    if !converged(grids, grid_ptr) # safety
		copyto!(grids[grid_ptr].x, grids[grid_ptr].A\grids[grid_ptr].b) # exact solve
    end
    sol[grid_ptr] = copy(grids[grid_ptr].x)
    iters[grid_ptr] = iter

    return sol, iters
end

converged(grid::Grid) = norm_of_residu(grid) < 1/prod(size(grid))

converged(grids::Vector{<:Grid}, grid_ptr) = converged(grids[grid_ptr])

μ(::V) = 1

μ(::W) = 2

#
# custom FMG implementation for MSG that returns the coarse solutions
#
function FMG!(mg::MultigridIterable{<:Any, <:AbstractMatrix}, grid_ptr::Int)
    grids = mg.grids
    R = CartesianIndices(size(grids))
    I1, Iend = first(R), last(R)
    if grid_ptr == sum(Tuple(Iend-I1)) + 1
        fill!(grids[grid_ptr].x, zero(eltype(grids[grid_ptr].x)))
        sol = Matrix{Vector{eltype(grids[1].x)}}(undef, size(grids)...)
        iters = Matrix{typeof(grid_ptr)}(undef, size(grids)...)
    else
        for I in grids_at_level(R, grid_ptr + 1)
            R_child = child_iter(R, I1, I)
			copyto!(grids[I].b, mean(map(i -> grids[last(i)].R[first(i)] * grids[last(i)].b, R_child)))
        end
        sol, iters = FMG!(mg, grid_ptr + 1)
        for I in grids_at_level(R, grid_ptr)
            R_parent = parent_iter(R, I1, I)
            λ = map(i -> grids[I].A * high_freq_mode(first(i), grids[I].sz), R_parent)
            λ² = broadcast(i->broadcast(j->j^2, i), λ)
            ω = map(i -> λ²[i] ./ sum(λ²), 1:length(λ))
            ip = map(i -> P̃(first(i), Cubic(), grids[last(i)].sz...) * grids[last(i)].x, R_parent)
			if length(R_parent) == 1
				d = SimpleMultigrid.residu(grids[I])
            	α = c'*d/(c'*grids[I].A*c)
            	α = isnan(α) ? one(eltype(c)) : min(1.1, max(0.7, α))
			else
				α = 1.
			end
			copyto!(grids[I].x, α * sum(map(i->ω[i].*ip[i], 1:length(ω))))
        end
    end
	iter = 0
	while !converged(grids, grid_ptr, R) && iter < 20
		μ_cycle!(grids, μ(mg.cycle_type), mg.cycle_type.ν₁, mg.cycle_type.ν₂, grid_ptr, mg.smoother)
		iter += 1
	end
	for I in grids_at_level(R, grid_ptr)
		if !converged(grids[I]) # safety
			copyto!(grids[I].x, grids[I].A\grids[I].b) # exact solve
		end
		sol[I] = copy(grids[I].x)
		iters[I] = iter
	end

    return sol, ν₀s
end

converged(grids::Matrix{<:Grid}, grid_ptr, R) = all([converged(grids[I]) for I in grids_at_level(R, grid_ptr)]) 

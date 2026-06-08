module LinearSolveDaggerExt

# Distributed dense/sparse linear solvers backed by Dagger.jl.
#
# Design overview
# ---------------
# * Inputs that are already `Dagger.DArray`s are used as-is; everything else is
#   wrapped in a `DArray` so the heavy linear-algebra runs distributed. Dense
#   in-memory arrays are wrapped with `view(A, Blocks(bs, bs))`, which takes
#   *views* of the original storage (no copy, aliasing preserved). Sparse inputs
#   are `distribute`d so each tile keeps its sparse storage (`DSparseArray`).
# * We keep data as `DArray`s for the whole solve and only materialize back into
#   the (possibly non-`DArray`) `cache.u` at the very end, so the pipeline stays
#   asynchronous.
# * A *square* tile size is always used: Dagger's tiled factorizations re-buffer
#   to square tiles internally, and the iterative solvers/preconditioners
#   *require* square diagonal tiles. `AutoBlocks` would instead give a 1-D column
#   partition (non-square tiles), which is both suboptimal for 2-D blocked
#   factorizations and unusable by the iterative path, so we deliberately pick a
#   square `Blocks(bs, bs)` instead.
# * Task placement / GPU usage / data locality are left entirely to the user via
#   `Dagger.with_options(; scope = ...)`; this extension never sets a scope.

using LinearSolve
using LinearSolve: LinearCache, OperatorAssumptions, LinearVerbosity,
    AbstractDaggerLinearSolveAlgorithm, DaggerLUFactorization,
    DaggerCholeskyFactorization, DaggerQRFactorization, DaggerKrylovJL
using LinearSolve.SciMLBase: SciMLBase, ReturnCode
using SciMLOperators: IdentityOperator
using LinearAlgebra
using Dagger: Dagger

# Mutable cache container so `solve!` can stash the factorization (direct
# methods) or the `(DMatrix, preconditioner)` pair (iterative methods) without
# reassigning the strongly-typed `cache.cacheval` field. `init_cacheval` returns
# this for every Dagger algorithm, so the cache field type stays stable.
mutable struct DaggerSolveCache
    state::Any
end

function LinearSolve.init_cacheval(
        ::AbstractDaggerLinearSolveAlgorithm, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return DaggerSolveCache(nothing)
end

# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------

# We handle all wrapping ourselves and never require a plain dense `Matrix`.
LinearSolve.needs_concrete_A(::AbstractDaggerLinearSolveAlgorithm) = false

# Alias the inputs: the dense wrap is a non-copying `view`, the direct
# factorizations are out-of-place (they never mutate `A`), and the RHS is copied
# into a private working vector before any in-place solve. Aliasing therefore
# avoids redundant copies while preserving the original arrays, exactly as
# desired for `DArray`/dense inputs.
LinearSolve.default_alias_A(::AbstractDaggerLinearSolveAlgorithm, ::Any, ::Any) = true
LinearSolve.default_alias_b(::AbstractDaggerLinearSolveAlgorithm, ::Any, ::Any) = true

# ---------------------------------------------------------------------------
# Block-size selection and array wrapping
# ---------------------------------------------------------------------------

# Auto block size: aim for roughly `sqrt(np)` tiles along each dimension so the
# `t x t` tile grid spreads across the `~np` available processors. Falls back to
# a single tile when there is only one processor or the matrix is tiny.
function _auto_blocksize(n::Integer)
    n <= 1 && return max(1, Int(n))
    np = max(1, Dagger.num_processors())
    tiles = max(1, floor(Int, sqrt(np)))
    return clamp(cld(Int(n), tiles), 1, Int(n))
end

_blocksize(alg::AbstractDaggerLinearSolveAlgorithm, n::Integer) =
    alg.blocksize === nothing ? _auto_blocksize(n) : clamp(alg.blocksize, 1, Int(n))

# Wrap a matrix as a `DMatrix`. Existing `DArray`s pass through untouched; dense
# arrays are viewed (aliasing, no copy); other matrices (e.g. sparse) are
# distributed so their tiles keep the proper storage type.
_to_dmatrix(A::Dagger.DArray, ::AbstractDaggerLinearSolveAlgorithm) = A
function _to_dmatrix(A::AbstractMatrix, alg::AbstractDaggerLinearSolveAlgorithm)
    bs = _blocksize(alg, size(A, 1))
    part = Dagger.Blocks(bs, bs)
    return A isa DenseArray ? view(A, part) : Dagger.distribute(A, part)
end

# Produce a *fresh* `DVector` with the solver's square block size, copying the
# data so an in-place solve never mutates the user's `b`. Works for both plain
# arrays and `DArray`s (repartitioning the latter when needed).
function _working_dvector(b::AbstractVector, bs::Integer)
    x = Dagger.DVector{eltype(b)}(undef, Dagger.Blocks(Int(bs)), length(b))
    copyto!(x, b)
    return x
end

# Materialize the distributed solution `x` back into `cache.u`. For
# overdetermined least-squares solves only the leading `ncols` entries of the
# (length-`nrows`) working vector hold the solution.
function _store_solution!(u, x::Dagger.DArray, ncols::Integer)
    src = length(x) == ncols ? x : x[1:ncols]
    copyto!(u, src)
    return u
end

# ---------------------------------------------------------------------------
# Dense direct factorizations
# ---------------------------------------------------------------------------

function SciMLBase.solve!(cache::LinearCache, alg::DaggerLUFactorization; kwargs...)
    if cache.isfresh
        DA = _to_dmatrix(cache.A, alg)
        pivot = alg.pivot ? RowMaximum() : NoPivot()
        cache.cacheval.state = lu(DA, pivot)
        cache.isfresh = false
    end
    F = cache.cacheval.state
    bs = _blocksize(alg, size(cache.A, 1))
    x = _working_dvector(cache.b, bs)
    ldiv!(F, x)
    _store_solution!(cache.u, x, size(cache.A, 2))
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success
    )
end

function SciMLBase.solve!(cache::LinearCache, alg::DaggerCholeskyFactorization; kwargs...)
    if cache.isfresh
        DA = _to_dmatrix(cache.A, alg)
        cache.cacheval.state = cholesky(DA)
        cache.isfresh = false
    end
    F = cache.cacheval.state
    bs = _blocksize(alg, size(cache.A, 1))
    x = _working_dvector(cache.b, bs)
    ldiv!(F, x)
    _store_solution!(cache.u, x, size(cache.A, 2))
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success
    )
end

function SciMLBase.solve!(cache::LinearCache, alg::DaggerQRFactorization; kwargs...)
    if cache.isfresh
        DA = _to_dmatrix(cache.A, alg)
        # `qr!` mutates its argument, so factorize an independent copy; this keeps
        # both the wrapped view and the user's original `A` intact.
        cache.cacheval.state = qr!(copy(DA); ib = alg.inner_blocksize, p = alg.domains)
        cache.isfresh = false
    end
    F = cache.cacheval.state
    bs = _blocksize(alg, size(cache.A, 1))
    # Working vector has the RHS length (number of rows); for overdetermined
    # systems the solution occupies its leading `size(A, 2)` entries.
    x = _working_dvector(cache.b, bs)
    ldiv!(F, x)
    _store_solution!(cache.u, x, size(cache.A, 2))
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success
    )
end

# ---------------------------------------------------------------------------
# Iterative (Krylov) solvers
# ---------------------------------------------------------------------------

_ntiles(DA::Dagger.DArray) = size(DA.chunks, 1)

# Build a Dagger preconditioner object (applied as `mul!(y, M, x)`, i.e. the
# approximate inverse) from the `precond` selector on the algorithm.
function _build_precond(alg::DaggerKrylovJL, DA::Dagger.DArray)
    p = alg.precond
    if p === nothing || p === :none
        return nothing
    elseif p === :jacobi
        return Dagger.JacobiPreconditioner(DA)
    elseif p === :blockjacobi
        return Dagger.BlockJacobiPreconditioner(DA)
    elseif p === :auto
        # Cheap, robust default: a diagonal preconditioner whenever the operator
        # actually spans more than one diagonal tile (otherwise it is pointless).
        return _ntiles(DA) > 1 ? Dagger.JacobiPreconditioner(DA) : nothing
    elseif p isa Function
        return p(DA)
    else
        # Assume a ready-made preconditioner object was supplied.
        return p
    end
end

# Pick the concrete Krylov method. With no symmetry information available from
# `OperatorAssumptions`, `:auto` falls back to robust GMRES.
_select_method(alg::DaggerKrylovJL) = alg.method === :auto ? :gmres : alg.method

function _krylov_kwargs(alg::DaggerKrylovJL, cache::LinearCache, M, method::Symbol)
    kw = (; atol = cache.abstol, rtol = cache.reltol, itmax = cache.maxiters)
    M === nothing || (kw = merge(kw, (; M = M)))
    # NOTE: Dagger's iterative API forwards kwargs to `Krylov.krylov_solve!`, which
    # only accepts `restart` for GMRES. The Krylov subspace size (`memory`) is a
    # workspace-construction parameter that Dagger currently sets to its default
    # and does not expose, so `gmres_memory` cannot be plumbed through yet.
    if method === :gmres && alg.gmres_restart > 0
        kw = merge(kw, (; restart = true))
    end
    return merge(kw, alg.kwargs)
end

function _krylov_resid(stats)
    hasproperty(stats, :residuals) || return nothing
    r = stats.residuals
    return isempty(r) ? nothing : last(r)
end

function SciMLBase.solve!(cache::LinearCache, alg::DaggerKrylovJL; kwargs...)
    if (!(cache.Pl isa UniformScaling) && !isa_identity(cache.Pl)) ||
            (!(cache.Pr isa UniformScaling) && !isa_identity(cache.Pr))
        @warn "DaggerKrylovJL does not use LinearSolve's `Pl`/`Pr`; set `precond` \
               on the algorithm instead. Provided preconditioners are ignored." maxlog = 1
    end

    bs = _blocksize(alg, size(cache.A, 1))
    if cache.isfresh
        DA = _to_dmatrix(cache.A, alg)
        M = _build_precond(alg, DA)
        cache.cacheval.state = (DA, M)
        cache.isfresh = false
    end
    DA, M = cache.cacheval.state

    Db = _working_dvector(cache.b, bs)
    method = _select_method(alg)
    x, stats = Dagger.krylov_solve(method, DA, Db; _krylov_kwargs(alg, cache, M, method)...)
    _store_solution!(cache.u, x, size(cache.A, 2))

    retcode = stats.solved ? ReturnCode.Success : ReturnCode.ConvergenceFailure
    iters = hasproperty(stats, :niter) ? stats.niter : 0
    return SciMLBase.build_linear_solution(
        alg, cache.u, _krylov_resid(stats), cache; retcode = retcode, iters = iters
    )
end

isa_identity(x) = x isa IdentityOperator

# ---------------------------------------------------------------------------
# Default algorithm selection for `DArray` inputs
# ---------------------------------------------------------------------------

# Best-effort detection of a sparse-backed `DArray` (tiles are `DSparseArray`).
function _is_sparse_darray(A::Dagger.DArray)
    isempty(A.chunks) && return false
    T = try
        Dagger.chunktype(first(A.chunks))
    catch
        return false
    end
    return T <: Dagger.DSparseArray
end

function LinearSolve.defaultalg(
        A::Dagger.DArray, b, assump::OperatorAssumptions{Bool}
    )
    if assump.issq
        return _is_sparse_darray(A) ? DaggerKrylovJL() : DaggerLUFactorization()
    else
        # Non-square: least-squares via QR (sparse direct least-squares is not
        # available, so sparse non-square also routes through QR after densify).
        return DaggerQRFactorization()
    end
end

# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

LinearSolve.DaggerKrylovJL_CG(; kwargs...) = DaggerKrylovJL(; method = :cg, kwargs...)
LinearSolve.DaggerKrylovJL_MINRES(; kwargs...) = DaggerKrylovJL(; method = :minres, kwargs...)
LinearSolve.DaggerKrylovJL_GMRES(; kwargs...) = DaggerKrylovJL(; method = :gmres, kwargs...)
LinearSolve.DaggerKrylovJL_BICGSTAB(; kwargs...) = DaggerKrylovJL(; method = :bicgstab, kwargs...)

end # module LinearSolveDaggerExt

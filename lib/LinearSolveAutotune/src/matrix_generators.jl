# Test-problem generators and the problem-class abstraction.
#
# The autotuner historically only benchmarked *dense* random systems. This file
# adds a small abstraction (`BenchmarkProblem`) that describes a class of test
# problems, together with a set of sparse generators of various kinds (PDE-like,
# unstructured SPD, unstructured nonsymmetric, and banded). The same abstraction
# also drives which solvers are appropriate for a given class.
#
# Soundness note
# --------------
# The benchmark "GFLOPs" metric for *dense* problems is derived from the dense LU
# flop count (`luflop`). That model is meaningless for sparse problems (a sparse
# solve performs far fewer operations), so for sparse problems we instead report a
# nominal throughput based on `2 * nnz(A)` (one sparse mat-vec worth of work) per
# second. Within a single (problem class, size, eltype) group this is a constant
# multiple of `1/runtime`, so it is a faithful *relative* speed ranking of solvers;
# it should NOT be compared across problem classes or against the dense numbers.

using SparseArrays
using LinearAlgebra
using Random

"""
    BenchmarkProblem

Describes a class of linear-system test problems for the autotuner.

# Fields

  - `name::String`: short identifier, also used as the `matrix_type` column value
    in the results DataFrame (e.g. `"dense"`, `"sparse_laplace2d"`).
  - `generate`: a function `(rng, ::Type{T}, n) -> AbstractMatrix{T}` building an
    `n`-ish × `n`-ish matrix. Some generators (e.g. the 2D Laplacian) round `n` to
    the nearest compatible size, so callers must read the *actual* size from the
    returned matrix rather than assuming `n`.
  - `sparse::Bool`: whether the generated matrix is sparse.
  - `symmetric::Bool`: whether the matrix is (Hermitian-)symmetric.
  - `posdef::Bool`: whether the matrix is positive definite.
  - `min_size::Int`: smallest matrix size for which the class is meaningful.
  - `description::String`: human-readable description.
"""
struct BenchmarkProblem
    name::String
    generate::Function
    sparse::Bool
    symmetric::Bool
    posdef::Bool
    min_size::Int
    description::String
end

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# Dense random matrix - the historical default problem class.
gen_dense(rng, ::Type{T}, n) where {T} = rand(rng, T, n, n)

"""
    gen_spd_laplace2d(rng, T, n)

2D 5-point Laplacian (Poisson operator) on a `k×k` grid with `k = round(√n)`, so
the returned matrix is `k^2 × k^2`. Symmetric positive definite and structurally
banded - the canonical sparse PDE test problem. Its condition number grows like
`O(k^2)`, which makes it a good stress test for *unpreconditioned* iterative
solvers (see the soundness note above).
"""
function gen_spd_laplace2d(rng, ::Type{T}, n) where {T}
    k = max(2, round(Int, sqrt(n)))
    main = fill(T(2), k)
    off = fill(T(-1), k - 1)
    T1 = spdiagm(-1 => off, 0 => main, 1 => off)
    Ik = spdiagm(0 => ones(T, k))
    return kron(Ik, T1) + kron(T1, Ik)
end

"""
    gen_spd_random(rng, T, n; nnz_per_row = 6)

Unstructured symmetric, strictly diagonally dominant (hence SPD), sparse matrix
with roughly `nnz_per_row` off-diagonal entries per row. Well-conditioned, so it
is friendly to both sparse direct factorizations and (preconditioned) CG.
"""
function gen_spd_random(rng, ::Type{T}, n; nnz_per_row::Int = 6) where {T}
    n = max(2, n)
    p = min(0.5, nnz_per_row / n)
    S = sprandn(rng, n, n, p)
    Bsym = S + transpose(S)
    Bsym = Bsym - spdiagm(0 => diag(Bsym))      # zero the diagonal; we set it below
    B = T.(Bsym)
    d = vec(sum(abs, B; dims = 2)) .+ one(real(T))
    return B + spdiagm(0 => T.(d))
end

"""
    gen_nonsym_diagdom(rng, T, n; nnz_per_row = 6)

Unstructured *nonsymmetric*, strictly diagonally dominant (hence invertible and
well-conditioned) sparse matrix. Representative of advection-dominated or
circuit-like systems where CG/Cholesky do not apply, but GMRES/BiCGSTAB and
sparse LU (UMFPACK/KLU) do.
"""
function gen_nonsym_diagdom(rng, ::Type{T}, n; nnz_per_row::Int = 6) where {T}
    n = max(2, n)
    p = min(0.5, nnz_per_row / n)
    S = sprandn(rng, n, n, p)
    S = S - spdiagm(0 => diag(S))
    B = T.(S)
    d = vec(sum(abs, B; dims = 2)) .+ one(real(T))
    return B + spdiagm(0 => T.(d))
end

"""
    gen_tridiagonal(rng, T, n)

Symmetric positive-definite tridiagonal (banded) matrix - effectively a 1D
Laplacian made strictly diagonally dominant. Extremely sparse (`~3n` nonzeros), so
it scales to very large `N` and exercises the banded/iterative regime cheaply.
"""
function gen_tridiagonal(rng, ::Type{T}, n) where {T}
    n = max(2, n)
    main = fill(T(2.001), n)        # slightly > 2 ⇒ strictly diagonally dominant ⇒ SPD
    off = fill(T(-1), n - 1)
    return spdiagm(-1 => off, 0 => main, 1 => off)
end

"""
    dense_problem()

The default dense random-matrix problem class (historical autotune behavior).
"""
dense_problem() = BenchmarkProblem(
    "dense", gen_dense, false, false, false, 1, "Dense random matrix"
)

"""
    get_sparse_problems()

Return the list of sparse [`BenchmarkProblem`](@ref)s benchmarked by the
autotuner. The set spans structured/unstructured, symmetric/nonsymmetric, and
banded sparsity so that the relative ranking of sparse solvers can be assessed
across problem characteristics.
"""
function get_sparse_problems()
    return BenchmarkProblem[
        BenchmarkProblem(
            "sparse_laplace2d", gen_spd_laplace2d, true, true, true, 9,
            "2D 5-point Laplacian (SPD, structured PDE)"
        ),
        BenchmarkProblem(
            "sparse_spd_random", gen_spd_random, true, true, true, 16,
            "Unstructured symmetric diagonally-dominant (SPD)"
        ),
        BenchmarkProblem(
            "sparse_nonsym_diagdom", gen_nonsym_diagdom, true, false, false, 16,
            "Unstructured nonsymmetric diagonally-dominant"
        ),
        BenchmarkProblem(
            "sparse_tridiagonal", gen_tridiagonal, true, true, true, 4,
            "Tridiagonal SPD (banded; scales to very large N)"
        ),
    ]
end

# ---------------------------------------------------------------------------
# Per-class algorithm selection
# ---------------------------------------------------------------------------

"""
    get_sparse_algorithms(problem::BenchmarkProblem)

Return `(algs, names)` of CPU solvers appropriate for a sparse `problem`. Includes
sparse direct factorizations (UMFPACK, KLU, Sparspak; plus CHOLMOD for SPD
problems) and Krylov iterative methods (CG for SPD, BiCGSTAB for nonsymmetric,
and GMRES always). The element-type/sparsity compatibility of each candidate is
checked later via [`test_algorithm_compatibility`](@ref).
"""
function get_sparse_algorithms(problem::BenchmarkProblem)
    algs = Any[]
    names = String[]

    push!(algs, UMFPACKFactorization())
    push!(names, "UMFPACKFactorization")

    push!(algs, KLUFactorization())
    push!(names, "KLUFactorization")

    try
        push!(algs, SparspakFactorization())
        push!(names, "SparspakFactorization")
    catch e
        @warn "SparspakFactorization unavailable; skipping it for sparse benchmarks: $e"
    end

    if problem.posdef
        push!(algs, CHOLMODFactorization())
        push!(names, "CHOLMODFactorization")
    end

    if problem.symmetric && problem.posdef
        push!(algs, KrylovJL_CG())
        push!(names, "KrylovJL_CG")
    else
        push!(algs, KrylovJL_BICGSTAB())
        push!(names, "KrylovJL_BICGSTAB")
    end

    push!(algs, KrylovJL_GMRES())
    push!(names, "KrylovJL_GMRES")

    return algs, names
end

"""
    get_dagger_algorithms(problem::BenchmarkProblem)

Return `(algs, names)` of Dagger.jl-backed distributed solvers to compare against
the in-process solvers for a given `problem`, across all benchmark sizes. For dense
problems this is the tiled LU; for sparse problems it is a distributed Krylov
method (CG for SPD, GMRES otherwise), since Dagger currently has no distributed
sparse direct factorization.

Returns empty vectors (with a warning) if the Dagger extension cannot be loaded.

!!! note "Execution model"
    These benchmarks run Dagger with its default, thread-based scheduler (start
    Julia with multiple threads, e.g. `julia -t auto`, to actually parallelize).

    TODO: add multi-process Dagger runs (e.g. `addprocs` / `Distributed`) so the
    distributed-memory scaling can be measured.

    TODO: add GPU-accelerated Dagger runs by selecting a GPU scope
    (`Dagger.with_options(; scope = Dagger.scope(cuda_gpu = ...))`) once the GPU
    tiled factorizations are wired through this extension.
"""
function get_dagger_algorithms(problem::BenchmarkProblem)
    algs = Any[]
    names = String[]
    try
        if problem.sparse
            if problem.symmetric && problem.posdef
                push!(algs, DaggerKrylovJL_CG())
                push!(names, "DaggerKrylovJL_CG")
            else
                push!(algs, DaggerKrylovJL_GMRES())
                push!(names, "DaggerKrylovJL_GMRES")
            end
        else
            push!(algs, DaggerLUFactorization())
            push!(names, "DaggerLUFactorization")
            push!(algs, DaggerQRFactorization())
            push!(names, "DaggerQRFactorization")
        end
    catch e
        @warn "Dagger solvers unavailable; skipping them. Is Dagger loaded? Error: $e"
        return Any[], String[]
    end
    return algs, names
end

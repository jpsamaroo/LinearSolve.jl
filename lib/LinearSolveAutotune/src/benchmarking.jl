# Core benchmarking functionality

using ProgressMeter
using LinearAlgebra

"""
    eltype_precheck(alg_name::String, eltype::Type)

Fast, name-based screen for known-incompatible (algorithm, element type) pairs.
Returns `false` when the algorithm is known not to support `eltype` (e.g. BLAS
wrappers with non-BLAS floats, or sparse/GPU routines without `Float16`), `true`
otherwise. This avoids the cost (and crash risk) of actually attempting the solve.
"""
function eltype_precheck(alg_name::String, eltype::Type)
    # Define strict compatibility rules for BLAS-dependent algorithms
    # Standard BLAS algorithms that rely on LinearAlgebra.BLAS interface
    if !(eltype <: LinearAlgebra.BLAS.BlasFloat) && alg_name in [
            "LUFactorization", "QRFactorization", "CHOLMODFactorization",
        ]
        return false  # Standard BLAS algorithms not compatible with non-standard types
    end

    # Manual BLAS wrappers with explicit method signatures for specific types only
    if alg_name in [
            "BLISLUFactorization", "MKLLUFactorization", "AppleAccelerateLUFactorization",
            "OpenBLASLUFactorization",
        ] &&
            !(eltype in [Float32, Float64, ComplexF32, ComplexF64])
        return false
    end

    if alg_name == "BLISLUFactorization" && Sys.isapple()
        return false  # BLISLUFactorization has no Apple Silicon binary
    end

    # GPU algorithms with limited Float16 support - prevent usage to avoid segfaults/errors
    if alg_name == "MetalLUFactorization" && eltype == Float16
        return false
    end

    if alg_name in [
            "CudaOffloadLUFactorization", "CudaOffloadQRFactorization", "CudaOffloadFactorization",
        ] &&
            eltype == Float16
        return false
    end

    if alg_name in ["AMDGPUOffloadLUFactorization", "AMDGPUOffloadQRFactorization"] &&
            eltype == Float16
        return false
    end

    # Sparse factorization algorithms: Most don't support Float16
    if alg_name in ["UMFPACKFactorization", "KLUFactorization"] && eltype == Float16
        return false
    end

    # UMFPACK/KLU/CHOLMOD (SuiteSparse) only support Float64/ComplexF64
    if alg_name in ["UMFPACKFactorization", "KLUFactorization", "CHOLMODFactorization"] &&
            !(eltype in [Float64, ComplexF64])
        return false
    end

    # Dagger tiled factorizations are built on BLAS kernels
    if alg_name == "DaggerLUFactorization" && !(eltype <: LinearAlgebra.BLAS.BlasFloat)
        return false
    end

    if alg_name in [
            "MKLPardisoFactorize", "MKLPardisoIterate",
            "PanuaPardisoFactorize", "PanuaPardisoIterate", "PardisoJL",
        ] &&
            eltype == Float16
        return false
    end

    if alg_name == "CUSOLVERRFFactorization" && eltype == Float16
        return false
    end

    return true
end

"""
    test_algorithm_compatibility(alg, eltype::Type, test_size::Int=4)

Test if an algorithm is compatible with a given element type by solving a small
*dense* test problem. Returns true if compatible, false otherwise.
Uses more strict rules for BLAS-dependent algorithms with non-standard types.
"""
function test_algorithm_compatibility(alg, eltype::Type, test_size::Int = 4)
    alg_name = string(typeof(alg).name.name)
    eltype_precheck(alg_name, eltype) || return false

    # For standard types or algorithms that passed the strict check, test functionality
    try
        # Create a small test problem with the specified element type
        rng = MersenneTwister(123)
        A = rand(rng, eltype, test_size, test_size)
        b = rand(rng, eltype, test_size)
        u0 = rand(rng, eltype, test_size)

        prob = LinearProblem(A, b; u0 = u0)

        # Try to solve - if it works, the algorithm is compatible
        sol = solve(prob, alg)

        # Additional check: verify the solution is actually of the expected type
        if !isa(sol.u, AbstractVector{eltype})
            @debug "Algorithm $alg_name returned wrong element type for $eltype"
            return false
        end

        return true

    catch e
        # Algorithm failed - not compatible with this element type
        @debug "Algorithm $alg_name failed for $eltype: $e"
        return false
    end
end

"""
    test_algorithm_compatibility(alg, eltype::Type, problem::BenchmarkProblem)

Problem-class-aware compatibility test. Builds a small instance of `problem`'s
matrix class (dense or sparse) with the given element type and attempts a solve.
This is essential for sparse-only solvers (UMFPACK, KLU, …), which would
spuriously fail a dense compatibility probe.
"""
function test_algorithm_compatibility(alg, eltype::Type, problem::BenchmarkProblem)
    alg_name = string(typeof(alg).name.name)
    eltype_precheck(alg_name, eltype) || return false

    try
        rng = MersenneTwister(123)
        # Use a modest instance that is still a valid member of the problem class.
        probe_size = max(problem.min_size, problem.sparse ? 36 : 4)
        A = problem.generate(rng, eltype, probe_size)
        n = size(A, 1)
        b = rand(rng, eltype, n)
        u0 = rand(rng, eltype, n)

        sol = solve(LinearProblem(A, b; u0 = u0), alg)

        if !isa(sol.u, AbstractVector{eltype})
            @debug "Algorithm $alg_name returned wrong element type for $eltype on $(problem.name)"
            return false
        end
        return true
    catch e
        @debug "Algorithm $alg_name failed for $eltype on $(problem.name): $e"
        return false
    end
end

"""
    filter_compatible_algorithms(algorithms, alg_names, eltype::Type[, problem])

Filter algorithms to only those compatible with the given element type (and, when
`problem` is supplied, the given problem class). Returns filtered algorithms and
names.
"""
function filter_compatible_algorithms(algorithms, alg_names, eltype::Type)
    compatible_algs = []
    compatible_names = String[]

    for (alg, name) in zip(algorithms, alg_names)
        if test_algorithm_compatibility(alg, eltype)
            push!(compatible_algs, alg)
            push!(compatible_names, name)
        end
    end

    return compatible_algs, compatible_names
end

function filter_compatible_algorithms(
        algorithms, alg_names, eltype::Type, problem::BenchmarkProblem
    )
    compatible_algs = []
    compatible_names = String[]

    for (alg, name) in zip(algorithms, alg_names)
        if test_algorithm_compatibility(alg, eltype, problem)
            push!(compatible_algs, alg)
            push!(compatible_names, name)
        end
    end

    return compatible_algs, compatible_names
end

"""
    reference_solution_for(problem, A, b, u0, eltype)

Compute a trusted reference solution used for the correctness gate. For dense
problems this uses the standard LU factorization. For sparse problems it uses a
*sparse* direct solver (UMFPACK for `Float64`/`ComplexF64`, Sparspak otherwise),
so the reference is computed **without densifying** the matrix - critical for the
large sparse sizes, which only fit in memory in their sparse form.

Returns the solution object, or `nothing` if no suitable reference solver could
produce a result.
"""
function reference_solution_for(problem::BenchmarkProblem, A, b, u0, eltype::Type)
    ref_alg = if !problem.sparse
        LinearSolve.LUFactorization()
    elseif eltype in (Float64, ComplexF64)
        UMFPACKFactorization()
    else
        try
            SparspakFactorization()
        catch
            return nothing
        end
    end
    try
        return solve(LinearProblem(copy(A), copy(b); u0 = copy(u0)), ref_alg)
    catch e
        @debug "Reference solve failed for $(problem.name) ($eltype): $e"
        return nothing
    end
end

"""
    benchmark_algorithms(matrix_sizes, algorithms, alg_names, eltypes;
                        samples=5, seconds=0.5, sizes=[:small, :medium],
                        maxtime=100.0, problem=nothing,
                        alg_min_sizes=Dict{String,Int}(), solve_kwargs=(;))

Benchmark the given algorithms across different matrix sizes and element types.
Returns a DataFrame with columns `size, algorithm, eltype, matrix_type, gflops,
success, error`.

# Arguments

  - `maxtime::Float64 = 100.0`: Maximum time in seconds for each algorithm test (including accuracy check).
    If the accuracy check exceeds this time, the run is skipped and recorded as NaN.
  - `problem::Union{Nothing, BenchmarkProblem} = nothing`: the problem class to
    generate test matrices from. Defaults to the dense random-matrix class, which
    reproduces the historical behavior.
  - `alg_min_sizes::AbstractDict = Dict{String,Int}()`: optional per-algorithm
    minimum matrix size. Algorithms are silently skipped for sizes below their
    threshold (used to restrict Dagger solvers to large/big problems).
  - `solve_kwargs::NamedTuple = (;)`: keyword arguments forwarded to every `solve`
    call (e.g. tolerances/iteration caps for iterative sparse solvers).

# Metric note

`gflops` is computed from the dense-LU flop model for dense problems. For sparse
problems it is a *nominal* throughput, `2·nnz(A) / runtime`, which is only
meaningful as a relative speed ranking within a single (problem class, size,
eltype) group - never across classes or against the dense numbers.
"""
function benchmark_algorithms(
        matrix_sizes, algorithms, alg_names, eltypes;
        samples = 5, seconds = 0.5, sizes = [:tiny, :small, :medium, :large],
        check_correctness = true, correctness_tol = 1.0e0, maxtime = 100.0,
        problem::Union{Nothing, BenchmarkProblem} = nothing,
        alg_min_sizes::AbstractDict = Dict{String, Int}(),
        solve_kwargs::NamedTuple = NamedTuple()
    )

    prob_class = problem === nothing ? dense_problem() : problem
    matrix_type = prob_class.name

    # Note: We pass benchmark parameters directly to @benchmark instead of
    # modifying BenchmarkTools.DEFAULT_PARAMETERS to avoid const assignment
    # errors in Julia 1.12+

    # Initialize results DataFrame
    results_data = []

    # Track algorithms that have exceeded maxtime (per element type and size)
    # Structure: eltype => algorithm_name => max_size_tested
    blocked_algorithms = Dict{String, Dict{String, Int}}()  # eltype => Dict(algorithm_name => max_size)

    alg_min(name) = get(alg_min_sizes, name, 0)

    # Calculate total number of benchmarks for progress bar (accounting for the
    # per-algorithm minimum-size gating).
    total_benchmarks = 0
    for eltype in eltypes
        test_algs, test_names = filter_compatible_algorithms(
            algorithms, alg_names, eltype, prob_class
        )
        for n in matrix_sizes, name in test_names
            n >= alg_min(name) && (total_benchmarks += 1)
        end
    end

    # Create progress bar
    progress = Progress(
        total_benchmarks, desc = "Benchmarking ($matrix_type): ",
        barlen = 50, showspeed = true
    )

    for eltype in eltypes
        # Initialize blocked algorithms dict for this element type
        blocked_algorithms[string(eltype)] = Dict{String, Int}()

        # Filter algorithms for this element type and problem class
        compatible_algs,
            compatible_names = filter_compatible_algorithms(
            algorithms, alg_names, eltype, prob_class
        )

        if isempty(compatible_algs)
            @warn "No algorithms compatible with $eltype for $matrix_type, skipping..."
            continue
        end

        for n in matrix_sizes
            # Create test problem with specified element type. Some generators
            # round the size (e.g. the 2D Laplacian to a square grid), so read the
            # actual dimension back from the generated matrix.
            rng = MersenneTwister(123)  # Consistent seed for reproducibility
            A = prob_class.generate(rng, eltype, n)
            n_actual = size(A, 1)
            b = rand(rng, eltype, n_actual)
            u0 = rand(rng, eltype, n_actual)
            nnz_A = prob_class.sparse ? nnz(A) : n_actual * n_actual

            # Compute reference solution if correctness check is enabled. Guard the
            # reference solve itself against maxtime: for very large problems even a
            # direct reference may be infeasible, in which case we skip correctness
            # rather than aborting the benchmark.
            reference_solution = nothing
            if check_correctness
                ref_start = time()
                reference_solution = reference_solution_for(prob_class, A, b, u0, eltype)
                if reference_solution !== nothing && (time() - ref_start) > maxtime
                    @warn "Reference solve for $matrix_type size $n_actual ($eltype) exceeded maxtime; skipping correctness check for this size."
                    reference_solution = nothing
                end
            end

            for (alg, name) in zip(compatible_algs, compatible_names)
                # Skip algorithms below their minimum size (e.g. Dagger on small
                # matrices). These are not recorded at all.
                if n_actual < alg_min(name)
                    continue
                end

                # Skip this algorithm if it has exceeded maxtime for a smaller or equal size matrix
                if haskey(blocked_algorithms[string(eltype)], name)
                    max_allowed_size = blocked_algorithms[string(eltype)][name]
                    if n_actual > max_allowed_size
                        # Clear progress line and show warning on new line
                        println()  # Ensure we're on a new line
                        @warn "Algorithm $name skipped for size $n_actual (exceeded maxtime on size $max_allowed_size matrix)"
                        # Still need to update progress bar
                        ProgressMeter.next!(progress)
                        # Record as skipped due to exceeding maxtime on smaller matrix
                        push!(
                            results_data,
                            (
                                size = n_actual,
                                algorithm = name,
                                eltype = string(eltype),
                                matrix_type = matrix_type,
                                gflops = NaN,
                                success = false,
                                error = "Skipped: exceeded maxtime on size $max_allowed_size matrix",
                            )
                        )
                        continue
                    end
                end

                # Update progress description
                ProgressMeter.update!(
                    progress,
                    desc = "Benchmarking $name on $(n_actual)×$(n_actual) $eltype $matrix_type: "
                )

                gflops = NaN  # Use NaN for failed/timed out runs
                success = true
                error_msg = ""
                passed_correctness = true
                exceeded_maxtime = false

                try
                    # Create the linear problem for this test
                    prob = LinearProblem(
                        copy(A), copy(b);
                        u0 = copy(u0),
                        alias = LinearAliasSpecifier(alias_A = true, alias_b = true)
                    )

                    # Time the warmup run and correctness check
                    start_time = time()

                    # Warmup run and correctness check - no interruption, just timing
                    warmup_sol = nothing

                    # Simply run the solve and measure time
                    warmup_sol = solve(prob, alg; solve_kwargs...)
                    elapsed_time = time() - start_time

                    # Check if we exceeded maxtime
                    if elapsed_time > maxtime
                        exceeded_maxtime = true
                        # Block this algorithm for larger matrices
                        # Store the last size that was allowed to complete
                        blocked_algorithms[string(eltype)][name] = n_actual
                        @warn "Algorithm $name exceeded maxtime ($(round(elapsed_time, digits = 2))s > $(maxtime)s) for size $n_actual, eltype $eltype. Will skip for larger matrices."
                        success = false
                        error_msg = "Exceeded maxtime ($(round(elapsed_time, digits = 2))s)"
                        gflops = NaN
                    else
                        # Successful completion within time limit

                        # Check correctness if reference solution is available
                        if check_correctness && reference_solution !== nothing
                            # Compute relative error
                            rel_error = norm(warmup_sol.u - reference_solution.u) /
                                norm(reference_solution.u)

                            if rel_error > correctness_tol
                                passed_correctness = false
                                @warn "Algorithm $name failed correctness check for size $n_actual, eltype $eltype ($matrix_type). " *
                                    "Relative error: $(round(rel_error, sigdigits = 3)) > tolerance: $correctness_tol. " *
                                    "Algorithm will be excluded from results."
                                success = false
                                error_msg = "Failed correctness check (rel_error = $(round(rel_error, sigdigits = 3)))"
                                gflops = 0.0
                            end
                        end

                        # Only benchmark if correctness check passed and we didn't exceed maxtime
                        if passed_correctness && !exceeded_maxtime
                            # Check if we have enough time remaining for benchmarking
                            # Allow at least 2x the warmup time for benchmarking
                            remaining_time = maxtime - elapsed_time
                            if remaining_time < 2 * elapsed_time
                                @warn "Algorithm $name: insufficient time remaining for benchmarking (warmup took $(round(elapsed_time, digits = 2))s). Recording as NaN."
                                gflops = NaN
                                success = false
                                error_msg = "Insufficient time for benchmarking"
                            else
                                # Actual benchmark
                                # Create benchmark with custom parameters
                                bench_params = BenchmarkTools.Parameters(; seconds = seconds, samples = samples)
                                _bench = @benchmarkable solve($prob, $alg; $(solve_kwargs)...) setup = (
                                    prob = LinearProblem(
                                        copy($A), copy($b);
                                        u0 = copy($u0),
                                        alias = LinearAliasSpecifier(alias_A = true, alias_b = true)
                                    )
                                )
                                bench = BenchmarkTools.run(_bench, bench_params)

                                # Calculate GFLOPs. Dense problems use the dense-LU
                                # flop model; sparse problems use a nominal
                                # 2·nnz(A) throughput (see the metric note above).
                                min_time_sec = minimum(bench.times) / 1.0e9
                                flops = prob_class.sparse ? 2.0 * nnz_A : luflop(n_actual, n_actual)
                                gflops = flops / min_time_sec / 1.0e9
                            end
                        end
                    end

                catch e
                    success = false
                    error_msg = string(e)
                    gflops = NaN
                    # Don't warn for each failure, just record it
                end

                # Store result with element type and matrix-type information
                push!(
                    results_data,
                    (
                        size = n_actual,
                        algorithm = name,
                        eltype = string(eltype),
                        matrix_type = matrix_type,
                        gflops = gflops,
                        success = success,
                        error = error_msg,
                    )
                )

                # Update progress
                ProgressMeter.next!(progress)
            end
        end
    end

    return DataFrame(results_data)
end

"""
    get_benchmark_sizes(size_categories; sparse = false)

Get the matrix sizes to benchmark based on the requested size categories.

Dense size categories (`sparse = false`):

  - `:tiny` - 5:5:20 (for very small problems)
  - `:small` - 20:20:100 (for small problems)
  - `:medium` - 100:50:300 (for typical problems)
  - `:large` - 300:100:1000 (for larger problems)
  - `:big` - vcat(1000:2000:10000, 10000:5000:15000) (for very large/GPU problems, capped at 15000)

Sparse size categories (`sparse = true`) use much larger, perfect-square sizes
because sparse problems have `O(N)` (rather than `O(N^2)`) memory footprints. The
sizes are perfect squares so that the 2D-Laplacian generator maps them onto exact
`k×k` grids:

  - `:tiny`   - [100, 400]            (10², 20²)
  - `:small`  - [900, 2_500]          (30², 50²)
  - `:medium` - [10_000, 40_000]      (100², 200²)
  - `:large`  - [90_000, 160_000]     (300², 400²)
  - `:big`    - [250_000, 562_500]    (500², 750²)
"""
function get_benchmark_sizes(size_categories; sparse::Bool = false)
    sizes = Int[]

    if sparse
        for category in size_categories
            if category == :tiny
                append!(sizes, [100, 400])
            elseif category == :small
                append!(sizes, [900, 2_500])
            elseif category == :medium
                append!(sizes, [10_000, 40_000])
            elseif category == :large
                append!(sizes, [90_000, 160_000])
            elseif category == :big
                append!(sizes, [250_000, 562_500])
            else
                @warn "Unknown size category: $category. Skipping."
            end
        end
        return sort(unique(sizes))
    end

    for category in size_categories
        if category == :tiny
            append!(sizes, 5:5:20)
        elseif category == :small
            append!(sizes, 20:20:100)
        elseif category == :medium
            append!(sizes, 100:50:300)
        elseif category == :large
            append!(sizes, 300:100:1000)
        elseif category == :big
            append!(sizes, vcat(1000:2000:10000, 10000:5000:15000))  # Capped at 15000
        else
            @warn "Unknown size category: $category. Skipping."
        end
    end

    # Remove duplicates and sort
    return sort(unique(sizes))
end

"""
    categorize_results(df::DataFrame)

Categorize the benchmark results into size ranges and find the best algorithm for each range and element type.
For complex types, avoids RFLUFactorization if possible due to known issues.
"""
function categorize_results(df::DataFrame)
    # Algorithm preferences in LinearSolve drive the *dense* default solver
    # selection, and the dense-LU GFLOPs metric is not comparable to the nominal
    # sparse throughput. So only dense results feed the categorization/preferences.
    if hasproperty(df, :matrix_type)
        df = filter(row -> row.matrix_type == "dense", df)
    end

    # Filter successful results and exclude NaN values
    successful_df = filter(row -> row.success && !isnan(row.gflops), df)

    if nrow(successful_df) == 0
        @warn "No successful benchmark results found!"
        return Dict{String, String}()
    end

    categories = Dict{String, String}()

    # Define size ranges based on actual benchmark categories
    # These align with the sizes defined in get_benchmark_sizes()
    ranges = [
        ("tiny (5-20)", 5:20),
        ("small (20-100)", 21:100),
        ("medium (100-300)", 101:300),
        ("large (300-1000)", 301:1000),
        ("big (1000+)", 1000:typemax(Int)),
    ]

    # Get unique element types
    eltypes = unique(successful_df.eltype)

    for eltype in eltypes
        @info "Categorizing results for element type: $eltype"

        # Filter results for this element type
        eltype_df = filter(row -> row.eltype == eltype, successful_df)

        if nrow(eltype_df) == 0
            continue
        end

        for (range_name, range) in ranges
            # Get results for this size range and element type
            range_df = filter(row -> row.size in range, eltype_df)

            if nrow(range_df) == 0
                continue
            end

            # Calculate average GFLOPs for each algorithm in this range, excluding NaN values
            avg_results = combine(
                groupby(range_df, :algorithm),
                :gflops => (x -> mean(filter(!isnan, x))) => :avg_gflops
            )

            # Sort by performance
            sort!(avg_results, :avg_gflops, rev = true)

            # Find the best algorithm (for complex types, avoid RFLU if possible)
            if nrow(avg_results) > 0
                best_alg = avg_results.algorithm[1]

                # For complex types, check if best is RFLU and we have alternatives
                if (eltype == "ComplexF32" || eltype == "ComplexF64") &&
                        (
                        contains(best_alg, "RFLU") ||
                            contains(best_alg, "RecursiveFactorization")
                    )

                    # Look for the best non-RFLU algorithm
                    for i in 2:nrow(avg_results)
                        alt_alg = avg_results.algorithm[i]
                        if !contains(alt_alg, "RFLU") &&
                                !contains(alt_alg, "RecursiveFactorization")
                            # Check if performance difference is not too large (within 20%)
                            perf_ratio = avg_results.avg_gflops[i] /
                                avg_results.avg_gflops[1]
                            if perf_ratio > 0.8
                                @info "Using $alt_alg instead of $best_alg for $eltype at $range_name ($(round(100 * perf_ratio, digits = 1))% of RFLU performance) to avoid complex number issues"
                                best_alg = alt_alg
                                break
                            else
                                @warn "RFLUFactorization is best for $eltype at $range_name but has complex number issues. Alternative algorithms are >20% slower."
                            end
                        end
                    end
                end

                category_key = "$(eltype)_$(range_name)"
                categories[category_key] = best_alg
                best_idx = findfirst(==(best_alg), avg_results.algorithm)
                @info "Best algorithm for $eltype size range $range_name: $best_alg ($(round(avg_results.avg_gflops[best_idx], digits = 2)) GFLOPs avg)"
            end
        end
    end

    return categories
end

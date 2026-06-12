module LinearSolveAutotune

# Ensure MKL is available for benchmarking by setting the preference before loading LinearSolve
using Preferences
using MKL_jll
using OpenBLAS_jll

# Set MKL preference to true for benchmarking if MKL is available
# We need to use UUID instead of the module since LinearSolve isn't loaded yet
const LINEARSOLVE_UUID = Base.UUID("7ed4a6bd-45f5-4d41-b270-4a48e9bafcae")
if MKL_jll.is_available()
    # Force load MKL for benchmarking to ensure we can test MKL algorithms
    # The autotune results will determine the final preference setting
    current_pref = Preferences.load_preference(LINEARSOLVE_UUID, "LoadMKL_JLL", nothing)
    if current_pref !== true
        Preferences.set_preferences!((LINEARSOLVE_UUID, "LinearSolve"), "LoadMKL_JLL" => true; force = true)
        @info "Temporarily setting LoadMKL_JLL=true for benchmarking (was $(current_pref))"
    end
end

using LinearSolve
using BenchmarkTools
using DataFrames
using PrettyTables
using Statistics
using Random
using LinearAlgebra
using SparseArrays
using Printf
using Dates
using Base64
using ProgressMeter
using CPUSummary

# Loaded so their LinearSolve extensions (sparse direct + Dagger solvers) activate.
using Sparspak
using Dagger

# Hard dependency to ensure RFLUFactorization others solvers are available
using RecursiveFactorization
using blis_jll
using LAPACK_jll
using cuSOLVER
using Metal
using FastLapackInterface


# Optional dependencies for telemetry and plotting
using GitHub
using gh_cli_jll
using Plots

export autotune_setup, share_results, AutotuneResults, plot

include("algorithms.jl")
include("gpu_detection.jl")
include("matrix_generators.jl")
include("benchmarking.jl")
include("plotting.jl")
include("telemetry.jl")
include("preferences.jl")

"""
    flops_unit_info(units::Symbol) -> (scale, label)

Map a display-unit symbol to a multiplicative scale (applied to the stored GFLOPs
values) and a label. Supports `:gflops` and `:mflops`. `:mflops` is handy for
sparse results, whose nominal GFLOPs are often small enough to truncate to `0.00`.
"""
function flops_unit_info(units::Symbol)
    if units === :gflops
        return (1.0, "GFLOPs")
    elseif units === :mflops
        return (1.0e3, "MFLOPs")
    else
        error("Unknown units = :$units; use :gflops or :mflops")
    end
end

# Define the AutotuneResults struct
struct AutotuneResults
    results_df::DataFrame
    sysinfo::Dict
    units::Symbol
end

# Backwards-compatible constructor (defaults display units to GFLOPs)
AutotuneResults(results_df::DataFrame, sysinfo::Dict) =
    AutotuneResults(results_df, sysinfo, :gflops)

# Display method for AutotuneResults
function Base.show(io::IO, results::AutotuneResults)
    println(io, "="^60)
    println(io, "LinearSolve.jl Autotune Results")
    println(io, "="^60)

    # System info summary
    println(io, "\n📊 System Information:")
    # Use cpu_model if available, otherwise fall back to cpu_name
    cpu_display = get(results.sysinfo, "cpu_model", get(results.sysinfo, "cpu_name", "Unknown"))
    println(io, "  • CPU: ", cpu_display)
    cpu_speed = get(results.sysinfo, "cpu_speed_mhz", 0)
    if cpu_speed > 0
        println(io, "  • Speed: ", cpu_speed, " MHz")
    end
    println(io, "  • OS: ", get(results.sysinfo, "os_name", "Unknown"), " (", get(results.sysinfo, "os", "Unknown"), ")")
    println(io, "  • Julia: ", get(results.sysinfo, "julia_version", "Unknown"))
    println(io, "  • Threads: ", get(results.sysinfo, "num_threads", "Unknown"), " (BLAS: ", get(results.sysinfo, "blas_num_threads", "Unknown"), ")")

    # Results summary - include all results to show what was attempted
    unit_scale, unit_label = flops_unit_info(getfield(results, :units))
    all_results = results.results_df
    successful_results = filter(row -> row.success && !isnan(row.gflops), results.results_df)
    if nrow(successful_results) > 0
        println(io, "\n🏆 Top Performing Algorithms:")
        summary = combine(
            groupby(successful_results, :algorithm),
            :gflops => (x -> mean(filter(!isnan, x))) => :avg_gflops,
            :gflops => (x -> maximum(filter(!isnan, x))) => :max_gflops,
            nrow => :num_tests
        )
        sort!(summary, :avg_gflops, rev = true)

        # Show top 5
        for (i, row) in enumerate(eachrow(first(summary, 5)))
            println(
                io, "  ", i, ". ", row.algorithm, ": ",
                @sprintf("%.2f %s avg", row.avg_gflops * unit_scale, unit_label)
            )
        end
    end

    # Show algorithms that had failures/timeouts to make it clear what was attempted
    failed_results = filter(row -> !row.success, all_results)
    if nrow(failed_results) > 0
        failed_algs = unique(failed_results.algorithm)
        println(io, "\n⚠️  Algorithms with failures/timeouts: ", join(failed_algs, ", "))
    end

    # Element types tested
    eltypes = unique(results.results_df.eltype)
    println(io, "\n🔬 Element Types Tested: ", join(eltypes, ", "))

    # Matrix (problem) types tested
    if hasproperty(results.results_df, :matrix_type)
        mtypes = unique(results.results_df.matrix_type)
        println(io, "🧩 Matrix Types Tested: ", join(mtypes, ", "))
    end

    # Matrix sizes tested
    sizes = unique(results.results_df.size)
    println(
        io, "📏 Matrix Sizes: ", minimum(sizes), "×", minimum(sizes),
        " to ", maximum(sizes), "×", maximum(sizes)
    )

    # Report tests that exceeded maxtime if any
    exceeded_results = filter(row -> isnan(row.gflops) && contains(get(row, :error, ""), "Exceeded maxtime"), results.results_df)
    if nrow(exceeded_results) > 0
        println(io, "⏱️  Exceeded maxtime: ", nrow(exceeded_results), " tests exceeded time limit")
    end

    # Call to action - reordered
    println(io, "\n" * "="^60)
    println(io, "🚀 For comprehensive results, consider running:")
    println(io, "   results_full = autotune_setup(")
    println(io, "       sizes = [:tiny, :small, :medium, :large, :big],")
    println(io, "       eltypes = (Float32, Float64, ComplexF32, ComplexF64)")
    println(io, "   )")
    println(io, "\n📈 See community results at:")
    println(io, "   https://github.com/SciML/LinearSolve.jl/issues/725")
    println(io, "\n💡 To share your results with the community, run:")
    println(io, "   share_results(results)")
    return println(io, "="^60)
end

# Plot method for AutotuneResults
function Plots.plot(results::AutotuneResults; units::Symbol = getfield(results, :units), kwargs...)
    # Generate plots from the results data
    plots_dict = create_benchmark_plots(results.results_df; units = units)

    if plots_dict === nothing || isempty(plots_dict)
        @warn "No data available for plotting"
        return nothing
    end

    # Create a composite plot from all element type plots
    plot_list = []
    for (eltype_name, p) in plots_dict
        push!(plot_list, p)
    end

    # Create composite plot
    n_plots = length(plot_list)
    if n_plots == 1
        return plot_list[1]
    elseif n_plots == 2
        return plot(plot_list..., layout = (1, 2), size = (1200, 500); kwargs...)
    elseif n_plots <= 4
        return plot(plot_list..., layout = (2, 2), size = (1200, 900); kwargs...)
    else
        ncols = ceil(Int, sqrt(n_plots))
        nrows = ceil(Int, n_plots / ncols)
        return plot(
            plot_list..., layout = (nrows, ncols),
            size = (400 * ncols, 400 * nrows); kwargs...
        )
    end
end

"""
    autotune_setup(; 
        sizes = [:small, :medium, :large],
        set_preferences::Bool = true,
        samples::Int = 5,
        seconds::Float64 = 0.5,
        eltypes = (Float32, Float64, ComplexF32, ComplexF64),
        skip_missing_algs::Bool = false,
        include_fastlapack::Bool = false,
        maxtime::Float64 = 100.0)

Run a comprehensive benchmark of all available LU factorization methods and optionally:

  - Create performance plots for each element type
  - Set Preferences for optimal algorithm selection
  - Support both CPU and GPU algorithms based on hardware detection
  - Test algorithm compatibility with different element types
  - Automatically manage MKL loading preference based on performance results

!!! note "MKL Preference Management"
    During benchmarking, MKL is temporarily enabled (if available) to test MKL algorithms.
    After benchmarking, the LoadMKL_JLL preference is set based on whether MKL algorithms
    performed best in any category. This optimizes startup time and memory usage.

# Arguments

  - `sizes = [:small, :medium, :large]`: Size categories to test. Options: :tiny (5-20), :small (20-100), :medium (100-300), :large (300-1000), :big (1000-15000)
  - `set_preferences::Bool = true`: Update LinearSolve preferences with optimal algorithms
  - `samples::Int = 5`: Number of benchmark samples per algorithm/size
  - `seconds::Float64 = 0.5`: Maximum time per benchmark
  - `eltypes = (Float32, Float64, ComplexF32, ComplexF64)`: Element types to benchmark
  - `skip_missing_algs::Bool = false`: If false, error when expected algorithms are missing; if true, warn instead
  - `include_fastlapack::Bool = false`: If true, includes FastLUFactorization in benchmarks
  - `include_dense::Bool = true`: If true, run the dense random-matrix benchmarks.
    Set to `false` to benchmark only sparse problems. Note that LinearSolve's
    default-algorithm preferences are derived from dense results only, so they will
    not be updated when dense benchmarks are disabled.
  - `units::Symbol = :gflops`: display units for the printed summary table, the
    `show` output, and plots. One of `:gflops` or `:mflops`. Use `:mflops` for
    sparse-focused runs, whose nominal GFLOPs are often small enough to display as
    `0.00`. (The underlying `results_df` always stores GFLOPs.)
  - `isolate_solvers::Bool = false`: when `true`, every individual solver run is
    executed in its own fresh `julia` subprocess that builds its own input data.
    If a solver exhausts memory, only that subprocess is killed (and the point is
    recorded as an out-of-memory failure) instead of the whole autotune run
    crashing. This is considerably slower because each run pays full process
    startup and package-load cost, so it is off by default and intended for
    memory-risky large runs. (It deliberately uses a plain OS subprocess rather
    than `Distributed`, so Dagger does not try to schedule onto the child.)
  - `include_sparse::Bool = true`: If true, also benchmark a suite of sparse problem
    classes (2D Laplacian, unstructured SPD, unstructured nonsymmetric, and
    tridiagonal) at large sparse sizes using sparse direct and Krylov solvers.
  - `include_dagger::Bool = true`: If true, also compare Dagger.jl distributed
    solvers (tiled LU for dense; distributed Krylov for sparse) across all
    benchmark sizes. Dagger runs with its default thread-based scheduler; start
    Julia with multiple threads (`julia -t auto`) to parallelize.
  - `maxtime::Float64 = 100.0`: Maximum time in seconds for each algorithm test (including accuracy check). 
    If exceeded, the run is skipped and recorded as NaN

!!! warning "Sparse vs. dense metric"
    For sparse problems the reported `gflops` is a *nominal* `2·nnz(A) / runtime`
    throughput, not a true flop rate. It ranks solvers correctly within a single
    (problem class, size, element type) group, but must not be compared across
    problem classes or against the dense GFLOPs. Only dense results are used to set
    LinearSolve's default-algorithm preferences.

# Returns

  - `AutotuneResults`: Object containing benchmark results, system info, and plots

# Examples

```julia
using LinearSolve
using LinearSolveAutotune

# Basic autotune with default sizes
results = autotune_setup()

# Test all size ranges
results = autotune_setup(sizes = [:small, :medium, :large, :big])

# Large matrices only
results = autotune_setup(sizes = [:large, :big], samples = 10, seconds = 1.0)

# Include FastLapackInterface.jl algorithms
results = autotune_setup(include_fastlapack = true)

# After running autotune, share results (requires gh CLI or GitHub token)
share_results(results)
```
"""
function autotune_setup(;
        sizes = [:tiny, :small, :medium, :large],
        set_preferences::Bool = true,
        samples::Int = 5,
        seconds::Float64 = 0.5,
        eltypes = (Float64,),
        skip_missing_algs::Bool = false,
        include_fastlapack::Bool = false,
        include_dense::Bool = true,
        include_sparse::Bool = true,
        include_dagger::Bool = true,
        units::Symbol = :gflops,
        isolate_solvers::Bool = false,
        maxtime::Float64 = 100.0
    )
    flops_unit_info(units)  # validate units early
    if !include_dense && !include_sparse
        error("Nothing to benchmark: both include_dense and include_sparse are false.")
    end
    @info "Starting LinearSolve.jl autotune setup..."
    @info "Configuration: sizes=$sizes, set_preferences=$set_preferences, include_dense=$include_dense, include_sparse=$include_sparse, include_dagger=$include_dagger, isolate_solvers=$isolate_solvers"
    @info "Element types to benchmark: $(join(eltypes, ", "))"

    # Get system information
    system_info = get_system_info()
    @info "System detected: $(system_info["os"]) $(system_info["arch"]) with $(system_info["num_cores"]) cores"

    result_frames = DataFrame[]

    # ----- Dense benchmarks --------------------------------------------------
    if include_dense
        # Get available algorithms (only needed for the dense suite)
        cpu_algs, cpu_names = get_available_algorithms(; skip_missing_algs = skip_missing_algs, include_fastlapack = include_fastlapack)
        @info "Found $(length(cpu_algs)) CPU algorithms: $(join(cpu_names, ", "))"

        # Add GPU algorithms if available
        gpu_algs, gpu_names = get_gpu_algorithms(; skip_missing_algs = skip_missing_algs)
        if !isempty(gpu_algs)
            @info "Found $(length(gpu_algs)) GPU algorithms: $(join(gpu_names, ", "))"
        end

        # Combine all dense algorithms
        dense_algs = vcat(cpu_algs, gpu_algs)
        dense_names = vcat(cpu_names, gpu_names)

        if isempty(dense_algs)
            error("No algorithms found! This shouldn't happen.")
        end

        dense_problem_class = dense_problem()
        if include_dagger
            dagg_algs, dagg_names = get_dagger_algorithms(dense_problem_class)
            if !isempty(dagg_algs)
                dense_algs = vcat(dense_algs, dagg_algs)
                dense_names = vcat(dense_names, dagg_names)
                @info "Comparing Dagger (dense) at all sizes: $(join(dagg_names, ", "))"
            end
        end

        dense_sizes = collect(get_benchmark_sizes(sizes; sparse = false))
        @info "Benchmarking $(length(dense_sizes)) dense matrix sizes from $(minimum(dense_sizes)) to $(maximum(dense_sizes))"
        @info "Running dense benchmarks (this may take several minutes)..."
        @info "Maximum time per algorithm test: $(maxtime)s"
        push!(
            result_frames,
            benchmark_algorithms(
                dense_sizes, dense_algs, dense_names, eltypes;
                samples = samples, seconds = seconds, sizes = sizes, maxtime = maxtime,
                problem = dense_problem_class, isolate = isolate_solvers
            )
        )
    else
        @info "Skipping dense benchmarks (include_dense = false)."
    end

    # ----- Sparse benchmarks -------------------------------------------------
    if include_sparse
        sparse_sizes = collect(get_benchmark_sizes(sizes; sparse = true))
        if isempty(sparse_sizes)
            @info "No sparse sizes for requested categories; skipping sparse benchmarks."
        else
            # Generous iterative-solver tolerances/iteration caps; ignored by the
            # direct solvers. Helps Krylov methods actually converge on the harder
            # (e.g. 2D-Laplacian) systems within the correctness gate.
            sparse_solve_kwargs = (; abstol = 1.0e-10, reltol = 1.0e-8, maxiters = 50_000)
            @info "Benchmarking $(length(sparse_sizes)) sparse matrix sizes from $(minimum(sparse_sizes)) to $(maximum(sparse_sizes))"
            for prob in get_sparse_problems()
                sp_algs, sp_names = get_sparse_algorithms(prob)
                if include_dagger
                    dagg_algs, dagg_names = get_dagger_algorithms(prob)
                    if !isempty(dagg_algs)
                        sp_algs = vcat(sp_algs, dagg_algs)
                        sp_names = vcat(sp_names, dagg_names)
                    end
                end
                @info "Running sparse benchmarks for '$(prob.name)' ($(prob.description))..."
                push!(
                    result_frames,
                    benchmark_algorithms(
                        sparse_sizes, sp_algs, sp_names, eltypes;
                        samples = samples, seconds = seconds, sizes = sizes, maxtime = maxtime,
                        problem = prob,
                        solve_kwargs = sparse_solve_kwargs, isolate = isolate_solvers
                    )
                )
            end
        end
    end

    results_df = vcat(result_frames...; cols = :union)

    # Display results table - show all results including NaN values to indicate what was tested
    all_results = results_df
    successful_results = filter(row -> row.success && !isnan(row.gflops), results_df)
    exceeded_maxtime_results = filter(row -> isnan(row.gflops) && contains(get(row, :error, ""), "Exceeded maxtime"), results_df)
    skipped_results = filter(row -> contains(get(row, :error, ""), "Skipped"), results_df)

    if nrow(exceeded_maxtime_results) > 0
        @info "$(nrow(exceeded_maxtime_results)) tests exceeded maxtime limit ($(maxtime)s)"
    end

    if nrow(skipped_results) > 0
        # Count unique algorithms that were skipped
        skipped_algs = unique([row.algorithm for row in eachrow(skipped_results)])
        @info "$(length(skipped_algs)) algorithms skipped for larger matrices after exceeding maxtime"
    end

    if nrow(successful_results) > 0
        @info "Benchmark completed successfully!"

        unit_scale, unit_label = flops_unit_info(units)

        # Create summary table for display - include algorithms with NaN values to show what was tested.
        # Group by matrix type as well, since the dense and sparse GFLOPs metrics
        # are not directly comparable.
        group_cols = hasproperty(all_results, :matrix_type) ?
            [:matrix_type, :algorithm] : [:algorithm]
        full_summary = combine(
            groupby(all_results, group_cols),
            :gflops => (
                x -> begin
                    valid_vals = filter(!isnan, x)
                    length(valid_vals) > 0 ? mean(valid_vals) : NaN
                end
            ) => :avg_gflops,
            :gflops => (
                x -> begin
                    valid_vals = filter(!isnan, x)
                    length(valid_vals) > 0 ? maximum(valid_vals) : NaN
                end
            ) => :max_gflops,
            :success => (x -> count(x)) => :successful_tests,
            nrow => :total_tests
        )

        # Sort by matrix type, then by average GFLOPs descending (NaN values last).
        # Use a temporary key mapping NaN -> -Inf so failed algorithms sort to the end.
        full_summary.sortkey = map(x -> isnan(x) ? -Inf : x, full_summary.avg_gflops)
        if hasproperty(full_summary, :matrix_type)
            sort!(full_summary, [order(:matrix_type), order(:sortkey, rev = true)])
            header = ["Matrix Type", "Algorithm", "Avg $unit_label", "Max $unit_label", "Success", "Total"]
            numcols = [3, 4]
        else
            sort!(full_summary, order(:sortkey, rev = true))
            header = ["Algorithm", "Avg $unit_label", "Max $unit_label", "Success", "Total"]
            numcols = [2, 3]
        end
        select!(full_summary, Not(:sortkey))

        println("\n" * "="^60)
        println("BENCHMARK RESULTS SUMMARY (including failed attempts)")
        println("Note: for sparse matrix types, $unit_label is a nominal 2·nnz/runtime")
        println("throughput - compare solvers only within the same matrix type.")
        println("="^60)
        pretty_table(
            full_summary,
            header = header,
            formatters = (v, i, j) -> begin
                if j in numcols && isa(v, Float64)
                    return isnan(v) ? "NaN" : @sprintf("%.2f", v * unit_scale)
                else
                    return v
                end
            end,
            crop = :none
        )
    else
        @warn "No successful benchmark results!"
        # Still show what was attempted
        if nrow(all_results) > 0
            failed_algs = unique(all_results.algorithm)
            @info "Algorithms tested (all failed): $(join(failed_algs, ", "))"
        end
        return results_df, nothing
    end

    # Categorize results and find best algorithms per size range
    categories = categorize_results(results_df)

    # Set preferences if requested
    if set_preferences && !isempty(categories)
        set_algorithm_preferences(categories, results_df)
    end

    @info "Autotune setup completed!"

    sysinfo_df = get_detailed_system_info()
    # Convert DataFrame to Dict for AutotuneResults
    sysinfo = Dict{String, Any}()
    if nrow(sysinfo_df) > 0
        for col in names(sysinfo_df)
            sysinfo[col] = sysinfo_df[1, col]
        end
    end

    # Return AutotuneResults object
    return AutotuneResults(results_df, sysinfo, units)
end

"""
    share_results(results::AutotuneResults; auto_login::Bool = true)

Share your benchmark results with the LinearSolve.jl community to help improve 
automatic algorithm selection across different hardware configurations.

This function will authenticate with GitHub (using gh CLI or token) and post
your results as a comment to the community benchmark collection issue.

If authentication is not found and `auto_login` is true, the function will
offer to run `gh auth login` interactively to set up authentication.

# Arguments
- `results`: AutotuneResults object from autotune_setup
- `auto_login`: If true, prompts to authenticate if not already authenticated (default: true)

# Authentication Methods

## Automatic (New!)
If gh is not authenticated, the function will offer to run authentication for you.

## Method 1: GitHub CLI (Recommended)
1. Install GitHub CLI: https://cli.github.com/
2. Run: `gh auth login` 
3. Follow the prompts to authenticate
4. Run this function - it will automatically use your gh session

## Method 2: GitHub Token
1. Go to: https://github.com/settings/tokens/new
2. Add description: "LinearSolve.jl Telemetry"
3. Select scope: "public_repo" (for commenting on issues)
4. Click "Generate token" and copy it
5. Set environment variable: `ENV["GITHUB_TOKEN"] = "your_token_here"`
6. Run this function

# Examples
```julia
# Run benchmarks
results = autotune_setup()

# Share results with automatic authentication prompt
share_results(results)

# Share results without authentication prompt
share_results(results; auto_login = false)
```
"""
function share_results(results::AutotuneResults; auto_login::Bool = true)
    @info "📤 Preparing to share benchmark results with the community..."

    # Extract from AutotuneResults
    results_df = results.results_df
    sysinfo = results.sysinfo

    # Get system info
    system_info = sysinfo

    # Categorize results
    categories = categorize_results(results_df)

    # Set up authentication (with auto-login prompt if enabled)
    @info "🔗 Checking GitHub authentication..."

    github_auth = setup_github_authentication(; auto_login = auto_login)

    if github_auth === nothing || github_auth[1] === nothing
        # Save results locally as fallback
        timestamp = replace(string(Dates.now()), ":" => "-")
        fallback_file = "autotune_results_$(timestamp).md"
        markdown_content = format_results_for_github(results_df, system_info, categories)
        open(fallback_file, "w") do f
            write(f, markdown_content)
        end
        @info "📁 Results saved locally to $fallback_file"
        @info "    You can manually share this file on the issue tracker:"
        @info "    https://github.com/SciML/LinearSolve.jl/issues/725"
        return
    end

    # Format results
    markdown_content = format_results_for_github(results_df, system_info, categories)

    # Upload to GitHub (without plots)
    upload_to_github(markdown_content, nothing, github_auth, results_df, system_info, categories)

    return @info "✅ Thank you for contributing to the LinearSolve.jl community!"
end

end

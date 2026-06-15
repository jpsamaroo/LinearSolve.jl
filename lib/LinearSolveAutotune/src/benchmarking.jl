# Core benchmarking functionality

using ProgressMeter
using LinearAlgebra
using Serialization

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
    run_single_benchmark(prob_class, A, b, u0, nnz_A, n_actual, alg, name, eltype; kwargs...)

Run the warmup, correctness check, and timing loop for a *single* `(algorithm,
size, eltype)` point and return a `NamedTuple` `(; gflops, success, error,
exceeded_maxtime)`. This is the shared core used by both the in-process benchmark
loop and the isolated-subprocess path, so the two stay behaviorally identical.

If `compute_reference = true` (used by the subprocess path) and no
`reference_solution` is supplied, a trusted reference is computed here so the
correctness gate still applies inside the isolated process.
"""
function run_single_benchmark(
        prob_class::BenchmarkProblem, A, b, u0, nnz_A, n_actual, alg, name, eltype::Type;
        samples, seconds, maxtime, check_correctness, correctness_tol,
        solve_kwargs::NamedTuple = NamedTuple(),
        reference_solution = nothing, compute_reference::Bool = false,
    )
    gflops = NaN
    success = true
    error_msg = ""
    passed_correctness = true
    exceeded_maxtime = false

    if compute_reference && check_correctness && reference_solution === nothing
        ref_start = time()
        reference_solution = reference_solution_for(prob_class, A, b, u0, eltype)
        if reference_solution !== nothing && (time() - ref_start) > maxtime
            @warn "Reference solve for $(prob_class.name) size $n_actual ($eltype) exceeded maxtime; skipping correctness check for this run."
            reference_solution = nothing
        end
    end

    try
        # Create the linear problem for this test
        prob = LinearProblem(
            copy(A), copy(b);
            u0 = copy(u0),
            alias = LinearAliasSpecifier(alias_A = true, alias_b = true)
        )

        # Time the warmup run and correctness check
        start_time = time()
        warmup_sol = solve(prob, alg; solve_kwargs...)
        elapsed_time = time() - start_time

        if elapsed_time > maxtime
            exceeded_maxtime = true
            @warn "Algorithm $name exceeded maxtime ($(round(elapsed_time, digits = 2))s > $(maxtime)s) for size $n_actual, eltype $eltype. Will skip for larger matrices."
            success = false
            error_msg = "Exceeded maxtime ($(round(elapsed_time, digits = 2))s)"
            gflops = NaN
        else
            # Check correctness if reference solution is available
            if check_correctness && reference_solution !== nothing
                rel_error = norm(warmup_sol.u - reference_solution.u) /
                    norm(reference_solution.u)

                if rel_error > correctness_tol
                    passed_correctness = false
                    @warn "Algorithm $name failed correctness check for size $n_actual, eltype $eltype ($(prob_class.name)). " *
                        "Relative error: $(round(rel_error, sigdigits = 3)) > tolerance: $correctness_tol. " *
                        "Algorithm will be excluded from results."
                    success = false
                    error_msg = "Failed correctness check (rel_error = $(round(rel_error, sigdigits = 3)))"
                    gflops = 0.0
                end
            end

            # Only benchmark if correctness check passed and we didn't exceed maxtime
            if passed_correctness && !exceeded_maxtime
                remaining_time = maxtime - elapsed_time
                if remaining_time < 2 * elapsed_time
                    @warn "Algorithm $name: insufficient time remaining for benchmarking (warmup took $(round(elapsed_time, digits = 2))s). Recording as NaN."
                    gflops = NaN
                    success = false
                    error_msg = "Insufficient time for benchmarking"
                else
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
    end

    return (; gflops, success, error = error_msg, exceeded_maxtime)
end

# ---------------------------------------------------------------------------
# Persistent isolated worker
#
# Isolated benchmarking runs each solver in a long-lived child `julia` process
# rather than spawning a fresh process per point. This amortizes the (large)
# process-startup + package-load cost across many benchmark points, while still
# giving us OOM isolation: if a solver exhausts memory and the OS OOM killer
# SIGKILLs the worker, only that worker dies. We detect the death, report the
# offending point as an out-of-memory failure, and transparently respawn a new
# worker for the remaining points.
#
# A plain OS subprocess is used (Pipes + `Base.julia_cmd`), **not** `Distributed`:
# a Distributed worker would make Dagger treat the child as an extra worker and
# schedule computation onto it, which we explicitly do not want here.
# ---------------------------------------------------------------------------

"""
    run_job(job) -> NamedTuple

Run one benchmark point described by `job` (a `NamedTuple`). The input data is
**regenerated deterministically** here (same RNG seed/order as the in-process
path), so large matrices are never sent over the pipe. Returns
`(; n_actual, gflops, success, error, exceeded_maxtime)`.
"""
function run_job(job)
    prob_class = job.problem
    eltype = job.eltype
    n = job.n

    rng = MersenneTwister(123)
    A = prob_class.generate(rng, eltype, n)
    n_actual = size(A, 1)
    b = rand(rng, eltype, n_actual)
    u0 = rand(rng, eltype, n_actual)
    nnz_A = prob_class.sparse ? nnz(A) : n_actual * n_actual

    res = run_single_benchmark(
        prob_class, A, b, u0, nnz_A, n_actual, job.alg, job.name, eltype;
        samples = job.samples, seconds = job.seconds, maxtime = job.maxtime,
        check_correctness = job.check_correctness, correctness_tol = job.correctness_tol,
        solve_kwargs = job.solve_kwargs, compute_reference = true,
    )
    return merge((; n_actual = n_actual), res)
end

"""
    worker_main()

Entry point for a persistent isolated worker process. Reads serialized jobs from
`stdin`, runs each via [`run_job`](@ref), and writes the serialized result back on
the *original* `stdout` (the protocol channel). All ordinary output produced while
solving is redirected to `stderr` so it can never corrupt the result stream. The
loop exits when `stdin` is closed or a `:shutdown` sentinel is received.
"""
function worker_main()
    # Capture the real stdout as the private result channel, then send any stray
    # solver/log output to stderr so it cannot corrupt the serialized stream.
    proto = stdout
    redirect_stdout(stderr)

    while true
        job = try
            Serialization.deserialize(stdin)
        catch
            break  # stdin closed (parent gone) → exit
        end
        job === :shutdown && break

        result = try
            run_job(job)
        catch e
            (; n_actual = get(job, :n, -1), gflops = NaN, success = false,
                error = "Worker error: " * sprint(showerror, e), exceeded_maxtime = false)
        end

        Serialization.serialize(proto, result)
        flush(proto)
    end
    return nothing
end

"""
    BenchmarkWorker

Handle for a running persistent worker: the child process plus the pipe we write
jobs to (`in`) and read results from (`out`).
"""
mutable struct BenchmarkWorker
    proc::Base.Process
    in::Base.Pipe
    out::Base.Pipe
end

"""
    start_worker(project, nthreads) -> BenchmarkWorker

Launch a fresh persistent worker process. The worker loads `LinearSolveAutotune`
from `project` and runs with `nthreads` threads (so Dagger stays multithreaded
inside it). The worker's `stderr` is shared with the parent so its warnings remain
visible.
"""
function start_worker(project::AbstractString, nthreads::Int)
    code = "using LinearSolveAutotune; LinearSolveAutotune.worker_main()"
    cmd = `$(Base.julia_cmd()) --project=$project --threads=$nthreads --startup-file=no -e $code`
    inp = Base.Pipe()
    outp = Base.Pipe()
    proc = run(pipeline(cmd; stdin = inp, stdout = outp, stderr = stderr); wait = false)
    # Parent only writes `inp` and reads `outp`; close the ends it doesn't use.
    close(inp.out)
    close(outp.in)
    return BenchmarkWorker(proc, inp, outp)
end

"""
    stop_worker!(w; grace = 5.0)

Politely shut a worker down (sentinel + close stdin), then hard-kill it if it does
not exit within `grace` seconds.
"""
function stop_worker!(w::BenchmarkWorker; grace::Float64 = 5.0)
    try
        if process_running(w.proc)
            try
                Serialization.serialize(w.in, :shutdown)
                flush(w.in)
            catch
            end
        end
    catch
    end
    try; close(w.in); catch; end

    if process_running(w.proc)
        t = @async (try; wait(w.proc); catch; end)
        deadline = time() + grace
        while !istaskdone(t) && time() < deadline
            sleep(0.05)
        end
        # SIGKILL (9): a worker hung in native code may ignore SIGTERM.
        process_running(w.proc) && (try; kill(w.proc, 9); catch; end)
    end
    try; close(w.out); catch; end
    return nothing
end

# Hard-kill a worker immediately (used on watchdog timeout). SIGKILL so it dies
# even if it is stuck in non-Julia code.
function kill_worker!(w::BenchmarkWorker)
    try; process_running(w.proc) && kill(w.proc, 9); catch; end
    try; close(w.in); catch; end
    try; close(w.out); catch; end
    return nothing
end

"""
    submit_job!(w, job; timeout) -> result | :died | :timeout

Send `job` to worker `w` and wait (up to `timeout` seconds) for its serialized
result. Returns the result `NamedTuple` on success, `:died` if the worker process
exited/closed the pipe before answering (the hallmark of an OOM SIGKILL), or
`:timeout` if it produced nothing within the watchdog window.
"""
function submit_job!(w::BenchmarkWorker, job; timeout::Float64)
    try
        Serialization.serialize(w.in, job)
        flush(w.in)
    catch
        return :died  # pipe already broken → worker gone
    end

    reader = @async begin
        try
            Serialization.deserialize(w.out)
        catch e
            e  # EOFError etc. → signal death to the caller
        end
    end

    deadline = time() + timeout
    while !istaskdone(reader) && time() < deadline
        sleep(0.05)
    end

    istaskdone(reader) || return :timeout

    res = fetch(reader)
    return res isa Exception ? :died : res
end

"""
    run_point_isolated!(worker, job, project, nthreads, timeout) -> (result, worker)

Run a single benchmark point on the persistent `worker`, lazily starting one if
needed and respawning it if it dies (OOM) or hangs (watchdog `timeout`). Returns
the result `NamedTuple` and the (possibly new) worker to use next.
"""
function run_point_isolated!(
        worker::Union{Nothing, BenchmarkWorker}, job, project::AbstractString,
        nthreads::Int, timeout::Float64
    )
    if worker === nothing || !process_running(worker.proc)
        worker = start_worker(project, nthreads)
    end

    outcome = submit_job!(worker, job; timeout = timeout)

    if outcome === :timeout
        kill_worker!(worker)
        worker = start_worker(project, nthreads)
        result = (; n_actual = job.n, gflops = NaN, success = false,
            error = "Exceeded isolation watchdog timeout ($(round(timeout, digits = 1))s); " *
                "worker killed and restarted (possible hang or out-of-memory thrash)",
            exceeded_maxtime = true)
        return result, worker
    elseif outcome === :died
        sig = 0
        ec = 0
        try
            wait(worker.proc)
            sig = worker.proc.termsignal
            ec = worker.proc.exitcode
        catch
        end
        worker = start_worker(project, nthreads)
        oom = sig != 0 || ec == 137
        msg = oom ?
            "Worker process killed (signal $sig, exit $ec) — likely out-of-memory; restarted" :
            "Worker process exited unexpectedly (signal $sig, exit $ec); restarted"
        result = (; n_actual = job.n, gflops = NaN, success = false,
            error = msg, exceeded_maxtime = false)
        return result, worker
    else
        return outcome, worker
    end
end

"""
    benchmark_algorithms(matrix_sizes, algorithms, alg_names, eltypes;
                        samples=5, seconds=0.5, sizes=[:small, :medium],
                        maxtime=100.0, problem=nothing,
                        alg_min_sizes=Dict{String,Int}(), solve_kwargs=(;),
                        isolate=false)

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
  - `isolate::Bool = false`: when `true`, solver runs are executed on a *persistent
    isolated worker* — a long-lived child `julia` process that builds its own input
    data and is reused across all points (so the startup + package-load cost is paid
    once, not per point). If a solver exhausts memory and the OS kills the worker,
    only that worker dies: the offending point is recorded as an out-of-memory
    failure and a fresh worker is spawned for the remaining points. A watchdog also
    restarts the worker if a point hangs. Off by default; intended for memory-risky
    large runs. Uses a plain OS subprocess (not `Distributed`) so Dagger does not
    schedule onto the worker.

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
        solve_kwargs::NamedTuple = NamedTuple(),
        isolate::Bool = false
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

    # Persistent isolated worker setup. The worker is started lazily on the first
    # point and reused across the whole sweep (respawned by `run_point_isolated!`
    # if it dies or hangs), so we pay startup cost once rather than per point.
    iso_project = isolate ? dirname(Base.active_project()) : ""
    iso_nthreads = max(1, Threads.nthreads())
    # Watchdog: generous upper bound on one point's wall time (reference solve +
    # warmup are each bounded by maxtime; the timing loop by ~samples·seconds).
    iso_timeout = 2.0 * maxtime + samples * seconds + 120.0
    worker = nothing

    try
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
            # In isolated mode the input data (and reference solution) are built
            # inside each child process, so the parent never allocates the large
            # matrices and can't itself be OOM-killed. Outside isolated mode we
            # build the data once per size and share it across algorithms.
            #
            # Some generators round the size (e.g. the 2D Laplacian to a square
            # grid), so read the actual dimension back from the generated matrix.
            local A, b, u0, nnz_A, reference_solution
            if isolate
                A = b = u0 = nothing
                nnz_A = 0
                n_actual = n
                reference_solution = nothing
            else
                rng = MersenneTwister(123)  # Consistent seed for reproducibility
                A = prob_class.generate(rng, eltype, n)
                n_actual = size(A, 1)
                b = rand(rng, eltype, n_actual)
                u0 = rand(rng, eltype, n_actual)
                nnz_A = prob_class.sparse ? nnz(A) : n_actual * n_actual

                # Compute reference solution if correctness check is enabled. Guard
                # the reference solve itself against maxtime: for very large
                # problems even a direct reference may be infeasible, in which case
                # we skip correctness rather than aborting the benchmark.
                reference_solution = nothing
                if check_correctness
                    ref_start = time()
                    reference_solution = reference_solution_for(prob_class, A, b, u0, eltype)
                    if reference_solution !== nothing && (time() - ref_start) > maxtime
                        @warn "Reference solve for $matrix_type size $n_actual ($eltype) exceeded maxtime; skipping correctness check for this size."
                        reference_solution = nothing
                    end
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

                # Run the benchmark either in-process or on the persistent isolated
                # worker (so an OOM in one solver doesn't take down the run).
                if isolate
                    job = (; problem = prob_class, n = n, eltype = eltype, alg = alg, name = name,
                        samples = samples, seconds = seconds, maxtime = maxtime,
                        check_correctness = check_correctness, correctness_tol = correctness_tol,
                        solve_kwargs = solve_kwargs)
                    res, worker = run_point_isolated!(
                        worker, job, iso_project, iso_nthreads, iso_timeout
                    )
                    row_size = res.n_actual
                else
                    res = run_single_benchmark(
                        prob_class, A, b, u0, nnz_A, n_actual, alg, name, eltype;
                        samples = samples, seconds = seconds, maxtime = maxtime,
                        check_correctness = check_correctness, correctness_tol = correctness_tol,
                        solve_kwargs = solve_kwargs, reference_solution = reference_solution,
                    )
                    row_size = n_actual
                end

                # Block this algorithm for larger matrices if it timed out.
                if res.exceeded_maxtime
                    blocked_algorithms[string(eltype)][name] = row_size
                end

                # Store result with element type and matrix-type information
                push!(
                    results_data,
                    (
                        size = row_size,
                        algorithm = name,
                        eltype = string(eltype),
                        matrix_type = matrix_type,
                        gflops = res.gflops,
                        success = res.success,
                        error = res.error,
                    )
                )

                # Update progress
                ProgressMeter.next!(progress)
            end
        end
    end
    finally
        worker isa BenchmarkWorker && stop_worker!(worker)
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
  - `:verybig` - [20_000, 35_000, 50_000] (beyond `:big`)
  - `:superbig` - [70_000, 100_000] (largest dense tier)

Sparse size categories (`sparse = true`) use larger, perfect-square sizes because
sparse problems have `O(N)` (rather than `O(N^2)`) storage. The sizes are perfect
squares so that the 2D-Laplacian generator maps them onto exact `k×k` grids.

These are kept deliberately conservative: sparse *direct* factorizations of the
unstructured (non-PDE) test matrices suffer heavy fill-in, so the peak memory of a
factorization can be far larger than the matrix itself. The ladder below avoids
out-of-memory kills on typical workstations; raise it if you have the headroom.

  - `:tiny`     - [100, 400]            (10², 20²)
  - `:small`    - [900, 2_500]          (30², 50²)
  - `:medium`   - [4_900, 10_000]       (70², 100²)
  - `:large`    - [22_500, 40_000]      (150², 200²)
  - `:big`      - [90_000, 160_000]       (300², 400²)
  - `:verybig`  - [360_000, 810_000]      (600², 900²)
  - `:superbig` - [1_440_000, 1_960_000]  (1200², 1400², ≈2M)
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
                append!(sizes, [4_900, 10_000])
            elseif category == :large
                append!(sizes, [22_500, 40_000])
            elseif category == :big
                append!(sizes, [90_000, 160_000])
            elseif category == :verybig
                append!(sizes, [360_000, 810_000])
            elseif category == :superbig
                append!(sizes, [1_440_000, 1_960_000])
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
        elseif category == :verybig
            append!(sizes, [20_000, 35_000, 50_000])
        elseif category == :superbig
            append!(sizes, [70_000, 100_000])
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

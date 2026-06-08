using LinearSolve
using LinearAlgebra
using SparseArrays
using Dagger
using Krylov
using IncompleteLU
using AlgebraicMultigrid
using Test
import SciMLBase

const DaggerExt = Base.get_extension(LinearSolve, :LinearSolveDaggerExt)

# Deterministic, well-conditioned test problems.
function spd_system(n)
    M = rand(n, n)
    A = M * M' + n * I        # symmetric positive definite
    b = rand(n)
    return Matrix(A), b
end

function general_system(n)
    A = rand(n, n) + n * I    # diagonally dominant, nonsymmetric
    b = rand(n)
    return A, b
end

@testset "extension loaded" begin
    @test DaggerExt !== nothing
end

@testset "DaggerLUFactorization (dense, general)" begin
    n = 64
    A, b = general_system(n)
    Aref = copy(A)
    xref = A \ b

    sol = solve(LinearProblem(A, b), DaggerLUFactorization())
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test sol.u ≈ xref rtol = 1e-8
    # Out-of-place factorization must not mutate the aliased input.
    @test A == Aref

    # Explicit (small) block size path.
    sol2 = solve(LinearProblem(A, b), DaggerLUFactorization(blocksize = 16))
    @test sol2.u ≈ xref rtol = 1e-8

    # No-pivot path on a diagonally dominant matrix.
    sol3 = solve(LinearProblem(A, b), DaggerLUFactorization(pivot = false))
    @test sol3.u ≈ xref rtol = 1e-6
end

@testset "DaggerCholeskyFactorization (dense, SPD)" begin
    n = 48
    A, b = spd_system(n)
    xref = A \ b
    sol = solve(LinearProblem(A, b), DaggerCholeskyFactorization(blocksize = 16))
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test sol.u ≈ xref rtol = 1e-8
end

@testset "DaggerQRFactorization (square + least squares)" begin
    n = 40
    A, b = general_system(n)
    sol = solve(LinearProblem(A, b), DaggerQRFactorization(blocksize = 16))
    @test sol.u ≈ A \ b rtol = 1e-7

    # Overdetermined least-squares.
    m, k = 60, 30
    Aover = rand(m, k)
    bover = rand(m)
    solls = solve(LinearProblem(Aover, bover), DaggerQRFactorization(blocksize = 15))
    @test solls.u ≈ Aover \ bover rtol = 1e-6
end

@testset "DaggerKrylovJL (dense)" begin
    n = 64
    A, b = spd_system(n)
    xref = A \ b

    solcg = solve(LinearProblem(A, b), DaggerKrylovJL_CG(blocksize = 16);
        reltol = 1e-10, abstol = 1e-12)
    @test solcg.retcode == SciMLBase.ReturnCode.Success
    @test solcg.u ≈ xref rtol = 1e-6

    # GMRES on a general system with a Jacobi preconditioner.
    Ag, bg = general_system(n)
    solg = solve(LinearProblem(Ag, bg),
        DaggerKrylovJL_GMRES(blocksize = 16, precond = :jacobi);
        reltol = 1e-10, abstol = 1e-12)
    @test solg.u ≈ Ag \ bg rtol = 1e-6

    # Block-Jacobi preconditioner.
    solbj = solve(LinearProblem(A, b),
        DaggerKrylovJL_CG(blocksize = 16, precond = :blockjacobi);
        reltol = 1e-10, abstol = 1e-12)
    @test solbj.u ≈ xref rtol = 1e-6
end

@testset "DaggerKrylovJL (sparse)" begin
    n = 80
    # SPD sparse tridiagonal-ish matrix.
    A = sparse(SymTridiagonal(fill(4.0, n), fill(-1.0, n - 1)))
    b = rand(n)
    xref = Matrix(A) \ b
    sol = solve(LinearProblem(A, b), DaggerKrylovJL_CG(blocksize = 20);
        reltol = 1e-10, abstol = 1e-12)
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test sol.u ≈ xref rtol = 1e-6
end

@testset "DArray inputs + defaultalg" begin
    n = 48
    A, b = general_system(n)
    xref = A \ b

    DA = distribute(A, Blocks(16, 16))
    Db = distribute(b, Blocks(16))

    # defaultalg should route a dense square DArray to LU.
    alg = LinearSolve.defaultalg(DA, Db, LinearSolve.OperatorAssumptions(true))
    @test alg isa DaggerLUFactorization

    sol = solve(LinearProblem(DA, Db))
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test collect(sol.u) ≈ xref rtol = 1e-7

    # Non-square DArray routes to QR.
    algqr = LinearSolve.defaultalg(distribute(rand(20, 10), Blocks(10, 10)), Db,
        LinearSolve.OperatorAssumptions(false))
    @test algqr isa DaggerQRFactorization
end

@testset "preconditioners (ILU + AMG) on sparse SPD" begin
    n = 120
    A = sparse(SymTridiagonal(fill(4.0, n), fill(-1.0, n - 1)))
    b = rand(n)
    xref = Matrix(A) \ b

    for pc in (:jacobi, :blockjacobi, :ilu, :amg, :ruge_stuben, :smoothed_aggregation)
        sol = solve(LinearProblem(A, b),
            DaggerKrylovJL_CG(blocksize = 30, precond = pc);
            reltol = 1e-10, abstol = 1e-12)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test sol.u ≈ xref rtol = 1e-6
    end

    # A callable preconditioner with custom options.
    sol = solve(LinearProblem(A, b),
        DaggerKrylovJL_CG(blocksize = 30,
            precond = DA -> Dagger.BlockILUPreconditioner(DA; τ = 1e-4));
        reltol = 1e-10, abstol = 1e-12)
    @test sol.u ≈ xref rtol = 1e-6
end

@testset "array-aware auto method + precond selection" begin
    n = 100
    DaggerExt = Base.get_extension(LinearSolve, :LinearSolveDaggerExt)

    # Symmetric (SPD) sparse system -> auto picks CG and an ILU preconditioner.
    As = sparse(SymTridiagonal(fill(4.0, n), fill(-1.0, n - 1)))
    bs = rand(n)
    cache = SciMLBase.init(LinearProblem(As, bs), DaggerKrylovJL(blocksize = 25);
        reltol = 1e-10, abstol = 1e-12)
    sol = solve!(cache)
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test sol.u ≈ Matrix(As) \ bs rtol = 1e-6
    # Resolved method is cached as the 3rd element of cacheval.state.
    @test cache.cacheval.state[3] == :cg
    @test cache.cacheval.state[2] isa Dagger.BlockILUPreconditioner

    # Nonsymmetric system -> auto picks GMRES.
    An = sparse(rand(n, n) + n * I)
    bn = rand(n)
    cachen = SciMLBase.init(LinearProblem(An, bn), DaggerKrylovJL(blocksize = 25);
        reltol = 1e-10, abstol = 1e-12)
    soln = solve!(cachen)
    @test cachen.cacheval.state[3] == :gmres
    @test soln.u ≈ Matrix(An) \ bn rtol = 1e-5

    # Dense symmetric -> CG with Jacobi (no sparse backend, so ILU/AMG skipped).
    Ad = rand(n, n); Ad = Ad * Ad' + n * I
    bd = rand(n)
    cached = SciMLBase.init(LinearProblem(Matrix(Ad), bd), DaggerKrylovJL(blocksize = 25);
        reltol = 1e-10, abstol = 1e-12)
    sold = solve!(cached)
    @test cached.cacheval.state[3] == :cg
    @test cached.cacheval.state[2] isa Dagger.JacobiPreconditioner
    @test sold.u ≈ Matrix(Ad) \ bd rtol = 1e-6
end

@testset "cache reuse across right-hand sides" begin
    n = 40
    A, b = general_system(n)
    cache = SciMLBase.init(LinearProblem(A, b), DaggerLUFactorization(blocksize = 16))
    sol1 = solve!(cache)
    @test sol1.u ≈ A \ b rtol = 1e-8

    b2 = rand(n)
    cache.b = b2
    sol2 = solve!(cache)
    @test sol2.u ≈ A \ b2 rtol = 1e-8
end

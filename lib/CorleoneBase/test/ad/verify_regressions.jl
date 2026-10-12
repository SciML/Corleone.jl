# A scratch-only environment: never resolve or write package metadata in the checkout.
using Pkg
if isempty(ARGS)
    environment = mktempdir("/tmp/opencode"; prefix = "ad-regressions-$(VERSION)-", cleanup = false)
    cp(joinpath(@__DIR__, "Project.toml"), joinpath(environment, "Project.toml"))
    Pkg.activate(environment)
    Pkg.develop(path = abspath(joinpath(@__DIR__, "../..")))
    # Core tests import ArrayInterface directly; the AD declarations need not change.
    Pkg.add("ArrayInterface")
    Pkg.instantiate()
    Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)
    println("Julia=", VERSION, " threads=", Threads.nthreads(), " environment=", environment)
    command = `$(Base.julia_cmd()) --startup-file=no --threads=$(Threads.nthreads()) --project=$environment $(abspath(@__FILE__)) resolved`
    run(command)
else

    using SafeTestsets
    # Expand these macros only after loading SafeTestsets in the resolved child.
    @eval begin
        @safetestset "Sequential Core regression" begin
            include("../core/sequential.jl")
        end
        @safetestset "Parallel Core regression" begin
            include("../core/parallel.jl")
        end
        Base.include(Main, abspath(joinpath(@__DIR__, "../../ext/CorleoneBaseMooncakeExtension.jl")))
        @safetestset "Sequential AD regression" begin
            include("runtests.jl")
        end
    end
end

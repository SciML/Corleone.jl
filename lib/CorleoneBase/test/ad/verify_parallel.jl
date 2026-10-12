# Run from the repository root, with --threads=4. The scratch environment uses
# the AD group's declarations without touching any checked-in package metadata.
using Pkg
environment = mktempdir("/tmp/opencode"; prefix = "parallel-ad-$(VERSION)-", cleanup = false)
cp(joinpath(@__DIR__, "Project.toml"), joinpath(environment, "Project.toml"))
Pkg.activate(environment)
Pkg.develop(path = abspath(joinpath(@__DIR__, "../..")))
Pkg.instantiate()
Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)
println("Julia=", VERSION, " threads=", Threads.nthreads(), " environment=", environment)
# Resolution may load dependencies at versions different from the final manifest.
# Start a clean process so driver and workers use exactly the recorded environment.
command = `$(Base.julia_cmd()) --startup-file=no --threads=$(Threads.nthreads()) --project=$environment $(joinpath(@__DIR__, "parallel.jl"))`
run(command)

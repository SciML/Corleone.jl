using Pkg
using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

# Detect sublibrary test groups.
# GROUP can be a bare sublibrary name (Core test group) or
# "{sublibrary}_{TEST_GROUP}" for any custom group (e.g., QA, GPU, etc.).
# Sublibraries declare their groups in test/test_groups.toml.
const _LIB_DIR = joinpath(dirname(@__DIR__), "lib")

# Check if GROUP matches a sublibrary, possibly with a _SUFFIX for the test group.
# Scan underscores right-to-left to find the longest matching sublibrary prefix.
function _detect_sublibrary_group(group, lib_dir)
    isdir(joinpath(lib_dir, group)) && return (group, "Core")
    for i in length(group):-1:1
        if group[i] == '_' && isdir(joinpath(lib_dir, group[1:(i - 1)]))
            return (group[1:(i - 1)], group[(i + 1):end])
        end
    end
    return (group, "Core")
end
base_group, test_group = _detect_sublibrary_group(GROUP, _LIB_DIR)

if isdir(joinpath(_LIB_DIR, base_group))
    Pkg.activate(joinpath(_LIB_DIR, base_group))
    # On Julia < 1.11, the [sources] section in Project.toml is not supported.
    # Manually Pkg.develop local path dependencies so CI tests the PR branch code.
    # We resolve transitively: each developed dependency's own [sources] are also
    # developed, so that packages like OrdinaryDiffEqRosenbrockTableaus (a source
    # dependency of OrdinaryDiffEqRosenbrock) are correctly found even when testing
    # a higher-level sublibrary like OrdinaryDiffEqDefault.
    if VERSION < v"1.11.0-DEV.0"
        developed = Set{String}()
        # Never develop the active project: when sublibraries cyclically
        # reference each other via [sources] (e.g. DiffEqDevTools points
        # back at OrdinaryDiffEqCore), the transitive walk below would
        # otherwise try to `Pkg.develop` the active project itself, which
        # Pkg refuses with "package <X> has the same name or UUID as the
        # active project".
        push!(developed, normpath(joinpath(_LIB_DIR, base_group)))
        specs = Pkg.PackageSpec[]
        queue = [joinpath(_LIB_DIR, base_group)]
        while !isempty(queue)
            pkg_dir = popfirst!(queue)
            toml_path = joinpath(pkg_dir, "Project.toml")
            isfile(toml_path) || continue
            toml = Pkg.TOML.parsefile(toml_path)
            if haskey(toml, "sources")
                for (dep_name, source_spec) in toml["sources"]
                    if source_spec isa Dict && haskey(source_spec, "path")
                        dep_path = normpath(joinpath(pkg_dir, source_spec["path"]))
                        if isdir(dep_path) && !(dep_path in developed)
                            push!(developed, dep_path)
                            @info "Queuing local source dependency" dep_name dep_path
                            push!(specs, Pkg.PackageSpec(path = dep_path))
                            # Queue this dependency so its own [sources] are also resolved.
                            push!(queue, dep_path)
                        end
                    end
                end
            end
        end
        # Batch the develop call so Pkg resolves all path deps together;
        # calling it one-at-a-time would re-resolve the active project and
        # fail to find unregistered siblings.
        isempty(specs) || Pkg.develop(specs)
    end
    withenv("CORLEONE_TEST_GROUP" => test_group) do
        Pkg.test(base_group; julia_args = ["--check-bounds=auto", "--depwarn=yes"], force_latest_compatible_version = false, allow_reresolve = true)
    end
else
    using JET
    @testset "Corleone.jl" begin
        @safetestset "Code quality (Aqua.jl)" begin
            using Aqua
            using Corleone
            Aqua.test_all(Corleone)
        end
        @safetestset "Local controls" begin
            include("local_controls.jl")
        end
        @safetestset "Multiple shooting" begin
            include("multiple_shooting.jl")
        end
        @testset "Examples" begin
            @safetestset "Lotka" begin
                include("examples/lotka_oc.jl")
            end
            @safetestset "Lotka MS" begin
                include("examples/lotka_ms.jl")
            end
            @safetestset "Lotka MTK" begin
                include("examples/mtk.jl")
            end
        end
    end
end

# What to test?
# local_controls.jl:
#   - construction of index_grid, get_subvector_indices -> Julius
# general:
#   - more convergence? Lotka OED

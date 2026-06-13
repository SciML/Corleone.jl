using Pkg
using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

# Centralized sublibrary CI (SciML/.github sublibrary-project-tests.yml@v1) tests
# each lib/<name> via the project model and never routes through this file. This
# dispatcher only matters when the root suite is invoked with a GROUP that names
# a sublibrary (e.g. local `GROUP=CorleoneOED julia test/runtests.jl`): the bare
# sublibrary name selects that sublibrary's "Core" group and "<sublibrary>_<grp>"
# selects a named group. We then activate the sublibrary's own test environment
# and hand off to its runtests.jl via CORLEONE_TEST_GROUP. Otherwise the
# main-package suite below runs: GROUP="All" (the local default) and "Core" run
# the light Core group; "Examples" and "QA" each activate their own dep-adding
# test/<group> environment.
const LIB_DIR = joinpath(@__DIR__, "..", "lib")

# QA (Aqua) runs in an isolated environment (test/qa) so its tooling deps never
# enter the main test target's resolve. On Julia < 1.11 the [sources] table is
# ignored, so develop the package by path to test the PR branch code.
function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..")))
    return Pkg.instantiate()
end

# Examples (dep-adding group): the example scripts need the heavy optimal-control
# stack (ModelingToolkit, Ipopt/OptimizationMOI, SciMLSensitivity), isolated in
# test/examples so they never enter the main test target's resolve. On Julia
# < 1.11 the [sources] table is ignored, so develop Corleone by path.
function activate_examples_env()
    Pkg.activate(joinpath(@__DIR__, "examples"))
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..")))
    return Pkg.instantiate()
end

# Reserved main-suite group names: these route to the main-package branch below,
# never to a sublibrary, even if a lib/<name> directory happened to match.
const _RESERVED_GROUPS = ("All", "Core", "Examples", "QA")

# Scan underscores right-to-left to find the longest matching sublibrary prefix,
# returning (sublibrary, test_group) where test_group defaults to "Core". An empty
# or reserved GROUP is never a sublibrary (an empty name would match lib/ itself).
function _detect_sublibrary_group(group, lib_dir)
    (isempty(group) || group in _RESERVED_GROUPS) && return (group, "Core")
    isdir(joinpath(lib_dir, group)) && return (group, "Core")
    for i in length(group):-1:1
        if group[i] == '_' && isdir(joinpath(lib_dir, group[1:(i - 1)]))
            return (group[1:(i - 1)], group[(i + 1):end])
        end
    end
    return (group, "Core")
end

const _SUBLIB, _SUB_GROUP = _detect_sublibrary_group(GROUP, LIB_DIR)

if !isempty(_SUBLIB) && !(_SUBLIB in _RESERVED_GROUPS) && isdir(joinpath(LIB_DIR, _SUBLIB))
    sublib_path = joinpath(LIB_DIR, _SUBLIB)
    Pkg.activate(sublib_path)
    # On Julia < 1.11 the [sources] table in Project.toml is ignored, so develop
    # the local path dependencies (e.g. Corleone) to test the PR branch code.
    # Walk [sources] transitively in case a developed dependency carries its own.
    if VERSION < v"1.11.0-DEV.0"
        developed = Set{String}()
        push!(developed, normpath(sublib_path))
        specs = Pkg.PackageSpec[]
        queue = [sublib_path]
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
                            push!(specs, Pkg.PackageSpec(path = dep_path))
                            push!(queue, dep_path)
                        end
                    end
                end
            end
        end
        isempty(specs) || Pkg.develop(specs)
    end
    # Hand the resolved test group to the sublibrary runtests.jl, which reads
    # CORLEONE_TEST_GROUP (matching the SublibraryCI group-env-name).
    withenv("CORLEONE_TEST_GROUP" => _SUB_GROUP) do
        Pkg.test(_SUBLIB; allow_reresolve = true)
    end
else
    @testset "Corleone.jl" begin
        # Core: light tests that resolve in the main test target's environment.
        # The default GROUP="All" runs only these no-extra-dep groups; the
        # dep-adding groups below (Examples, QA) each activate their own
        # isolated test/<group> environment and run only when selected.
        if GROUP == "All" || GROUP == "Core"
            @safetestset "Local controls" begin
                include("core/local_controls.jl")
            end
            @safetestset "Multiple shooting" begin
                include("core/multiple_shooting.jl")
            end
        end

        # Examples: the optimal-control example scripts pull in the heavy
        # MTK + Ipopt/MOI + SciMLSensitivity stack, isolated in test/examples
        # so those deps never enter the light main test target's resolve.
        if GROUP == "Examples"
            activate_examples_env()
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

        if GROUP == "QA"
            activate_qa_env()
            @safetestset "Code quality (Aqua.jl)" include("qa/qa.jl")
        end
    end
end

# What to test?
# local_controls.jl:
#   - construction of index_grid, get_subvector_indices -> Julius
# general:
#   - more convergence? Lotka OED

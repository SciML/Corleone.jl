using Pkg
using SafeTestsets, Test
using SciMLTesting

const GROUP = current_group()
const LIB_DIR = joinpath(dirname(@__DIR__), "lib")

# Centralized sublibrary CI (SciML/.github sublibrary-project-tests.yml@v1) tests
# each lib/<name> via the project model and never routes through this file. This
# dispatcher only matters when the root suite is invoked with a GROUP that names
# a sublibrary (e.g. local `GROUP=CorleoneOED julia test/runtests.jl`): the bare
# sublibrary name selects that sublibrary's "Core" group and "<sublibrary>_<grp>"
# selects a named group. We then activate the sublibrary's own test environment
# and hand off to its runtests.jl via CORLEONE_TEST_GROUP. The sublibrary Pkg.test
# is done explicitly here (rather than via run_tests's built-in lib_dir path) so the
# Julia < 1.11 transitive [sources] develop walk is preserved verbatim. Otherwise
# the main-package suite below runs via run_tests: GROUP="All"/"Core" run the light
# Core group; "Examples" and "QA" each activate their own test/<group> environment.
base_group, test_group = detect_sublibrary_group(GROUP, LIB_DIR)

if !isempty(base_group) && isdir(joinpath(LIB_DIR, base_group))
    sublib_path = joinpath(LIB_DIR, base_group)
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
                            push!(specs, Pkg.PackageSpec(path=dep_path))
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
    withenv("CORLEONE_TEST_GROUP" => test_group) do
        Pkg.test(base_group; allow_reresolve=true)
    end
else
    run_tests(;
        # Core: light tests that resolve in the main test target's environment.
        core=function ()
            #=
            @safetestset "Local controls" begin
                include("core/local_controls.jl")
            end
            return @safetestset "Multiple shooting" begin
                include("core/multiple_shooting.jl")
            end
            =#
        end,
        groups=Dict(
            # Examples: the optimal-control example scripts pull in the heavy
            # MTK + Ipopt/MOI + SciMLSensitivity stack, isolated in test/examples
            # so those deps never enter the light main test target's resolve.
            "Examples" => (;
                env=joinpath(@__DIR__, "examples"),
                body=function ()
                    #=
                    @safetestset "Lotka" begin
                        include("examples/lotka_oc.jl")
                    end
                    @safetestset "Lotka MS" begin
                        include("examples/lotka_ms.jl")
                    end
                    return @safetestset "Lotka MTK" begin
                        include("examples/mtk.jl")
                    end
                    =#
                end,
            ),
        ),
        qa=(;
            env=joinpath(@__DIR__, "qa"),
            body=function ()
                return @safetestset "Code quality (Aqua.jl)" include("qa/qa.jl")
            end,
        ),
        # The original ran only the Core group for the default GROUP="All"; the
        # dep-adding Examples and QA groups ran only when explicitly selected.
        all=["Core"],
        sublib_env="CORLEONE_TEST_GROUP",
        lib_dir=LIB_DIR,
    )
end

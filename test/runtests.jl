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
# and hand off to its runtests.jl via CORLEONE_TEST_GROUP. Otherwise the main-package
# suite below runs via run_tests: GROUP="All"/"Core" run the light
# Core group; "Examples" and "QA" each activate their own test/<group> environment.
base_group, test_group = detect_sublibrary_group(GROUP, LIB_DIR)

if !isempty(base_group) && isdir(joinpath(LIB_DIR, base_group))
    sublib_path = joinpath(LIB_DIR, base_group)
    Pkg.activate(sublib_path)
    withenv("CORLEONE_TEST_GROUP" => test_group) do
        Pkg.test(base_group; allow_reresolve = true)
    end
else
    run_tests(;
        # Core: light tests that resolve in the main test target's environment.
        core = function ()
            @safetestset "Annotations" begin
                include("core/annotations.jl")
            end
            @safetestset "Annotation initialization" begin
                include("core/annotation_initialization.jl")
            end
            @safetestset "Symbolic bookkeeping" begin
                include("core/symbolic_metadata.jl")
            end
            @safetestset "Per-variable initial conditions" begin
                include("core/initial_conditions.jl")
            end
            @safetestset "Initial-condition differentiation" begin
                include("core/initial_conditions_ad.jl")
            end
            @safetestset "Optimization contracts" begin
                include("core/optimization_contracts.jl")
            end
            @safetestset "Optimization contract differentiation" begin
                include("core/optimization_contracts_ad.jl")
            end
            @safetestset "Cached shooting timestops" begin
                include("core/shooting_timestops.jl")
            end
            @safetestset "Parallel shooting" begin
                include("core/shooting_parallel.jl")
            end
            @safetestset "Parallel shooting differentiation" begin
                include("core/shooting_parallel_ad.jl")
            end
            @safetestset "Legacy single-shooting constructor" begin
                include("core/legacy_single_shooting.jl")
            end
            @safetestset "Trajectory" begin
                include("core/trajectory.jl")
            end
            return @safetestset "Trajectory differentiation" begin
                include("core/trajectory_ad.jl")
            end
            #@safetestset "Local controls" begin
            #    include("core/local_controls.jl")
            #end
            #@safetestset "Precompile workload" begin
            #    include("core/precompile_workload.jl")
            #end
            #@safetestset "Generic layer interface" begin
            #    include("core/interface_contracts.jl")
            #end
            #@safetestset "Multiple shooting" begin
            #    include("core/multiple_shooting.jl")
            #end
            #return @safetestset "Segment state handoff" begin
            #    include("core/state_handoff.jl")
            #end
        end,
        groups = Dict(
            # Examples: the optimal-control example scripts pull in the heavy
            # MTK + Ipopt/MOI + SciMLSensitivity stack, isolated in test/examples
            # so those deps never enter the light main test target's resolve.
            "Examples" => (;
                env = joinpath(@__DIR__, "examples"),
                body = function ()
                    @safetestset "Lotka" begin
                        include("examples/lotka_oc.jl")
                    end
                    return @safetestset "Lotka MS" begin
                        include("examples/lotka_ms.jl")
                    end
                    #return @safetestset "Lotka MTK" begin
                    #    include("examples/mtk.jl")
                    #end
                end,
            ),
        ),
        qa = (;
            env = joinpath(@__DIR__, "qa"),
            body = function ()
                return @safetestset "Code quality (Aqua.jl)" include("qa/qa.jl")
            end,
        ),
        # The original ran only the Core group for the default GROUP="All"; the
        # dep-adding Examples and QA groups ran only when explicitly selected.
        all = ["Core"],
        sublib_env = "CORLEONE_TEST_GROUP",
        lib_dir = LIB_DIR,
    )
end

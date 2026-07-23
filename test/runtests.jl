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
            @safetestset "Solutions" begin
                include("core/solutions.jl")
            end
            return @safetestset "Layers" begin
                include("core/layers.jl")
            end
        end,
        groups = Dict(
            # Examples: the optimal-control example scripts pull in the heavy
            # MTK + Ipopt/MOI + SciMLSensitivity stack, isolated in test/examples
            # so those deps never enter the light main test target's resolve.
            "Examples" => (;
                env = joinpath(@__DIR__, "examples"),
                body = function ()
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

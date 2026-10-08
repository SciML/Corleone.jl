using SafeTestsets
using SciMLTesting

const REQUESTED_GROUP = get(ENV, "CORLEONE_TEST_GROUP", get(ENV, "GROUP", "All"))
const GROUP = REQUESTED_GROUP == "CorleoneBase" ? "Core" :
    startswith(REQUESTED_GROUP, "CorleoneBase_") ? REQUESTED_GROUP[14:end] : REQUESTED_GROUP

GROUP in ("All", "Core", "AD", "QA", "Everything") ||
    error("Unknown CorleoneBase test group: $GROUP")

withenv("GROUP" => GROUP) do
    run_tests(;
        core = function ()
            return @safetestset "Sequential Core contracts" begin
                include("core/sequential.jl")
            end
        end,
        groups = Dict(
            "AD" => (;
                env = joinpath(@__DIR__, "ad"),
                body = function ()
                    return @safetestset "Sequential AD" begin
                        include("ad/runtests.jl")
                    end
                end,
            ),
        ),
        qa = (;
            env = joinpath(@__DIR__, "qa"),
            body = function ()
                return @safetestset "Code quality" begin
                    include("qa/qa.jl")
                end
            end,
        ),
        all = ["Core"],
    )
end

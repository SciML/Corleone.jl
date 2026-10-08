using Test

const REQUESTED_GROUP = get(ENV, "CORLEONE_TEST_GROUP", get(ENV, "GROUP", "All"))
const GROUP = REQUESTED_GROUP == "CorleoneBase" ? "Core" :
    startswith(REQUESTED_GROUP, "CorleoneBase_") ? REQUESTED_GROUP[14:end] : REQUESTED_GROUP

GROUP in ("All", "Core", "AD") || error("Unknown CorleoneBase test group: $GROUP")

@testset "CorleoneBase" begin
    include("ad/runtests.jl")
end

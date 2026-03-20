using CorleoneOED
using SafeTestsets

# Old API tests - commented out until migration is complete
# @safetestset "1D Example" begin
#     include("1d_oed.jl")
# end
# @safetestset "Lotka Volterra" begin
#     include("lotka_oed.jl")
# end
# @safetestset "Lotka Volterra SVD" begin
#     include("lotka_oed_svd.jl")
# end

# New API tests
@safetestset "Example Corrected" begin
    include("test_example_corrected.jl")
end
@safetestset "Augmentation V2 New API" begin
    include("test_augmentation_v2_new_api.jl")
end
@safetestset "Differentiability" begin
    include("test_differentiability.jl")
end
@safetestset "2D System" begin
    include("test_2d_system.jl")
end
@safetestset "Discrete Fisher" begin
    include("test_discrete_fisher.jl")
end
@safetestset "Helper Functions" begin
    include("test_helper_functions.jl")
end
@safetestset "Multiparameter Integration" begin
    include("test_multiparameter_integration.jl")
end

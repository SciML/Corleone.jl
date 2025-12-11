using CorleoneOED
using SafeTestsets

@safetestset "1D Example" begin
  include("1d_oed.jl")
end
@safetestset "Lotka Volterra" begin
  include("lotka_oed.jl")
end

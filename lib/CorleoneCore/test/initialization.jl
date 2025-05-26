using CorleoneCore
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Test

@variables x(..) = 0.5 [bounds=(0.0,5.0)]
@variables c(..)=0.5 [input = true, bounds=(-1,1)]
@parameters p = 1.0 [tunable = false]
tspan = (0.0,1.0)
@named first_order = ODESystem(
    [
        D(x(t)) ~ - p * x(t) + c(t),
    ], t, [x(t), c(t)], [p];
    costs=Num[∫((x(t) - 1)^2)],
    consolidate=sum,
    tspan=tspan,
    constraints = []
)

N = 10

shooting_points = [0.0, 0.5, 1.0]

@testset "Basic shooting initialization tests" begin

    for init_method in [DefaultsInitialization, ForwardSolveInitialization, RandomInitialization,
                        LinearInterpolationInitialization, ConstantInitialization, CustomInitialization]

        alg = Tsit5()

        init_values = begin
            if init_method == ConstantInitialization || init_method == LinearInterpolationInitialization
                [x(t) => 1.0]
            elseif init_method == CustomInitialization
                [x(t) => [0.5, 0.75, 1.0]]
            elseif init_method in [DefaultsInitialization, ForwardSolveInitialization, RandomInitialization]
                nothing
            end
        end

        inits = map([DirectControlCallback, IfElseControl]) do controlmethod


            control = controlmethod(c(t) =>  (; timepoints=collect(0.0:0.1:0.9), defaults=collect(0.0:0.1:0.9)))

            newsys, subs = CorleoneCore.extend_costs(first_order)
            sys = tearing(control(newsys))

            initializer = @test_nowarn begin
                if init_method == ForwardSolveInitialization
                    init_method(sys, shooting_points, alg; init_values = init_values)
                else
                    init_method(sys, shooting_points; init_values = init_values)
                end
            end

            initializer
        end

        @info [x.init for x in inits]
        are_inits_equal = map(x->isapprox(first(inits).init[1].second, first(x.init).second), inits)
        if init_method != RandomInitialization
            @test all(are_inits_equal)
        end
    end
end
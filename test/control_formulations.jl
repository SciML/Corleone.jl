using Test
using Corleone
using ModelingToolkit

import ModelingToolkit.D_nounits as D, ModelingToolkit.t_nounits as t

## Helper Functions

function test_controlspecs(control;
    kwargs...
)
    for k in keys(kwargs)
        @test hasproperty(control, k)
        @test isequal(getproperty(control, k), kwargs[k])
    end
end
test_controlspecs(x::Corleone.AbstractControlFormulation; kwargs...) = test_controlspecs(
    Corleone.get_control_specs(x)
    ; kwargs...)
test_controlspecs(x::Tuple; kwargs...) =
    foreach(x) do xi
        test_controlspecs(xi; kwargs...)
    end


function test_control_expansion(sys, controlspecs; kwargs...)
    newsys = controlspecs(sys)
    # Basic tests 
    @testset "Local controls" begin
        locals = Corleone.get_localcontrols(newsys)
        ts = Corleone.get_tstoppoints(newsys)
            @test length(locals) == length(controlspecs.controls)
        for (u, specs) in zip(locals, controlspecs.controls) 
            @test ModelingToolkit.getdefault(u)  == specs.defaults
        end
        @test ts == unique!(sort!(reduce(vcat, map(Base.Fix2(getproperty, :timepoints), controlspecs.controls))))
    end
end

## Setup 

@mtkmodel LQR begin
    @description "Linear quadratic regulator"
    @variables begin
        x(t) = 1.0, [description = "State variable", tunable = false]
        u(t) = 0.0, [description = "Control variable", input = true]
    end
    @parameters begin
        a = -1.0, [description = "Decay", tunable = false]
        b = 1.0, [description = "Input scale", tunable = false]
    end
    @equations begin
        D(x) ~ a * x + b * u
    end
    @costs begin
        Symbolics.Integral(t in (0., 10.)).(10.0 * (x - 3.0)^2 + 0.1 * u^2)
    end
    @consolidate begin
        (system_costs...) -> first(system_costs)[1]
    end
end;

lqr_model = LQR(; name=:LQR)
u = ModelingToolkit.getvar(lqr_model, :u, namespace=false)

direct_controls = [
    x(Num(u) => (; timepoints=collect(0.0:0.5:9.5))) for x in (DirectControlCallback, IfElseControl, TanhControl)
]
differential_controls = [
    x(Num(D(u)) => (; timepoints=collect(0.0:0.5:9.5))) for x in (DirectControlCallback, IfElseControl, TanhControl)
]

@testset "Control Formulation Tests" begin
    @testset "Direct Controls" begin
        for control in direct_controls
            test_controlspecs(control,
                variable=Symbolics.operation(u),
                timepoints=collect(0.0:0.5:9.5),
                differential=false,
                defaults=[0.0 for _ in 1:length(collect(0.0:0.5:9.5))],
                bounds=getbounds(Symbolics.operation(u))
            )
        end
    end
    @testset "Differential Controls" begin
        for control in differential_controls
            test_controlspecs(control,
                variable=Symbolics.operation(u),
                timepoints=collect(0.0:0.5:9.5),
                differential=true,
                defaults=[0.0 for _ in 1:length(collect(0.0:0.5:9.5))],
                bounds=getbounds(Symbolics.operation(u))
            )
        end
    end
end

@testset "Control Expansion Tests" begin
    @testset "Direct Controls" begin
        for controlspec in direct_controls
            test_control_expansion(lqr_model, controlspec)
        end
    end
    @testset "Differential Controls" begin
        for controlspec in differential_controls
            test_control_expansion(lqr_model, controlspec)
        end
    end
end

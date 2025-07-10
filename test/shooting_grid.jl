using Test
using Corleone
using ModelingToolkit
import ModelingToolkit.D_nounits as D, ModelingToolkit.t_nounits as t

## Setup 
@mtkmodel FO begin
    @description "Linear quadratic regulator"
    @variables begin
        x(t) = 1.0, [description = "State variable", tunable = false]
    end
    @parameters begin
        a = -1.0, [description = "Decay", tunable = false]
    end
    @equations begin
        D(x) ~ a * x 
    end
end;

model = FO(; name = :FO)

grid = ShootingGrid([0., 10., 20.])

new_model = grid(model)

@test Corleone.get_shootingpoints(new_model) == [0., 10., 20.]
xshoot = Corleone.get_shootingvariables(new_model) 
@test length(xshoot) == 1 
@test length(only(xshoot)) == 3


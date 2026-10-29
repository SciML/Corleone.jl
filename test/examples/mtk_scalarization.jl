using Corleone, ModelingToolkit, Test
using ModelingToolkit: Symbolics, t_nounits as t, D_nounits as D

@testset "MTK cost and constraint scalarization" begin
    extension = Base.get_extension(Corleone, :CorleoneModelingToolkitExtension)
    @variables scalarization_state(t) = 0.0
    @parameters scalarization_matrix[1:2, 1:2] = [1.0 2.0; 3.0 4.0]
    @named arrays = System(
        [D(scalarization_state) ~ sum(scalarization_matrix)], t;
        costs = [scalarization_matrix .^ 2, scalarization_matrix[1, 1] + scalarization_matrix[2, 2]],
        constraints = [
            scalarization_matrix ~ zeros(2, 2),
            Symbolics.Inequality(scalarization_matrix, fill(5.0, 2, 2), Symbolics.leq),
            scalarization_matrix[1, 1] ≳ 0.0,
        ]
    )
    scalar = extension.scalarize_system(arrays)
    components = vec(collect(Symbolics.scalarize(scalarization_matrix)))
    expected_costs = Symbolics.unwrap.(vcat(components .^ 2, [scalarization_matrix[1, 1] + scalarization_matrix[2, 2]]))
    expected_constraints = vcat(
        components .~ 0.0, components .≲ 5.0, [scalarization_matrix[1, 1] ≳ 0.0]
    )
    @test isequal(ModelingToolkit.get_costs(scalar), expected_costs)
    @test isequal(ModelingToolkit.get_constraints(scalar), expected_constraints)
    @test length(ModelingToolkit.get_costs(scalar)) == 5
    @test length(ModelingToolkit.get_constraints(scalar)) == 9

    # Array expressions also need expansion when their symbols are all scalar.
    @parameters scalarization_parameter = 1.0
    @named scalar_symbols = System(
        [D(scalarization_state) ~ scalarization_parameter], t;
        costs = [[scalarization_parameter, scalarization_parameter^2]],
        constraints = [[scalarization_parameter, scalarization_parameter + 1] ~ [0.0, 1.0]]
    )
    expanded = extension.scalarize_system(scalar_symbols)
    @test length(ModelingToolkit.get_costs(expanded)) == 2
    @test length(ModelingToolkit.get_constraints(expanded)) == 2
    @test isequal(ModelingToolkit.get_costs(expanded), Symbolics.unwrap.([scalarization_parameter, scalarization_parameter^2]))
    @test isequal(ModelingToolkit.get_constraints(expanded), [scalarization_parameter ~ 0.0, scalarization_parameter + 1 ~ 1.0])

    @named already_scalar = System(
        [D(scalarization_state) ~ scalarization_parameter], t;
        costs = [scalarization_parameter^2], constraints = [scalarization_parameter ≲ 2.0]
    )
    @test extension.scalarize_system(already_scalar) === already_scalar
end

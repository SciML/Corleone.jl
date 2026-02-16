"""
$(TYPEDEF)

A struct for capturing the internal definition of a dynamic optimization problem. 

# Fields 
$(FIELDS)
"""
struct CorleoneDynamicOptProblem{L, G, O, C, CB}
    "The resulting layer for the problem"
    layer::L
    "The getters which return the values of the trajectory"
    getters::G
    "The objective function"
    objective::O
    "The constraint function"
    contraints::C
    "Lower bounds for the constraints"
    lcons::CB
    "Upper bounds for the constraints"
    ucons::CB
end
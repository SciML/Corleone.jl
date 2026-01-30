links = InterLinks(
    "Julia" => (
        "https://docs.julialang.org/en/v1/",
        "https://docs.julialang.org/en/v1/objects.inv"
    ),
    "SciMLBase" => (
        "https://docs.sciml.ai/SciMLBase/stable/", 
        "https://docs.sciml.ai/SciMLBase/stable/objects.inv"
    ),
    "Optimization" => (
        "https://docs.sciml.ai/Optimization/stable/",
        "https://docs.sciml.ai/Optimization/stable/objects.inv"
    ),
    # Here we cheat a little and reuse optimization for the inventory
    "OptimizationMOI" => (
        "https://docs.sciml.ai/Optimization/stable/optimization_packages/mathoptinterface/",
        "https://docs.sciml.ai/Optimization/stable/objects.inv"
    ),
);
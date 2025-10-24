using Documenter, Corleone

mathengine = Documenter.MathJax(Dict(:TeX => Dict(
    :equationNumbers => Dict(:autoNumber => "AMS"),
    :Macros => Dict(
        :ket => ["|#1\\rangle", 1],
        :bra => ["\\langle#1|", 1],
    ),
)))

makedocs(  sitename = "Corleone.jl",
    format = Documenter.HTML(mathengine = mathengine,),
    modules = [Corleone],
    pages = [
        "Home" => "index.md",
        "Examples" => ["Optimal Control" => "./examples/lotka.md",
                        "Optimal Experimenal Design" => "./examples/lotka_oed.md",
                        "Multiexperiments" => "./examples/multiexperiments.md"
                        ],
        "API Reference" => "api.md",
    ],
    remotes = nothing)
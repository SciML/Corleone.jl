using OptimalControlBenchmarks, Documenter
makedocs(;
    sitename = "OptimalControlBenchmarks",
    modules = [OptimalControlBenchmarks],
    format = Documenter.HTML(edit_link = "main"),
    pages = ["API and usage" => "index.md"],
    checkdocs = :exports,
    warnonly = false,
)

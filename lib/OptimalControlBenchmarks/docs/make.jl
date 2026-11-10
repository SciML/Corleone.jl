using OptimalControlBenchmarks, Documenter
makedocs(;
    sitename = "OptimalControlBenchmarks",
    modules = [OptimalControlBenchmarks],
    format = Documenter.HTML(edit_link = "main"),
    pages = ["Overview" => "index.md", "Problem Catalog" => "problems.md"],
    checkdocs = :exports,
    warnonly = false,
)

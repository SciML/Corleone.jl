using CorleoneOED, Documenter

makedocs(;
    sitename = "CorleoneOED",
    modules = [CorleoneOED],
    format = Documenter.HTML(edit_link = "main"),
    pages = ["CorleoneOED" => "index.md"],
    checkdocs = :exports,
    warnonly = false,
)

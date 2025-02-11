using Corleone
using Documenter

DocMeta.setdocmeta!(Corleone, :DocTestSetup, :(using Corleone); recursive=true)

makedocs(;
    modules=[Corleone],
    authors="JuliusMartensen <julius.martensen@gmail.com> and contributors",
    sitename="Corleone.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

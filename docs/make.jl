using LowRankArithmetic
using Documenter

DocMeta.setdocmeta!(LowRankArithmetic, :DocTestSetup, :(using LowRankArithmetic); recursive=true)

makedocs(;
    modules=[LowRankArithmetic],
    authors="Acme Corp",
    repo="https://github.com/FHoltorf/LowRankArithmetic.jl/blob/{commit}{path}#{line}",
    sitename="LowRankArithmetic.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://FHoltorf.github.io/LowRankArithmetic.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/FHoltorf/LowRankArithmetic.jl",
    devbranch="main",
)

using EquivariantTensors
using Documenter

DocMeta.setdocmeta!(EquivariantTensors, :DocTestSetup, :(using EquivariantTensors); recursive=true)

makedocs(;
    modules=[EquivariantTensors],
    authors="Christoph Ortner <christophortner0@gmail.com> and contributors",
    sitename="EquivariantTensors.jl",
    format=Documenter.HTML(;
        canonical="https://ACEsuit.github.io/EquivariantTensors.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Public API" => "api.md",
        "Docstrings" => "docstrings.md",
        "Benchmarking" => "benchmarking.md",

    ],
    checkdocs=:exports
)

deploydocs(;
    repo="github.com/ACEsuit/EquivariantTensors.jl",
    devbranch="main",
)

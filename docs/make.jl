using HyperGraphNeuralNetworks
using Documenter

DocMeta.setdocmeta!(HyperGraphNeuralNetworks, :DocTestSetup, :(using HyperGraphNeuralNetworks); recursive=true)

makedocs(;
    modules=[HyperGraphNeuralNetworks],
    authors="Evan Walter Clark Spotte-Smith",
    sitename="HyperGraphNeuralNetworks.jl",
    format=Documenter.HTML(;
        canonical="https://espottesmith.github.io/HyperGraphNeuralNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/espottesmith/HyperGraphNeuralNetworks.jl",
    devbranch="main",
)

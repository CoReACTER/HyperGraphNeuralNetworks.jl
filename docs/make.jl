using HyperGraphNeuralNetworks
using Documenter

DocMeta.setdocmeta!(HyperGraphNeuralNetworks, :DocTestSetup, :(using HyperGraphNeuralNetworks); recursive=true)

makedocs(;
    modules=[HyperGraphNeuralNetworks],
    authors="Evan Walter Clark Spotte-Smith, Punna Amornvivat, CoReACTER",
    sitename="HyperGraphNeuralNetworks.jl",
    format=Documenter.HTML(;
        canonical="https://CoReACTER.org/HyperGraphNeuralNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="https://github.com/CoReACTER/HyperGraphNeuralNetworks.jl",
    branch="gh-pages",
    devbranch="main",
)
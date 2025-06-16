module HyperGraphNeuralNetworks

using Graphs
using GNNGraphs
using MLUtils
using SimpleHypergraphs
using SimpleDirectedHypergraphs

include("core/abstracttypes.jl")
include("core/hypergraphs.jl")
include("core/generate.jl")

export AbstractHGNNHypergraph, AbstractHGNNDiHypergraph
export HGNNHypergraph, HGNNDiHypergraph

export erdos_renyi_hypergraph, random_kuniform_hypergraph, random_dregular_hypergraph, random_preferential_hypergraph

end

module HyperGraphNeuralNetworks

using Graphs
using GNNGraphs
using MLUtils
using SimpleHypergraphs
using SimpleDirectedHypergraphs

export AbstractHGNNHypergraph, AbstractHGNNDiHypergraph
export HGNNHypergraph, HGNNDiHypergraph

include("core/abstracttypes.jl")
include("core/hypergraphs.jl")

end

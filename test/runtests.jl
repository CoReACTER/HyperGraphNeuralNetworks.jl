using Random
using StatsBase
using LinearAlgebra
using Test
using Graphs
using GNNGraphs
import GNNGraphs: getn, getdata, normalize_graphdata, cat_features, shortsummary
using MLUtils
using SimpleHypergraphs
using SimpleDirectedHypergraphs
using HyperGraphNeuralNetworks

# Necessary for MLDatasets
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

include("core/hypergraph.jl")
include("core/dihypergraph.jl")
include("core/generate.jl")
include("core/query.jl")
include("core/split.jl")
include("core/transform.jl")

include("layers/message_passing.jl")
include("layers/attention.jl")


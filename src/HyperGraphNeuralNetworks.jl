module HyperGraphNeuralNetworks

using Random
using DataStructures: counter
using StatsBase: FrequencyWeights, sample
using InvertedIndices
using Graphs
using GNNGraphs
import GNNGraphs: getn, getdata, normalize_graphdata, cat_features, shortsummary
using MLUtils
import MLDatasets
using SimpleHypergraphs
using SimpleDirectedHypergraphs

include("core/abstracttypes.jl")
include("core/hypergraphs.jl")

export AbstractHGNNHypergraph, AbstractHGNNDiHypergraph
export HGNNHypergraph, HGNNDiHypergraph
export add_vertex, add_vertices, remove_vertex, remove_vertices
export add_hyperedge, add_hyperedges, remove_hyperedge, remove_hyperedges

include("core/generate.jl")

export erdos_renyi_hypergraph, random_kuniform_hypergraph, random_dregular_hypergraph, random_preferential_hypergraph

include("core/query.jl")

export hyperedge_index, get_hyperedge_weight, get_hyperedge_weights
export has_vertex, vertices, isolated_vertices
export degree, indegree, outdegree
export all_neighbors, in_neighbors, out_neighbors, hyperedge_neighbors
export incidence_matrix, complex_incidence_matrix, hyperedge_weight_matrix, vertex_weight_matrix
export normalized_laplacian
export hypergraph_ids, has_self_loops, has_multi_hyperedges

include("core/transform.jl")

export add_self_loops, remove_self_loops, remove_multi_hyperedges
export rewire_hyperedges, to_undirected
export combine_hypergraphs, get_hypergraph, negative_sample
export random_split_vertices, random_split_hyperedges

include("core/utils.jl")

export check_num_vertices, check_num_hyperedges
export normalize_graphdata

include("datasets/datasets.jl")

export getHyperCora, getHyperCiteSeer

end

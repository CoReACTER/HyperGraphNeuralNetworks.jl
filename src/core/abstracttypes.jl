# TODO: special types for different types of matrices, e.g., adjacency lists?

abstract type AbstractHGNNHypergraph{T} <: AbstractSimpleHypergraph{T} end
abstract type AbstractHGNNDiHypergraph{T} <: AbstractDirectedHypergraph{Tuple{T, T}} end
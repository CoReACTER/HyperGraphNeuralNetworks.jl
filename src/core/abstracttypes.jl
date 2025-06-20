# TODO: special types for different types of matrices, e.g., adjacency lists?

"""
    AbstractHGNNHypergraph{T} <: AbstractSimpleHypergraph{T}

An abstract undirected hypergraph type for use in machine learning
"""
abstract type AbstractHGNNHypergraph{T} <: AbstractSimpleHypergraph{T} end

"""
    AbstractHGNNDiHypergraph{T} <: AbstractDirectedHypergraph{T}

An abstract directed hypergraph type for use in machine learning
"""
abstract type AbstractHGNNDiHypergraph{T} <: AbstractDirectedHypergraph{T} end
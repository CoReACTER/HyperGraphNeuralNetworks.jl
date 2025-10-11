"""
    abstract type HGNNLayer <: AbstractLuxLayer end

    An abstract layer type (compatible with Lux) used for hypergraph neural network layers.
"""
abstract type HGNNLayer <: AbstractLuxLayer end

"""
    abstract type HGNNContainerLayer{T} <: AbstractLuxContainerLayer{T} end

    An abstract container layer (compatible with Lux) for hypergraph neural networks.
"""
abstract type HGNNContainerLayer{T} <: AbstractLuxContainerLayer{T} end
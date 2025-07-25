
"""
    check_num_vertices(hg::H, x::AbstractArray) where {H <: AbstractHGNNHypergraph}
    check_num_vertices(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNHypergraph}

    Ensure that an array abstract array, tuple, or named tuple (i.e., tensor) `x` has the appropriate final dimension,
    equal to the number of vertices in the associated hypergraph `hg`
"""
function check_num_vertices(hg::H, x::AbstractArray) where {H <: AbstractHGNNHypergraph}
    @assert hg.num_vertices == size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension instead of num_vertices = $(hg.num_vertices)"
    return true
end

function check_num_vertices(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNHypergraph}
    map(q -> check_num_vertices(hg, q), x)
    return true
end

check_num_vertices(::H, ::Nothing) where {H <: AbstractHGNNHypergraph} = true


"""
    check_num_hyperedges(hg::H, x::AbstractArray) where {H <: AbstractHGNNHypergraph}
    check_num_hyperedges(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNHypergraph}

    Ensure that an array abstract array, tuple, or named tuple (i.e., tensor) `x` has the appropriate final dimension,
    equal to the number of hyperedges in the associated hypergraph `hg`
"""
function check_num_hyperedges(hg::H, x::AbstractArray) where {H <: AbstractHGNNHypergraph}
    @assert hg.num_hyperedges == size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension size instead of num_hyperedges=$(hg.num_hyperedges)"
    return true
end

function check_num_hyperedges(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNHypergraph}
    map(q -> check_num_hyperedges(hg, q), x)
    return true
end

check_num_hyperedges(::H, ::Nothing) where {H <: AbstractHGNNHypergraph} = true


"""
    check_num_vertices(hg::H, x::AbstractArray) where {H <: AbstractHGNNDiHypergraph}
    check_num_vertices(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNDiHypergraph}

    Ensure that an array abstract array, tuple, or named tuple (i.e., tensor) `x` has the appropriate final dimension,
    equal to the number of vertices in the associated directed hypergraph `hg`
"""
function check_num_vertices(hg::H, x::AbstractArray) where {H <: AbstractHGNNDiHypergraph}
    @assert hg.num_vertices == size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension instead of num_vertices = $(hg.num_vertices)"
    return true
end

function check_num_vertices(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNDiHypergraph}
    map(q -> check_num_vertices(hg, q), x)
    return true
end

check_num_vertices(::H, ::Nothing) where {H <: AbstractHGNNDiHypergraph} = true

"""
    check_num_hyperedges(hg::H, x::AbstractArray) where {H <: AbstractHGNNDiHypergraph}
    check_num_hyperedges(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNDiHypergraph}

    Ensure that an array abstract array, tuple, or named tuple (i.e., tensor) `x` has the appropriate final dimension,
    equal to the number of hyperedges in the associated directed hypergraph `hg`
"""
function check_num_hyperedges(hg::H, x::AbstractArray) where {H <: AbstractHGNNDiHypergraph}
    @assert hg.num_hyperedges == size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension size instead of num_hyperedges=$(hg.num_hyperedges)"
    return true
end

function check_num_hyperedges(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNDiHypergraph}
    map(q -> check_num_hyperedges(hg, q), x)
    return true
end

check_num_hyperedges(::H, ::Nothing) where {H <: AbstractHGNNDiHypergraph} = true
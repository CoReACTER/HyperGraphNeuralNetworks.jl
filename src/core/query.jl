"""
TODO: docstring

TODO: pad things like hyperedge indices with zeros to make linalg more efficient?
"""

function hyperedge_index(hg::H) where {H <: AbstractHGNNHypergraph}
    collect.(keys.(hg.he2v))
end

function hyperedge_index(hg::H) where {H <: AbstractHGNNDiHypergraph}
    ind_tail = collect.(keys.(hg.hg_tail.he2v))
    ind_head = collect.(keys.(hg.hg_head.he2v))
    return ind_tail, ind_head
end

function get_hyperedge_weights(hg::H) where {H <: AbstractHGNNHypergraph}
    map(x -> filter(y -> !isnothing(y), x), eachcol(hg))
end

function get_hyperedge_weights(hg::H, op::Function) where {H <: AbstractHGNNHypergraph}
    weights = get_hyperedge_weights(hg)
    op.(weights)
end

# TODO: add check for array bounds
function get_hyperedge_weight(hg::H, he_ind::Int) where {H <: AbstractHGNNHypergraph}
    filter(x -> !isnothing(x), hg[:, he_ind])
end

function get_hyperedge_weight(hg::H, he_ind::Int, op::Function) where {H <: AbstractHGNNHypergraph}
    weights = get_hyperedge_weight(hg, he_ind)
    op(weights)
end

function get_hyperedge_weights(hg::H; side::Symbol = :both) where {H <: AbstractHGNNDiHypergraph}
    if side === :both
        tweights = map(x -> filter(y -> !isnothing(y), x), eachcol(hg.hg_tail))
        hweights = map(x -> filter(y -> !isnothing(y), x), eachcol(hg.hg_head))
        return tweights, hweights
    elseif side === :tail
        return map(x -> filter(y -> !isnothing(y), x), eachcol(hg.hg_tail))
    elseif side === :head
        return map(x -> filter(y -> !isnothing(y), x), eachcol(hg.hg_head))
    else
        throw(ArgumentError("Argument `side` must be one of :head, :tail, or :both!"))
    end
end

# TODO: how to handle case where user wants a single weight value per hyperedge, using :both?
function get_hyperedge_weights(hg::H, op::Function; side::Symbol = :both) where {H <: AbstractHGNNDiHypergraph}
    weights = get_hyperedge_weights(hg; side=side)

    if side === :both
        return op.(weights[1]), op.(weights[2])
    else
        op.(weights)
    end
end

function get_hyperedge_weight(hg::H, he_ind::Int; side::Symbol = :both) where {H <: AbstractHGNNDiHypergraph}
    if side === :both
        tweight = filter(y -> !isnothing(y), hg.hg_tail[:, he_ind])
        hweight = filter(y -> !isnothing(y), hg.hg_head[:, he_ind])
        return tweight, hweight
    elseif side === :tail
        return filter(y -> !isnothing(y), hg.hg_tail[:, he_ind])
    elseif side === :head
        return filter(y -> !isnothing(y), hg.hg_head[:, he_ind])
    else
        throw(ArgumentError("Argument `side` must be one of :head, :tail, or :both!"))
    end
end

# TODO: see `get_hyperedge_weights` above
function get_hyperedge_weight(hg::H, he_ind::Int, op::Function; side::Symbol = :both) where {H <: AbstractHGNNDiHypergraph}
    weight = get_hyperedge_weight(hg, he_ind; side=side)
    
    if side === :both
        return op(weight[1]), op(weight[2])
    else
        op(weight)
    end
end

function Base.eltype()

end

# has_vertex
# vertices
# degree
# indegree
# outdegree
# isolated_vertices
# laplacian_matrix
# normalized_laplacian
# scaled_laplacian
# eigenvalues
# eigenvalues_laplacian
# hypergraph_ids
# vertex_features
# hyperedge_features
# hypergraph_features
# has_self_loops
# has_multi_hyperedges

# ??? can I do khop_adj? does the trick of nth nearest neighbors being related to exponentiating the adjacency matrix work here?


# neighbors (within, incoming, outgoing)

function incidence_matrix()

end


"""
TODO: pad things like hyperedge indices with zeros to make linalg more efficient?
"""


"""
    hyperedge_index(hg::H) where {H <: AbstractHGNNHypergraph}

    Obtain the hyperedge index of an undirected hypergraph `hg`. The index is a vector of vectors, where the `i`th
    element of the index contains the indices of all vertices present in hyperedge `i`.
"""
function hyperedge_index(hg::H) where {H <: AbstractHGNNHypergraph}
    collect.(keys.(hg.he2v))
end

"""
    hyperedge_index(hg::H) where {H <: AbstractHGNNHypergraph}

    Obtain the hyperedge index of a directed hypergraph `hg`. The index is returned as two vectors of vectors, one for
    the hyperedge tails and the other for the hyperedge heads. The `i`th element of `ind_tail` contains the indices of
    the vertices present in the tail of hyperedge `i`, and likewise, the `i`th element of `ind_head` contains the
    indices of the vertices in the head of hyperedge `i`.
"""
function hyperedge_index(hg::H) where {H <: AbstractHGNNDiHypergraph}
    ind_tail = collect.(keys.(hg.hg_tail.he2v))
    ind_head = collect.(keys.(hg.hg_head.he2v))
    return ind_tail, ind_head
end

"""
    get_hyperedge_weights(hg::H) where {H <: AbstractHGNNHypergraph}
    get_hyperedge_weights(hg::H, op::Function) where {H <: AbstractHGNNHypergraph}

    Get the weights of each hyperedge in the hypergraph `hg`. This function returns a vector of vectors, where each
    element contains the non-`nothing` weights of the associated hyperedge. If the function `op` is provided, then
    the weights are transformed using `op` before being returned. Note that `op` should take only one argument.
"""
function get_hyperedge_weights(hg::H) where {H <: AbstractHGNNHypergraph}
    map(x -> filter(y -> !isnothing(y), x), eachcol(hg))
end

function get_hyperedge_weights(hg::H, op::Function) where {H <: AbstractHGNNHypergraph}
    weights = get_hyperedge_weights(hg)
    op.(weights)
end

"""
    get_hyperedge_weight(hg::H, he_ind::Int) where {H <: AbstractHGNNHypergraph}
    get_hyperedge_weight(hg::H, he_ind::Int, op::Function) where {H <: AbstractHGNNHypergraph}

    Obtain the non-`nothing` weights associated with a hyperedge in `hg` given by index `he_ind`. See
    `get_hyperedge_weights` for more detail

    TODO: add check for array bounds
"""
function get_hyperedge_weight(hg::H, he_ind::Int) where {H <: AbstractHGNNHypergraph}
    filter(x -> !isnothing(x), hg[:, he_ind])
end

function get_hyperedge_weight(hg::H, he_ind::Int, op::Function) where {H <: AbstractHGNNHypergraph}
    weights = get_hyperedge_weight(hg, he_ind)
    op(weights)
end

"""
    get_hyperedge_weights(hg::H; side::Symbol = :both) where {H <: AbstractHGNNDiHypergraph}
    get_hyperedge_weights(hg::H, op::Function; side::Symbol = :both) where {H <: AbstractHGNNDiHypergraph}

    Return the weights of a directed hypergraph `hg`. The user can choose to obtain the weights of the hyperedge tails
    (`side=:tail`), the heads (`side=:head`), or both (`side=:both`). The tail weights and head weights are both
    vectors of vectors, where the `i`th element of one such vector corresponds to the non-`nothing` weights of the tail
    or head of the `i`th hyperedge. If function `op` is provided, then the weights are transformed using `op` before
    being returned. Note that `op` should take only one argument.

    TODO: how to handle case where user wants a single weight value per hyperedge, using :both?
"""
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

function get_hyperedge_weights(hg::H, op::Function; side::Symbol = :both) where {H <: AbstractHGNNDiHypergraph}
    weights = get_hyperedge_weights(hg; side=side)

    if side === :both
        return op.(weights[1]), op.(weights[2])
    else
        op.(weights)
    end
end

"""
    get_hyperedge_weight(hg::H, he_ind::Int; side::Symbol = :both) where {H <: AbstractHGNNDiHypergraph}
    get_hyperedge_weight(hg::H, he_ind::Int, op::Function; side::Symbol = :both) where {H <: AbstractHGNNDiHypergraph}

    Obtain the non-`nothing` weights associated with a directed hyperedge in `hg` given by index `he_ind`. See
    `get_hyperedge_weights` for more detail.

    TODO: add check for array bounds
"""
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


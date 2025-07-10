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

function Base.eltype(hg::HGNNHypergraph)
    return eltype(hyperedge_index(hg))
end

# From Graphs.jl, but not directly implemented for hypergraphs
has_vertex(hg::H, i::Int) where {H <: AbstractHGNNHypergraph} = 1 <= i <= hg.num_vertices
has_vertex(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph} = 1 <= i <= hg.num_vertices

vertices(hg::H) where {H <: AbstractHGNNHypergraph} = 1:(hg.num_vertices)
vertices(hg::H) where {H <: AbstractHGNNDiHypergraph} = 1:(hg.num_vertices)

#TODO: docstrings

degree(hg::H) where {H <: AbstractHGNNHypergraph} = length.(keys.(hg.v2he))
degree(hg::H, i::Int) where {H <: AbstractHGNNHypergraph} = length(hg.v2he[i])
degree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNHypergraph} = [degree(hg, i) for i in inds]

degree(hg::H) where {H <: AbstractHGNNDiHypergraph} = indegree(hg) .+ outdegree(hg)
degree(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph} = indegree(hg, i) + outdegree(hg, i)
degree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNDiHypergraph} = [degree(hg, i) for i in inds]

# TODO: do these need to be implemented for undirected hypergraphs? Seems like the answer is "no"
indegree(hg::H) where {H <: AbstractHGNNDiHypergraph} = length.(keys.(hg.hg_head.v2he))
indegree(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph} = length(hg.hg_head.v2he[i])
indegree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNDiHypergraph} = [indegree(hg, i) for i in inds]

outdegree(hg::H) where {H <: AbstractHGNNDiHypergraph} = length.(keys.(hg.hg_tail.v2he))
outdegree(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph} = length(hg.hg_tail.v2he[i])
outdegree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNDiHypergraph} = [indegree(hg, i) for i in inds]

function all_neighbors(hg::H) where {H <: AbstractHGNNHypergraph}

    # For undirected hypergraph, neighbor vertices are those that share a hyperedge with the vertex of interest
    hes = collect.(keys.(hg.v2he))

    # There's probably a more efficient way to do this
    neighbors = Vector{Vector{Int}}(undef,length(hes))
    for (i, hes_i) in enumerate(hes)
        for he in hes_i
            cat(neighbors[i], collect(keys(hg.he2v[he])))
        end
        neighbors[i] = sort!(unique!(neighbors[i]))
    end

    neighbors
end

function all_neighbors(hg::H, i::Int) where {H <: AbstractHGNNHypergraph}

    hes = collect(keys(hg.v2he[i]))

    # There's probably a more efficient way to do this
    neighbors = Vector{Int}[]
    for he in hes
        cat(neighbors, collect(keys(hg.he2v[he])))
    end
    
    sort!(unique!(neighbors))
end

function all_neighbors(hg::H; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    incoming = in_neighbors(hg; same_side=same_side)
    outgoing = out_neighbors(hg; same_side=same_side)

    sort!(unique!([incoming; outgoing]))

end

function all_neighbors(hg::H, i::Int; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    incoming = in_neighbors(hg, i; same_side=same_side)
    outgoing = out_neighbors(hg, i; same_side=same_side)

    sort!(unique!([incoming; outgoing]))

end

function in_neighbors(hg::H; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}
    if same_side
        all_neighbors(hg.hg_head)
    else
        # Find the hyperedges for which each node is in the head
        hes = collect.(keys.(hg.hg_head.v2he))

        # Neighbors are the vertices in the associated tails
        neighbors = Vector{Vector{Int}}(undef,length(hes))
        for (i, hes_i) in enumerate(hes)
            for he in hes_i
                cat(neighbors[i], collect(keys(hg.hg_tail.he2v[he])))
            end
            neighbors[i] = sort!(unique!(neighbors[i]))
        end

        neighbors
    end
end

function in_neighbors(hg::H, i::Int; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}
    if same_side
        all_neighbors(hg.hg_head, i)
    else
        # Find the hyperedges for which this node is in the head
        hes = collect(keys(hg.hg_head.v2he[i]))

        # Neighbors are the vertices in the associated tails
        neighbors = Vector{Int}[]
        for he in hes
            cat(neighbors, collect(keys(hg.hg_tail.he2v[he])))
        end
        
        sort!(unique!(neighbors))
    end
end

function out_neighbors(hg::H; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}
    if same_side
        all_neighbors(hg.hg_tail)
    else
        # Find the hyperedges for which each node is in the tail
        hes = collect.(keys.(hg.hg_tail.v2he))

        # Neighbors are the vertices in the associated heads
        neighbors = Vector{Vector{Int}}(undef,length(hes))
        for (i, hes_i) in enumerate(hes)
            for he in hes_i
                cat(neighbors[i], collect(keys(hg.hg_head.he2v[he])))
            end
            neighbors[i] = sort!(unique!(neighbors[i]))
        end

        neighbors
    end
end

function out_neighbors(hg::H, i::Int; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}
    if same_side
        all_neighbors(hg.hg_tail, i)
    else
        # Find the hyperedges for which this node is in the head
        hes = collect(keys(hg.hg_head.v2he[i]))

        # Neighbors are the vertices in the associated tails
        neighbors = Vector{Int}[]
        for he in hes
            cat(neighbors, collect(keys(hg.hg_tail.he2v[he])))
        end
        
        sort!(unique!(neighbors))
    end
end


# TODO: docstrings
function hyperedge_neighbors(hg::H) where {H <: AbstractHGNNHypergraph}
    # Two hyperedges are considered "neighbors" if they share at least one vertex
    vs = Set.(collect.(keys.(hg.he2v)))

    neighbors = Vector{Vector{Int}}(undef,length(vs))
    for i in eachindex(vs)
        for j in eachindex(vs)[i+1:end]
            if length(intersect(vs[i], vs[j])) > 0
                push!(neighbors[i], j)
                push!(neighbors[j], i)
            end
        end
    end
    sort!.(neighbors)
end

function hyperedge_neighbors(hg::H, i::Int) where {H <: AbstractHGNNHypergraph}
    # Two hyperedges are considered "neighbors" if they share at least one vertex
    vs = Set(collect(keys(hg.he2v[i])))

    neighbors = Vector{Int}[]
    for j in eachindex(vs)
        if i == j
            continue
        end

        if length(intersect(vs[i], vs[j])) > 0
            push!(neighbors, j)
        end
    end
    sort!(neighbors)
end

function hyperedge_neighbors(hg::H; directed::Bool=false) where {H <: AbstractHGNNDiHypergraph}
    if !directed
        # Two directed hyperedges are considered "neighbors" if they share at least one vertex in EITHER their tails or
        # heads
        vs = Set.(cat.(collect.(keys.(hg.hg_tail.he2v)), collect.(keys.(hg.hg_head.he2v))))

        neighbors = Vector{Vector{Int}}(undef,length(vs))
        for i in eachindex(vs)
            for j in eachindex(vs)[i+1:end]
                if length(intersect(vs[i], vs[j])) > 0
                    push!(neighbors[i], j)
                    push!(neighbors[j], i)
                end
            end
        end
    else
        # Two directed hyperedges Ea and Eb are considered "directed neighbors" (TODO: term?) there is at least one vertex
        # in the head of Ea that is in the tail of Eb
        vs_tail = Set.(collect.(keys.(hg.hg_tail.he2v)))
        vs_head = Set.(collect.(keys.(hg.hg_head.he2v)))

        neighbors = Vector{Vector{Int}}(undef,length(vs_tail))
        for i in eachindex(vs_head)
            for j in eachindex(vs_tail)
                if i == j
                    continue
                end

                if length(intersect(vs_head[i], vs_tail[j])) > 0
                    push!(neighbors[i], j)
                end
            end
        end
    end

    sort!.(neighbors)
end

function hyperedge_neighbors(hg::H, i::Int; directed::Bool=false) where {H <: AbstractHGNNDiHypergraph}
    if !directed
        # Two directed hyperedges are considered "neighbors" if they share at least one vertex in EITHER their tails or
        # heads
        vs = Set.(cat.(collect.(keys.(hg.hg_tail.he2v)), collect.(keys.(hg.hg_head.he2v))))

        neighbors = Vector{Int}[]
        for j in eachindex(vs)
            if i == j
                continue
            end

            if length(intersect(vs[i], vs[j])) > 0
                push!(neighbors, j)
            end
        end
    else
        # Two directed hyperedges Ea and Eb are considered "directed neighbors" (TODO: term?) there is at least one vertex
        # in the head of Ea that is in the tail of Eb
        vs_tail = Set.(collect.(keys.(hg.hg_tail.he2v)))

        vs_head = Set(collect(keys(hg.hg_head.he2v[i])))

        neighbors = Vector{Int}[]
        for j in eachindex(vs_tail)
            if i == j
                continue
            end

            if length(intersect(vs_head, vs_tail[j])) > 0
                push!(neighbors, j)
            end
        end
    end

    sort!(neighbors)
end



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


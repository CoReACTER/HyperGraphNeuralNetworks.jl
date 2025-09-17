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
Graphs.has_vertex(hg::H, i::Int) where {H <: AbstractHGNNHypergraph} = 1 <= i <= hg.num_vertices
Graphs.has_vertex(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph} = 1 <= i <= hg.num_vertices

Graphs.vertices(hg::H) where {H <: AbstractHGNNHypergraph} = 1:(hg.num_vertices)
Graphs.vertices(hg::H) where {H <: AbstractHGNNDiHypergraph} = 1:(hg.num_vertices)


"""
    degree(hg::H) where {H <: AbstractHGNNHypergraph}
    degree(hg::H, i::Int) where {H <: AbstractHGNNHypergraph}
    degree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNHypergraph}
    
    Return the degree of all vertices (if no index is provided) or of a specific group of vertices.
"""
degree(hg::H) where {H <: AbstractHGNNHypergraph} = length.(keys.(hg.v2he))
degree(hg::H, i::Int) where {H <: AbstractHGNNHypergraph} = length(hg.v2he[i])
degree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNHypergraph} = [degree(hg, i) for i in inds]

"""
    degree(hg::H) where {H <: AbstractHGNNDiHypergraph}
    degree(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph}
    degree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNDiHypergraph}

    Return the degree of all vertices (if no index is provided) or of a specific group of indices
    For directed hypergraphs, the total degree is the sum of the incoming degree (see `indegree`) and the
    outgoing degree (see `outdegree`).
"""
degree(hg::H) where {H <: AbstractHGNNDiHypergraph} = indegree(hg) .+ outdegree(hg)
degree(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph} = indegree(hg, i) + outdegree(hg, i)
degree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNDiHypergraph} = [degree(hg, i) for i in inds]

"""
    indegree(hg::H) where {H <: AbstractHGNNDiHypergraph}
    indegree(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph}
    indegree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNDiHypergraph}

    Return the incoming degree of all vertices (if no index is provided) or of a specific group of indices for a
    directed hypergraph. The incoming degree is the number of directed hyperedges containing a vertex in the head.
"""
indegree(hg::H) where {H <: AbstractHGNNDiHypergraph} = length.(keys.(hg.hg_head.v2he))
indegree(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph} = length(hg.hg_head.v2he[i])
indegree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNDiHypergraph} = [indegree(hg, i) for i in inds]

"""
    outdegree(hg::H) where {H <: AbstractHGNNDiHypergraph}
    outdegree(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph}
    outdegree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNDiHypergraph}

    Return the outgoing degree of all vertices (if no index is provided) or of a specific group of indices for a
    directed hypergraph. The outgoing degree is the number of directed hyperedges containing a vertex in the tail.
"""
outdegree(hg::H) where {H <: AbstractHGNNDiHypergraph} = length.(keys.(hg.hg_tail.v2he))
outdegree(hg::H, i::Int) where {H <: AbstractHGNNDiHypergraph} = length(hg.hg_tail.v2he[i])
outdegree(hg::H, inds::AbstractVector{Int}) where {H <: AbstractHGNNDiHypergraph} = [indegree(hg, i) for i in inds]

"""
    all_neighbors(hg::H) where {H <: AbstractHGNNHypergraph}

    Collect all neighbors for all vertices in an undirected hypergraph.
"""
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

"""
    all_neighbors(hg::H, i::Int) where {H <: AbstractHGNNHypergraph}

    Returns the neighbors of vertex `i` in undirected hypergraph `hg`.
"""
function all_neighbors(hg::H, i::Int) where {H <: AbstractHGNNHypergraph}

    hes = collect(keys(hg.v2he[i]))

    # There's probably a more efficient way to do this
    neighbors = Vector{Int}[]
    for he in hes
        cat(neighbors, collect(keys(hg.he2v[he])))
    end
    
    sort!(unique!(neighbors))
end

"""
    all_neighbors(hg::H; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    Returns all neighbors for all vertices in a directed hypergraph. The set of all neighbors for a vertex is the union
    of the set of all incoming neighbors and outgoing neighbors. Note that the definition of `incoming` and `outgoing`
    neighbor depends on if `same_side` is true or false; see below.

    Args:
        hg::H where {H <: AbstractHGNNDiHypergraph} : Hypergraph
        same_side::Bool : If true, return the neighbors within the same side of a hyperedge; i.e., if vertex
        `i` and vertex `j` are both in the tail of hyperedge `e`, they are neighbors. If false, instead return
        the neighbors on the opposite side of the hyperedge; i.e., if vertex `i` is in the tail of `e` and vertex
        `j` is in the head of `e`, then `i` and `j` are neighbors. Default is false.

"""
function all_neighbors(hg::H; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    incoming = in_neighbors(hg; same_side=same_side)
    outgoing = out_neighbors(hg; same_side=same_side)

    sort!(unique!([incoming; outgoing]))

end

"""
    all_neighbors(hg::H, i::Int; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    Returns the neighbors for vertex `i` in a directed hypergraph `hg`. The set of all neighbors for a vertex is the
    union of the set of all incoming neighbors and outgoing neighbors. Note that the definition of `incoming` and
    `outgoing` neighbor depends on if `same_side` is true or false; see below.

    Args:
        hg::H where {H <: AbstractHGNNDiHypergraph} : Hypergraph
        i::Int : Vertex index
        same_side::Bool : If true, return the neighbors within the same side of a hyperedge; i.e., if vertex
        `i` and vertex `j` are both in the tail of hyperedge `e`, they are neighbors. If false, instead return
        the neighbors on the opposite side of the hyperedge; i.e., if vertex `i` is in the tail of `e` and vertex
        `j` is in the head of `e`, then `i` and `j` are neighbors. Default is false.

"""
function all_neighbors(hg::H, i::Int; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    incoming = in_neighbors(hg, i; same_side=same_side)
    outgoing = out_neighbors(hg, i; same_side=same_side)

    sort!(unique!([incoming; outgoing]))

end

"""
    in_neighbors(hg::H; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    Return the incoming neighbors for a directed hypergraph `hg`. Note that the definition of `incoming` depends on
    if `same_side` is true or false; see below.

    Args:
        hg::H where {H <: AbstractHGNNDiHypergraph} : Hypergraph
        same_side::Bool : If true, return the neighbors that share a hyperedge head with each vertex; i.e., if vertex
        `i` and vertex `j` are both in the head of some hyperedge `e`, they are incoming neighbors. If false, instead
        return the neighbors on the opposite side of the hyperedge; i.e., if vertex `i` is in the head of `e` and vertex
        `j` is in the tail of `e`, then `j` is an incoming neighbor to `i`. Default is false.
"""
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

"""
    in_neighbors(hg::H, i::Int; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    Return the incoming neighbors for a vertex `i` of a directed hypergraph `hg`. Note that the definition of
    `incoming` depends on if `same_side` is true or false; see below.

    Args:
        hg::H where {H <: AbstractHGNNDiHypergraph} : Hypergraph
        i::Int : Vertex index
        same_side::Bool : If true, return the neighbors that share a hyperedge head with each vertex; i.e., if vertex
        `i` and vertex `j` are both in the head of some hyperedge `e`, they are incoming neighbors. If false, instead
        return the neighbors on the opposite side of the hyperedge; i.e., if vertex `i` is in the head of `e` and vertex
        `j` is in the tail of `e`, then `j` is an incoming neighbor to `i`. Default is false.
"""
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

"""
    out_neighbors(hg::H; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    Return the outgoing neighbors for a directed hypergraph `hg`. Note that the definition of `outgoing` depends on
    if `same_side` is true or false; see below.

    Args:
        hg::H where {H <: AbstractHGNNDiHypergraph} : Hypergraph
        same_side::Bool : If true, return the neighbors that share a hyperedge tail with each vertex; i.e., if vertex
        `i` and vertex `j` are both in the tail of some hyperedge `e`, they are outgoing neighbors. If false, instead
        return the neighbors on the opposite side of the hyperedge; i.e., if vertex `i` is in the tail of `e` and vertex
        `j` is in the head of `e`, then `j` is an outgoing neighbor to `i`. Default is false.
"""
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

"""
    out_neighbors(hg::H, i::Int; same_side::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    Return the outgoing neighbors for a vertex `i` of a directed hypergraph `hg`. Note that the definition of
    `outgoing` depends on if `same_side` is true or false; see below.

    Args:
        hg::H where {H <: AbstractHGNNDiHypergraph} : Hypergraph
        i::Int : Vertex index
        same_side::Bool : If true, return the neighbors that share a hyperedge tail with each vertex; i.e., if vertex
        `i` and vertex `j` are both in the tail of some hyperedge `e`, they are outgoing neighbors. If false, instead
        return the neighbors on the opposite side of the hyperedge; i.e., if vertex `i` is in the tail of `e` and vertex
        `j` is in the head of `e`, then `j` is an outgoing neighbor to `i`. Default is false.
"""
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


"""
    hyperedge_neighbors(hg::H) where {H <: AbstractHGNNHypergraph}

    Returns the neighbors of each hyperedge in an undirected hypergraph `hg`. A hyperedge `e` is neighbors with a
    hyperedge `f` if there is at least one vertex contained in both `e` and `f`.

"""
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

"""
    hyperedge_neighbors(hg::H, i::Int) where {H <: AbstractHGNNHypergraph}

    Returns the neighbors of hyperedge `i` in an undirected hypergraph `hg`. A hyperedge `i` is neighbors with a
    hyperedge `e` if there is at least one vertex contained in both `i` and `e`.

"""
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

"""
    hyperedge_neighbors(hg::H; directed::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    Returns the neighbors of each hyperedge in directed hypergraph `hg`. If `directed` is true (default false), then
    hyperedge `i` is neighbors with hyperedge `j` if and only if there is at least one vertex in the head of `i` that
    is also in the tail of `j`. Otherwise, two hyperedges are neighbors if they share any vertices, irrespective of
    whether those vertices are in the hyperedges' heads or tails.
"""
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

"""
    hyperedge_neighbors(hg::H, i::Int; directed::Bool=false) where {H <: AbstractHGNNDiHypergraph}

    Returns the neighbors of hyperedge `i` in directed hypergraph `hg`. If `directed` is true (default false), then
    hyperedge `i` is neighbors with hyperedge `j` if and only if there is at least one vertex in the head of `i` that
    is also in the tail of `j`. Otherwise, two hyperedges are neighbors if they share any vertices, irrespective of
    whether those vertices are in the hyperedges' heads or tails.
"""
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

isolated_vertices(hg::H) where {H <: AbstractHGNNHypergraph} = [i for i in 1:nhv(hg) if length(hg.v2he[i]) == 0]

isolated_vertices(hg::H) where {H <: AbstractHGNNDiHypergraph} = [
    i for i in 1:nhv(hg) if length(hg.hg_tail.v2he[i]) == 0 && length(hg.hg_head.v2he[i]) == 0
]

"""
    incidence_matrix(hg::H) where {H <: AbstractSimpleHypergraph}

    Convert the matrix representation of an undirected hypergraph `hg` to an incidence matrix. The (i,j)-th element of
    the incidence matrix `M` is 1 if vertex `i` is in hyperedge `j` and 0 otherwise.
"""
function incidence_matrix(hg::H) where {H <: AbstractSimpleHypergraph}
    M = Matrix(hg)
    M[M .!== nothing] .= 1
    M[M .=== nothing] .= 0

    M
end

"""
    incidence_matrix(hg::H) where {H <: AbstractDirectedHypergraph}

    Convert the matrix representation of an undirected hypergraph `hg` into two incidence matrices, `Mt` (the tail
    incidence matrix) and `Mh` (the head incidence matrix). The (i,j)-th element of `Mt` is 1 if vertex `i` is in the
    tail of hyperedge `j` and 0 otherwise. Likewise, `Mh`[i, j] is 1 if `i` is in the head of `j` and 0 otherwise
"""
function incidence_matrix(hg::H) where {H <: AbstractDirectedHypergraph}
    Mt = incidence_matrix(hg.hg_tail)
    Mh = incidence_matrix(hg.hg_head)

    Mt, Mh
end

"""
    complex_incidence_matrix(hg::H) where {H <: AbstractDirectedHypergraph}

    Convert the matrix representation of an undirected hypergraph `hg` into a single complex-valued incidence matrix
    `M`. The (i,j)-th element of `M` is 1 if vertex `i` is in the head of hyperedge `j`, -im if `i` is in the tail of
    `j`, and 0 otherwise. If there are any hyperedges where any vertex is in both the tail and the head, an error is
    thrown.

    Reference:
        Fiorini, S., Coniglio, S., Ciavotta, M., Del Bue, A., Let There be Direction in Hypergraph Neural Networks.
        Transactions on Machine Learning Research, 2024.
"""
function complex_incidence_matrix(hg::H) where {H <: AbstractDirectedHypergraph}
    Mt, Mh = incidence_matrix(hg)

    M = convert(Complex{typeof(Mh)}, Mh) .- (im .* convert(Complex{typeof(Mt)}, Mt))

    # Means that there's overlap between head and hg_tail
    if any(abs.(M) .> 1)
        throw("Cannot have vertices in the tail and the head of the same directed hyperedge!")
    end

    M
end

"""
    vertex_weight_matrix(hg::H; weighting_function::Function=sum) where {H <: AbstractSimpleHypergraph}

    Return an NxN diagonal weight matrix for the undirected hypergraph `hg`, where N is the number of vertices in `hg`.
    Because SimpleHypergraphs hypergraphs can have different weights for each vertex-hyperedge pair, the weight of a
    vertex is ambiguous. The user can specify a `weighting_function` (default is `sum`) that operates on each row of
    the hypergraph weighted incidence matrix.
"""
function vertex_weight_matrix(hg::H; weighting_function::Function=sum) where {H <: AbstractSimpleHypergraph}
    # Weight matrix is the diagonal matrix of vertex weights
    weights = [weighting_function(hg[i,:]) for i in 1:nhv(hg)]
    
    Diagonal(weights)
end

"""
    hyperedge_weight_matrix(hg::H; weighting_function::Function=sum) where {H <: AbstractSimpleHypergraph}

    Return an NxN diagonal weight matrix for the undirected hypergraph `hg`, where N is the number of hyperedges in
    `hg`. Because SimpleHypergraphs hypergraphs can have different weights for each vertex-hyperedge pair, the weight
    of a hyperedge is ambiguous. The user can specify a `weighting_function` (default is `sum`) that operates on each
    column of the hypergraph weighted incidence matrix.
"""
function hyperedge_weight_matrix(hg::H; weighting_function::Function=sum) where {H <: AbstractSimpleHypergraph}
    # Weight matrix is the diagonal matrix of hyperedge weights
    weights = [weighting_function(hg[:,i]) for i in 1:nhe(hg)]
    
    Diagonal(weights)
end

"""
    vertex_weight_matrix(hg::H; weighting_function::Function=sum) where {H <: AbstractDirectedHypergraph}

    Return two NxN diagonal weight matrices for the directed hypergraph `hg`, where N is the number of vertices in
    `hg`. One matrix is based on the hyperedge tails that each vertex is included in; the other is based on the
    hyperedge heads that each vertex is included in. Because SimpleHypergraphs hypergraphs can have different weights
    for each vertex-hyperedge pair, the weight of a vertex is ambiguous. The user can specify a `weighting_function`
    (default is `sum`) that operates on each row of the hypergraph tail/head weighted incidence matrix.
"""
function vertex_weight_matrix(hg::H; weighting_function::Function=sum) where {H <: AbstractDirectedHypergraph}
    Wt = vertex_weight_matrix(hg.hg_tail; weighting_function=weighting_function)
    Wh = vertex_weight_matrix(hg.hg_head; weighting_function=weighting_function)

    (Wt, Wh)
end

"""
    hyperedge_weight_matrix(h::H; weighting_function::Function=sum) where {H <: AbstractDirectedHypergraph}

    Return two NxN diagonal weight matrices for the directed hypergraph `hg`, where N is the number of hyperedges in
    `hg`. One matrix is based on the hyperedge tails, and the other is based on the hyperedge heads. Because
    SimpleHypergraphs hypergraphs can have different weights for each vertex-hyperedge pair, the weight of a hyperedge
    is ambiguous. The user can specify a `weighting_function` (default is `sum`) that operates on each row of the
    hypergraph tail/head weighted incidence matrix.
"""
function hyperedge_weight_matrix(hg::H; weighting_function::Function=sum) where {H <: AbstractDirectedHypergraph}
    Wt = hyperedge_weight_matrix(hg.hg_tail; weighting_function=weighting_function)
    Wh = hyperedge_weight_matrix(hg.hg_head; weighting_function=weighting_function)

    (Wt, Wh)
end

vertex_degree_matrix(hg::H) where {H <: AbstractSimpleHypergraph} = Diagonal(length.(keys.(hg.v2he)))

vertex_degree_matrix(hg::H) where {H <: AbstractDirectedHypergraph} = (
    Diagonal(length.(keys.(hg.hg_tail.v2he))),
    Diagonal(length.(keys.(hg.hg_head.v2he)))
)

hyperedge_degree_matrix(hg::H) where {H <: AbstractSimpleHypergraph} = Diagonal(length.(keys.(hg.he2v)))

hyperedge_degree_matrix(hg::H) where {H <: AbstractDirectedHypergraph} = (
    Diagonal(length.(keys.(hg.hg_tail.he2v))),
    Diagonal(length.(keys.(hg.hg_head.he2v)))
)

"""
    normalized_laplacian(hg::H; weighting_function::Function=sum) where {H <: AbstractSimpleHypergraph}

    Returns the normalized Laplacian for an undirected hypergraph `hg`.

    Because of the ambiguity of defining hyperedge weight in the SimpleHypergraphs formalism, the user can
    specify a `weighting_function` that, for each hyperedge, acts on the associated row of the matrix representation of
    `hg`; the default is `sum`.

    Ln = I - Qn,
    Qn = Dv^(-1/2) A W De^(-1) A* Dv^(-1/2),

    where
        Ln = normalized signed Laplacian (MxM)
        I = identity matrix (MxM)
        Qn = normalized Laplacian (MxM)
        Dv = vertex degree matrix (MxM)
        De = hyperedge degree matrix (NxN)
        A = incidence matrix (MxN)
        A* = transpose of A (NxM)
        W = hyperedge weight matrix (NxN)

        M = # vertices in `hg`
        N = # hyperedges in `hg`

    Reference:
        Fiorini, S., Coniglio, S., Ciavotta, M., Del Bue, A., Let There be Direction in Hypergraph Neural Networks.
        Transactions on Machine Learning Research, 2024.
"""
function normalized_laplacian(hg::H; weighting_function::Function=sum) where {H <: AbstractSimpleHypergraph}
    Dv = vertex_degree_matrix(hg)
    De = hyperedge_degree_matrix(hg)
    W = hyperedge_weight_matrix(hg; weighting_function=weighting_function)
    A = incidence_matrix(hg)

    # From "Let There be Direction in Hypergraph Neural Networks" by Fiorini et al.
    # https://openreview.net/forum?id=h48Ri6pmvi
    I - Dv^(-1/2) * A * W * (inv(De)) * transpose(A) * Dv^(-1/2)
end

_matrix_avg(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where {T <: Real} = (a .+ b) ./ 2

"""
    normalized_laplacian(
        h::H;
        weighting_function::Function=sum,
        combining_function::Function=_matrix_avg
    )

    Returns the normalized Laplacian for an undirected hypergraph `hg`.

    Because of the ambiguity of defining hyperedge weight in the SimpleHypergraphs formalism, the user can
    specify a `weighting_function` that, for each hyperedge, acts on the associated row of the matrix representation of
    `hg`; the default is `sum`. The user can also specify a `combining_function` for how the hyperedge tail and head
    weights should be combined to a single matrix; the default is to average the two matrices.

    Ln = I - Qn,
    Qn = Dv^(-1/2) A W De^(-1) A* Dv^(-1/2),

    where
        Ln = normalized signed Laplacian (MxM)
        I = identity matrix (MxM)
        Qn = normalized Laplacian (MxM)
        Dv = vertex degree matrix (MxM) <-- obtained by summing the tail and head degree matrices
        De = hyperedge degree matrix (NxN) <-- obtained by summing the tail and head degree matrices
        A = incidence matrix (MxN)
        A* = conjugate transpose of A (NxM)
        W = hyperedge weight matrix (NxN) <-- obtained by applying `combining_function` (default: averaging) to the
            tail and head weight matrices

        M = # vertices in `hg`
        N = # hyperedges in `hg`

    Reference:
        Fiorini, S., Coniglio, S., Ciavotta, M., Del Bue, A., Let There be Direction in Hypergraph Neural Networks.
        Transactions on Machine Learning Research, 2024.
"""
function normalized_laplacian(
    h::H;
    weighting_function::Function=sum,
    combining_function::Function=_matrix_avg
) where {H <: AbstractDirectedHypergraph}
    V = vertex_degree_matrix(h)
    Dv = V[1] .+ V[2]
    E = hyperedge_degree_matrix(h)
    De = E[1] .+ E[2]
    W = combining_function(
        hyperedge_weight_matrix(h; weighting_function=weighting_function)...
    )
    A = complex_incidence_matrix(h)

    # From "Let There be Direction in Hypergraph Neural Networks" by Fiorini et al.
    # https://openreview.net/forum?id=h48Ri6pmvi
    I - Dv^(-1/2) * A * W * (inv(De)) * A' * Dv^(-1/2)
end

"""
    hypergraph_ids(hg::H) where {H <: AbstractHGNNHypergraph}
    hypergraph_ids(hg::H) where {H <: AbstractHGNNDiHypergraph}

    Returns a vector containing the graph membership of each vertex in the hypergraph `hg`.
"""
function hypergraph_ids(hg::H) where {H <: AbstractHGNNHypergraph}
    if isnothing(hg.hypergraph_ids)
        gi = ones(Int, hg.num_vertices)
    else
        gi = hg.hypergraph_ids
    end
end

function hypergraph_ids(hg::H) where {H <: AbstractHGNNDiHypergraph}
    if isnothing(hg.hypergraph_ids)
        gi = ones(Int, hg.num_vertices)
    else
        gi = hg.hypergraph_ids
    end
end

# TODO: be consistent in handling; see transform.jl
"""
    has_self_loops(_::H) where {H <: AbstractHGNNHypergraph}
    has_self_loops(hg::H) where {H <: AbstractHGNNDiHypergraph}

    Does the hypergraph contain self-loops? For an undirected hypergraph, this is defined to always be `false`. For a
    directed hypergraph, this function checks the intersection between the tail and the head of each hyperedge in `hg`.
    If any intersections are nonempty, then the dihypergraph has a self-loop.
"""
has_self_loops(_::H) where {H <: AbstractHGNNHypergraph} = false

function has_self_loops(hg::H) where {H <: AbstractHGNNDiHypergraph}
    vs_tail = Set.(collect.(keys.(hg.hg_tail.he2v)))
    vs_head = Set.(collect.(keys.(hg.hg_head.he2v)))

    any([length(intersect(vs_tail[i], vs_head[i])) for i in eachindex(vs_tail)])
end

"""
    has_multi_hyperedges(hg::H) where {H <: AbstractHGNNHypergraph}
    has_multi_hyperedges(hg::H) where {H <: AbstractHGNNDiHypergraph}

    Checks if there are any hyperedges with multiplicity greater than 1, i.e., if there are two or more hyperedges
    containing identical vertices. For directed hyperedges, both the tail vertices and the head vertices have to be
    identical to be considered duplicates.
"""
function has_multi_hyperedges(hg::H) where {H <: AbstractHGNNHypergraph}
    vs = sort!.(collect.(keys.(hg.he2v)))

    length(Set(vs)) < hg.num_hyperedges
end

function has_multi_hyperedges(hg::H) where {H <: AbstractHGNNDiHypergraph}
    vs = [
        (
            sort!(collect(keys(hg.hg_tail.he2v[i]))),
            sort!(collect(keys(hg.hg_head.he2v[i]))),
        )
        for i in eachindex(hg.hg_tail.he2v)
    ]

    length(Set(vs)) < hg.num_hyperedges
end

# ??? can I do khop_adj? does the trick of nth nearest neighbors being related to exponentiating the adjacency matrix work here?

"""
    add_self_loops(
        hg::HGNNHypergraph{T, D};
        add_repeated_hyperedge::Bool = false
    ) where {T<:Real, D<:AbstractDict{Int,T}}

    Add self-loops (hyperedges containing a single vertex) to an undirected hypergraph. If `add_repeated_hyperedge` is
    true (default is false), then new self-loops will be added, even when a self-loop already exists for some vertex.

    NOTE: this function will throw an AssertionError if hg.hedata is not empty
"""
function add_self_loops(hg::HGNNHypergraph{T, D}; add_repeated_hyperedge::Bool = false) where {T<:Real, D<:AbstractDict{Int,T}}
    
    @assert isempty(hg.hedata)
    
    vertices = [1:hg.num_vertices]
    v2he = deepcopy(hg.v2he)    
    he2v = deepcopy(hg.he2v)

    all_he_keys = collect.(keys.(he2v))
    if add_repeated_hyperedge
        for v in vertices
            push!(he2v, D(v => convert(T, 1)))
            v2he[v][length(he2v)] = convert(T, 1)
        end
    else
        for v in vertices
            if !([v] in all_he_keys)
                push!(he2v, D(v => convert(T, 1)))
                v2he[v][length(he2v)] = convert(T, 1)
            end
        end
    end

    return HGNNHypergraph(
        v2he,
        he2v,
        hg.num_vertices,
        length(he2v),
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        hg.vdata,
        DataStore(),
        hg.hgdata
    )
end

"""
    add_self_loops(
        hg::HGNNDiHypergraph{T, D};
        add_repeated_hyperedge::Bool = false
    ) where {T<:Real, D<:AbstractDict{Int,T}}

    Add self-loops (hyperedges with the tail and the head both containing only a single vertex `v`) to a directed
    hypergraph. If `add_repeated_hyperedge` is true (default is false), then new self-loops will be added, even when
    a self-loop already exists for some vertex.

    NOTE: this function will throw an AssertionError if hg.hedata is not empty
"""
function add_self_loops(hg::HGNNDiHypergraph{T, D}; add_repeated_hyperedge::Bool = false) where {T<:Real, D<:AbstractDict{Int,T}}

    @assert isempty(hg.hedata)

    vertices = [1:hg.num_vertices]
    v2he_tail = deepcopy(hg.hg_tail.v2he)
    v2he_head = deepcopy(hg.hg_head.v2he)
    he2v_tail = deepcopy(hg.hg_tail.he2v)
    he2v_head = deepcoyp(hg.hg_head.he2v)

    all_he_keys = collect(zip(collect.(keys.(he2v_tail)), collect.(keys.(he2v_head))))
    if add_repeated_hyperedge
        for v in vertices
            push!(he2v_tail, D(v => convert(T, 1)))
            push!(he2v_head, D(v => convert(T, 1)))

            v2he_tail[v][length(he2v_tail)] = convert(T, 1)
            v2he_head[v][length(he2v_head)] = convert(T, 1)
        end
    else
        for v in vertices
            if !(([v], [v]) in all_he_keys)
                push!(he2v_tail, D(v => convert(T, 1)))
                push!(he2v_head, D(v => convert(T, 1)))
    
                v2he_tail[v][length(he2v_tail)] = convert(T, 1)
                v2he_head[v][length(he2v_head)] = convert(T, 1)
            end
        end
    end

    return HGNNDiHypergraph(
        Hypergraph(v2he_tail, he2v_tail, Vector{Nothing}(nothing, length(v2he_tail)), Vector{Nothing}(nothing, length(he2v_tail))),
        Hypergraph(v2he_head, he2v_head, Vector{Nothing}(nothing, length(v2he_head)), Vector{Nothing}(nothing, length(he2v_head))),
        hg.num_vertices,
        length(he2v_tail),
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        hg.vdata,
        DataStore(),
        hg.hgdata
    )
end

"""
    remove_self_loops(hg::HGNNHypergraph{T, D}) where {T<:Real, D<:AbstractDict{Int,T}}

    Remove self-loops (hyperedges containing only one vertex) from an undirected hypergraph `hg`.
"""
function remove_self_loops(hg::HGNNHypergraph{T, D}) where {T<:Real, D<:AbstractDict{Int,T}}
    @assert isempty(hg.hedata)
    
    v2he = deepcopy(hg.v2he)    
    he2v = deepcopy(hg.he2v)

    to_remove = Int[]
    for (i, he) in enumerate(he2v)
        # Hyperedge only goes from a -> a
        if length(he) == 1
            push!(to_remove, i)
            delete!(v2he[collect(keys(he))[1]], i)
        end
    end

    he2v = he2v[Not(to_remove)]

    return HGNNHypergraph(
        v2he,
        he2v,
        hg.num_vertices,
        length(he2v),
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        hg.vdata,
        DataStore(),
        hg.hgdata
    )
end

"""
    remove_self_loops(hg::HGNNDiHypergraph{T, D}) where {T<:Real, D<:AbstractDict{Int,T}}

    Remove self-loops (hyperedges where the tail and head both contain only a single, shared vertex `v`) from a
    directed hypergraph `hg`.
"""
function remove_self_loops(hg::HGNNDiHypergraph{T, D}) where {T<:Real, D<:AbstractDict{Int,T}}
    @assert isempty(hg.hedata)

    v2he_tail = deepcopy(hg.hg_tail.v2he)
    v2he_head = deepcopy(hg.hg_head.v2he)
    he2v_tail = deepcopy(hg.hg_tail.he2v)
    he2v_head = deepcoyp(hg.hg_head.he2v)

    to_remove = Int[]
    for (i, (he_tail, he_head)) in enumerate(zip(he2v_tail, he2v_head))
        # Hyperedge only goes from a -> a
        if length(he_tail) == 1 && sort(collect(keys(he_tail))) == sort(collect(keys(he_head)))
            vertex = collect(keys(he_tail))[1]
            push!(to_remove, i)
            delete!(v2he_tail[vertex], i)
            delete!(v2he_head[vertex], i)
        end
    end

    he2v_tail = he2v_tail[Not(to_remove)]
    he2v_head = he2v_head[Not(to_remove)]

    return HGNNDiHypergraph(
        Hypergraph(v2he_tail, he2v_tail, Vector{Nothing}(nothing, length(v2he_tail)), Vector{Nothing}(nothing, length(he2v_tail))),
        Hypergraph(v2he_head, he2v_head, Vector{Nothing}(nothing, length(v2he_head)), Vector{Nothing}(nothing, length(he2v_head))),
        hg.num_vertices,
        length(he2v_tail),
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        hg.vdata,
        DataStore(),
        hg.hgdata
    )
end

"""
    remove_multi_hyperedges(hg::HGNNHypergraph)

    Remove duplicate hyperedges (hyperedges containing identical vertices) from an undirected hypergraph `hg`.
"""
function remove_multi_hyperedges(hg::HGNNHypergraph)
    unique_vs = Set{Set{Int}}()

    v2he = deepcopy(hg.v2he)

    # Note: this function (arbitrarily) keeps the lowest-index instance of a particular hyperedge
    mask_to_keep = trues(nhe(hg))
    for (i, he) in enumerate(hg.he2v)
        vs = Set(collect(keys(he)))
        if vs in unique_vs
            mask_to_keep[i] = false
            for v in vs
                delete!(v2he[v], i)
            end
        else
            push!(unique_vs, vs)
        end
    end

    he2v = hg.he2v[mask_to_keep]

    hedata = getobs(hg.hedata, mask_to_keep)

    return HGNNHypergraph(
        v2he,
        he2v,
        hg.num_vertices,
        length(he2v),
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        hg.vdata,
        hedata,
        hg.hgdata
    )
end

"""
    remove_multi_hyperedges(hg::HGNNHypergraph)

    Remove duplicate hyperedges, i.e., hyperedges ei = (ti, hi), ej = (tj, hj), where t represents a hyperedge tail, h
    represents a hyperedge head, and ti = tj and hi = hj, from a directed hypergraph `hg`.
"""
function remove_multi_hyperedges(hg::HGNNDiHypergraph)
    unique_vs = Set{Tuple{Set{Int}, Set{Int}}}()

    v2he_tail = deepcopy(hg.hg_tail.v2he)
    v2he_head = deepcopy(hg.hg_head.v2he)

    mask_to_keep = trues(nhe(hg))
    for (i, (he_tail, he_head)) in enumerate(zip(hg.hg_tail.he2v, hg.hg_head.he2v))
        vs_tail = Set(collect(keys(he_tail)))
        vs_head = Set(collect(keys(he_head)))
        if (vs_tail, vs_head) in unique_vs
            mask_to_keep[i] = false
            for v in vs_tail
                delete!(v2he_tail[v], i)
            end
            for v in vs_head
                delete!(v2he_head[v], i)
            end
        else
            push!(unique_vs, (vs_tail, vs_head))
        end
    end

    he2v_tail = hg.hg_tail.he2v[mask_to_keep]
    he2v_head = hg.hg_head.he2v[mask_to_keep]

    hedata = getobs(hg.hedata, mask_to_keep)

    HGNNDiHypergraph(
        Hypergraph(v2he_tail, he2v_tail, Vector{Nothing}(nothing, length(v2he_tail)), Vector{Nothing}(nothing, length(he2v_tail))),
        Hypergraph(v2he_head, he2v_head, Vector{Nothing}(nothing, length(v2he_head)), Vector{Nothing}(nothing, length(he2v_head))),
        hg.num_vertices,
        length(he2v_tail),
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        hg.vdata,
        hedata,
        hg.hgdata
    )
end

"""
    to_undirected(hg::HGNNDiHypergraph{T,D}) where {T <: Real, D <: AbstractDict{Int, T}}

    Converts a directed hypergraph into an undirected hypergraph.
    Tail and head hyperedges are combined; that is, for all hyperedges he_orig in
    the directed hypergraph h, all vertices in the head or tail are added to a
    corresponding undirected hyperedge he_new in the undirected hypergraph h'.

    Vertex, hyperedge, and hypergraph features, as well as the hypergraph IDs, are undisturbed.

    Because vertex-hyperedge weights are restricted to real numbers, we cannot
    combine the weights, so we simply set the values to 1.0 if a given vertex
    is in a given hyperedge 

"""
function to_undirected(hg::HGNNDiHypergraph{T,D}) where {T <: Real, D <: AbstractDict{Int, T}}

    incidence = Matrix{Union{T, Nothing}}(nothing, nhv(hg), nhe(hg))

    this_nhe = nhe(hg)

    for row in 1:nhv(hg)
        for column in 1:this_nhe
            tail_val, head_val = h[row, column]
            if tail_val === nothing && head_val === nothing
                incidence[row, column] = nothing
            else
                incidence[row, column] = convert(T, 1.0)
            end
        end
    end

    HGNNHypergraph{T, D}(
        incidence;
        hypergraph_ids=hg.hypergraph_ids,
        vdata=hg.vdata,
        hedata=hg.hedata,
        hgdata=hg.hgdata
    )
end

#TODO: docstrings
function combine_hypergraphs(hg1::HGNNHypergraph{T, D}, hg2::HGNNHypergraph{T, D}) where {T <: Real, D <: AbstractDict{Int, T}}
    num_vertices = hg1.num_vertices + hg2.num_vertices
    num_hyperedges = hg1.num_hyperedges + hg2.num_hyperedges
    num_hypergraphs = hg1.num_hypergraphs + hg2.num_hypergraphs
    
    v_increment = hg1.num_vertices
    he_increment = hg1.num_hyperedges

    v2he_new = deepcopy(hg1.v2he)
    for v in hg2.v2he
        newv = D()
        for (he, val) in v
            newv[he + he_increment] = val
        end
        push!(v2he_new, newv)
    end

    he2v_new = deepcopy(hg1.he2v)
    for he in hg2.he2v
        newhe = D()
        for (v, val) in he
            newhe[v + v_increment] = val
        end
        push!(he2v_new, newhe)
    end

    hgid_increment = hg1.num_hypergraphs
    if isnothing(hg1.hypergraph_ids) && isnothing(hg2.hypergraph_ids)
        hypergraph_ids = ones(num_vertices)
    elseif isnothing(hg1.hypergraph_ids)
        hypergraph_ids = cat(ones(hg1.num_vertices), (hg2.hypergraph_ids .+ hgid_increment))
    elseif isnothing(hg2.hypergraph_ids)
        hypergraph_ids = cat(hg1.hypergraph_ids, (ones(hg2.num_vertices) .+ hgid_increment))
    else
        hypergraph_ids = cat(hg1.hypergraph_ids, (hg2.hypergraph_ids .+ hgid_increment))
    end

    HGNNHypergraph(
        v2he_new,
        he2v_new,
        num_vertices,
        num_hyperedges,
        num_hypergraphs,
        hypergraph_ids,
        cat_features(hg1.vdata, hg2.vdata),
        cat_features(hg1.hedata, hg2.hedata),
        cat_features(hg1.hgdata, hg2.hgdata)
    )
end

function combine_hypergraphs(hg1::HGNNHypergraph{T,D}, hgothers::HGNNHypergraph{T,D}...) where {T <: Real, D <: AbstractDict{Int, T}}
    hg = hg1

    for hgo in hgothers
        hg = combine_hypergraphs(hg, hgo)
    end

    hg
end

function combine_hypergraphs(hgs::AbstractVector{HGNNHypergraph{T,D}}) where {T <: Real, D <: AbstractDict{Int, T}}
    num_vs = [hg.num_vertices for hg in hgs]
    num_hes = [hg.num_hyperedges for hg in hgs]
    num_hgs = [hg.num_hypergraphs for hg in hgs]
    
    vsum = cumsum([0; num_vs])[1:(end - 1)]
    hesum = cumsum([0; num_hes])[1:(end - 1)]
    hgsum = cumsum([0; num_hgs])[1:(end - 1)]

    v2hes = [hg.v2he for hg in hgs]
    he2vs = [hg.he2v for hg in hgs]

    v2he = D[]
    he2v = D[]

    for (i, v) in enumerate(v2hes)
        new_v = D()
        for (he, val) in v
            new_v[he + hesum[i]] = val
        end
        push!(v2he, new_v)
    end

    for (i, he) in enumerate(he2vs)
        new_he = D()
        for (v, val) in he
            new_he[v + vsum[i]] = val
        end
        push!(he2v, new_he)
    end

    function obtain_hg_inds(hg)
        hg.hypergraph_ids === nothing ? ones(hg.num_vertices) : hg.hypergraph_ids
    end
    
    hypergraph_id_vecs = obtain_hg_inds.(hgs)
    hypergraph_ids = cat_features([nhg .+ inc for (nhg, inc) in zip(hgsum, hypergraph_id_vecs)])

    HGNNHypergraph(
        v2he,
        he2v,
        sum(num_vs),
        sum(num_hes),
        sum(num_hgs),
        hypergraph_ids,
        cat_features([hg.vdata for hg in hgs]),
        cat_features([hg.hedata for hg in hgs]),
        cat_features([hg.hgdata for hg in hgs])
    )
end

function combine_hypergraphs(hg1::HGNNDiHypergraph{T, D}, hg2::HGNNDiHypergraph{T, D}) where {T <: Real, D <: AbstractDict{Int, T}}
    num_vertices = hg1.num_vertices + hg2.num_vertices
    num_hyperedges = hg1.num_hyperedges + hg2.num_hyperedges
    num_hypergraphs = hg1.num_hypergraphs + hg2.num_hypergraphs
    
    v_increment = hg1.num_vertices
    he_increment = hg1.num_hyperedges

    v2he_tail = deepcopy(hg1.hg_tail.v2he)
    for v in hg2.hg_tail.v2he
        newv = D()
        for (he, val) in v
            newv[he + he_increment] = val
        end
        push!(v2he_tail, newv)
    end

    v2he_head = deepcopy(hg1.hg_head.v2he)
    for v in hg2.hg_head.v2he
        newv = D()
        for (he, val) in v
            newv[he + he_increment] = val
        end
        push!(v2he_head, newv)
    end

    he2v_tail = deepcopy(hg1.hg_tail.he2v)
    for he in hg2.hg_tail.he2v
        newhe = D()
        for (v, val) in he
            newhe[v + v_increment] = val
        end
        push!(he2v_tail, newhe)
    end

    he2v_head = deepcopy(hg1.hg_head.he2v)
    for he in hg2.hg_head.he2v
        newhe = D()
        for (v, val) in he
            newhe[v + v_increment] = val
        end
        push!(he2v_head, newhe)
    end

    hgid_increment = hg1.num_hypergraphs
    if isnothing(hg1.hypergraph_ids) && isnothing(hg1.hypergraph_ids)
        hypergraph_ids = ones(num_vertices)
    elseif isnothing(hg1.hypergraph_ids)
        hypergraph_ids = cat(ones(hg1.num_vertices), (hg2.hypergraph_ids .+ hgid_increment))
    elseif isnothing(hg2.hypergraph_ids)
        hypergraph_ids = cat(hg1.hypergraph_ids, (ones(hg2.num_vertices) .+ hgid_increment))
    else
        hypergraph_ids = cat(hg1.hypergraph_ids, (hg2.hypergraph_ids .+ hgid_increment))
    end

    HGNNDiHypergraph(
        Hypergraph(v2he_tail, he2v_tail, Vector{Nothing}(undef, num_vertices), Vector{Nothing}(undef, num_hyperedges)),
        Hypergraph(v2he_head, he2v_head, Vector{Nothing}(undef, num_vertices), Vector{Nothing}(undef, num_hyperedges)),
        num_vertices,
        num_hyperedges,
        num_hypergraphs,
        hypergraph_ids,
        cat_features(hg1.vdata, hg2.vdata),
        cat_features(hg1.hedata, hg2.hedata),
        cat_features(hg1.hgdata, hg2.hgdata)
    )
end

function combine_hypergraphs(hg1::HGNNDiHypergraph{T,D}, hgothers::HGNNDiHypergraph{T,D}...) where {T <: Real, D <: AbstractDict{Int, T}}
    hg = hg1

    for hgo in hgothers
        hg = combine_hypergraphs(hg, hgo)
    end

    hg
end

function combine_hypergraphs(hgs::AbstractVector{HGNNDiHypergraph{T,D}}) where {T <: Real, D <: AbstractDict{Int, T}}
    num_vs = [hg.num_vertices for hg in hgs]
    num_hes = [hg.num_hyperedges for hg in hgs]
    num_hgs = [hg.num_hypergraphs for hg in hgs]
    
    vsum = cumsum([0; num_vs])[1:(end - 1)]
    hesum = cumsum([0; num_hes])[1:(end - 1)]
    hgsum = cumsum([0; num_hgs])[1:(end - 1)]

    v2he_tails = [hg.hg_tail.v2he for hg in hgs]
    v2he_heads = [hg.hg_head.v2he for hg in hgs]
    he2v_tails = [hg.hg_tail.he2v for hg in hgs]
    he2v_heads = [hg.hg_tail.he2v for hg in hgs]

    v2he_tail = D[]
    v2he_head = D[]
    he2v_tail = D[]
    he2v_head = D[]

    for (i, v) in enumerate(v2he_tails)
        new_v = D()
        for (he, val) in v
            new_v[he + hesum[i]] = val
        end
        push!(v2he_tail, new_v)
    end

    for (i, v) in enumerate(v2he_heads)
        new_v = D()
        for (he, val) in v
            new_v[he + hesum[i]] = val
        end
        push!(v2he_head, new_v)
    end

    for (i, he) in enumerate(he2v_tails)
        new_he = D()
        for (v, val) in he
            new_he[v + vsum[i]] = val
        end
        push!(he2v_tail, new_he)
    end

    for (i, he) in enumerate(he2v_heads)
        new_he = D()
        for (v, val) in he
            new_he[v + vsum[i]] = val
        end
        push!(he2v_head, new_he)
    end

    function obtain_hg_inds(hg)
        hg.hypergraph_ids === nothing ? ones(hg.num_vertices) : hg.hypergraph_ids
    end
    
    hypergraph_id_vecs = obtain_hg_inds.(hgs)
    hypergraph_ids = cat_features([nhg .+ inc for (nhg, inc) in zip(hgsum, hypergraph_id_vecs)])

    HGNNDiHypergraph(
        Hypergraph(v2he_tail, he2v_tail, Vector{Nothing}(undef, num_vertices), Vector{Nothing}(undef, num_hyperedges)),
        Hypergraph(v2he_head, he2v_head, Vector{Nothing}(undef, num_vertices), Vector{Nothing}(undef, num_hyperedges)),
        sum(num_vs),
        sum(num_hes),
        sum(num_hgs),
        hypergraph_ids,
        cat_features([hg.vdata for hg in hgs]),
        cat_features([hg.hedata for hg in hgs]),
        cat_features([hg.hgdata for hg in hgs])
    )
end

function MLUtils.batch(hgs::AbstractVector{HGNNHypergraph{T,D}}) where {T <: Real, D <: AbstractDict{Int, T}}
    combine_hypergraphs(hgs)
end

function MLUtils.batch(hgs::AbstractVector{HGNNDiHypergraph{T,D}}) where {T <: Real, D <: AbstractDict{Int, T}}
    combine_hypergraphs(hgs)
end

get_hypergraph(hg::HGNNHypergraph, i::Int; kws...) = getgraph(hg, [i]; kws...)

function get_hypergraph(hg::HGNNHypergraph, i::AbstractVector{Int}; map_vertices::Bool = false)
    if hg.hypergraph_ids === nothing
        @assert i == [1]

        if map_vertices
            return hg, 1:(hg.num_vertices)
        else
            return hg
        end
    end

    vertex_mask = hg.hypergraph_ids .∈ Ref(i)
    vertices = (1:(hg.num_vertices))[vertex_mask]
    vertex_map = Dict(v => vnew for (vnew, v) in enumerate(vertices))

    hgmap = Dict(i => inew for (inew, i) in enumerate(i))
    hypergraph_ids = [hgmap[i] for i in hg.hypergraph_ids[vertex_mask]]

    he_mask = all.(keys.(hg.he2v) .∈ Ref(vertices))
    hyperedges = (1:(hg.num_hyperedges))[he_mask]
    hyperedge_map = Dict(he => henew for (henew, he) in enumerate(hyperedges))

    he2v = hg.he2v[he_mask]
    for (i, he) in enumerate(he2v)
        new_he = D()
        for (v, val) in he
            new_he[vertex_map[v]] = val
        end
        he2v[i] = new_he
    end

    v2he = hg.v2he[vertex_mask]
    for (i, v) in enumerate(v2he)
        new_v = D()
        for (he, val) in v
            if he_mask[he]
                new_v[hyperedge_map[he]] = val
            end
        end
        v2he[i] = new_v
    end

    vdata = getobs(hg.vdata, vertex_mask)
    hedata = getobs(hg.hedata, he_mask)
    hgdata = getobs(hg.hgdata, i)

    num_vertices = length(vertices)
    num_hyperedges = length(hyperedges)
    num_hypergraphs = length(i)

    HGNNHypergraph(
        v2he, he2v,
        num_vertices, num_hyperedges, num_hypergraphs,
        hypergraph_ids,
        vdata, hedata, hgdata
    )

    if map_vertices
        return hg_new, vertices
    else
        return hg_new
    end
end

function MLUtils.unbatch(hg::HGNNHypergraph)
    return [get_hypergraph(hg, i) for i in 1:(hg.num_hypergraphs)]
end

get_hypergraph(hg::HGNNDiHypergraph, i::Int; kws...) = getgraph(hg, [i]; kws...)

function get_hypergraph(hg::HGNNDiHypergraph, i::AbstractVector{Int}; map_vertices::Bool = false)
    if hg.hypergraph_ids === nothing
        @assert i == [1]

        if map_vertices
            return hg, 1:(hg.num_vertices)
        else
            return hg
        end
    end

    vertex_mask = hg.hypergraph_ids .∈ Ref(i)
    vertices = (1:(hg.num_vertices))[vertex_mask]
    vertex_map = Dict(v => vnew for (vnew, v) in enumerate(vertices))

    hgmap = Dict(i => inew for (inew, i) in enumerate(i))
    hypergraph_ids = [hgmap[i] for i in hg.hypergraph_ids[vertex_mask]]

    he_mask = all.(keys.(hg.hg_tail.he2v) .∈ Ref(vertices)) .* all.(keys.(hg.hg_head.he2v) .∈ Ref(vertices))
    hyperedges = (1:(hg.num_hyperedges))[he_mask]
    hyperedge_map = Dict(he => henew for (henew, he) in enumerate(hyperedges))

    he2v_tail = hg.hg_tail.he2v[he_mask]
    for (i, he) in enumerate(he2v_tail)
        new_he = D()
        for (v, val) in he
            new_he[vertex_map[v]] = val
        end
        he2v_tail[i] = new_he
    end

    he2v_head = hg.hg_head.he2v[he_mask]
    for (i, he) in enumerate(he2v_head)
        new_he = D()
        for (v, val) in he
            new_he[vertex_map[v]] = val
        end
        he2v_head[i] = new_he
    end

    v2he_tail = hg.hg_tail.v2he[vertex_mask]
    for (i, v) in enumerate(v2he_tail)
        new_v = D()
        for (he, val) in v
            if he_mask[he]
                new_v[hyperedge_map[he]] = val
            end
        end
        v2he_tail[i] = new_v
    end

    v2he_head = hg.hg_head.v2he[vertex_mask]
    for (i, v) in enumerate(v2he_head)
        new_v = D()
        for (he, val) in v
            if he_mask[he]
                new_v[hyperedge_map[he]] = val
            end
        end
        v2he_head[i] = new_v
    end

    vdata = getobs(hg.vdata, vertex_mask)
    hedata = getobs(hg.hedata, he_mask)
    hgdata = getobs(hg.hgdata, i)

    num_vertices = length(vertices)
    num_hyperedges = length(hyperedges)
    num_hypergraphs = length(i)

    HGNNDiHypergraph(
        Hypergraph(v2he_tail, he2v_tail, Vector{Nothing}(undef, num_vertices), Vector{Nothing}(undef, num_hyperedges)),
        Hypergraph(v2he_head, he2v_head, Vector{Nothing}(undef, num_vertices), Vector{Nothing}(undef, num_hyperedges)),
        num_vertices, num_hyperedges, num_hypergraphs,
        hypergraph_ids,
        vdata, hedata, hgdata
    )

    if map_vertices
        return hg_new, vertices
    else
        return hg_new
    end
end

function MLUtils.unbatch(hg::HGNNDiHypergraph)
    return [get_hypergraph(hg, i) for i in 1:(hg.num_hypergraphs)]
end


abstract type AbstractNegativeSamplingStrategy end
struct UniformSample <: AbstractNegativeSamplingStrategy end
struct SizedSample <: AbstractNegativeSamplingStrategy end
struct MotifSample <: AbstractNegativeSamplingStrategy end
struct CliqueSample <: AbstractNegativeSamplingStrategy end

function uniform_negative_sample(
    hg::HGNNHypergraph{T, D},
    n::Int,
    rng::AbstractRNG;
    max_trials::Int = 10
) where {T <: Real, D <: AbstractDict{Int, T}}
    vertices = 1:hg.num_vertices
    he2v = Set.(keys.(hg.he2v))

    choices = Set{Set{Int}}()

    # Multiple attempts to generate n negative hyperedges
    for _ in 1:max_trials
        for _ in 1:(n - length(choices))
            size = rand(rng, vertices)
            verts = Set(sample(rng, vertices, size; replace=false))
            if !((verts in he2v) || (verts in choices))
                push!(choices, verts)
            end
        end

        if length(choices) >= n
            break
        end
    end

    base_h = Hypergraph{T, D}(hg.num_vertices, length(choices))
    for (i, choice) in enumerate(choices)
        base_h[collect(choice), i] .= convert(T, 1)
    end

    return HGNNHypergraph(base_h)

end

function uniform_negative_sample(
    hg::HGNNDiHypergraph{T, D},
    n::Int,
    rng::AbstractRNG;
    max_trials::Int = 10
) where {T <: Real, D <: AbstractDict{Int, T}}
    vertices = 1:hg.num_vertices

    he2v_tail = Set.(keys.(hg.hg_tail.he2v))
    he2v_head = Set.(keys.(hg.hg_head.he2v))

    he2v = collect(zip(he2v_tail, he2v_head))

    choices = Tuple{Set{Int}, Set{Int}}[]

    for _ in 1:max_trials
        for _ in 1:(n - length(choices))
            total_size = rand(rng, vertices)
            size_tail = rand(rng, 1:total_size)
            size_head = total_size - size_tail
            
            verts_tail = Set(sample(rng, vertices, size_tail; replace=false))
            verts_head = Set(sample(rng, setdiff(collect(vertices), collect(verts_tail)), size_head; replace=false))
            verts = (verts_tail, verts_head)
            if !(verts in he2v || verts in choices)
                push!(choices, verts)
            end
        end

        if length(choices) >= n
            break
        end
    end

    hg_tail = Hypergraph{T, D}(hg.num_vertices, length(choices))
    hg_head = Hypergraph{T, D}(hg.num_vertices, length(choices))

    for (i, choice) in enumerate(choices)
        hg_tail[collect(choice[1]), i] .= convert(T, 1)
        hg_head[collect(choice[2]), i] .= convert(T, 1)
    end

    return HGNNDiHypergraph(DirectedHypergraph(hg_tail, hg_head))
end


function sized_negative_sample(
    hg::HGNNHypergraph{T, D},
    n::Int,
    rng::AbstractRNG;
    max_trials::Int = 10
) where {T <: Real, D <: AbstractDict{Int, T}}
    vertices = 1:hg.num_vertices

    # Likelihood of hyperedge of size s is based on how often s-sized hyperedges appear in hg
    he2v = Set.(keys.(hg.he2v))
    c = counter(length.(he2v))
    size_dist = FrequencyWeights([c[i] for i in vertices])

    choices = Set{Set{Int}}()

    for _ in 1:max_trials
        for _ in 1:(n - length(choices))
            size = sample(rng, vertices, size_dist)
            verts = Set(sample(rng, vertices, size; replace=false))
            if !((verts in he2v) || (verts in choices))
                push!(choices, verts)
            end
        end

        if length(choices) >= n
            break
        end
    end

    base_h = Hypergraph{T, D}(hg.num_vertices, length(choices))
    for (i, choice) in enumerate(choices)
        base_h[collect(choice), i] .= convert(T, 1)
    end

    return HGNNHypergraph(base_h)

end

function sized_negative_sample(
    hg::HGNNDiHypergraph{T, D},
    n::Int,
    rng::AbstractRNG;
    max_trials::Int = 10
) where {T <: Real, D <: AbstractDict{Int, T}}

    vertices = 1:hg.num_vertices

    he2v_tail = Set.(keys.(hg.hg_tail.he2v))
    he2v_head = Set.(keys.(hg.hg_head.he2v))

    # Likelihood of hyperedge of size s is based on how often s-sized hyperedges appear in hg
    c_total = counter(length.(he2v_tail) .+ length.(he2v_head))
    size_dist = FrequencyWeights([c_total[i] for i in vertices])

    # Tail size distribution follows similar logic
    c_tail = counter(length.(he2v_tail))
    size_dist_tail = FrequencyWeights([c_tail[i] for i in vertices])

    he2v = collect(zip(he2v_tail, he2v_head))

    choices = Tuple{Set{Int}, Set{Int}}[]

    for _ in 1:max_trials
        for _ in 1:(n - length(choices))
            total_size = sample(rng, vertices, size_dist)
            size_tail = sample(rng, 1:total_size, size_dist_tail)
            size_head = total_size - size_tail
            
            verts_tail = Set(sample(rng, vertices, size_tail; replace=false))
            verts_head = Set(sample(rng, setdiff(collect(vertices), collect(verts_tail)), size_head; replace=false))
            verts = (verts_tail, verts_head)
            if !(verts in he2v || verts in choices)
                push!(choices, verts)
            end
        end

        if length(choices) >= n
            break
        end
    end

    hg_tail = Hypergraph{T, D}(hg.num_vertices, length(choices))
    hg_head = Hypergraph{T, D}(hg.num_vertices, length(choices))

    for (i, choice) in enumerate(choices)
        hg_tail[collect(choice[1]), i] .= convert(T, 1)
        hg_head[collect(choice[2]), i] .= convert(T, 1)
    end

    return HGNNDiHypergraph(DirectedHypergraph(hg_tail, hg_head))

end

function motif_negative_sample(
    hg::HGNNHypergraph{T, D},
    n::Int,
    rng::AbstractRNG;
    max_trials::Int = 10
) where {T <: Real, D <: AbstractDict{Int, T}}

    vertices = 1:hg.num_vertices

    # Likelihood of hyperedge of size s is based on how often s-sized hyperedges appear in hg
    he2v = Set.(keys.(hg.he2v))
    c = counter(length.(he2v))
    size_dist = FrequencyWeights([c[i] for i in vertices])

    choices = Set{Set{Int}}()

    adjmat = get_twosection_adjacency_mx(hg)
    g = SimpleGraph(adjmat)
    edges = [Set([src(e), dst(e)]) for e in edges(g)]

    for _ in 1:max_trials
        for _ in 1:(n - length(choices))
            size = sample(rng, vertices, size_dist)
            e0 = rand(rng, edges)
            he = deepcopy(e0)
            while length(he) < size
                e_options = [e for e in edges if length(intersect(he, e)) == 1]
                if length(e_options) == 0
                    break
                end

                e = rand(rng, e_options)
                union!(he, e)
            end
            if length(he) >= size
                push!(choices, he)
            end
        end

        if length(choices) >= n
            break
        end
    end

    base_h = Hypergraph{T, D}(hg.num_vertices, length(choices))
    for (i, choice) in enumerate(choices)
        base_h[collect(choice), i] .= convert(T, 1)
    end

    return HGNNHypergraph(base_h)
end

function motif_negative_sample(
    hg::HGNNDiHypergraph{T, D},
    n::Int,
    rng::AbstractRNG;
    max_trials::Int = 10
) where {T <: Real, D <: AbstractDict{Int, T}}

    vertices = 1:hg.num_vertices

    he2v_tail = Set.(keys.(hg.hg_tail.he2v))
    he2v_head = Set.(keys.(hg.hg_head.he2v))

    # Likelihood of hyperedge of size s is based on how often s-sized hyperedges appear in hg
    c_total = counter(length.(he2v_tail) .+ length.(he2v_head))
    size_dist = FrequencyWeights([c_total[i] for i in vertices])

    he2v = collect(zip(he2v_tail, he2v_head))

    adjmat = get_twosection_adjacency_mx(hg; replace_weights=1)
    g = SimpleDiGraph(adjmat)

    sources = [src(e) for e in edges(g)]
    destinations = [dst(e) for e in edges(g)]

    choices = Tuple{Set{Int}, Set{Int}}[]

    for _ in 1:max_trials
        for _ in 1:(n - length(choices))
            total_size = sample(rng, vertices, size_dist)

            i = rand(rng, 1:ne(g))
            he_tail = Set{Int}(sources[i])
            he_head = Set{Int}(destinations[i])

            while length(he) < total_size
                e_options = [
                    i for i in 1:ne(g) if 
                    (sources[i] ∈ he_tail && destinations[i] ∉ he_head) || (destinations[i] ∈ he_head&& sources[i] ∉ he_tail)
                ]

                if length(e_options) == 0
                    break
                end

                e = rand(rng, e_options)
                push!(he_tail, sources[e])
                push!(he_head, destinations[e])
            end

            if length(he_tail) + length(he_head) >= total_size
                push!(choices, (he_tail, he_head))
            end

        end

        if length(choices) >= n
            break
        end
    end

    hg_tail = Hypergraph{T, D}(hg.num_vertices, length(choices))
    hg_head = Hypergraph{T, D}(hg.num_vertices, length(choices))

    for (i, choice) in enumerate(choices)
        hg_tail[collect(choice[1]), i] .= convert(T, 1)
        hg_head[collect(choice[2]), i] .= convert(T, 1)
    end

    return HGNNDiHypergraph(DirectedHypergraph(hg_tail, hg_head))
end

# TODO: you are here
function clique_negative_sample(
    hg::HGNNHypergraph{T, D},
    n::Int,
    rng::AbstractRNG;
    max_trials::Int = 10
) where {T <: Real, D <: AbstractDict{Int, T}}
end

function clique_negative_sample(
    hg::HGNNDiHypergraph{T, D},
    n::Int,
    rng::AbstractRNG;
    max_trials::Int = 10
) where {T <: Real, D <: AbstractDict{Int, T}}
end


function negative_sample_hyperedge(hg::H, n::Int, rng::AbstractRNG, ::S; max_trials::Int = 10) where {H <: AbstractSimpleHypergraph, S <: AbstractNegativeSamplingStrategy}
    if S <: UniformSample
        return uniform_negative_sample(hg, n, rng; max_trials=max_trials)
    elseif S <: SizedSample
        return sized_negative_sample(hg, n, rng; max_trials=max_trials)
    elseif S <: MotifSample
        return motif_negative_sample(hg, n, rng; max_trials=max_trials)
    elseif S <: CliqueSample
        return clique_negative_sample(hg, n, rng; max_trials=max_trials)
    else
        throw("negative_sample not implemented for strategy of type $S; please call a sampling function directly.")
    end
end

function negative_sample_hyperedge(hg::H, n::Int, rng::AbstractRNG, ::S; max_trials::Int = 10) where {H <: AbstractDirectedHypergraph, S <: AbstractNegativeSamplingStrategy}
    if S <: UniformSample
        return uniform_negative_sample(hg, n, rng; max_trials=max_trials)
    elseif S <: SizedSample
        return sized_negative_sample(hg, n, rng; max_trials=max_trials)
    elseif S <: MotifSample
        return motif_negative_sample(hg, n, rng; max_trials=max_trials)
    elseif S <: CliqueSample
        return clique_negative_sample(hg, n, rng; max_trials=max_trials)
    else
        throw("negative_sample not implemented for strategy of type $S; please call a sampling function directly.")
    end
end

function random_split_vertices()
end

function random_split_hyperedges()
end

function random_split_hypergraphs()
end

# TODO: PageRank diffusion?
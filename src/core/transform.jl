
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

# TODO: does this have the same nice properties re: edge distribution as the undirected case?
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


function clique_negative_sample(
    hg::HGNNHypergraph{T, D},
    n::Int,
    rng::AbstractRNG;
    max_trials::Int = 10
) where {T <: Real, D <: AbstractDict{Int, T}}

    vertices = 1:hg.num_vertices

    # Likelihood of hyperedge of size s is based on how often s-sized hyperedges appear in hg
    he2v = Set.(keys.(hg.he2v))

    choices = Set{Set{Int}}()

    for _ in 1:max_trials
        for _ in 1:(n - length(choices))
            he = rand(rng, he2v)

            elim = rand(rng, he)

            heminus = setdiff(he, elim)

            # TODO: I'm sure there's a more efficient way to implement this
            neighbors = Set{Int}()
            for i in setdiff(Set(vertices), he)
                for j in heminus
                    if any(map(x -> i in x && j in x, he2v))
                        push!(neighbors, i)
                    end
                end
            end

            if length(neighbors) == 0
                continue
            end

            incl = rand(rng, neighbors)

            push!(choices, union(heminus, incl))
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

# TODO: clique only defined for undirected graph
# Is there a way to adapt this for dihypergraphs?
# function clique_negative_sample(
#     hg::HGNNDiHypergraph{T, D},
#     n::Int,
#     rng::AbstractRNG;
#     max_trials::Int = 10
# ) where {T <: Real, D <: AbstractDict{Int, T}}
# end

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
    else
        throw("negative_sample not implemented for strategy of type $S; please call a sampling function directly.")
    end
end

"""
    split_vertices(
        hg::HGNNHypergraph{T,D},
        masks::AbstractVector{BitVector}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_vertices(
        hg::HGNNHypergraph{T,D},
        train_mask::BitVector,
        test_mask::BitVector;
        val_mask::Union{BitVector, Nothing} = nothing,
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_vertices(
        hg::HGNNHypergraph{T,D},
        index_groups::AbstractVector{AbstractVector{Int}}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_vertices(
        hg::HGNNHypergraph{T,D},
        train_inds::AbstractVector{Int},
        test_inds::AbstractVector{Int};
        val_inds::Union{AbstractVector{Int}, Nothing} = nothing
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    Split a single `HGNNHypergraph` into an arbitrary number of `HGNNHypergraph`s by partitioning vertices. This will
    also partition hyperedges and hypergraphs. If no vertex in a particular partition is incident on a hyperedge, then
    that hyperedge and its associated features will not be present in the resulting `HGNNHypergraph`. Similarly, if no
    vertex in the partition belongs to a particular sub-hypergraph (based on `hypergraph_id`s), then that hypergraph
    and its associated features will not be present in the resulting `HGNNHypergraph`.
    
    Users can provide partitions as `BitVector` masks or vectors of indices (which will be converted into masks). To
    facilitate train-val-test splits, users can specify which mask/indices correspond to the train set, test set, and
    (optionally) validation set. In these cases, the output of `split_vertices` will be a `NamedTuple` with keys
    "train", "test", and (optionally) "val" and values of `HGNNHypergraph`s; otherwise, the output is a vector of
    `HGNNHypergraph`s.
"""
function split_vertices(
    hg::HGNNHypergraph{T,D},
    masks::AbstractVector{BitVector}
) where {T <: Real, D <: AbstractDict{Int, T}}

    @assert all(length.(masks) .== hg.num_vertices)

    res = HGNNHypergraph{T,D}[]

    # Partition v2he and he2v, being careful of indices
    for mask in masks
        v2he = hg.v2he[mask]
        he2v = D[]

        vmap = Dict{Int, Int}()
        index = 1
        for (i, x) in enumerate(mask)
            if x
                vmap[i] = index
                index += 1
            end
        end

        hemap = Dict{Int, Int}()
        hgmap = Dict{Int, Int}()

        for (i, he) in enumerate(hg.he2v)
            newhe = filter(((k,v), ) -> mask[k], he)
            if length(newhe) > 0
                newhe = D(vmap[k] => v for (k, v) in newhe)
                push!(he2v, newhe)
                hemap[i] = length(he2v)
            end
        end

        for i in eachindex(v2he)
            v2he[i] = D(hemap[k] => v for (k, v) in v2he[i])
        end

        unique_hgids = sort(collect(Set(hg.hypergraph_ids[mask])))
        for (i, e) in enumerate(unique_hgids)
            hgmap[e] = i
        end
        hypergraph_ids = [hgmap[x] for x in hg.hypergraph_ids[mask]]

        push!(
            res,
            HGNNHypergraph{T,D}(
                v2he,
                he2v,
                length(v2he),
                length(he2v),
                length(unique_hgids),
                hypergraph_ids,
                getobs(hg.vdata, mask),
                getobs(hg.hedata, collect(keys(hemap))),
                getobs(hg.hgdata, unique_hgids)
            )
        )
    end

    return res
end

function split_vertices(
    hg::HGNNHypergraph{T,D},
    train_mask::BitVector,
    test_mask::BitVector;
    val_mask::Union{BitVector, Nothing} = nothing,
) where {T <: Real, D <: AbstractDict{Int, T}}

    if !isnothing(val_mask)
        masks = [train_mask, val_mask, test_mask]
    else
        masks = [train_mask, test_mask]
    end

    hgs = split_vertices(hg, masks)

    val_data = isnothing(val_mask) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

function split_vertices(
    hg::HGNNHypergraph{T,D},
    index_groups::AbstractVector{AbstractVector{Int}}
) where {T <: Real, D <: AbstractDict{Int, T}}
    masks = BitVector[]

    for indgroup in index_groups
        push!(masks, BitArray(i in indgroup for i in 1:hg.num_vertices))
    end

    split_vertices(hg, masks)
end

function split_vertices(
    hg::HGNNHypergraph{T,D},
    train_inds::AbstractVector{Int},
    test_inds::AbstractVector{Int};
    val_inds::Union{AbstractVector{Int}, Nothing} = nothing
) where {T <: Real, D <: AbstractDict{Int, T}}
    if !isnothing(val_inds)
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_vertices),
            BitArray(i in val_inds for i in 1:hg.num_vertices),
            BitArray(i in test_inds for i in 1:hg.num_vertices)
        ]
    else
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_vertices),
            BitArray(i in test_inds for i in 1:hg.num_vertices)
        ]
    end

    hgs = split_vertices(hg, masks)

    val_data = isnothing(val_inds) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

"""
    split_vertices(
        hg::HGNNDiHypergraph{T,D},
        masks::AbstractVector{BitVector}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_vertices(
        hg::HGNNDiHypergraph{T,D},
        train_mask::BitVector,
        test_mask::BitVector;
        val_mask::Union{BitVector, Nothing} = nothing,
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_vertices(
        hg::HGNNDiHypergraph{T,D},
        index_groups::AbstractVector{AbstractVector{Int}}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_vertices(
        hg::HGNNDiHypergraph{T,D},
        train_inds::AbstractVector{Int},
        test_inds::AbstractVector{Int};
        val_inds::Union{AbstractVector{Int}, Nothing} = nothing
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    Split a single `HGNNDiHypergraph` into an arbitrary number of `HGNNDiHypergraph`s by partitioning vertices. This
    will also partition hyperedges and hypergraphs. If no vertex in a particular partition is incident on a hyperedge,
    then that hyperedge and its associated features will not be present in the resulting `HGNNDiHypergraph`. Similarly,
    if no vertex in the partition belongs to a particular sub-hypergraph (based on `hypergraph_id`s), then that
    hypergraph and its associated features will not be present in the resulting `HGNNDiHypergraph`.
    
    Users can provide partitions as `BitVector` masks or vectors of indices (which will be converted into masks). To
    facilitate train-val-test splits, users can specify which mask/indices correspond to the train set, test set, and
    (optionally) validation set. In these cases, the output of `split_vertices` will be a `NamedTuple` with keys
    "train", "test", and (optionally) "val" and values of `HGNNDiHypergraph`s; otherwise, the output is a vector of
    `HGNNDiHypergraph`s.
"""
function split_vertices(
    hg::HGNNDiHypergraph{T,D},
    masks::AbstractVector{BitVector}
) where {T <: Real, D <: AbstractDict{Int, T}}
    res = HGNNDiHypergraph{T,D}[]

    # Partition v2he and he2v, being careful of indices
    for mask in masks
        v2he_tail = hg.hg_tail.v2he[mask]
        v2he_head = hg.hg_head.v2he[mask]

        he2v_tail = D[]
        he2v_head = D[]

        vmap = Dict{Int, Int}()
        index = 1
        for (i, x) in enumerate(mask)
            if x
                vmap[i] = index
                index += 1
            end
        end

        hemap = Dict{Int, Int}()
        hgmap = Dict{Int, Int}()

        for i in 1:hg.num_hyperedges
            he_tail = hg.hg_tail.he2v[i]
            he_head = hg.hg_head.he2v[i]
            
            newhe_tail = filter(((k,v), ) -> mask[k], he_tail)
            newhe_head = filter(((k,v), ) -> mask[k], he_head)
            if length(newhe_tail) > 0 || length(newhe_head) > 0
                newhe_tail = D(vmap[k] => v for (k, v) in newhe_tail)
                newhe_head = D(vmap[k] => v for (k, v) in newhe_head)
                push!(he2v_tail, newhe_tail)
                push!(he2v_head, newhe_head)
                hemap[i] = length(he2v_tail)
            end
        end

        for i in eachindex(v2he_tail)
            v2he_tail[i] = D(hemap[k] => v for (k, v) in v2he_tail[i])
            v2he_head[i] = D(hemap[k] => v for (k, v) in v2he_head[i])
        end

        unique_hgids = sort(collect(Set(hg.hypergraph_ids[mask])))
        for (i, e) in enumerate(unique_hgids)
            hgmap[e] = i
        end

        hypergraph_ids = [hgmap[x] for x in hg.hypergraph_ids[mask]]

        push!(
            res,
            HGNNDiHypergraph{T,D}(
                Hypergraph{T, Nothing, Nothing, D}(
                    v2he_tail,
                    he2v_tail,
                    Vector{Nothing}(undef, length(part)),
                    Vector{Nothing}(undef, length(newhe_tail))
                    ),
                Hypergraph{T, Nothing, Nothing, D}(
                    v2he_head,
                    he2v_head,
                    Vector{Nothing}(undef, length(part)),
                    Vector{Nothing}(undef, length(newhe_head))
                    ),
                length(v2he_tail),
                length(he2v_tail),
                length(unique_hgids),
                hypergraph_ids,
                getobs(hg.vdata, mask),
                getobs(hg.hedata, collect(keys(hemap))),
                getobs(hg.hgdata, unique_hgids)
            )
        )
    end

    return res
end

function split_vertices(
    hg::HGNNDiHypergraph{T,D},
    train_mask::BitVector,
    test_mask::BitVector;
    val_mask::Union{BitVector, Nothing} = nothing,
) where {T <: Real, D <: AbstractDict{Int, T}}
    if !isnothing(val_mask)
        masks = [train_mask, val_mask, test_mask]
    else
        masks = [train_mask, test_mask]
    end

    hgs = split_vertices(hg, masks)

    val_data = isnothing(val_mask) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

function split_vertices(
    hg::HGNNHypergraph{T,D},
    index_groups::AbstractVector{AbstractVector{Int}}
) where {T <: Real, D <: AbstractDict{Int, T}}
    masks = BitVector[]

    for indgroup in index_groups
        push!(masks, BitArray(i in indgroup for i in 1:hg.num_vertices))
    end

    split_vertices(hg, masks)
end

function split_vertices(
    hg::HGNNHypergraph{T,D},
    train_inds::AbstractVector{Int},
    test_inds::AbstractVector{Int};
    val_inds::Union{AbstractVector{Int}, Nothing} = nothing
) where {T <: Real, D <: AbstractDict{Int, T}}
    if !isnothing(val_inds)
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_vertices),
            BitArray(i in val_inds for i in 1:hg.num_vertices),
            BitArray(i in test_inds for i in 1:hg.num_vertices)
        ]
    else
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_vertices),
            BitArray(i in test_inds for i in 1:hg.num_vertices)
        ]
    end

    hgs = split_vertices(hg, masks)

    val_data = isnothing(val_inds) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

"""
    split_hyperedges(
        hg::HGNNHypergraph{T,D},
        masks::AbstractVector{BitVector}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hyperedges(
        hg::HGNNHypergraph{T, D},
        train_mask::BitVector,
        test_mask::BitVector;
        val_mask::Union{BitVector, Nothing} = nothing,
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hyperedges(
        hg::HGNNHypergraph{T,D},
        index_groups::AbstractVector{AbstractVector{Int}}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hyperedges(
        hg::HGNNHypergraph{T,D},
        train_inds::AbstractVector{Int},
        test_inds::AbstractVector{Int};
        val_inds::Union{AbstractVector{Int}, Nothing} = nothing
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    Split a single `HGNNHypergraph` into an arbitrary number of `HGNNHypergraph`s by partitioning hyperedges. This will
    also partition vertices and hypergraphs. If a vertex is not incident on any hyperedge in a particular partition,
    then that vertex and its associated features will not be present in the resulting `HGNNHypergraph`. If none of the
    relevant vertices in a partition belong to a particular sub-hypergraph (based on `hypergraph_id`s), then that
    hypergraph and its associated features will not be present in the resulting `HGNNHypergraph`.
    
    Users can provide partitions as `BitVector` masks or vectors of indices (which will be converted into masks). To
    facilitate train-val-test splits, users can specify which mask/indices correspond to the train set, test set, and
    (optionally) validation set. In these cases, the output of `split_vertices` will be a `NamedTuple` with keys
    "train", "test", and (optionally) "val" and values of `HGNNHypergraph`s; otherwise, the output is a vector of
    `HGNNHypergraph`s.
"""
function split_hyperedges(
    hg::HGNNHypergraph{T,D},
    masks::AbstractVector{BitVector}
) where {T <: Real, D <: AbstractDict{Int, T}}
    @assert all(length.(masks) .== hg.num_hyperedges)

    res = HGNNHypergraph{T,D}[]

    # Partition v2he and he2v, being careful of indices
    for mask in masks
        he2v = hg.he2v[mask]
        v2he = D[]

        hemap = Dict{Int, Int}()
        index = 1
        for (i, x) in enumerate(mask)
            if x
                hemap[i] = index
                index += 1
            end
        end

        vmap = Dict{Int, Int}()
        hgmap = Dict{Int, Int}()

        for (i, v) in enumerate(hg.v2he)
            newv = filter(((key,val), ) -> mask[key], v)
            if length(newv) > 0
                newv = D(hemap[key] => val for (key, val) in newv)
                push!(v2he, newv)
                vmap[i] = length(v2he)
            end
        end

        for i in eachindex(he2v)
            he2v[i] = D(vmap[key] => val for (key, val) in he2v[i])
        end

        rel_vs = collect(keys(vmap))

        unique_hgids = sort(collect(Set(hg.hypergraph_ids[rel_vs])))
        for (i, e) in enumerate(unique_hgids)
            hgmap[e] = i
        end

        hypergraph_ids = [hgmap[x] for x in hg.hypergraph_ids[rel_vs]]

        push!(
            res,
            HGNNHypergraph{T,D}(
                v2he,
                he2v,
                length(v2he),
                length(he2v),
                length(unique_hgids),
                hypergraph_ids,
                getobs(hg.vdata, rel_vs),
                getobs(hg.hedata, mask),
                getobs(hg.hgdata, unique_hgids)
            )
        )
    end

    return res
end

function split_hyperedges(
    hg::HGNNHypergraph{T, D},
    train_mask::BitVector,
    test_mask::BitVector;
    val_mask::Union{BitVector, Nothing} = nothing,
) where {T <: Real, D <: AbstractDict{Int, T}}

    if !isnothing(val_mask)
        masks = [train_mask, val_mask, test_mask]
    else
        masks = [train_mask, test_mask]
    end

    hgs = split_hyperedges(hg, masks)

    val_data = isnothing(val_mask) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

function split_hyperedges(
    hg::HGNNHypergraph{T,D},
    index_groups::AbstractVector{AbstractVector{Int}}
) where {T <: Real, D <: AbstractDict{Int, T}}

    masks = BitVector[]

    for indgroup in index_groups
        push!(masks, BitArray(i in indgroup for i in 1:hg.num_hyperedges))
    end

    split_hyperedges(hg, masks)
end

function split_hyperedges(
    hg::HGNNHypergraph{T,D},
    train_inds::AbstractVector{Int},
    test_inds::AbstractVector{Int};
    val_inds::Union{AbstractVector{Int}, Nothing} = nothing
) where {T <: Real, D <: AbstractDict{Int, T}}
    if !isnothing(val_inds)
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_hyperedges),
            BitArray(i in val_inds for i in 1:hg.num_hyperedges),
            BitArray(i in test_inds for i in 1:hg.num_hyperedges)
        ]
    else
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_hyperedges),
            BitArray(i in test_inds for i in 1:hg.num_hyperedges)
        ]
    end

    hgs = split_hyperedges(hg, masks)

    val_data = isnothing(val_inds) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

"""
    split_hyperedges(
        hg::HGNNDiHypergraph{T,D},
        masks::AbstractVector{BitVector}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hyperedges(
        hg::HGNNDiHypergraph{T, D},
        train_mask::BitVector,
        test_mask::BitVector;
        val_mask::Union{BitVector, Nothing} = nothing,
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hyperedges(
        hg::HGNNDiHypergraph{T,D},
        index_groups::AbstractVector{AbstractVector{Int}}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hyperedges(
        hg::HGNNDiHypergraph{T,D},
        train_inds::AbstractVector{Int},
        test_inds::AbstractVector{Int};
        val_inds::Union{AbstractVector{Int}, Nothing} = nothing
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    Split a single `HGNNDiHypergraph` into an arbitrary number of `HGNNDiHypergraph`s by partitioning hyperedges. This
    will also partition vertices and hypergraphs. If a vertex is not incident on any hyperedge in a particular
    partition, then that vertex and its associated features will not be present in the resulting `HGNNDiHypergraph`.
    Similarly, if no relevant vertex in the partition belongs to a particular sub-hypergraph (based on
    `hypergraph_id`s), then that hypergraph and its associated features will not be present in the resulting
    `HGNNDiHypergraph`.
    
    Users can provide partitions as `BitVector` masks or vectors of indices (which will be converted into masks). To
    facilitate train-val-test splits, users can specify which mask/indices correspond to the train set, test set, and
    (optionally) validation set. In these cases, the output of `split_vertices` will be a `NamedTuple` with keys
    "train", "test", and (optionally) "val" and values of `HGNNDiHypergraph`s; otherwise, the output is a vector of
    `HGNNDiHypergraph`s.
"""
function split_hyperedges(
    hg::HGNNDiHypergraph{T,D},
    masks::AbstractVector{BitVector}
) where {T <: Real, D <: AbstractDict{Int, T}}
    res = HGNNDiHypergraph{T,D}[]

    # Partition v2he and he2v, being careful of indices
    for mask in masks
        he2v_tail = hg.hg_tail.he2v[mask]
        he2v_head = hg.hg_head.he2v[mask]

        v2he_tail = D[]
        v2he_head = D[]

        hemap = Dict{Int, Int}()
        index = 1
        for (i, x) in enumerate(mask)
            if x
                hemap[i] = index
                index += 1
            end
        end

        vmap = Dict{Int, Int}()
        hgmap = Dict{Int, Int}()

        for i in 1:hg.num_vertices
            v_tail = hg.hg_tail.v2he[i]
            v_head = hg.hg_head.v2he[i]
            
            newv_tail = filter(((k,v), ) -> mask[k], v_tail)
            newv_head = filter(((k,v), ) -> mask[k], v_head)
            if length(newv_tail) > 0 || length(newv_head) > 0
                newv_tail = D(hemap[k] => v for (k, v) in newv_tail)
                newv_head = D(hemap[k] => v for (k, v) in newv_head)
                push!(v2he_tail, newv_tail)
                push!(v2he_head, newv_head)
                vmap[i] = length(v2he_tail)
            end
        end

        for i in eachindex(he2v_tail)
            he2v_tail[i] = D(vmap[k] => v for (k, v) in he2v_tail[i])
            he2v_head[i] = D(vmap[k] => v for (k, v) in he2v_head[i])
        end

        rel_vs = collect(keys(vmap))

        unique_hgids = sort(collect(Set(hg.hypergraph_ids[rel_vs])))
        for (i, e) in enumerate(unique_hgids)
            hgmap[e] = i
        end

        hypergraph_ids = [hgmap[x] for x in hg.hypergraph_ids[rel_vs]]

        push!(
            res,
            HGNNDiHypergraph{T,D}(
                Hypergraph{T, Nothing, Nothing, D}(
                    v2he_tail,
                    he2v_tail,
                    Vector{Nothing}(undef, length(part)),
                    Vector{Nothing}(undef, length(newhe_tail))
                    ),
                Hypergraph{T, Nothing, Nothing, D}(
                    v2he_head,
                    he2v_head,
                    Vector{Nothing}(undef, length(part)),
                    Vector{Nothing}(undef, length(newhe_head))
                    ),
                length(v2he_tail),
                length(he2v_tail),
                length(unique_hgids),
                hypergraph_ids,
                getobs(hg.vdata, collect(keys(vmap))),
                getobs(hg.hedata, mask),
                getobs(hg.hgdata, unique_hgids)
            )
        )
    end

    return res
end

function split_hyperedges(
    hg::HGNNDiHypergraph{T, D},
    train_mask::BitVector,
    test_mask::BitVector;
    val_mask::Union{BitVector, Nothing} = nothing,
) where {T <: Real, D <: AbstractDict{Int, T}}
    if !isnothing(val_mask)
        masks = [train_mask, val_mask, test_mask]
    else
        masks = [train_mask, test_mask]
    end

    hgs = split_hyperedges(hg, masks)

    val_data = isnothing(val_mask) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

function split_hyperedges(
    hg::HGNNDiHypergraph{T,D},
    index_groups::AbstractVector{AbstractVector{Int}}
) where {T <: Real, D <: AbstractDict{Int, T}}
    masks = BitVector[]

    for indgroup in index_groups
        push!(masks, BitArray(i in indgroup for i in 1:hg.num_hyperedges))
    end

    split_hyperedges(hg, masks)
end

function split_hyperedges(
    hg::HGNNDiHypergraph{T,D},
    train_inds::AbstractVector{Int},
    test_inds::AbstractVector{Int};
    val_inds::Union{AbstractVector{Int}, Nothing} = nothing
) where {T <: Real, D <: AbstractDict{Int, T}}
    if !isnothing(val_inds)
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_hyperedges),
            BitArray(i in val_inds for i in 1:hg.num_hyperedges),
            BitArray(i in test_inds for i in 1:hg.num_hyperedges)
        ]
    else
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_hyperedges),
            BitArray(i in test_inds for i in 1:hg.num_hyperedges)
        ]
    end

    hgs = split_hyperedges(hg, masks)

    val_data = isnothing(val_inds) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

"""
    split_hypergraphs(
        hg::HGNNHypergraph{T,D},
        masks::AbstractVector{BitVector}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hypergraphs(
        hg::HGNNHypergraph{T,D},
        train_mask::BitVector,
        test_mask::BitVector;
        val_mask::Union{BitVector, Nothing} = nothing,
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hyperedges(
        hg::HGNNHypergraph{T,D},
        index_groups::AbstractVector{AbstractVector{Int}}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hyperedges(
        hg::HGNNHypergraph{T,D},
        train_inds::AbstractVector{Int},
        test_inds::AbstractVector{Int};
        val_inds::Union{AbstractVector{Int}, Nothing} = nothing
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    Split a single `HGNNHypergraph` into an arbitrary number of `HGNNHypergraph`s by partitioning hypergraphs (from
    vertex `hypergraph_id`s). This will also partition vertices and hyperedges. If a vertex's `hypergraph_id` is not
    included in the partition, then that vertex and its associated features will not be present in the resulting
    `HGNNHypergraph`. If none of the relevant vertices in a partition are incident on a particular hyperedge, then that
    hyperedge and its associated features will not be present in the resulting `HGNNHypergraph`.
    
    Users can provide partitions as `BitVector` masks or vectors of indices (which will be converted into masks). To
    facilitate train-val-test splits, users can specify which mask/indices correspond to the train set, test set, and
    (optionally) validation set. In these cases, the output of `split_vertices` will be a `NamedTuple` with keys
    "train", "test", and (optionally) "val" and values of `HGNNHypergraph`s; otherwise, the output is a vector of
    `HGNNHypergraph`s.
"""
function split_hypergraphs(
    hg::HGNNHypergraph{T,D},
    masks::AbstractVector{BitVector}
) where {T <: Real, D <: AbstractDict{Int, T}}
    @assert all(length.(masks) .== hg.num_hypergraphs)

    res = HGNNHypergraph{T,D}[]

    # Partition v2he and he2v, being careful of indices
    for mask in masks
        rel_vs = [i for (i, e) in enumerate(hg.hypergraph_ids) if mask[e]]
                
        v2he = hg.v2he[rel_vs]
        he2v = D[]
        
        vmap = Dict{Int, Int}(x => i for (i, x) in enumerate(rel_vs))
        hemap = Dict{Int, Int}()
        hgmap = Dict{Int, Int}()
        index = 1
        for (i, x) in enumerate(mask)
            if x
                hgmap[i] = index
                index += 1
            end
        end

        for (i, he) in enumerate(hg.he2v)
            newhe = filter(((k,v), ) -> k in rel_vs, he)
            if length(newhe) > 0
                newhe = D(vmap[k] => v for (k, v) in newhe)
                push!(he2v, newhe)
                hemap[i] = length(he2v)
            end
        end

        for i in eachindex(v2he)
            v2he[i] = D(hemap[k] => v for (k, v) in v2he[i])
        end

        hypergraph_ids = [hgmap[hg.hypergraph_ids[v]] for v in rel_vs]

        push!(
            res,
            HGNNHypergraph{T,D}(
                v2he,
                he2v,
                length(v2he),
                length(he2v),
                length(keys(hgmap)),
                hypergraph_ids,
                getobs(hg.vdata, rel_vs),
                getobs(hg.hedata, collect(keys(hemap))),
                getobs(hg.hgdata, mask)
            )
        )
    end

    return res
end

function split_hypergraphs(
    hg::HGNNHypergraph{T,D},
    train_mask::BitVector,
    test_mask::BitVector;
    val_mask::Union{BitVector, Nothing} = nothing,
) where {T <: Real, D <: AbstractDict{Int, T}}
    if !isnothing(val_mask)
        masks = [train_mask, val_mask, test_mask]
    else
        masks = [train_mask, test_mask]
    end

    hgs = split_hypergraphs(hg, masks)

    val_data = isnothing(val_mask) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

function split_hypergraphs(
    hg::HGNNHypergraph{T,D},
    index_groups::AbstractVector{AbstractVector{Int}}
) where {T <: Real, D <: AbstractDict{Int, T}} 
    masks = BitVector[]

    for indgroup in index_groups
        push!(masks, BitArray(i in indgroup for i in 1:hg.num_hypergraphs))
    end

    split_hyperedges(hg, masks)
end

function split_hypergraphs(
    hg::HGNNHypergraph{T,D},
    train_inds::AbstractVector{Int},
    test_inds::AbstractVector{Int};
    val_inds::Union{AbstractVector{Int}, Nothing} = nothing
) where {T <: Real, D <: AbstractDict{Int, T}}
    if !isnothing(val_inds)
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_hypergraphs),
            BitArray(i in val_inds for i in 1:hg.num_hypergraphs),
            BitArray(i in test_inds for i in 1:hg.num_hypergraphs)
        ]
    else
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_hypergraphs),
            BitArray(i in test_inds for i in 1:hg.num_hypergraphs)
        ]
    end

    hgs = split_hyperedges(hg, masks)

    val_data = isnothing(val_inds) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

"""
    split_hypergraphs(
        hg::HGNNDiHypergraph{T,D},
        masks::AbstractVector{BitVector}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hypergraphs(
        hg::HGNNDiHypergraph{T,D},
        train_mask::BitVector,
        test_mask::BitVector;
        val_mask::Union{BitVector, Nothing} = nothing,
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hypergraphs(
        hg::HGNNDiHypergraph{T,D},
        index_groups::AbstractVector{AbstractVector{Int}}
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    split_hypergraphs(
        hg::HGNNDiHypergraph{T,D},
        train_inds::AbstractVector{Int},
        test_inds::AbstractVector{Int};
        val_inds::Union{AbstractVector{Int}, Nothing} = nothing
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    Split a single `HGNNDiHypergraph` into an arbitrary number of `HGNNDiHypergraph`s by partitioning hypergraphs (from
    vertex `hypergraph_id`s). This will also partition vertices and hyperedges. If a vertex's `hypergraph_id` is not
    included in the partition, then that vertex and its associated features will not be present in the resulting
    `HGNNDiHypergraph`. If none of the relevant vertices in a partition are incident on a particular hyperedge, then
    that hyperedge and its associated features will not be present in the resulting `HGNNDiHypergraph`.
    
    Users can provide partitions as `BitVector` masks or vectors of indices (which will be converted into masks). To
    facilitate train-val-test splits, users can specify which mask/indices correspond to the train set, test set, and
    (optionally) validation set. In these cases, the output of `split_vertices` will be a `NamedTuple` with keys
    "train", "test", and (optionally) "val" and values of `HGNNDiHypergraph`s; otherwise, the output is a vector of
    `HGNNDiHypergraph`s.
"""
function split_hypergraphs(
    hg::HGNNDiHypergraph{T,D},
    masks::AbstractVector{BitVector}
) where {T <: Real, D <: AbstractDict{Int, T}}
    res = HGNNDiHypergraph{T,D}[]

    for mask in masks
        rel_vs = [i for (i, e) in enumerate(hg.hypergraph_ids) if mask[e]]
                    
        v2he_tail = hg.hg_tail.v2he[rel_vs]
        v2he_head = hg.hg_head.v2he[rel_vs]

        he2v_tail = D[]
        he2v_head = D[]
        
        vmap = Dict{Int, Int}(x => i for (i, x) in enumerate(rel_vs))
        hemap = Dict{Int, Int}()
        hgmap = Dict{Int, Int}()
        index = 1
        for (i, x) in enumerate(mask)
            if x
                hgmap[i] = index
                index += 1
            end
        end

        for (i, _) in enumerate(hg.hg_tail.he2v)
            newhe_tail = filter(((k,v), ) -> k in rel_vs, hg.hg_tail.he2v[i])
            newhe_head = filter(((k,v), ) -> k in rel_vs, hg.hg_head.he2v[i])
            if length(newhe_tail) > 0 || length(newhe_head) > 0
                newhe_tail = D(vmap[k] => v for (k, v) in newhe_tail)
                newhe_head = D(vmap[k] => v for (k, v) in newhe_head)
                push!(he2v_tail, newhe_tail)
                push!(he2v_head, newhe_head)
                hemap[i] = length(he2v_tail)
            end
        end

        for i in eachindex(v2he_tail)
            v2he_tail[i] = D(hemap[k] => v for (k, v) in v2he_tail[i])
            v2he_head[i] = D(hemap[k] => v for (k, v) in v2he_head[i])
        end

        hypergraph_ids = [hgmap[hg.hypergraph_ids[v]] for v in rel_vs]

        push!(
            res,
            HGNNDiHypergraph{T,D}(
                Hypergraph{T, Nothing, Nothing, D}(
                    v2he_tail,
                    he2v_tail,
                    Vector{Nothing}(undef, length(part)),
                    Vector{Nothing}(undef, length(newhe_tail))
                    ),
                Hypergraph{T, Nothing, Nothing, D}(
                    v2he_head,
                    he2v_head,
                    Vector{Nothing}(undef, length(part)),
                    Vector{Nothing}(undef, length(newhe_head))
                    ),
                length(v2he_tail),
                length(he2v_tail),
                length(keys(hgmap)),
                hypergraph_ids,
                getobs(hg.vdata, rel_vs),
                getobs(hg.hedata, collect(keys(hemap))),
                getobs(hg.hgdata, mask)
            )
        )
    end

    return res
end

function split_hypergraphs(
    hg::HGNNDiHypergraph{T,D},
    train_mask::BitVector,
    test_mask::BitVector;
    val_mask::Union{BitVector, Nothing} = nothing,
) where {T <: Real, D <: AbstractDict{Int, T}}
    if !isnothing(val_mask)
        masks = [train_mask, val_mask, test_mask]
    else
        masks = [train_mask, test_mask]
    end

    hgs = split_hypergraphs(hg, masks)

    val_data = isnothing(val_mask) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

function split_hypergraphs(
    hg::HGNNDiHypergraph{T,D},
    index_groups::AbstractVector{AbstractVector{Int}}
) where {T <: Real, D <: AbstractDict{Int, T}} 
    masks = BitVector[]

    for indgroup in index_groups
        push!(masks, BitArray(i in indgroup for i in 1:hg.num_hypergraphs))
    end

    split_hyperedges(hg, masks)
end

function split_hypergraphs(
    hg::HGNNDiHypergraph{T,D},
    train_inds::AbstractVector{Int},
    test_inds::AbstractVector{Int};
    val_inds::Union{AbstractVector{Int}, Nothing} = nothing
) where {T <: Real, D <: AbstractDict{Int, T}}
    if !isnothing(val_inds)
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_hypergraphs),
            BitArray(i in val_inds for i in 1:hg.num_hypergraphs),
            BitArray(i in test_inds for i in 1:hg.num_hypergraphs)
        ]
    else
        masks = [
            BitArray(i in train_inds for i in 1:hg.num_hypergraphs),
            BitArray(i in test_inds for i in 1:hg.num_hypergraphs)
        ]
    end

    hgs = split_hyperedges(hg, masks)

    val_data = isnothing(val_inds) ? nothing : hgs[2]

    return (train=hgs[1], val=val_data, test=hgs[end])
end

"""
    random_split_vertices(
        hg::HGNNHypergraph{T,D},
        fracs::AbstractVector{<:Real},
        rng::AbstractRNG
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    random_split_vertices(
        hg::HGNNDiHypergraph{T,D},
        fracs::AbstractVector{<:Real},
        rng::AbstractRNG
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    Randomly partition a hypergraph `hg` by dividing the vertices (see `split_vertices`). Users provide the (relative)
    sizes of the partitions via `fracs`, a vector of real numbers. The sum of the fractions must equal 1, and all
    fractions must be between 0 and 1. Users must additionally provide a random number generator (`rng`).
"""
function random_split_vertices(
    hg::HGNNHypergraph{T,D},
    fracs::AbstractVector{<:Real},
    rng::AbstractRNG
) where {T <: Real, D <: AbstractDict{Int, T}}
    # For all f ∈ fracs, 0 < f <= 1
    @assert all(fracs .> 0) && all(fracs .<= 1)
    # Fractions must sum to 1
    @assert abs(sum(fracs) - 1) <= 1e-5

    num_choices = round.(fracs .* hg.num_vertices)
    rand_inds = shuffle(rng, Vector(1:hg.num_vertices))
    
    masks = BitVector[]
    start_point = 1

    # Provide the (approximate) right amount of (randomly selected) vertex indices per partition
    for i in 1:(length(num_choices) - 1)
        part = rand_inds[start_point:start_point + num_choices[i] - 1]
        start_point += num_choices
        push!(masks, BitArray(i in part for i in 1:hg.num_vertices(hg)))
    end
    remainder = rand_inds[start_point:end]
    push!(masks, BitArray(i in remainder for i in 1:hg.num_vertices(hg)))

    split_vertices(hg, masks)
end

function random_split_vertices(
    hg::HGNNDiHypergraph{T,D},
    fracs::AbstractVector{<:Real},
    rng::AbstractRNG
) where {T <: Real, D <: AbstractDict{Int, T}}
    # For all f ∈ fracs, 0 < f <= 1
    @assert all(fracs .> 0) && all(fracs .<= 1)
    # Fractions must sum to 1
    @assert abs(sum(fracs) - 1) <= 1e-5

    num_choices = round.(fracs .* hg.num_vertices)
    rand_inds = shuffle(rng, Vector(1:hg.num_vertices))

    masks = BitVector[]
    start_point = 1

    # Provide the (approximate) right amount of (randomly selected) vertex indices per partition
    for i in 1:(length(num_choices) - 1)
        part = rand_inds[start_point:start_point + num_choices[i] - 1]
        start_point += num_choices
        push!(masks, BitArray(i in part for i in 1:hg.num_vertices(hg)))
    end
    remainder = rand_inds[start_point:end]
    push!(masks, BitArray(i in remainder for i in 1:hg.num_vertices(hg)))

    split_vertices(hg, masks)
end

"""
    random_split_hyperedges(
        hg::HGNNHypergraph{T,D},
        fracs::AbstractVector{<:Real},
        rng::AbstractRNG
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    random_split_hyperedges(
        hg::HGNNDiHypergraph{T,D},
        fracs::AbstractVector{<:Real},
        rng::AbstractRNG
    ) where {T <: Real, D <: AbstractDict{Int, T}}
    
    Randomly partition a hypergraph `hg` by dividing the hyperedges (see `split_hyperedges`). Users provide the
    (relative) sizes of the partitions via `fracs`, a vector of real numbers. The sum of the fractions must equal 1,
    and all fractions must be between 0 and 1. Users must additionally provide a random number generator (`rng`).
"""
function random_split_hyperedges(
    hg::HGNNHypergraph{T,D},
    fracs::AbstractVector{<:Real},
    rng::AbstractRNG
) where {T <: Real, D <: AbstractDict{Int, T}}
    # For all f ∈ fracs, 0 < f <= 1
    @assert all(fracs .> 0) && all(fracs .<= 1)
    # Fractions must sum to 1
    @assert abs(sum(fracs) - 1) <= 1e-5

    num_choices = round.(fracs .* hg.num_hyperedges)
    rand_inds = shuffle(rng, Vector(1:hg.num_hyperedges))
    
    masks = BitVector[]
    start_point = 1

    # Provide the (approximate) right amount of (randomly selected) vertex indices per partition
    for i in 1:(length(num_choices) - 1)
        part = rand_inds[start_point:start_point + num_choices[i] - 1]
        start_point += num_choices
        push!(masks, BitArray(i in part for i in 1:hg.num_hyperedges))
    end
    remainder = rand_inds[start_point:end]
    push!(masks, BitArray(i in remainder for i in 1:hg.num_hyperedges))

    split_hyperedges(hg, masks)
end

function random_split_hyperedges(
    hg::HGNNDiHypergraph{T,D},
    fracs::AbstractVector{<:Real},
    rng::AbstractRNG
) where {T <: Real, D <: AbstractDict{Int, T}}
    # For all f ∈ fracs, 0 < f <= 1
    @assert all(fracs .> 0) && all(fracs .<= 1)
    # Fractions must sum to 1
    @assert abs(sum(fracs) - 1) <= 1e-5

    num_choices = round.(fracs .* hg.num_hyperedges)
    rand_inds = shuffle(rng, Vector(1:hg.num_hyperedges))
    
    masks = BitVector[]
    start_point = 1

    # Provide the (approximate) right amount of (randomly selected) vertex indices per partition
    for i in 1:(length(num_choices) - 1)
        part = rand_inds[start_point:start_point + num_choices[i] - 1]
        start_point += num_choices
        push!(masks, BitArray(i in part for i in 1:hg.num_hyperedges))
    end
    remainder = rand_inds[start_point:end]
    push!(masks, BitArray(i in remainder for i in 1:hg.num_hyperedges))

    split_hyperedges(hg, masks)
end

"""
    random_split_hypergraphs(
        hg::HGNNHypergraph{T,D},
        fracs::AbstractVector{<:Real},
        rng::AbstractRNG
    ) where {T <: Real, D <: AbstractDict{Int, T}}

    random_split_hypergraphs(
        hg::HGNNDiHypergraph{T,D},
        fracs::AbstractVector{<:Real},
        rng::AbstractRNG
    ) where {T <: Real, D <: AbstractDict{Int, T}}
    
    Randomly partition a hypergraph `hg` by dividing the (sub)-hypergraphs (see `split_hypergraphs`). Users provide the
    (relative) sizes of the partitions via `fracs`, a vector of real numbers. The sum of the fractions must equal 1,
    and all fractions must be between 0 and 1. Users must additionally provide a random number generator (`rng`).
"""
function random_split_hypergraphs(
    hg::HGNNHypergraph{T,D},
    fracs::AbstractVector{<:Real},
    rng::AbstractRNG
) where {T <: Real, D <: AbstractDict{Int, T}}
    # For all f ∈ fracs, 0 < f <= 1
    @assert all(fracs .> 0) && all(fracs .<= 1)
    # Fractions must sum to 1
    @assert abs(sum(fracs) - 1) <= 1e-5

    num_choices = round.(fracs .* hg.num_hypergraphs)
    rand_inds = shuffle(rng, Vector(1:hg.num_hypergraphs))
    
    masks = BitVector[]
    start_point = 1

    # Provide the (approximate) right amount of (randomly selected) vertex indices per partition
    for i in 1:(length(num_choices) - 1)
        part = rand_inds[start_point:start_point + num_choices[i] - 1]
        start_point += num_choices
        push!(masks, BitArray(i in part for i in 1:hg.num_hypergraphs))
    end
    remainder = rand_inds[start_point:end]
    push!(masks, BitArray(i in remainder for i in 1:hg.num_hypergraphs))

    split_hypergraphs(hg, masks)
end

function random_split_hypergraphs(
    hg::HGNNDiHypergraph{T,D},
    fracs::AbstractVector{<:Real},
    rng::AbstractRNG
) where {T <: Real, D <: AbstractDict{Int, T}}
    # For all f ∈ fracs, 0 < f <= 1
    @assert all(fracs .> 0) && all(fracs .<= 1)
    # Fractions must sum to 1
    @assert abs(sum(fracs) - 1) <= 1e-5

    num_choices = round.(fracs .* hg.num_hypergraphs)
    rand_inds = shuffle(rng, Vector(1:hg.num_hypergraphs))
    
    masks = BitVector[]
    start_point = 1

    # Provide the (approximate) right amount of (randomly selected) vertex indices per partition
    for i in 1:(length(num_choices) - 1)
        part = rand_inds[start_point:start_point + num_choices[i] - 1]
        start_point += num_choices
        push!(masks, BitArray(i in part for i in 1:hg.num_hypergraphs))
    end
    remainder = rand_inds[start_point:end]
    push!(masks, BitArray(i in remainder for i in 1:hg.num_hypergraphs))

    split_hypergraphs(hg, masks)
end

"""
TODO: add docstrings
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

# TODO: you are here
function rewire_hyperedges()
end

function to_undirected()
end

# MLUtils.batch
function batch()
end

# MLUtils.unbatch
function unbatch()
end

function get_hypergraph()
end

function negative_sample()
end

function random_split_vertices()
end

function random_split_hyperedges()
end

# PageRank diffusion?
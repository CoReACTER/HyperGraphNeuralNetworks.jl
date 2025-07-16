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


function remove_self_loops()
end

function remove_hyperedges()
end

function remove_multi_hyperedges()
end

function remove_vertices()
end

function add_hyperedges()
end

function add_vertices()
end

function rewire_hyperedges()
end

function to_undirected()
end

function set_hyperedge_weight()
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
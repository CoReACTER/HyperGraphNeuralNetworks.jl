
"""
    erdos_renyi_hypergraph(
        nVertices::Int,
        nEdges::Int,
        HType::Type{H};
        seed::Int = -1,
        kws...
    ) where {H <: AbstractHGNNHypergraph}

    erdos_renyi_hypergraph(
        nVertices::Int,
        nEdges::Int,
        HType::Type{H};
        seed::Int = -1,
        no_self_loops::Bool = false,
        kws...
    ) where {H <: AbstractHGNNDiHypergraph}

    Generate a *random* undirected hypergraph (in the style of Erdős–Rényi random graphs) without any structural
    constraints. See `SimpleHypergraphs.random_model`. The user can optionally seed the random number
    generator with kwarg `seed`. The default value is -1; if the value is greater than or equal to 0, then `seed` will
    be used.
"""
function erdos_renyi_hypergraph(
    nVertices::Int,
    nEdges::Int,
    HType::Type{H};
    seed::Int = -1,
    kws...
) where {H <: AbstractHGNNHypergraph}
    if seed >= 0
        seed!(seed)
    end
    
    basic = random_model(nVertices, nEdges, Hypergraph)

    HType(basic; kws...)
end

function erdos_renyi_hypergraph(
    nVertices::Int,
    nEdges::Int,
    HType::Type{H};
    seed::Int = -1,
    no_self_loops::Bool = false,
    kws...
) where {H <: AbstractHGNNDiHypergraph}
    if seed >= 0
        seed!(seed)
    end
    
    basic = random_model(
        nVertices,
        nEdges,
        DirectedHypergraph;
        no_self_loops=no_self_loops
    )

    HType(basic; kws...)
end

"""
    random_kuniform_hypergraph(
        nVertices::Int,
        nEdges::Int,
        k::Int,
        HType::Type{H};
        seed::Int = -1,
        kws...
    ) where {H <: AbstractHGNNHypergraph}

    random_kuniform_hypergraph(
        nVertices::Int,
        nEdges::Int,
        k::Int,
        HType::Type{H};
        seed::Int = -1,
        no_self_loops::Bool = false,
        kws...
    ) where {H <: AbstractHGNNDiHypergraph}

    Generates a *k*-uniform hypergraph, i.e. an hypergraph where each hyperedge has size *k*. For a directed hypergraph,
    each hyperedge has size *k = k_tail + k_head*, where *k_tail* and *k_head* are not necessarily equal. See 
    `SimpleHypergraphs.random_kuniform_model`. The user can optionally seed the random number generator with kwarg
    `seed`. The default value is -1; if the value is greater than or equal to 0, then `seed` will be used.
"""
function random_kuniform_hypergraph(
    nVertices::Int,
    nEdges::Int,
    k::Int,
    HType::Type{H};
    seed::Int = -1,
    kws...
) where {H <: AbstractHGNNHypergraph}
    if seed >= 0
        seed!(seed)
    end

    kuniform = random_kuniform_model(nVertices, nEdges, k, Hypergraph)

    HType(kuniform; kws...)
end

function random_kuniform_hypergraph(
    nVertices::Int,
    nEdges::Int,
    k::Int,
    HType::Type{H};
    seed::Int = -1,
    no_self_loops::Bool = false,
    kws...
) where {H <: AbstractHGNNDiHypergraph}
    if seed >= 0
        seed!(seed)
    end

    kuniform = random_kuniform_model(
        nVertices,
        nEdges,
        k,
        DirectedHypergraph;
        no_self_loops=no_self_loops
    )

    HType(kuniform; kws...)
end

"""
    random_dregular_hypergraph(
        nVertices::Int,
        nEdges::Int,
        d::Int,
        HType::Type{H};
        seed::Int = -1,
        kws...
    ) where {H <: AbstractHGNNHypergraph}

    random_dregular_hypergraph(
        nVertices::Int,
        nEdges::Int,
        d::Int,
        HType::Type{H};
        seed::Int = -1,
        no_self_loops::Bool = false,
        kws...
    ) where {H <: AbstractHGNNDiHypergraph}

    Generates a *d*-regular hypergraph, where each node has degree *d*. See `SimpleHypergraphs.random_dregular_model`.
    The user can optionally seed the random number generator with kwarg `seed`. The default value is -1; if the value
    is greater than or equal to 0, then `seed` will be used.
"""
function random_dregular_hypergraph(
    nVertices::Int,
    nEdges::Int,
    d::Int,
    HType::Type{H};
    seed::Int = -1,
    kws...
) where {H <: AbstractHGNNHypergraph}
    if seed >= 0
        seed!(seed)
    end

    dregular = random_dregular_model(nVertices, nEdges, d, Hypergraph)

    HType(dregular; kws...)
end

function random_dregular_hypergraph(
    nVertices::Int,
    nEdges::Int,
    d::Int,
    HType::Type{H};
    seed::Int = -1,
    no_self_loops::Bool = false,
    kws...
) where {H <: AbstractHGNNDiHypergraph}
    if seed >= 0
        seed!(seed)
    end

    dregular = random_dregular_model(
        nVertices,
        nEdges,
        d,
        DirectedHypergraph;
        no_self_loops=no_self_loops
    )

    HType(dregular; kws...)
end

"""
    random_preferential_hypergraph(
        nVertices::Int,
        p::Real,
        HType::Type{HO};
        seed::Int = -1,
        HTypeStart::Type{HI} = Hypergraph,
        hg::HI = random_model(5,5, HI),
        kws...
    ) where {HI<:AbstractSimpleHypergraph, HO<:AbstractHGNNHypergraph}

    Generate a hypergraph with a preferential attachment rule between nodes, as presented in
    *Avin, C., Lotker, Z., and Peleg, D. Random preferential attachment hyper-graphs. Computer Science 23 (2015).*
    See `SimpleHypergraphs.random_preferential_model` for more details. The user can optionally seed the random number
    generator with kwarg `seed`. The default value is -1; if the value is greater than or equal to 0, then `seed` will
    be used.
"""
function random_preferential_hypergraph(
    nVertices::Int,
    p::Real,
    HType::Type{HO};
    seed::Int = -1,
    HTypeStart::Type{HI} = Hypergraph,
    hg::HI = random_model(5,5, HI),
    kws...
) where {HI<:AbstractSimpleHypergraph, HO<:AbstractHGNNHypergraph}
    if seed >= 0
        seed!(seed)
    end

    pref_hg = random_preferential_model(
        nVertices,
        p,
        HTypeStart;
        hg=hg
    )

    HType(pref_hg; kws...)
end
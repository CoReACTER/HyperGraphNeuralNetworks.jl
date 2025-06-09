"""
TODO: docstrings
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
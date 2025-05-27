"""
TODO: docstrings for datatypes
"""
struct HGNNHypergraph{T<:Real, D<:AbstractDict{Int,T}} <: AbstractHGNNHypergraph{T}
    v2he::Vector{D}
    he2v::Vector{D}
    num_vertices::Int
    num_hyperedges::Int
    num_hypergraphs::Int
    hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}}
    vdata::DataStore
    hedata::DataStore
    hgdata::DataStore
end

function HGNNHypergraph(
    h::AbstractSimpleHypergraph{T};
    hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    vdata::Union{DataStore, Nothing} = nothing,
    hedata::Union{DataStore, Nothing} = nothing,
    hgdata::Union{DataStore, Nothing} = nothing
) where {T<:Real}
    nhg = !isnothing(hypergraph_ids) ? maximum(hypergraph_ids) : 1

    # From GNNGraphs.jl
    vdata = normalize_graphdata(
        vdata,
        default_name = :x,
        n = nhv(h)
    )
    hedata = normalize_graphdata(
        hedata,
        default_name = :e,
        n = nhe(h),
        duplicate_if_needed = true
    )
    hgdata = normalize_graphdata(
        hgdata,
        default_name = :u,
        n = nhg,
        glob = true
    )

    HGNNHypergraph(
        deepcopy!(h.v2he),
        deepcopy!(h.he2v),
        nhv(h),
        nhe(h),
        nhg,
        hypergraph_ids,
        vdaata,
        hedata,
        hgdata
    )
end

function HGNNHypergraph(
    incidence::AbstractMatrix{Union{T, Nothing}};
    hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    vdata::Union{DataStore, Nothing} = nothing,
    hedata::Union{DataStore, Nothing} = nothing,
    hgdata::Union{DataStore, Nothing} = nothing
) where {T<:Real}
    h = Hypergraph(incidence)
    HGNNHypergraph(
        h; 
        hypergraph_ids=hypergraph_ids,
        vdata=vdata,
        hedata=hedata,
        hgdata=hgdata
    )
end

function HGNNHypergraph(num_nodes::T; vdata=nothing, kws...) where {T<:Integer}
    h = Hypergraph(num_nodes, 0)
    HGNNHypergraph(h; vdata, kws...)
end

function HGNNHypergraph(; num_nodes=nothing, vdata=nothing, kws...)
    if num_nodes === nothing
        if vdata === nothing
            num_nodes = 0
        else
            num_nodes = numobs(vdata)
        end
    end
    return HGNNHypergraph(num_nodes; vdata, kws...)
end


Base.zero(::Type{H}) where {H <: HGNNHypergraph} = H(0)

"""
    copy(hg::HGNNHypergraph; deep=false)

Create a copy of `hg`. If `deep` is `true`, then copy will be a deep copy (equivalent to `deepcopy(hg)`),
otherwise it will be a shallow copy with the same underlying hypergraph data.
"""
function Base.copy(hg::HGNNHypergraph; deep = false)
    if deep
        HGNNHypergraph(
            deepcopy(hg.v2he), deepcopy(hg.he2v),
            hg.num_vertices, hg.num_hyperedges, hg.num_hypergraphs,
            deepcopy(hg.hypergraph_ids),
            deepcopy(hg.vdata), deepcopy(hg.hedata), deepcopy(hg.hgdata)
        )
    else
        HGNNHypergraph(
            hg.v2he, hg.he2v,
            hg.num_vertices, hg.num_hyperedges, hg.num_hypergraphs,
            hg.hypergraph_ids,
            hg.vdata, hg.hedata, hg.hgdata
        )
    end
end

function Base.show(io::IO, hg::HGNNHypergraph)
    print(io, "HGNNHypergraph($(hg.num_vertices), $(hg.num_hyperedges), $(hg.num_hypergraphs)) with ")
    print_all_features(io, hg.vdata, hg.hedata, hg.hgdata)
    print(io, " data")
end

function Base.show(io::IO, ::MIME"text/plain", hg::HGNNHypergraph)
    if get(io, :compact, false)
        print(io, "HGNNHypergraph($(hg.num_vertices), $(hg.num_hyperedges), $(hg.num_hypergraphs)) with ")
        print_all_features(io, g.vdata, g.hedata, g.hgdata)
        print(io, " data")
    else
        print(io,
              "HGNNHypergraph:\n  num_vertices: $(hg.num_vertices)\n  num_hyperedges: $(hg.num_hyperedges)")
        hg.num_hypergraphs > 1 && print(io, "\n  num_hypergraphs: $(hg.num_graphs)")
        if !isempty(hg.vdata)
            print(io, "\n  vdata (vertex data):")
            for k in keys(hg.vdata)
                print(io, "\n    $k = $(shortsummary(hg.vdata[k]))")
            end
        end
        if !isempty(hg.hedata)
            print(io, "\n  hedata (hyperedge data):")
            for k in keys(hg.hedata)
                print(io, "\n    $k = $(shortsummary(hg.hedata[k]))")
            end
        end
        if !isempty(hg.hgdata)
            print(io, "\n  hgdata (hypergraph data):")
            for k in keys(hg.hgdata)
                print(io, "\n    $k = $(shortsummary(hg.hgdata[k]))")
            end
        end
    end
end

MLUtils.numobs(hg::HGNNHypergraph) = hg.num_hypergraphs
# TODO: implement gethypergraph function
# MLUtils.getobs(hg::HGNNHypergraph, i) = gethypergraph(hg, i)

function Base.:(==)(hg1::HGNNHypergraph, hg2::HGNNHypergraph)
    hg1 === hg2 && return true
    for k in fieldnames(typeof(hg1))
        k === :hypergraph_ids && continue
        getfield(hg1, k) != getfield(hg2, k) && return false
    end
    return true
end

function Base.hash(hg::T, h::UInt) where {T <: HGNNHypergraph}
    fs = (getfield(hg, k) for k in fieldnames(T) if k !== :hypergraph_ids)
    return foldl((h, f) -> hash(f, h), fs, init = hash(T, h))
end

function Base.getproperty(hg::HGNNHypergraph, s::Symbol)
    if s in fieldnames(HGNNHypergraph)
        return getfield(hg, s)
    end
    if (s in keys(hg.vdata)) + (s in keys(hg.hedata)) + (s in keys(hg.hgdata)) > 1
        throw(ArgumentError("Ambiguous property name $s"))
    end
    if s in keys(hg.vdata)
        return hg.vdata[s]
    elseif s in keys(hg.hedata)
        return hg.hedata[s]
    elseif s in keys(hg.hgdata)
        return hg.hgdata[s]
    else
        throw(ArgumentError("$(s) is not a field of HGNNHypergraph"))
    end
end


#####
"""
TODO: docstring
"""
struct HGNNDiHypergraph{T<:Real, D<:AbstractDict{Int,T}} <: AbstractHGNNDiHypergraph{T}
    hg_tail::Hypergraph{T, D}
    hg_head::Hypergraph{T, D}
    num_vertices::Int
    num_hyperedges::Int
    num_hypergraphs::Int
    hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}}
    vdata::DataStore
    hedata::DataStore
    hgdata::DataStore
end

function HGNNDiHypergraph(
    h::AbstractDirectedHypergraph{T};
    hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    vdata::Union{DataStore, Nothing} = nothing,
    hedata::Union{DataStore, Nothing} = nothing,
    hgdata::Union{DataStore, Nothing} = nothing
) where {T<:Real}
    nhg = !isnothing(hypergraph_ids) ? maximum(hypergraph_ids) : 1

    # From GNNGraphs.jl
    vdata = normalize_graphdata(
        vdata,
        default_name = :x,
        n = nhv(h)
    )
    hedata = normalize_graphdata(
        hedata,
        default_name = :e,
        n = nhe(h),
        duplicate_if_needed = true
    )
    hgdata = normalize_graphdata(
        hgdata,
        default_name = :u,
        n = nhg,
        glob = true
    )

    HGNNDiHypergraph(
        deepcopy!(h.hg_tail),
        deepcopy!(h.hg_head),
        nhv(h),
        nhe(h),
        nhg,
        hypergraph_ids,
        vdaata,
        hedata,
        hgdata
    )
end

function HGNNDiHypergraph(
    incidence_tail::AbstractMatrix{Union{T, Nothing}},
    incidence_head::AbstractMatrix{Union{T, Nothing}};
    hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    vdata::Union{DataStore, Nothing} = nothing,
    hedata::Union{DataStore, Nothing} = nothing,
    hgdata::Union{DataStore, Nothing} = nothing
) where {T<:Real}
    h = DirectedHypergraph(incidence_tail, incidence_head)
    HGNNDiHypergraph(
        h; 
        hypergraph_ids=hypergraph_ids,
        vdata=vdata,
        hedata=hedata,
        hgdata=hgdata
    )
end

function HGNNDiHypergraph(num_nodes::T; vdata=nothing, kws...) where {T<:Integer}
    h = DirectedHypergraph(num_nodes, 0)
    HGNNDiHypergraph(h; vdata, kws...)
end

function HGNNDiHypergraph(; num_nodes=nothing, vdata=nothing, kws...)
    if num_nodes === nothing
        if vdata === nothing
            num_nodes = 0
        else
            num_nodes = numobs(vdata)
        end
    end
    HGNNDiHypergraph(num_nodes; vdata, kws...)
end


Base.zero(::Type{H}) where {H <: HGNNDiHypergraph} = H(0)

function Base.show(io::IO, hg::HGNNDiHypergraph)
    print(io, "HGNNDiHypergraph($(hg.num_vertices), $(hg.num_hyperedges), $(hg.num_hypergraphs)) with ")
    print_all_features(io, hg.vdata, hg.hedata, hg.hgdata)
    print(io, " data")
end

function Base.show(io::IO, ::MIME"text/plain", hg::HGNNDiHypergraph)
    if get(io, :compact, false)
        print(io, "HGNNDiHypergraph($(hg.num_vertices), $(hg.num_hyperedges), $(hg.num_hypergraphs)) with ")
        print_all_features(io, hg.vdata, hg.hedata, hg.hgdata)
        print(io, " data")
    else
        print(io,
              "HGNNDiHypergraph:\n  num_vertices: $(hg.num_vertices)\n  num_hyperedges: $(hg.num_hyperedges)")
        hg.num_hypergraphs > 1 && print(io, "\n  num_hypergraphs: $(hg.num_graphs)")
        if !isempty(hg.vdata)
            print(io, "\n  vdata (vertex data):")
            for k in keys(hg.vdata)
                print(io, "\n    $k = $(shortsummary(hg.vdata[k]))")
            end
        end
        if !isempty(hg.hedata)
            print(io, "\n  hedata (hyperedge data):")
            for k in keys(hg.hedata)
                print(io, "\n    $k = $(shortsummary(hg.hedata[k]))")
            end
        end
        if !isempty(hg.hgdata)
            print(io, "\n  hgdata (hypergraph data):")
            for k in keys(hg.hgdata)
                print(io, "\n    $k = $(shortsummary(hg.hgdata[k]))")
            end
        end
    end
end

"""
    copy(hg::HGNNDiHypergraph; deep=false)

Create a copy of `hg`. If `deep` is `true`, then copy will be a deep copy (equivalent to `deepcopy(hg)`),
otherwise it will be a shallow copy with the same underlying hypergraph data.
"""
function Base.copy(hg::HGNNDiHypergraph; deep = false)
    if deep
        HGNNDiHypergraph(
            deepcopy(hg.hg_tail), deepcopy(hg.hg_head),
            hg.num_vertices, hg.num_hyperedges, hg.num_hypergraphs,
            deepcopy(hg.hypergraph_ids),
            deepcopy(hg.vdata), deepcopy(hg.hedata), deepcopy(hg.hgdata)
        )
    else
        HGNNDiHypergraph(
            hg.hg_tail, hg.hg_head,
            hg.num_vertices, hg.num_hyperedges, hg.num_hypergraphs,
            hg.hypergraph_ids,
            hg.vdata, hg.hedata, hg.hgdata
        )
    end
end

MLUtils.numobs(hg::HGNNDiHypergraph) = hg.num_hypergraphs
# TODO: implement gethypergraph function
# MLUtils.getobs(hg::HGNNDiHypergraph, i) = gethypergraph(hg, i)

function Base.:(==)(hg1::HGNNDiHypergraph, hg2::HGNNDiHypergraph)
    hg1 === hg2 && return true
    for k in fieldnames(typeof(hg1))
        k === :hypergraph_ids && continue
        getfield(hg1, k) != getfield(hg2, k) && return false
    end
    return true
end

function Base.hash(hg::T, h::UInt) where {T <: HGNNDiHypergraph}
    fs = (getfield(hg, k) for k in fieldnames(T) if k !== :hypergraph_ids)
    return foldl((h, f) -> hash(f, h), fs, init = hash(T, h))
end

function Base.getproperty(hg::HGNNDiHypergraph, s::Symbol)
    if s in fieldnames(HGNNDiHypergraph)
        return getfield(hg, s)
    end
    if (s in keys(hg.vdata)) + (s in keys(hg.hedata)) + (s in keys(hg.hgdata)) > 1
        throw(ArgumentError("Ambiguous property name $s"))
    end
    if s in keys(hg.vdata)
        return hg.vdata[s]
    elseif s in keys(hg.hedata)
        return hg.hedata[s]
    elseif s in keys(hg.hgdata)
        return hg.hgdata[s]
    else
        throw(ArgumentError("$(s) is not a field of HGNNDiHypergraph"))
    end
end
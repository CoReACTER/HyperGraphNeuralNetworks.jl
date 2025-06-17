"""
TODO: docstrings for datatypes

TODO: what functions from SimpleHypergraphs/SimpleDirectedHypergraphs need to be implemented to finish interface?

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

# TODO: setters and getters
hasvertexmeta(::Type{HGNNHypergraph}) = true
hasvertexmeta(X::HGNNHypergraph) = true
hashyperedgemeta(::Type{HGNNHypergraph}) = true
hashyperedgemeta(X::HGNNHypergraph) = true


"""
    add_vertex!(::HGNNHypergraph{T, D}; ::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}

    This function is not implemented for HGNNHypergraph.
        
    The basic hypergraph structure of HGNNHypergraph (i.e., the number of vertices, the hyperedges, and the hypergraph
    IDs) are not mutable. Users can change the features in the `vdata`, `hedata`, and `hgdata` DataStore
    objects, but the number of vertices, number of hyperedges, and number of hypergraphs cannot change.

    To create a new HGNNHypergraph object with an additional vertex, use `add_vertex`.

"""
function add_vertex!(::HGNNHypergraph{T, D}; ::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}
    throw("Not implemented! Number of vertices in HGNNHypergraph is fixed.")
end

function add_vertex(hg::HGNNHypergraph{T, D}, features::DataStore; hyperedges::D = D(), hypergraph_id::Int = 1) where {T <: Real, D <: AbstractDict{Int,T}}
    @boundscheck (checkbounds(hg,1,k) for k in keys(hyperedges))

    # Verify that all all expected properties are present
    # Additional properties in `features` that are not in `hg` will be ignored

    data = Dict{Symbol, Any}()
    for key in keys(hg.vdata)
        @assert key in keys(features) && numobs(features.key) == 1
        @assert typeof(features.key) === typeof(hg.vdata.key)
        data[key] = cat_features(hg.vdata.key, features.key)
    end

    v2he = deepcopy(hg.v2he)
    he2v = deepcopy(hg.he2v)

    push!(v2he, hyperedges)

    ix = length(v2he)
    for k in keys(hyperedges)
        he2v[k][ix] = hyperedges[k]
    end

    if isnothing(hg.hypergraph_ids)
        hypergraph_ids = nothing
    else
        hypergraph_ids = cat(
            hg.hypergraph_ids,
            convert(typeof(hg.hypergraph_ids), [hypergraph_id]);
            dims=1
        )
    end

    return HGNNHypergraph(
        v2he,
        he2v,
        ix,
        hg.num_hyperedges,
        hg.num_hypergraphs,
        hypergraph_ids,
        data,
        hg.hedata,
        hg.hgdata
    )
end

# TODO: you are here


"""
    remove_vertex!(::HGNNHypergraph, ::Int)

    This function is not implemented for HGNNHypergraph.
        
    The basic hypergraph structure of HGNNHypergraph (i.e., the number of vertices, the hyperedges, and the hypergraph
    IDs) are not mutable. Users can change the features in the `vdata`, `hedata`, and `hgdata` DataStore
    objects, but the number of vertices, number of hyperedges, and number of hypergraphs cannot change.

    To create a new HGNNHypergraph object with a vertex removed, use `remove_vertex`.

"""
function remove_vertex!(::HGNNHypergraph, ::Int)
    throw("Not implemented! Number of vertices in HGNNHypergraph is fixed.")
end


"""
    add_hyperedge!(::HGNNHypergraph{T, D}; ::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}

    This function is not implemented for HGNNHypergraph.
        
    The basic hypergraph structure of HGNNHypergraph (i.e., the number of vertices, the hyperedges, and the hypergraph
    IDs) are not mutable. Users can change the features in the `vdata`, `hedata`, and `hgdata` DataStore
    objects, but the number of vertices, number of hyperedges, and number of hypergraphs cannot change.

    To create a new HGNNHypergraph object with an additional hyperedge, use `add_hyperedge`.

"""
function add_hyperedge!(::HGNNHypergraph{T, D}; ::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}
    throw("Not implemented! Number of hyperedges in HGNNHypergraph is fixed.")
end


"""
    remove_hyperedge!(::HGNNHypergraph, ::Int)
    
    This function is not implemented for HGNNHypergraph.
        
    The basic hypergraph structure of HGNNHypergraph (i.e., the number of vertices, the hyperedges, and the hypergraph
    IDs) are not mutable. Users can change the features in the `vdata`, `hedata`, and `hgdata` DataStore
    objects, but the number of vertices, number of hyperedges, and number of hypergraphs cannot change.

    To create a new HGNNHypergraph object with a hyperedge removed, use `remove_hyperedge`.
"""
function remove_hyperedge!(::HGNNHypergraph, ::Int)
    throw("Not implemented! Number of hyperedges in HGNNHypergraph is fixed.")
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

# TODO: setters and getters
hasvertexmeta(::Type{HGNNDiHypergraph}) = true
hasvertexmeta(X::HGNNDiHypergraph) = true
hashyperedgemeta(::Type{HGNNDiHypergraph}) = true
hashyperedgemeta(X::HGNNDiHypergraph) = true


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
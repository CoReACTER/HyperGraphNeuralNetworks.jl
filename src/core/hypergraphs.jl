# TODO: what functions from SimpleHypergraphs/SimpleDirectedHypergraphs need to be implemented to finish interface?


"""
   HGNNHypergraph{T<:Real, D<:AbstractDict{Int,T}} <: AbstractHGNNHypergraph{Union{T, Nothing}}

An undirected hypergraph type for use in hypergraph neural networks

**Constructors**

    HGNNHypergraph(
        h::AbstractSimpleHypergraph{T};
        hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        vdata::Union{DataStore, Nothing} = nothing,
        hedata::Union{DataStore, Nothing} = nothing,
        hgdata::Union{DataStore, Nothing} = nothing
    ) where {T<:Real}

    Construct an `HGNNHypergraph` from a previously constructed hypergraph. Optionally, the user can specify
    what hypergraph each vertex belongs to (if multiple distinct hypergraphs are included), as well as vertex,
    hyperedge, and hypergraph features.

    function HGNNHypergraph(
        incidence::AbstractMatrix{Union{T, Nothing}};
        hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        vdata::Union{DataStore, Nothing} = nothing,
        hedata::Union{DataStore, Nothing} = nothing,
        hgdata::Union{DataStore, Nothing} = nothing
    ) where {T<:Real}

    Construct an `HGNNHypergraph` from an incidence matrix. The incidence matrix is an `M`x`N` matrix, where `M` is the
    number of vertices and `N` is the number of hyperedges.

    function HGNNHypergraph(num_nodes::T; vdata=nothing, kws...) where {T<:Integer}

    Construct an `HGNNHypergraph` with no hyperedges and `num_nodes` vertices.

    function HGNNHypergraph(; num_nodes=nothing, vdata=nothing, kws...)

    Construct an `HGNNHypergraph` with minimal (perhaps no) information.


**Arguments**

    * `T` : type of weight values stored in the hypergraph's incidence matrix
    * `D` : dictionary type for storing values; the default is `Dict{Int, T}`
    * `hypergraph_ids` : Nothing (implying that all vertices belong to the same hypergraph) or a vector of ID integers
    * `vdata` : an optional DataStore (from GNNGraphs.jl) containing vertex-level features. Each entry in `vdata`
        should have `M` entries/observations, where `M` is the number of vertices in the hypergraph
    * `hedata` : an optional DataStore containing hyperedge-level features. Each entry in `hedata` should have `N`
        entries/observations, where `N` is the number of hyperedges in the hypergraph
    * `hgdata` : an optional DataStore containing hypergraph-level features. Each entry in `hgdata` should have `G`
        entries/observations, where `G` is the number of hypergraphs in the HGNNHypergraph (note: the maximum index
        in `hypergraph_ids` should be `G`)
    * `incidence` : a matrix representation; rows are vertices and columns are hyperedges
    * `num_nodes` : the number of vertices in the hypergraph (i.e., `M`)
"""
struct HGNNHypergraph{T<:Real, D<:AbstractDict{Int,T}} <: AbstractHGNNHypergraph{Union{T, Nothing}}
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
    h::AbstractSimpleHypergraph;
    hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    vdata = nothing,
    hedata = nothing,
    hgdata = nothing
)
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
        deepcopy(h.v2he),
        deepcopy(h.he2v),
        nhv(h),
        nhe(h),
        nhg,
        hypergraph_ids,
        vdata,
        hedata,
        hgdata
    )
end

function HGNNHypergraph(
    incidence::Matrix{Union{T, Nothing}};
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
function add_vertex!(hg::HGNNHypergraph{T, D}; hyperedges::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}
    throw("Not implemented! Number of vertices in HGNNHypergraph is fixed.")
end


"""
    add_vertex(
        hg::HGNNHypergraph{T, D},
        features::DataStore;
        hyperedges::D = D(),
        hypergraph_id::Int = 1
    ) where {T <: Real, D <: AbstractDict{Int,T}}

    Create a new HGNNHypergraph that adds a vertex to an existing hypergraph `hg`. Note that the `features` DataStore
    is not optional, but if the input hypergraph has no vertex data, this can be empty. Optionally, the vertex can be
    added to existing hyperedges. The `hyperedges` parameter presents a dictionary of hyperedge identifiers and values
    stored at the hyperedges.
"""
function add_vertex(
    hg::HGNNHypergraph{T, D},
    features::DataStore;
    hyperedges::D = D(),
    hypergraph_id::Int = 1
) where {T <: Real, D <: AbstractDict{Int,T}}
    @boundscheck (checkbounds(hg,1,k) for k in keys(hyperedges))
    @assert isnothing(hg.hypergraph_ids) || hypergraph_id <= hg.num_hypergraphs

    # Verify that all all expected properties are present
    # Additional properties in `features` that are not in `hg` will be ignored
    if !isnothing(hg.vdata)
        data_dict = Dict{Symbol, Any}()
        for key in keys(hg.vdata)
            @assert key in keys(features) && numobs(features.key) == 1
            @assert typeof(features.key) === typeof(hg.vdata.key)
            data_dict[key] = cat_features(hg.vdata.key, features.key)
        end
        data = DataStore(data_dict)
    else
        data = nothing
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
    remove_vertex(hg::HGNNHypergraph, v::Int)

    Removes the vertex `v` from a given `HGNNHypergraph` `hg`. Note that this creates a new HGNNHypergraph, as
    HGNNHypergraph objects are immutable.
"""
function remove_vertex(hg::HGNNHypergraph, v::Int)
    n = nhv(hg)

    mask_to_keep = trues(nhv(hg))
    mask_to_keep[v] = false

    # Extract all data NOT for the given vertex
    if !isnothing(hg.vdata)
        data = getobs(hg.hedata, mask_to_keep)
    else
        data = nothing
    end

    v2he = deepcopy(hg.v2he)[mask_to_keep]

    # Decrement vertex indices where needed
    he2v = deepcopy(hg.he2v)
    for he in he2v
        if v < n
            delete!(he, v)
            for key in keys(he)
                if key > v
                    he[key - 1] = he[key]
                    delete!(he, key)
                end
            end
        else
            delete!(he, v)
        end
    end

    if isnothing(hg.hypergraph_ids)
        hypergraph_ids = nothing
    else
        hypergraph_ids = hg.hypergraph_ids[mask_to_keep]
    end

    return HGNNHypergraph(
        v2he,
        he2v,
        n - 1,
        hg.num_hyperedges,
        hg.num_hypergraphs,
        hypergraph_ids,
        data,
        hg.hedata,
        hg.hgdata
    )

end


"""
    add_hyperedge!(::HGNNHypergraph{T, D}; ::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}

    This function is not implemented for HGNNHypergraph.
        
    The basic hypergraph structure of HGNNHypergraph (i.e., the number of vertices, the hyperedges, and the hypergraph
    IDs) are not mutable. Users can change the features in the `vdata`, `hedata`, and `hgdata` DataStore
    objects, but the number of vertices, number of hyperedges, and number of hypergraphs cannot change.

    To create a new HGNNHypergraph object with an additional hyperedge, use `add_hyperedge`.

"""
function add_hyperedge!(hg::HGNNHypergraph{T, D}; vertices::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}
    throw("Not implemented! Number of hyperedges in HGNNHypergraph is fixed.")
end

"""
    add_hyperedge(
        hg::HGNNHypergraph{T, D},
        features::DataStore;
        vertices::D = D(),
    ) where {T <: Real, D <: AbstractDict{Int,T}}

    Adds a hyperedge to a given `HGNNHypergraph`. Because `HGNNHypergraph` is immutable, this creates a new
    `HGNNHypergraph`. Optionally, existing vertices can be added to the created hyperedge. The paramater `vertices`
    represents a dictionary of vertex identifiers andvalues stored at the hyperedges. Note that the `features`
    DataStore is not optional; however, if `hg` has no `hedata` (i.e., if `hedata` is nothing), this can be empty.
"""
function add_hyperedge(
    hg::HGNNHypergraph{T, D},
    features::DataStore;
    vertices::D = D(),
) where {T <: Real, D <: AbstractDict{Int,T}}

    @boundscheck (checkbounds(hg,k,1) for k in keys(vertices))

    # Verify that all all expected properties are present
    # Additional properties in `features` that are not in `hg` will be ignored
    if !isnothing(hg.hedata)
        data = Dict{Symbol, Any}()
        for key in keys(hg.hedata)
            @assert key in keys(features) && numobs(features.key) == 1
            @assert typeof(features.key) === typeof(hg.hedata.key)
            data[key] = cat_features(hg.hedata.key, features.key)
        end
    else
        data = nothing
    end

    v2he = deepcopy(hg.v2he)
    he2v = deepcopy(hg.he2v)

    push!(he2v, vertices)

    ix = length(he2v)
    for k in keys(vertices)
        v2he[k][ix] = vertices[k]
    end

    return HGNNHypergraph(
        v2he,
        he2v,
        hg.num_vertices,
        ix,
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        hg.vdata,
        DataStore(data),
        hg.hgdata
    )

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

"""
    remove_hyperedge(hg::HGNNHypergraph, e::Int)

    Removes the hyperedge `e` from a given undirected HGNNHypergraph `hg`. Note that this function creates a new
    HGNNHypergraph.
"""
function remove_hyperedge(hg::HGNNHypergraph, e::Int)
    ne = nhe(hg)
	@assert(e <= ne)

    # Extract all data NOT for the given hyperedge
    # TODO: simplify this; see transform.jl
    mask_to_keep = trues(nhe(hg))
    mask_to_keep[e] = false

    data = getobs(hg.hedata, mask_to_keep)

    he2v = deepcopy(hg.he2v)[mask_to_keep]

    # Decrement vertex indices where needed
    v2he = deepcopy(hg.v2he)
    for v in v2he
        if e < ne
            delete!(v, e)
            for key in keys(v)
                if key > e
                    v[key - 1] = v[key]
                    delete!(v, key)
                end
            end
        else
            delete!(v, e)
        end
    end
    
    return HGNNHypergraph(
        v2he,
        he2v,
        hg.num_vertices,
        ne - 1,
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        hg.vdata,
        data,
        hg.hgdata
    )

end

"""
    remove_vertices(hg::HGNNHypergraph, to_remove::AbstractVector{Int})

    Removes a set of vertices (`to_remove`) from an undirected hypergraph `hg` by index
"""
function remove_vertices(hg::HGNNHypergraph, to_remove::AbstractVector{Int})
    mask_to_keep = trues(nhv(hg))
    mask_to_keep[to_remove] .= false
    
    he2v = deepcopy(hg.he2v)
    for i in to_remove
        for he in keys(hg.v2he[i])
            delete!(he2v[he], i)
        end
    end

    v2he = hg.v2he[mask_to_keep]

    vdata = getobs(hg.vdata, mask_to_keep)

    return HGNNHypergraph(
        v2he,
        he2v,
        length(v2he),
        hg.num_hyperedges,
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        vdata,
        hg.hedata,
        hg.hgdata
    )
end

"""
    remove_hyperedges(hg::HGNNHypergraph, to_remove::AbstractVector{Int})

    Removes a set of hyperedges (`to_remove`) from an undirected hypergraph `hg` by index
"""
function remove_hyperedges(hg::HGNNHypergraph, to_remove::AbstractVector{Int})
    mask_to_keep = trues(nhe(hg))
    mask_to_keep[to_remove] .= false
    
    v2he = deepcopy(hg.v2he)
    for i in to_remove
        for v in keys(hg.he2v[i])
            delete!(v2he[v], i)
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

Base.zero(::Type{H}) where {H <: HGNNHypergraph} = H(0)

"""
    copy(hg::HGNNHypergraph; deep=false)

    Create a copy of `hg`. If `deep` is `true`, then copy will be a deep copy (equivalent to `deepcopy(hg)`),
    therwise it will be a shallow copy with the same underlying hypergraph data.
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


"""
   HGNNDiHypergraph{T<:Real, D<:AbstractDict{Int,T}} <: AbstractHGNNDiHypergraph{Tuple{Union{T, Nothing}, Union{T, Nothing}}}

A directed hypergraph type for use in hypergraph neural networks

**Constructors**

    HGNNDiHypergraph(
        h::AbstractDirectedHypergraph{T};
        hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        vdata::Union{DataStore, Nothing} = nothing,
        hedata::Union{DataStore, Nothing} = nothing,
        hgdata::Union{DataStore, Nothing} = nothing
    ) where {T<:Real}

    Construct a `HGNNDiHypergraph` from a previously constructed directed hypergraph. Optionally, the user can specify
    what hypergraph each vertex belongs to (if multiple distinct hypergraphs are included), as well as vertex,
    hyperedge, and hypergraph features.

    HGNNDiHypergraph(
        incidence_tail::AbstractMatrix{Union{T, Nothing}},
        incidence_head::AbstractMatrix{Union{T, Nothing}};
        hypergraph_ids::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        vdata::Union{DataStore, Nothing} = nothing,
        hedata::Union{DataStore, Nothing} = nothing,
        hgdata::Union{DataStore, Nothing} = nothing
    ) where {T<:Real}

    Construct a `HGNNDiHypergraph` from incidence matrices `incidence_tail` (containing information regarding which 
    vertices are in the tails of which hyperedges) and `incidence_head` (containing the corresponding information
    about the heads). The incidence matrices have dimensions `M`x`N`, where `M` is the
    number of vertices and `N` is the number of hyperedges.

    function HGNNDiHypergraph(num_nodes::T; vdata=nothing, kws...) where {T<:Integer}

    Construct a `HGNNDiHypergraph` with no hyperedges and `num_nodes` vertices.

    function HGNNDiHypergraph(; num_nodes=nothing, vdata=nothing, kws...)

    Construct a `HGNNDiHypergraph` with minimal (perhaps no) information.


**Arguments**

    * `T` : type of weight values stored in the hypergraph's incidence matrix
    * `D` : dictionary type for storing values; the default is `Dict{Int, T}`
    * `hypergraph_ids` : Nothing (implying that all vertices belong to the same hypergraph) or a vector of ID integers
    * `vdata` : an optional DataStore (from GNNGraphs.jl) containing vertex-level features. Each entry in `vdata`
        should have `M` entries/observations, where `M` is the number of vertices in the hypergraph
    * `hedata` : an optional DataStore containing hyperedge-level features. Each entry in `hedata` should have `N`
        entries/observations, where `N` is the number of hyperedges in the hypergraph
    * `hgdata` : an optional DataStore containing hypergraph-level features. Each entry in `hgdata` should have `G`
        entries/observations, where `G` is the number of hypergraphs in the HGNNHypergraph (note: the maximum index
        in `hypergraph_ids` should be `G`)
    * `incidence_tail` : a matrix representation of the tails of the hypergraph(s); rows are vertices and columns are
        hyperedges
    * `incidence_head` : a matrix representation of the heads of the hypergraph(s); rows are vertices and columns are
        hyperedges
    * `num_nodes` : the number of vertices in the hypergraph (i.e., `M`)
"""

struct HGNNDiHypergraph{T<:Real, D<:AbstractDict{Int,T}} <: AbstractHGNNDiHypergraph{Tuple{Union{T, Nothing}, Union{T, Nothing}}}
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
    vdata = nothing,
    hedata = nothing,
    hgdata = nothing
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

# TODO: modification functions

"""
    (::HGNNDiHypergraph{T, D}; ::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}

    This function is not implemented for HGNNDiHypergraph.
        
    The basic hypergraph structure of HGNNDiHypergraph (i.e., the number of vertices, the hyperedges, and the
    hypergraph IDs) are not mutable. Users can change the features in the `vdata`, `hedata`, and `hgdata` DataStore
    objects, but the number of vertices, number of hyperedges, and number of hypergraphs cannot change.

    To create a new HGNNDiHypergraph object with an additional vertex, use `add_vertex`.

"""
function add_vertex!(hg::HGNNDiHypergraph{T, D}; hyperedges::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}
    throw("Not implemented! Number of vertices in HGNNDiHypergraph is fixed.")
end


"""
    add_vertex(
        hg::HGNNDiHypergraph{T, D},
        features::DataStore;
        hyperedges_tail::D = D(),
        hyperedges_head::D = D(),
        hypergraph_id::Int = 1
    ) where {T <: Real, D <: AbstractDict{Int,T}}

    Create a new HGNNDiHypergraph that adds a vertex to an existing directed hypergraph `hg`. Note that the `features`
    DataStore is not optional, but if the input hypergraph has no vertex data, this can be empty. Optionally, the
    vertex can be added to existing hyperedges. The `hyperedges_tail` and `hyperedges_head` parameters include
    dictionaries of hyperedge identifiers and values stored at the hyperedges.
"""
function add_vertex(
    hg::HGNNDiHypergraph{T, D},
    features::DataStore;
    hyperedges_tail::D = D(),
    hyperedges_head::D = D(),
    hypergraph_id::Int = 1
) where {T <: Real, D <: AbstractDict{Int,T}}
    @boundscheck (checkbounds(hg,1,k) for k in keys(hyperedges_tail))
    @boundscheck (checkbounds(hg,1,k) for k in keys(hyperedges_head))
    @assert isnothing(hg.hypergraph_ids) || hypergraph_id <= hg.num_hypergraphs

    # Verify that all all expected properties are present
    # Additional properties in `features` that are not in `hg` will be ignored
    if !isnothing(hg.vdata)
        data = Dict{Symbol, Any}()
        for key in keys(hg.vdata)
            @assert key in keys(features) && numobs(features.key) == 1
            @assert typeof(features.key) === typeof(hg.vdata.key)
            data[key] = cat_features(hg.vdata.key, features.key)
        end
    else
        data = nothing
    end

    v2he_tail = deepcopy(hg.hg_tail.v2he)
    v2he_head = deepcopy(hg.hg_head.v2he)

    he2v_tail = deepcopy(hg.hg_tail.he2v)
    he2v_head = deepcopy(hg.hg_head.he2v)

    push!(v2he_tail, hyperedges_tail)
    push!(v2he_head, hyperedges_head)

    ix = length(v2he_tail)
    for k in keys(hyperedges_tail)
        he2v_tail[k][ix] = hyperedges_tail[k]
    end

    for k in keys(hyperedges_head)
        he2v_head[k][ix] = hyperedges_head[k]
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

    return HGNNDiHypergraph(
        Hypergraph(v2he_tail, he2v_tail, nothing, nothing),
        Hypergraph(v2he_tail, he2v_tail, nothing, nothing),
        ix,
        hg.num_hyperedges,
        hg.num_hypergraphs,
        hypergraph_ids,
        DataStore(data),
        hg.hedata,
        hg.hgdata
    )
end


"""
    remove_vertex!(::HGNNDiHypergraph, ::Int)

    This function is not implemented for HGNNDiHypergraph.
        
    The basic hypergraph structure of HGNNDiHypergraph (i.e., the number of vertices, the hyperedges, and the hypergraph
    IDs) are not mutable. Users can change the features in the `vdata`, `hedata`, and `hgdata` DataStore
    objects, but the number of vertices, number of hyperedges, and number of hypergraphs cannot change.

    To create a new HGNNDiHypergraph object with a vertex removed, use `remove_vertex`.

"""
function remove_vertex!(hg::HGNNDiHypergraph, v::Int)
    throw("Not implemented! Number of vertices in HGNNDiHypergraph is fixed.")
end

"""
    remove_vertex(hg::HGNNDiHypergraph, v::Int)

    Removes the vertex `v` from a given `HGNNDiHypergraph` `hg`. Note that this creates a new HGNNDiHypergraph, as
    HGNNDiHypergraph objects are immutable.
"""
function remove_vertex(hg::HGNNDiHypergraph, v::Int)
    n = nhv(hg)

    # Extract all data NOT for the given vertex
    mask_to_keep = trues(nhv(hg))
    mask_to_keep[v] = false
    if !isnothing(hg.vdata)
        data = getobs(hg.vdata, mask_to_keep)
    else
        data = nothing
    end

    v2he_tail = deepcopy(hg.hg_tail.v2he)[mask_to_keep]
    v2he_head = deepcopy(hg.hg_head.v2he)[mask_to_keep]

    # Decrement vertex indices where needed
    he2v_tail = deepcopy(hg.hg_tail.he2v)
    for he in he2v_tail
        if v < n
            delete!(he, v)
            for key in keys(he)
                if key > v
                    he[key - 1] = he[key]
                    delete!(he, key)
                end
            end
        else
            delete!(he, v)
        end
    end

    # Decrement vertex indices where needed
    he2v_head = deepcopy(hg.hg_head.he2v)
    for he in he2v_head
        if v < n && haskey(he, n)
            for i in v:n-1
                he[i] = he[i+1]
            end
            delete!(he, n)
        else
            delete!(he, v)
        end
    end

    if isnothing(hg.hypergraph_ids)
        hypergraph_ids = nothing
    else
        hypergraph_ids = hg.hypergraph_ids[Not(v)]
    end

    return HGNNDiHypergraph(
        Hypergraph(v2he_tail, he2v_tail, nothing, nothing),
        Hypergraph(v2he_head, he2v_head, nothing, nothing),
        n - 1,
        hg.num_hyperedges,
        hg.num_hypergraphs,
        hypergraph_ids,
        data,
        hg.hedata,
        hg.hgdata
    )

end


"""
    add_hyperedge!(::HGNNDiHypergraph{T, D}; ::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}

    This function is not implemented for HGNNDiHypergraph.
        
    The basic hypergraph structure of HGNNDiHypergraph (i.e., the number of vertices, the hyperedges, and the hypergraph
    IDs) are not mutable. Users can change the features in the `vdata`, `hedata`, and `hgdata` DataStore
    objects, but the number of vertices, number of hyperedges, and number of hypergraphs cannot change.

    To create a new HGNNDiHypergraph object with an additional hyperedge, use `add_hyperedge`.

"""
function add_hyperedge!(hg::HGNNDiHypergraph{T, D}; vertices::D = D()) where {T <: Real, D <: AbstractDict{Int,T}}
    throw("Not implemented! Number of hyperedges in HGNNHypergraph is fixed.")
end

"""
    add_hyperedge(
        hg::HGNNDiHypergraph{T, D},
        features::DataStore;
        vertices_tail::D = D(),
        vertices_head::D = D(),
    ) where {T <: Real, D <: AbstractDict{Int,T}}

    Adds a hyperedge to a given `HGNNDiHypergraph`. Because `HGNNDiHypergraph` is immutable, this creates a new
    `HGNNDiHypergraph`. Optionally, existing vertices can be added to the tail and/or head of the hyperedge. The
    paramaters `vertices_tail` and `vertices_head` represent dictionaries of vertex identifiers and values stored at
    the tail and head of hyperedges, respectively. Note that the `features` DataStore is not optional; however, if `hg`
    has no `hedata` (i.e., if `hedata` is nothing), this can be empty.
"""
function add_hyperedge(
    hg::HGNNDiHypergraph{T, D},
    features::DataStore;
    vertices_tail::D = D(),
    vertices_head::D = D(),
) where {T <: Real, D <: AbstractDict{Int,T}}
    @boundscheck (checkbounds(hg,k,1) for k in keys(vertices_tail))
    @boundscheck (checkbounds(hg,k,1) for k in keys(vertices_head))

    # Verify that all all expected properties are present
    # Additional properties in `features` that are not in `hg` will be ignored
    if !isnothing(hg.hedata)
        data_dict = Dict{Symbol, Any}()
        for key in keys(hg.hedata)
            @assert key in keys(features) && numobs(features.key) == 1
            @assert typeof(features.key) === typeof(hg.hedata.key)
            data_dict[key] = cat_features(hg.hedata.key, features.key)
        end
        data = DataStore(data_dict)
    else
        data = nothing
    end

    v2he_tail = deepcopy(hg.hg_tail.v2he)
    v2he_head = deepcopy(hg.hg_head.v2he)

    he2v_tail = deepcopy(hg.hg_tail.he2v)
    he2v_head = deepcopy(hg.hg_head.he2v)

    push!(he2v_tail, vertices_tail)
    push!(he2v_head, vertices_head)

    ix = length(he2v_tail)
    for k in keys(vertices_tail)
        v2he_tail[k][ix] = vertices_tail[k]
    end

    for k in keys(vertices_head)
        v2he_head[k][ix] = vertices_head[k]
    end

    return HGNNHypergraph(
        Hypergraph(v2he_tail, he2v_tail, nothing, nothing),
        Hypergraph(v2he_head, he2v_head, nothing, nothing),
        he.num_vertices,
        ix,
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        hg.vdata,
        data,
        hg.hgdata
    )

end

"""
    remove_hyperedge!(::HGNNHypergraph, ::Int)
    
    This function is not implemented for HGNNHypergraph.
        
    The basic hypergraph structure of HGNNHypergraph (i.e., the number of vertices, the hyperedges, and the hypergraph
    IDs) are not mutable. Users can change the features in the `vdata`, `hedata`, and `hgdata` DataStore
    objects, but the number of vertices, number of hyperedges, and number of hypergraphs cannot change.

    To create a new HGNNHypergraph object with a hyperedge removed, use `remove_hyperedge`.
"""
function remove_hyperedge!(::HGNNDiHypergraph, ::Int)
    throw("Not implemented! Number of hyperedges in HGNNHypergraph is fixed.")
end

"""
    remove_hyperedge(hg::HGNNDiHypergraph, e::Int)

Removes the hyperedge `e` from a given undirected HGNNDiHypergraph `hg`. Note that this function creates a new
HGNNDiHypergraph.
"""
function remove_hyperedge(hg::HGNNDiHypergraph, e::Int)
    ne = nhe(hg)
	@assert(e <= ne)

    # Extract all data NOT for the given hyperedge
    mask_to_keep = trues(nhe(hg))
    mask_to_keep[e] = false
    if !isnothing(hg.hedata)
        data = getobs(hg.hedata, mask_to_keep)
    else
        data = nothing
    end

    he2v_tail = deepcopy(hg.hg_tail.he2v)[mask_to_keep]
    he2v_head = deepcopy(hg.hg_head.he2v)[mask_to_keep]

    # Decrement vertex indices where needed
    v2he_tail = deepcopy(hg.hg_tail.v2he)
    for v in v2he_tail
        if e < ne
            delete!(v, e)
            for key in keys(v)
                if key > e
                    v[key - 1] = v[key]
                    delete!(v, key)
                end
            end
        else
            delete!(v, e)
        end
    end

    v2he_head = deepcopy(hg.hg_head.v2he)
    for v in v2he_head
        if e < ne
            delete!(v, e)
            for key in keys(v)
                if key > e
                    v[key - 1] = v[key]
                    delete!(v, key)
                end
            end
        else
            delete!(v, e)
        end
    end

    if isnothing(hg.hypergraph_ids)
        hypergraph_ids = nothing
    else
        hypergraph_ids = hg.hypergraph_ids[Not(e)]
    end
    
    return HGNNDiHypergraph(
        Hypergraph(v2he_tail, he2v_tail, nothing, nothing),
        Hypergraph(v2he_head, he2v_head, nothing, nothing),
        he.num_vertices,
        ne - 1,
        hg.num_hypergraphs,
        hypergraph_ids,
        hg.vdata,
        data,
        hg.hgdata
    )

end

"""
    remove_vertices(hg::HGNNDiHypergraph, to_remove::AbstractVector{Int})

    Removes a set of vertices (`to_remove`) from a directed hypergraph `hg` by index
"""
function remove_vertices(hg::HGNNDiHypergraph, to_remove::AbstractVector{Int})
    mask_to_keep = trues(nhe(hg))
    mask_to_keep[to_remove] .= false

    he2v_tail = deepcopy(hg.hg_tail.he2v)
    he2v_head = deepcopy(hg.hg_head.he2v)

    for i in to_remove
        for he in keys(hg.hg_tail.v2he[i])
            delete!(he2v_tail[he], i)
        end
        for he in keys(hg.hg_head.v2he[i])
            delete!(he2v_head[he], i)
        end
    end

    v2he_tail = hg.hg_tail.v2he[mask_to_keep]
    v2he_head = hg.hg_head.v2he[mask_to_keep]

    vdata = getobs(hg.vdata, mask_to_keep)

    HGNNDiHypergraph(
        Hypergraph(v2he_tail, he2v_tail, Vector{Nothing}(nothing, length(v2he_tail)), Vector{Nothing}(nothing, length(he2v_tail))),
        Hypergraph(v2he_head, he2v_head, Vector{Nothing}(nothing, length(v2he_head)), Vector{Nothing}(nothing, length(he2v_head))),
        length(v2he_tail),
        hg.num_hyperedges,
        hg.num_hypergraphs,
        hg.hypergraph_ids,
        vdata,
        hg.hedata,
        hg.hgdata
    )
end

"""
    remove_hyperedges(hg::HGNNDiHypergraph, to_remove::AbstractVector{Int})

    Removes a set of hyperedges (`to_remove`) from a directed hypergraph `hg` by index
"""
function remove_hyperedges(hg::HGNNDiHypergraph, to_remove::AbstractVector{Int})
    mask_to_keep = trues(nhe(hg))
    mask_to_keep[to_remove] .= false

    v2he_tail = deepcopy(hg.hg_tail.v2he)
    v2he_head = deepcopy(hg.hg_head.v2he)

    for i in to_remove
        for v in keys(hg.hg_tail.he2v[i])
            delete!(v2he_tail[v], i)
        end
        for v in keys(hg.hg_head.he2v[i])
            delete!(v2he_head[v], i)
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

function add_vertices(
    hg::HGNNHypergraph{T, D},
    n::Int,
    features::DataStore;
    hyperedges::AbstractVector{D} = Vector{D}(D(), n),
    hypergraph_ids::AbstractVector{Int} = ones(n)
) where {T <: Real, D <: AbstractDict{Int, T}}

    for i in 1:n
        hg = add_vertex(hg, getobs(features, i); hyperedges=hyperedges, hypergraph_id=hypergraph_ids[i])
    end

    hg

end

function add_vertices(
    hg::HGNNDiHypergraph{T, D},
    n::Int,
    features::DataStore;
    hyperedges_tail::AbstractVector{D} = Vector{D}(D(), n),
    hyperedges_head::AbstractVector{D} = Vector{D}(D(), n),
    hypergraph_ids::AbstractVector{Int} = ones(n)
) where {T <: Real, D <: AbstractDict{Int, T}}

    for i in 1:n
        hg = add_vertex(
            hg,
            getobs(features, i);
            hyperedges_tail = hyperedges_tail[i],
            hyperedges_head = hyperedges_head[i],
            hypergraph_id = hypergraph_ids[i]
        )
    end

    hg

end

function add_hyperedges(
    hg::HGNNHypergraph{T, D},
    n::Int,
    features::DataStore;
    vertices::AbstractVector{D} = Vector{D}(D(), n)    
) where {T <: Real, D <: AbstractDict{Int, T}}

    for i in 1:n
        hg = add_hyperedge(hg, getobs(features, i); vertices=vertices[i])
    end

    hg
end

function add_hyperedges(
    hg::HGNNDiHypergraph{T, D},
    n::Int,
    features::DataStore;
    vertices_tail::AbstractVector{D} = Vector{D}(D(), n),
    vertices_head::AbstractVector{D} = Vector{D}(D(), n)
) where {T <: Real, D <: AbstractDict{Int, T}}

    for i in 1:n
        hg = add_hyperedge(hg, getobs(features, i); vertices_tail=vertices_tail[i], vertices_head=vertices_head[i])
    end

    hg
end


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
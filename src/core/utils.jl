
"""
    check_num_vertices(hg::H, x::AbstractArray) where {H <: AbstractHGNNHypergraph}
    check_num_vertices(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNHypergraph}

    Ensure that an array abstract array, tuple, or named tuple (i.e., tensor) `x` has the appropriate final dimension,
    equal to the number of vertices in the associated hypergraph `hg`
"""
function check_num_vertices(hg::H, x::AbstractArray) where {H <: AbstractHGNNHypergraph}
    @assert hg.num_vertices == size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension instead of num_vertices = $(hg.num_vertices)"
    return true
end

function check_num_vertices(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNHypergraph}
    map(q -> check_num_vertices(hg, q), x)
    return true
end

check_num_vertices(::H, ::Nothing) where {H <: AbstractHGNNHypergraph} = true


"""
    check_num_hyperedges(hg::H, x::AbstractArray) where {H <: AbstractHGNNHypergraph}
    check_num_hyperedges(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNHypergraph}

    Ensure that an array abstract array, tuple, or named tuple (i.e., tensor) `x` has the appropriate final dimension,
    equal to the number of hyperedges in the associated hypergraph `hg`
"""
function check_num_hyperedges(hg::H, x::AbstractArray) where {H <: AbstractHGNNHypergraph}
    @assert hg.num_hyperedges == size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension size instead of num_hyperedges=$(hg.num_hyperedges)"
    return true
end

function check_num_hyperedges(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNHypergraph}
    map(q -> check_num_hyperedges(hg, q), x)
    return true
end

check_num_hyperedges(::H, ::Nothing) where {H <: AbstractHGNNHypergraph} = true


"""
    check_num_vertices(hg::H, x::AbstractArray) where {H <: AbstractHGNNDiHypergraph}
    check_num_vertices(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNDiHypergraph}

    Ensure that an array abstract array, tuple, or named tuple (i.e., tensor) `x` has the appropriate final dimension,
    equal to the number of vertices in the associated directed hypergraph `hg`
"""
function check_num_vertices(hg::H, x::AbstractArray) where {H <: AbstractHGNNDiHypergraph}
    @assert hg.num_vertices == size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension instead of num_vertices = $(hg.num_vertices)"
    return true
end

function check_num_vertices(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNDiHypergraph}
    map(q -> check_num_vertices(hg, q), x)
    return true
end

check_num_vertices(::H, ::Nothing) where {H <: AbstractHGNNDiHypergraph} = true

"""
    check_num_hyperedges(hg::H, x::AbstractArray) where {H <: AbstractHGNNDiHypergraph}
    check_num_hyperedges(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNDiHypergraph}

    Ensure that an array abstract array, tuple, or named tuple (i.e., tensor) `x` has the appropriate final dimension,
    equal to the number of hyperedges in the associated directed hypergraph `hg`
"""
function check_num_hyperedges(hg::H, x::AbstractArray) where {H <: AbstractHGNNDiHypergraph}
    @assert hg.num_hyperedges == size(x, ndims(x)) "Got $(size(x, ndims(x))) as last dimension size instead of num_hyperedges=$(hg.num_hyperedges)"
    return true
end

function check_num_hyperedges(hg::H, x::Union{Tuple, NamedTuple}) where {H <: AbstractHGNNDiHypergraph}
    map(q -> check_num_hyperedges(hg, q), x)
    return true
end

check_num_hyperedges(::H, ::Nothing) where {H <: AbstractHGNNDiHypergraph} = true


# COPIED FROM GNNGraphs.jl

# Turns generic type into named tuple
normalize_graphdata(data::Nothing; n, kws...) = DataStore(n)

function normalize_graphdata(data; default_name::Symbol, kws...)
    normalize_graphdata(NamedTuple{(default_name,)}((data,)); default_name, kws...)
end

function normalize_graphdata(data::DataStore)
end

function normalize_graphdata(data::NamedTuple; default_name::Symbol, n::Int, 
        duplicate_if_needed::Bool = false, glob::Bool = false)
    # This had to workaround two Zygote bugs with NamedTuples
    # https://github.com/FluxML/Zygote.jl/issues/1071
    # https://github.com/FluxML/Zygote.jl/issues/1072 # TODO fixed. Can we simplify something?


    if n > 1
        @assert all(x -> x isa AbstractArray, data) "Non-array features provided."
    end

    if n <= 1 && glob == true
        @assert n == 1
        n = -1 # relax the case of a single graph, allowing to store arbitrary types
    #     # # If last array dimension is not 1, add a new dimension.
    #     # # This is mostly useful to reshape global feature vectors
    #     # # of size D to Dx1 matrices.
        # TODO remove this and handle better the batching of global features
        unsqz_last(v::AbstractArray) = size(v)[end] != 1 ? reshape(v, size(v)..., 1) : v
        unsqz_last(v) = v

        data = map(unsqz_last, data)
    end

    if n > 0
        if duplicate_if_needed
            function duplicate(v)
                if v isa AbstractArray && size(v)[end] == n ÷ 2
                    v = cat(v, v, dims = ndims(v))
                end
                return v
            end
            data = map(duplicate, data)
        end

        for x in data
            if x isa AbstractArray
                @assert size(x)[end] == n "Wrong size in last dimension for feature array, expected $n but got $(size(x)[end])."
            end
        end
    end

    return DataStore(n, data)
end
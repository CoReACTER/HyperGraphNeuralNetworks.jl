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
        index_groups::AbstractVector{V}
    ) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}}

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
                getobs(hg.hedata, sort(collect(keys(hemap)))),
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
    index_groups::AbstractVector{V}
) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}}
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
        index_groups::AbstractVector{V}
    ) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}}

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
                getobs(hg.hedata, sort(collect(keys(hemap)))),
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
    hg::HGNNDiHypergraph{T,D},
    index_groups::AbstractVector{V}
) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}}
    masks = BitVector[]

    for indgroup in index_groups
        push!(masks, BitArray(i in indgroup for i in 1:hg.num_vertices))
    end

    split_vertices(hg, masks)
end

function split_vertices(
    hg::HGNNDiHypergraph{T,D},
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
        index_groups::AbstractVector{V}
    ) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}}

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

        rel_vs = sort(collect(keys(vmap)))

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
    index_groups::AbstractVector{V}
) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}}

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
        index_groups::AbstractVector{V}
    ) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}}

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

        rel_vs = sort(collect(keys(vmap)))

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
    index_groups::AbstractVector{V}
) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}}
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
        index_groups::AbstractVector{V}
    ) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}}

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
                getobs(hg.hedata, sort(collect(keys(hemap)))),
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
    index_groups::AbstractVector{V}
) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}} 
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
        index_groups::AbstractVector{V}
    ) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}}

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
                getobs(hg.hedata, sort(collect(keys(hemap)))),
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
    index_groups::AbstractVector{V}
) where {T <: Real, D <: AbstractDict{Int, T}, V <: AbstractVector{Int}} 
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
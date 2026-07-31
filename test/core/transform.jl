using Random
using StatsBase
using LinearAlgebra
using Test
using Graphs
using GNNGraphs
import GNNGraphs: cat_features
using MLUtils
using SimpleHypergraphs
using SimpleDirectedHypergraphs
using HyperGraphNeuralNetworks

@testset "HyperGraphNeuralNetworks                                    transforms" begin
    # TODO: directed hyperedges

    hgnn = HGNNHypergraph(uh1.v2he, uh1.he2v, 11, 5, 2, uid1, DataStore(), DataStore(), DataStore())
    
    dhgnn = HGNNDiHypergraph(
        dh1;
        hypergraph_ids = did1,
        vdata = nothing,
        hedata = nothing,
        hgdata = nothing
    )

    # add_selfloops
    hgnn0 = add_selfloops(hgnn)
    @test hgnn0.num_hyperedges == 16
    for i in 1:hgnn0.num_vertices
        @test Dict{Int, Float64}(i => 1.0) in hgnn0.he2v
    end

    hgnn1 = add_hyperedge(hgnn, DataStore(); vertices=Dict{Int, Float64}(1 => 1.0))
    hgnn2 = add_selfloops(hgnn1)
    @test hgnn2.num_hyperedges == 16
    hgnn2 = add_selfloops(hgnn1; add_repeated_hyperedge=true)
    @test hgnn2.num_hyperedges == 17

    dhgnn0 = add_selfloops(dhgnn)
    @test dhgnn0.num_hyperedges == 16
    he_verts = collect(zip(Set.(keys.(dhgnn0.hg_tail.he2v)), Set.(keys.(dhgnn0.hg_head.he2v))))
    for i in 1:dhgnn0.num_vertices
        @test (Set(i), Set(i)) in he_verts
    end

    dhgnn1 = add_hyperedge(
        dhgnn,
        DataStore();
        vertices_tail=Dict{Int, Float64}(1 => 1.0),
        vertices_head=Dict{Int, Float64}(1 => 1.0)
    )
    dhgnn2 = add_selfloops(dhgnn1)
    @test dhgnn2.num_hyperedges == 16
    dhgnn2 = add_selfloops(dhgnn1; add_repeated_hyperedge=true)
    @test dhgnn2.num_hyperedges == 17

    # remove_selfloops
    hgnn1 = remove_selfloops(hgnn2)
    @test hgnn1.num_hyperedges == 5

    dhgnn1 = remove_selfloops(dhgnn2)
    @test dhgnn1.num_hyperedges == 5

    # remove_multihyperedges
    hgnn2 = add_hyperedge(hgnn, DataStore(); vertices=Dict{Int, Float64}(1 => 1.0, 2 => 2.0, 4 => 4.0))
    @test remove_multihyperedges(hgnn2).num_hyperedges == 5

    dhgnn2 = add_hyperedge(
        dhgnn,
        DataStore();
        vertices_tail=Dict{Int, Float64}(1 => 1.0, 2 => 2.0),
        vertices_head=Dict{Int, Float64}(4 => 4.0)
    )
    @test remove_multihyperedges(dhgnn2).num_hyperedges == 5

    # to_undirected
    @test Set.(keys.(to_undirected(dhgnn).he2v)) == Set.(keys.(hgnn.he2v))

    # combine_hypergraphs / MLUtils.batch
    hg1 = HGNNHypergraph(
        [
            1.0     nothing
            nothing 2.0
            nothing 3.0
        ];
        hypergraph_ids=[1,2,2],
        vdata = rand(Float64, 5, 3),
        hedata = rand(Float64, 5, 2),
        hgdata = rand(Float64, 5, 2)
    )
    hg2 = HGNNHypergraph(
        [
            1.0     nothing     1.0
            nothing 2.0         3.0
        ];
        hypergraph_ids=[1,1],
        vdata = rand(Float64, 5, 2),
        hedata = rand(Float64, 5, 3),
        hgdata = rand(Float64, 5, 1)
    )

    hg_comb1 = combine_hypergraphs(hg1, hg2)
    @test hg_comb1.num_vertices == 5
    @test hg_comb1.num_hyperedges == 5
    @test hg_comb1.num_hypergraphs == 3
    @test hg_comb1.hypergraph_ids == [1, 2, 2, 3, 3]
    @test hg_comb1.vdata == cat_features(hg1.vdata, hg2.vdata)
    @test hg_comb1.hedata == cat_features(hg1.hedata, hg2.hedata)
    @test hg_comb1.hgdata == cat_features(hg1.hgdata, hg2.hgdata)

    hg_comb2 = combine_hypergraphs(hg1, hg1, hg2, hg2)
    @test hg_comb2.num_vertices == 10
    @test hg_comb2.num_hyperedges == 10
    @test hg_comb2.num_hypergraphs == 6
    @test hg_comb2.hypergraph_ids == [1, 2, 2, 3, 4, 4, 5, 5, 6, 6]
    @test hg_comb2.vdata == cat_features([hg1.vdata, hg1.vdata, hg2.vdata, hg2.vdata])
    @test hg_comb2.hedata == cat_features([hg1.hedata, hg1.hedata, hg2.hedata, hg2.hedata])
    @test hg_comb2.hgdata == cat_features([hg1.hgdata, hg1.hgdata, hg2.hgdata, hg2.hgdata])

    @test isnothing(combine_hypergraphs(HGNNHypergraph{Float64, Dict{Int,Float64}}[]))
    @test combine_hypergraphs([hg1]) == hg1
    @test combine_hypergraphs([hg1, hg1, hg2, hg2]) == hg_comb2

    @test combine_hypergraphs([hg1, hg2]) == batch([hg1, hg2])

    dhg1 = HGNNDiHypergraph(
        [
            1.0     nothing
            nothing 2.0
            nothing 3.0
        ],
        [
            nothing nothing
            2.0     nothing
            nothing 6.0
        ];
        hypergraph_ids=[1,2,2],
        vdata = rand(Float64, 5, 3),
        hedata = rand(Float64, 5, 2),
        hgdata = rand(Float64, 5, 2)
    )
    dhg2 = HGNNDiHypergraph(
        [
            1.0     nothing     nothing
            nothing 2.0         3.0
        ],
        [
            nothing nothing 1.0
            2.0     4.0     nothing
        ];
        hypergraph_ids=[1,1],
        vdata = rand(Float64, 5, 2),
        hedata = rand(Float64, 5, 3),
        hgdata = rand(Float64, 5, 1)
    )

    dhg_comb1 = combine_hypergraphs(dhg1, dhg2)
    @test dhg_comb1.num_vertices == 5
    @test dhg_comb1.num_hyperedges == 5
    @test dhg_comb1.num_hypergraphs == 3
    @test dhg_comb1.hypergraph_ids == [1, 2, 2, 3, 3]
    @test dhg_comb1.vdata == cat_features(dhg1.vdata, dhg2.vdata)
    @test dhg_comb1.hedata == cat_features(dhg1.hedata, dhg2.hedata)
    @test dhg_comb1.hgdata == cat_features(dhg1.hgdata, dhg2.hgdata)

    dhg_comb2 = combine_hypergraphs(dhg1, dhg1, dhg2, dhg2)
    @test dhg_comb2.num_vertices == 10
    @test dhg_comb2.num_hyperedges == 10
    @test dhg_comb2.num_hypergraphs == 6
    @test dhg_comb2.hypergraph_ids == [1, 2, 2, 3, 4, 4, 5, 5, 6, 6]
    @test dhg_comb2.vdata == cat_features([dhg1.vdata, dhg1.vdata, dhg2.vdata, dhg2.vdata])
    @test dhg_comb2.hedata == cat_features([dhg1.hedata, dhg1.hedata, dhg2.hedata, dhg2.hedata])
    @test dhg_comb2.hgdata == cat_features([dhg1.hgdata, dhg1.hgdata, dhg2.hgdata, dhg2.hgdata])

    @test isnothing(combine_hypergraphs(HGNNDiHypergraph{Float64, Dict{Int,Float64}}[]))
    @test combine_hypergraphs([dhg1]) == dhg1
    @test combine_hypergraphs([dhg1, dhg1, dhg2, dhg2]) == dhg_comb2

    @test combine_hypergraphs([dhg1, dhg2]) == batch([dhg1, dhg2])

    # get_hypergraph / MLUtils.unbatch
    hg1_1 = get_hypergraph(hg1, 1)
    @test hg1_1 == get_hypergraph(hg1, [1])
    @test get_hypergraph(hg1, [1,2]) == hg1
    @test hg1_1.num_vertices == 1
    @test hg1_1.num_hyperedges == 1
    @test hg1_1.num_hypergraphs == 1
    @test get_hypergraph(hg1, 2; map_vertices=true)[2] == [2, 3]
    @test unbatch(hg1) == [get_hypergraph(hg1, 1), get_hypergraph(hg1, 2)]

    dhg1_1 = get_hypergraph(dhg1, 1)
    @test dhg1_1 == get_hypergraph(dhg1, [1])
    @test get_hypergraph(dhg1, [1,2]) == dhg1
    @test dhg1_1.num_vertices == 1
    @test dhg1_1.num_hyperedges == 0
    @test dhg1_1.num_hypergraphs == 1
    @test get_hypergraph(dhg1, 2; map_vertices=true)[2] == [2, 3]
    @test unbatch(dhg1) == [get_hypergraph(dhg1, 1), get_hypergraph(dhg1, 2)]

    start_he_keys = Set.(keys.(hgnn.he2v))

    # uniform_negative_sample
    hgnn_u = negative_sample_hyperedge(hgnn, 3, Xoshiro(42), UniformSample(); max_trials=100)
    @test hgnn_u.num_vertices == 11
    @test hgnn_u.num_hyperedges == 3
    # No hyperedge should be in the original hypergraph
    for he in hgnn_u.he2v
        @test Set(keys(he)) ∉ start_he_keys
    end
    # All hyperedges should be unique
    @test length(Set(Set.(keys.(hgnn_u.he2v)))) == 3

    start_he_keys = collect(
        zip(
            Set.(keys.(dhgnn.hg_tail.he2v)),
            Set.(keys.(dhgnn.hg_head.he2v))
        )
    )

    dhgnn_u = negative_sample_hyperedge(dhgnn, 3, Xoshiro(42), UniformSample(); max_trials=100)
    @test dhgnn_u.num_vertices == 11
    @test dhgnn_u.num_hyperedges == 3
    # No hyperedge should be in the original hypergraph
    
    all_he_inds = Set{Tuple{Set{Int}, Set{Int}}}()

    for (he_tail, he_head) in zip(dhgnn_u.hg_tail.he2v, dhgnn_u.hg_head.he2v)
        he_inds = (Set(keys(he_tail)), Set(keys(he_head)))
        @test he_inds ∉ start_he_keys
        push!(all_he_inds, he_inds)
    end
    # All hyperedges should be unique
    @test length(all_he_inds) == 3


    # sized_negative_sample
    hgnn_s = negative_sample_hyperedge(hgnn, 3, Xoshiro(42), SizedSample(); max_trials=100)
    @test hgnn_s.num_vertices == 11
    @test hgnn_s.num_hyperedges == 3
    for he in hgnn_s.he2v
        @test Set(keys(he)) ∉ start_he_keys
    end
    @test length(Set(Set.(keys.(hgnn_s.he2v)))) == 3

    dhgnn_s = negative_sample_hyperedge(dhgnn, 3, Xoshiro(42), SizedSample(); max_trials=100)
    @test dhgnn_s.num_vertices == 11
    @test dhgnn_s.num_hyperedges == 3
    # No hyperedge should be in the original hypergraph

    all_he_inds = Set{Tuple{Set{Int}, Set{Int}}}()

    for (he_tail, he_head) in zip(dhgnn_s.hg_tail.he2v, dhgnn_s.hg_head.he2v)
        he_inds = (Set(keys(he_tail)), Set(keys(he_head)))
        @test he_inds ∉ start_he_keys
        push!(all_he_inds, he_inds)
    end
    # All hyperedges should be unique
    @test length(all_he_inds) == 3


    # motif_negative_sample
    hgnn_m = negative_sample_hyperedge(hgnn, 3, Xoshiro(42), MotifSample(); max_trials=100)
    @test hgnn_m.num_vertices == 11
    @test hgnn_m.num_hyperedges == 3
    for he in hgnn_m.he2v
        @test Set(keys(he)) ∉ start_he_keys
    end
    @test length(Set(Set.(keys.(hgnn_m.he2v)))) == 3

    dhgnn_m = negative_sample_hyperedge(dhgnn, 3, Xoshiro(42), MotifSample(); max_trials=100)
    @test dhgnn_m.num_vertices == 11
    @test dhgnn_m.num_hyperedges == 3
    # No hyperedge should be in the original hypergraph

    all_he_inds = Set{Tuple{Set{Int}, Set{Int}}}()

    for (he_tail, he_head) in zip(dhgnn_m.hg_tail.he2v, dhgnn_m.hg_head.he2v)
        he_inds = (Set(keys(he_tail)), Set(keys(he_head)))
        @test he_inds ∉ start_he_keys
        push!(all_he_inds, he_inds)
    end
    # All hyperedges should be unique
    @test length(all_he_inds) == 3

    # clique_negative_sample
    hgnn_c = negative_sample_hyperedge(hgnn, 3, Xoshiro(42), CliqueSample(); max_trials=100)
    @test hgnn_c.num_vertices == 11
    @test hgnn_c.num_hyperedges == 3
    for he in hgnn_c.he2v
        @test Set(keys(he)) ∉ start_he_keys
    end
    @test length(Set(Set.(keys.(hgnn_c.he2v)))) == 3

    @test_throws "negative_sample not implemented for strategy of type CliqueSample" negative_sample_hyperedge(
        dhgnn,
        1,
        Xoshiro(42),
        CliqueSample()
    )

    # negative_sample_hyperedge
    struct NewSample <: AbstractNegativeSamplingStrategy end
    @test_throws "negative_sample not implemented for strategy of type NewSample" negative_sample_hyperedge(
        hgnn,
        1,
        Xoshiro(42),
        NewSample()
    )

    @test_throws "negative_sample not implemented for strategy of type NewSample" negative_sample_hyperedge(
        dhgnn,
        1,
        Xoshiro(42),
        NewSample()
    )

end


using Random
using StatsBase
using LinearAlgebra
using Test
using Graphs
using GNNGraphs
using MLUtils
using SimpleHypergraphs
using SimpleDirectedHypergraphs
using HyperGraphNeuralNetworks

@testset "HyperGraphNeuralNetworks                                data splitting" begin
    @testset "    split vertices" begin    
	# Split vertices of undirected hypergraphs
	hgnn1 = HGNNHypergraph(
	    uh1;
	    hypergraph_ids = uid1,
	    vdata = rand(Float64, 5, 11),
	    hedata = rand(Float64, 5, 5),
	    hgdata = rand(Float64, 5, 2)
	)

	vmasks = [
	    BitVector((false, true, true, false, true, false, true, false, false, false, true)),
	    BitVector((false, false, false, true, false, true, false, true, false, true, false)),
	    BitVector((true, false, false, false, false, false, false, false, true, false, false))
	]

	# Split vertices using masks
	hgnns = split_vertices(hgnn1, vmasks)
	@test length(hgnns) == 3
	@test hgnns[1].num_vertices == 5
	@test hgnns[1].num_hyperedges == 3
	@test hgnns[1].num_hypergraphs == 2
	@test getobs(hgnns[1].vdata, 1).x == getobs(hgnn1.vdata, 2).x
	@test getobs(hgnns[1].hedata, 1).e == getobs(hgnn1.hedata, 1).e
	@test getobs(hgnns[1].hgdata, 1).u == getobs(hgnn1.hgdata, 1).u
	@test hgnns[2].num_vertices == 4
	@test hgnns[2].num_hyperedges == 4
	@test hgnns[2].num_hypergraphs == 2
	@test getobs(hgnns[2].vdata, 1).x == getobs(hgnn1.vdata, 4).x
	@test getobs(hgnns[2].hedata, 2).e == getobs(hgnn1.hedata, 3).e
	@test getobs(hgnns[2].hgdata, 2).u == getobs(hgnn1.hgdata, 2).u
	@test hgnns[3].num_vertices == 2
	@test hgnns[3].num_hyperedges == 2
	@test hgnns[3].num_hypergraphs == 2
	@test getobs(hgnns[3].vdata, 2).x == getobs(hgnn1.vdata, 9).x
	@test getobs(hgnns[3].hedata, 2).e == getobs(hgnn1.hedata, 5).e
	@test getobs(hgnns[3].hgdata, 1).u == getobs(hgnn1.hgdata, 1).u

	# Split vertices by train-val-test labeled masks
	hgnns_tvt = split_vertices(hgnn1, vmasks[1], vmasks[3]; val_mask=vmasks[2])
	@test hgnns_tvt.train == hgnns[1]
	@test hgnns_tvt.val == hgnns[2]
	@test hgnns_tvt.test == hgnns[3]

	# Split without validation set
	hgnns_tvt_noval = split_vertices(hgnn1, vmasks[1], vmasks[3])
	@test hgnns_tvt_noval.train == hgnns[1]
	@test hgnns_tvt_noval.val === nothing
	@test hgnns_tvt_noval.test == hgnns[3]

	vinds = [
	    [2, 3, 5, 7, 11],
	    [4, 6, 8, 10],
	    [1, 9]
	]

	# Split vertices using vertex indices
	hgnns_ind = split_vertices(hgnn1, vinds)
	@test length(hgnns_ind) == 3
	@test hgnns_ind[1] == hgnns[1]
	@test hgnns_ind[2] == hgnns[2]
	@test hgnns_ind[3] == hgnns[3]

	# Split vertices by train-val-test labeled indices
	hgnns_ind_tvt = split_vertices(hgnn1, vinds[1], vinds[3]; val_inds=vinds[2])
	@test hgnns_ind_tvt.train == hgnns[1]
	@test hgnns_ind_tvt.val == hgnns[2]
	@test hgnns_ind_tvt.test == hgnns[3]

	# Split without validation set
	hgnns_ind_tvt_noval = split_vertices(hgnn1, vinds[1], vinds[3])
	@test hgnns_ind_tvt_noval.train == hgnns[1]
	@test hgnns_ind_tvt_noval.val === nothing
	@test hgnns_ind_tvt_noval.test == hgnns[3]

	# "Random" split
	rng = Xoshiro(42)
	hgnns_rand = random_split_vertices(hgnn1, [0.7, 0.1, 0.2], rng)
	@test length(hgnns_rand) == 3
	@test hgnns_rand[1].num_vertices == 8
	@test hgnns_rand[2].num_vertices == 1
	@test hgnns_rand[3].num_vertices == 2

	# Split vertices of directed hypergraphs
	dhgnn1 = HGNNDiHypergraph(
	    dh1;
	    hypergraph_ids = did1,
	    vdata = rand(Float64, 5, 11),
	    hedata = rand(Float64, 5, 5),
	    hgdata = rand(Float64, 5, 2)
	)

	vmasks = [
	    BitVector((false, true, true, false, true, false, true, false, false, false, true)),
	    BitVector((false, false, false, true, false, true, false, true, false, true, false)),
	    BitVector((true, false, false, false, false, false, false, false, true, false, false))
	]

	# Split vertices using masks
	dhgnns = split_vertices(dhgnn1, vmasks)
	@test length(dhgnns) == 3
	@test dhgnns[1].num_vertices == 5
	@test dhgnns[1].num_hyperedges == 3
	@test dhgnns[1].num_hypergraphs == 2
	@test getobs(dhgnns[1].vdata, 1).x == getobs(dhgnn1.vdata, 2).x
	@test getobs(dhgnns[1].hedata, 1).e == getobs(dhgnn1.hedata, 1).e
	@test getobs(dhgnns[1].hgdata, 1).u == getobs(dhgnn1.hgdata, 1).u
	@test dhgnns[2].num_vertices == 4
	@test dhgnns[2].num_hyperedges == 4
	@test dhgnns[2].num_hypergraphs == 2
	@test getobs(dhgnns[2].vdata, 1).x == getobs(dhgnn1.vdata, 4).x
	@test getobs(dhgnns[2].hedata, 2).e == getobs(dhgnn1.hedata, 3).e
	@test getobs(dhgnns[2].hgdata, 2).u == getobs(dhgnn1.hgdata, 2).u
	@test dhgnns[3].num_vertices == 2
	@test dhgnns[3].num_hyperedges == 2
	@test dhgnns[3].num_hypergraphs == 2
	@test getobs(dhgnns[3].vdata, 2).x == getobs(dhgnn1.vdata, 9).x
	@test getobs(dhgnns[3].hedata, 2).e == getobs(dhgnn1.hedata, 5).e
	@test getobs(dhgnns[3].hgdata, 1).u == getobs(dhgnn1.hgdata, 1).u

	# Split vertices by train-val-test labeled masks
	dhgnns_tvt = split_vertices(dhgnn1, vmasks[1], vmasks[3]; val_mask=vmasks[2])
	@test dhgnns_tvt.train == dhgnns[1]
	@test dhgnns_tvt.val == dhgnns[2]
	@test dhgnns_tvt.test == dhgnns[3]

	# Split without validation set
	dhgnns_tvt_noval = split_vertices(dhgnn1, vmasks[1], vmasks[3])
	@test dhgnns_tvt_noval.train == dhgnns[1]
	@test dhgnns_tvt_noval.val === nothing
	@test dhgnns_tvt_noval.test == dhgnns[3]

	vinds = [
	    [2, 3, 5, 7, 11],
	    [4, 6, 8, 10],
	    [1, 9]
	]

	# Split vertices using vertex indices
	dhgnns_ind = split_vertices(dhgnn1, vinds)
	@test length(dhgnns_ind) == 3
	@test dhgnns_ind[1] == dhgnns[1]
	@test dhgnns_ind[2] == dhgnns[2]
	@test dhgnns_ind[3] == dhgnns[3]

	# Split vertices by train-val-test labeled indices
	dhgnns_ind_tvt = split_vertices(dhgnn1, vinds[1], vinds[3]; val_inds=vinds[2])
	@test dhgnns_ind_tvt.train == dhgnns[1]
	@test dhgnns_ind_tvt.val == dhgnns[2]
	@test dhgnns_ind_tvt.test == dhgnns[3]

	# Split without validation set
	dhgnns_ind_tvt_noval = split_vertices(dhgnn1, vinds[1], vinds[3])
	@test dhgnns_ind_tvt_noval.train == dhgnns[1]
	@test dhgnns_ind_tvt_noval.val === nothing
	@test dhgnns_ind_tvt_noval.test == dhgnns[3]

	# "Random" split
	rng = Xoshiro(42)
	dhgnns_rand = random_split_vertices(dhgnn1, [0.7, 0.1, 0.2], rng)
	@test length(dhgnns_rand) == 3
	@test dhgnns_rand[1].num_vertices == 8
	@test dhgnns_rand[2].num_vertices == 1
	@test dhgnns_rand[3].num_vertices == 2
    end

    @testset "    split hyperedges" begin    
	# Split hyperedges of undirected hypergraphs
	hgnn1 = HGNNHypergraph(
	    uh1;
	    hypergraph_ids = uid1,
	    vdata = rand(Float64, 5, 11),
	    hedata = rand(Float64, 5, 5),
	    hgdata = rand(Float64, 5, 2)
	)

	hemasks = [
	    BitVector((false, true, true, false, true)),
	    BitVector((false, false, false, true, false)),
	    BitVector((true, false, false, false, false))
	]

	# Split hyperedges using masks
	hgnns = split_hyperedges(hgnn1, hemasks)
	@test length(hgnns) == 3
	@test hgnns[1].num_vertices == 8
	@test hgnns[1].num_hyperedges == 3
	@test hgnns[1].num_hypergraphs == 2
	@test getobs(hgnns[1].vdata, 1).x == getobs(hgnn1.vdata, 2).x
	@test getobs(hgnns[1].hedata, 1).e == getobs(hgnn1.hedata, 2).e
	@test getobs(hgnns[1].hgdata, 1).u == getobs(hgnn1.hgdata, 1).u
	@test hgnns[2].num_vertices == 3
	@test hgnns[2].num_hyperedges == 1
	@test hgnns[2].num_hypergraphs == 1
	@test getobs(hgnns[2].vdata, 1).x == getobs(hgnn1.vdata, 7).x
	@test getobs(hgnns[2].hedata, 1).e == getobs(hgnn1.hedata, 4).e
	@test getobs(hgnns[2].hgdata, 1).u == getobs(hgnn1.hgdata, 2).u
	@test hgnns[3].num_vertices == 3
	@test hgnns[3].num_hyperedges == 1
	@test hgnns[3].num_hypergraphs == 1
	@test getobs(hgnns[3].vdata, 1).x == getobs(hgnn1.vdata, 1).x
	@test getobs(hgnns[3].hedata, 1).e == getobs(hgnn1.hedata, 1).e
	@test getobs(hgnns[3].hgdata, 1).u == getobs(hgnn1.hgdata, 1).u

	# Split hyperedges by train-val-test labeled masks
	hgnns_tvt = split_hyperedges(hgnn1, hemasks[1], hemasks[3]; val_mask=hemasks[2])
	@test hgnns_tvt.train == hgnns[1]
	@test hgnns_tvt.val == hgnns[2]
	@test hgnns_tvt.test == hgnns[3]

	# Split without validation set
	hgnns_tvt_noval = split_hyperedges(hgnn1, hemasks[1], hemasks[3])
	@test hgnns_tvt_noval.train == hgnns[1]
	@test hgnns_tvt_noval.val === nothing
	@test hgnns_tvt_noval.test == hgnns[3]

	heinds = [
	    [2, 3, 5],
	    [4],
	    [1]
	]

	# Split hyperedges using hyperedge indices
	hgnns_ind = split_hyperedges(hgnn1, heinds)
	@test length(hgnns_ind) == 3
	@test hgnns_ind[1] == hgnns[1]
	@test hgnns_ind[2] == hgnns[2]
	@test hgnns_ind[3] == hgnns[3]

	# Split hyperedges by train-val-test labeled indices
	hgnns_ind_tvt = split_hyperedges(hgnn1, heinds[1], heinds[3]; val_inds=heinds[2])
	@test hgnns_ind_tvt.train == hgnns[1]
	@test hgnns_ind_tvt.val == hgnns[2]
	@test hgnns_ind_tvt.test == hgnns[3]

	# Split without validation set
	hgnns_ind_tvt_noval = split_hyperedges(hgnn1, heinds[1], heinds[3])
	@test hgnns_ind_tvt_noval.train == hgnns[1]
	@test hgnns_ind_tvt_noval.val === nothing
	@test hgnns_ind_tvt_noval.test == hgnns[3]

	# "Random" split
	rng = Xoshiro(42)
	hgnns_rand = random_split_hyperedges(hgnn1, [0.7, 0.3], rng)
	@test length(hgnns_rand) == 2
	@test hgnns_rand[1].num_hyperedges == 4
	@test hgnns_rand[2].num_hyperedges == 1

	# Split hyperedges of directed hypergraphs
	dhgnn1 = HGNNDiHypergraph(
	    dh1;
	    hypergraph_ids = did1,
	    vdata = rand(Float64, 5, 11),
	    hedata = rand(Float64, 5, 5),
	    hgdata = rand(Float64, 5, 2)
	)

	hemasks = [
	    BitVector((false, true, true, false, true)),
	    BitVector((false, false, false, true, false)),
	    BitVector((true, false, false, false, false))
	]

	# Split hyperedges using masks
	dhgnns = split_hyperedges(dhgnn1, hemasks)
	@test length(dhgnns) == 3
	@test dhgnns[1].num_vertices == 8
	@test dhgnns[1].num_hyperedges == 3
	@test dhgnns[1].num_hypergraphs == 2
	@test getobs(dhgnns[1].vdata, 1).x == getobs(dhgnn1.vdata, 2).x
	@test getobs(dhgnns[1].hedata, 1).e == getobs(dhgnn1.hedata, 2).e
	@test getobs(dhgnns[1].hgdata, 1).u == getobs(dhgnn1.hgdata, 1).u
	@test dhgnns[2].num_vertices == 3
	@test dhgnns[2].num_hyperedges == 1
	@test dhgnns[2].num_hypergraphs == 1
	@test getobs(dhgnns[2].vdata, 1).x == getobs(dhgnn1.vdata, 7).x
	@test getobs(dhgnns[2].hedata, 1).e == getobs(dhgnn1.hedata, 4).e
	@test getobs(dhgnns[2].hgdata, 1).u == getobs(dhgnn1.hgdata, 2).u
	@test dhgnns[3].num_vertices == 3
	@test dhgnns[3].num_hyperedges == 1
	@test dhgnns[3].num_hypergraphs == 1
	@test getobs(dhgnns[3].vdata, 1).x == getobs(dhgnn1.vdata, 1).x
	@test getobs(dhgnns[3].hedata, 1).e == getobs(dhgnn1.hedata, 1).e
	@test getobs(dhgnns[3].hgdata, 1).u == getobs(dhgnn1.hgdata, 1).u

	# Split hyperedges by train-val-test labeled masks
	dhgnns_tvt = split_hyperedges(dhgnn1, hemasks[1], hemasks[3]; val_mask=hemasks[2])
	@test dhgnns_tvt.train == dhgnns[1]
	@test dhgnns_tvt.val == dhgnns[2]
	@test dhgnns_tvt.test == dhgnns[3]

	# Split without validation set
	dhgnns_tvt_noval = split_hyperedges(dhgnn1, hemasks[1], hemasks[3])
	@test dhgnns_tvt_noval.train == dhgnns[1]
	@test dhgnns_tvt_noval.val === nothing
	@test dhgnns_tvt_noval.test == dhgnns[3]

	heinds = [
	    [2, 3, 5],
	    [4],
	    [1]
	]

	# Split hyperedges using hyperedge indices
	dhgnns_ind = split_hyperedges(dhgnn1, heinds)
	@test length(dhgnns_ind) == 3
	@test dhgnns_ind[1] == dhgnns[1]
	@test dhgnns_ind[2] == dhgnns[2]
	@test dhgnns_ind[3] == dhgnns[3]

	# Split hyperedges by train-val-test labeled indices
	dhgnns_ind_tvt = split_hyperedges(dhgnn1, heinds[1], heinds[3]; val_inds=heinds[2])
	@test dhgnns_ind_tvt.train == dhgnns[1]
	@test dhgnns_ind_tvt.val == dhgnns[2]
	@test dhgnns_ind_tvt.test == dhgnns[3]

	# Split without validation set
	dhgnns_ind_tvt_noval = split_hyperedges(dhgnn1, heinds[1], heinds[3])
	@test dhgnns_ind_tvt_noval.train == dhgnns[1]
	@test dhgnns_ind_tvt_noval.val === nothing
	@test dhgnns_ind_tvt_noval.test == dhgnns[3]

	# "Random" split
	rng = Xoshiro(42)
	dhgnns_rand = random_split_hyperedges(dhgnn1, [0.7, 0.3], rng)
	@test length(dhgnns_rand) == 2
	@test dhgnns_rand[1].num_hyperedges == 4
	@test dhgnns_rand[2].num_hyperedges == 1
    end

    @testset "    split hypergraphs" begin
	uid2 = [1,1,1,1,2,2,3,3,3,3,3]

	# Split hypergraphs of undirected hypergraphs
	hgnn1 = HGNNHypergraph(
	    uh1;
	    hypergraph_ids = uid2,
	    vdata = rand(Float64, 5, 11),
	    hedata = rand(Float64, 5, 5),
	    hgdata = rand(Float64, 5, 3)
	)

	hgmasks = [
	    BitVector((true, false, false)),
	    BitVector((false, true, false)),
	    BitVector((false, false, true))
	]

	# Split hypergraphs using masks
	hgnns = split_hypergraphs(hgnn1, hgmasks)
	@test length(hgnns) == 3
	@test hgnns[1].num_vertices == 4
	@test hgnns[1].num_hyperedges == 3
	@test hgnns[1].num_hypergraphs == 1
	@test getobs(hgnns[1].vdata, 1).x == getobs(hgnn1.vdata, 1).x
	@test getobs(hgnns[1].hedata, 2).e == getobs(hgnn1.hedata, 2).e
	@test getobs(hgnns[1].hgdata, 1).u == getobs(hgnn1.hgdata, 1).u
	@test hgnns[2].num_vertices == 2
	@test hgnns[2].num_hyperedges == 2
	@test hgnns[2].num_hypergraphs == 1
	@test getobs(hgnns[2].vdata, 1).x == getobs(hgnn1.vdata, 5).x
	@test getobs(hgnns[2].hedata, 2).e == getobs(hgnn1.hedata, 3).e
	@test getobs(hgnns[2].hgdata, 1).u == getobs(hgnn1.hgdata, 2).u
	@test hgnns[3].num_vertices == 5
	@test hgnns[3].num_hyperedges == 2
	@test hgnns[3].num_hypergraphs == 1
	@test getobs(hgnns[3].vdata, 1).x == getobs(hgnn1.vdata, 7).x
	@test getobs(hgnns[3].hedata, 2).e == getobs(hgnn1.hedata, 5).e
	@test getobs(hgnns[3].hgdata, 1).u == getobs(hgnn1.hgdata, 3).u

	# Split hypergraphs by train-val-test labeled masks
	hgnns_tvt = split_hypergraphs(hgnn1, hgmasks[1], hgmasks[3]; val_mask=hgmasks[2])
	@test hgnns_tvt.train == hgnns[1]
	@test hgnns_tvt.val == hgnns[2]
	@test hgnns_tvt.test == hgnns[3]

	# Split without validation set
	hgnns_tvt_noval = split_hypergraphs(hgnn1, hgmasks[1], hgmasks[3])
	@test hgnns_tvt_noval.train == hgnns[1]
	@test hgnns_tvt_noval.val === nothing
	@test hgnns_tvt_noval.test == hgnns[3]

	hginds = [[1], [2], [3]]

	# Split hypergraphs using vertex indices
	hgnns_ind = split_hypergraphs(hgnn1, hginds)
	@test length(hgnns_ind) == 3
	@test hgnns_ind[1] == hgnns[1]
	@test hgnns_ind[2] == hgnns[2]
	@test hgnns_ind[3] == hgnns[3]

	# Split hypergraphs by train-val-test labeled indices
	hgnns_ind_tvt = split_hypergraphs(hgnn1, hginds[1], hginds[3]; val_inds=hginds[2])
	@test hgnns_ind_tvt.train == hgnns[1]
	@test hgnns_ind_tvt.val == hgnns[2]
	@test hgnns_ind_tvt.test == hgnns[3]

	# Split without validation set
	hgnns_ind_tvt_noval = split_hypergraphs(hgnn1, hginds[1], hginds[3])
	@test hgnns_ind_tvt_noval.train == hgnns[1]
	@test hgnns_ind_tvt_noval.val === nothing
	@test hgnns_ind_tvt_noval.test == hgnns[3]

	# "Random" split
	rng = Xoshiro(42)
	hgnns_rand = random_split_hypergraphs(hgnn1, [0.34, 0.33, 0.33], rng)
	@test length(hgnns_rand) == 3
	@test hgnns_rand[1].num_hypergraphs == 1
	@test hgnns_rand[2].num_hypergraphs == 1
	@test hgnns_rand[3].num_hypergraphs == 1


	# Split hypergraphs of directed hypergraphs
	dhgnn1 = HGNNDiHypergraph(
	    dh1;
	    hypergraph_ids = uid2,
	    vdata = rand(Float64, 5, 11),
	    hedata = rand(Float64, 5, 5),
	    hgdata = rand(Float64, 5, 3)
	)

	hgmasks = [
	    BitVector((true, false, false)),
	    BitVector((false, true, false)),
	    BitVector((false, false, true))
	]

	# Split hypergraphs using masks
	dhgnns = split_hypergraphs(dhgnn1, hgmasks)
	@test length(dhgnns) == 3
	@test dhgnns[1].num_vertices == 4
	@test dhgnns[1].num_hyperedges == 3
	@test dhgnns[1].num_hypergraphs == 1
	@test getobs(dhgnns[1].vdata, 1).x == getobs(dhgnn1.vdata, 1).x
	@test getobs(dhgnns[1].hedata, 2).e == getobs(dhgnn1.hedata, 2).e
	@test getobs(dhgnns[1].hgdata, 1).u == getobs(dhgnn1.hgdata, 1).u
	@test dhgnns[2].num_vertices == 2
	@test dhgnns[2].num_hyperedges == 2
	@test dhgnns[2].num_hypergraphs == 1
	@test getobs(dhgnns[2].vdata, 1).x == getobs(dhgnn1.vdata, 5).x
	@test getobs(dhgnns[2].hedata, 2).e == getobs(dhgnn1.hedata, 3).e
	@test getobs(dhgnns[2].hgdata, 1).u == getobs(dhgnn1.hgdata, 2).u
	@test dhgnns[3].num_vertices == 5
	@test dhgnns[3].num_hyperedges == 2
	@test dhgnns[3].num_hypergraphs == 1
	@test getobs(dhgnns[3].vdata, 1).x == getobs(dhgnn1.vdata, 7).x
	@test getobs(dhgnns[3].hedata, 2).e == getobs(dhgnn1.hedata, 5).e
	@test getobs(dhgnns[3].hgdata, 1).u == getobs(dhgnn1.hgdata, 3).u

	# Split hypergraphs by train-val-test labeled masks
	dhgnns_tvt = split_hypergraphs(dhgnn1, hgmasks[1], hgmasks[3]; val_mask=hgmasks[2])
	@test dhgnns_tvt.train == dhgnns[1]
	@test dhgnns_tvt.val == dhgnns[2]
	@test dhgnns_tvt.test == dhgnns[3]

	# Split without validation set
	dhgnns_tvt_noval = split_hypergraphs(dhgnn1, hgmasks[1], hgmasks[3])
	@test dhgnns_tvt_noval.train == dhgnns[1]
	@test dhgnns_tvt_noval.val === nothing
	@test dhgnns_tvt_noval.test == dhgnns[3]

	hginds = [[1], [2], [3]]

	# Split hypergraphs using vertex indices
	dhgnns_ind = split_hypergraphs(dhgnn1, hginds)
	@test length(dhgnns_ind) == 3
	@test dhgnns_ind[1] == dhgnns[1]
	@test dhgnns_ind[2] == dhgnns[2]
	@test dhgnns_ind[3] == dhgnns[3]

	# Split hypergraphs by train-val-test labeled indices
	dhgnns_ind_tvt = split_hypergraphs(dhgnn1, hginds[1], hginds[3]; val_inds=hginds[2])
	@test dhgnns_ind_tvt.train == dhgnns[1]
	@test dhgnns_ind_tvt.val == dhgnns[2]
	@test dhgnns_ind_tvt.test == dhgnns[3]

	# Split without validation set
	dhgnns_ind_tvt_noval = split_hypergraphs(dhgnn1, hginds[1], hginds[3])
	@test dhgnns_ind_tvt_noval.train == dhgnns[1]
	@test dhgnns_ind_tvt_noval.val === nothing
	@test dhgnns_ind_tvt_noval.test == dhgnns[3]

	# "Random" split
	rng = Xoshiro(42)
	dhgnns_rand = random_split_hypergraphs(dhgnn1, [0.34, 0.33, 0.33], rng)
	@test length(dhgnns_rand) == 3
	@test dhgnns_rand[1].num_hypergraphs == 1
	@test dhgnns_rand[2].num_hypergraphs == 1
	@test dhgnns_rand[3].num_hypergraphs == 1
    end
end;

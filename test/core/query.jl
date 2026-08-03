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

@testset "HyperGraphNeuralNetworks                                         query" begin
    hgnn = HGNNHypergraph(uh1.v2he, uh1.he2v, 11, 5, 2, uid1, DataStore(), DataStore(), DataStore())
    dhgnn = HGNNDiHypergraph(
        dh1;
        hypergraph_ids = did1,
        vdata = rand(Float64, 5, 11),
        hedata = rand(Float64, 5, 5),
        hgdata = rand(Float64, 5, 2)
    )

    # hyperedge_index
    @test hyperedge_index(hgnn) == [
        [1, 2, 4],
        [2, 3, 5],
        [4, 6],
        [7, 10, 11],
        [8, 9, 10],
    ]
    @test hyperedge_index(dhgnn) == (
        [[1, 2], [2, 5], [4], [7, 10], [10]],
        [[4], [3], [6], [11], [8, 9]]
    )
    
    # get_hyperedge_weights
    @test get_hyperedge_weights(hgnn) == [
        [1.0, 2.0, 4.0],
        [3.0, 0.0, 12.0],
        [1.0, 4.0],
        [3.5, 1.0, 4.0],
        [1.0, 5.0, 7.0]
    ]
    @test get_hyperedge_weights(hgnn, sum) == [7.0, 15.0, 5.0, 8.5, 13.0]

    dweights = (
        [[1.0, 2.0], [3.0, 12.0], [1.0], [3.5, 1.0], [7.0]],
        [[4.0], [0.0], [4.0], [4.0], [1.0, 5.0]]
    )

    @test get_hyperedge_weights(dhgnn) == dweights
    @test get_hyperedge_weights(dhgnn; side=:both) == dweights
    @test get_hyperedge_weights(dhgnn; side=:tail) == dweights[1]
    @test get_hyperedge_weights(dhgnn; side=:head) == dweights[2]
    @test get_hyperedge_weights(dhgnn, sum) == ([3.0, 15.0, 1.0, 4.5, 7.0], [4.0, 0.0, 4.0, 4.0, 6.0])

    # get_hyperedge_weight
    @test get_hyperedge_weight(hgnn, 2) == [3.0, 0.0, 12.0]
    @test get_hyperedge_weight(hgnn, 2, sum) == 15.0

    @test get_hyperedge_weight(dhgnn, 2) == ([3.0, 12.0], [0.0])
    @test get_hyperedge_weight(dhgnn, 2; side=:both) == ([3.0, 12.0], [0.0])
    @test get_hyperedge_weight(dhgnn, 2; side=:tail) == [3.0, 12.0]
    @test get_hyperedge_weight(dhgnn, 2; side=:head) == [0.0]
    @test get_hyperedge_weight(dhgnn, 2, sum) == (15.0, 0.0)

    # has_vertex
    @test has_vertex(hgnn, 10)
    @test !(has_vertex(hgnn, 12))

    @test has_vertex(dhgnn, 10)
    @test !(has_vertex(dhgnn, 25))

    # vertices
    @test vertices(hgnn) == 1:11
    @test vertices(dhgnn) == 1:11

    # degree
    @test degree(hgnn) == [1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
    @test degree(hgnn, 2) == 2
    @test degree(hgnn, [1,3,5]) == [1, 1, 1]

    @test degree(dhgnn) == [1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
    @test degree(dhgnn, 1) == 1
    @test degree(dhgnn, [1,2,3]) == [1, 2, 1]

    # indegree
    @test indegree(dhgnn) == [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
    @test indegree(dhgnn, 5) == 0
    @test indegree(dhgnn, [1,2,4,8]) == [0, 0, 1, 1]

    # outdegree
    @test outdegree(dhgnn) == [1, 2, 0, 1, 1, 0, 1, 0, 0, 2, 0]
    @test outdegree(dhgnn, 2) == 2
    @test outdegree(dhgnn, [5, 10]) == [1, 2]

    # all_neighbors
    @test all_neighbors(hgnn) == [
        [2, 4],
        [1, 3, 4, 5],
        [2, 5],
        [1, 2, 6],
        [2, 3],
        [4],
        [10, 11],
        [9, 10],
        [8, 10],
        [7, 8, 9, 11],
        [7, 10]
    ]
    @test all_neighbors(hgnn, 2) == [1,3,4,5]

    @test all_neighbors(dhgnn) == [
        [4],
        [3, 4],
        [2, 5],
        [1, 2, 6],
        [3],
        [4],
        [11],
        [10],
        [10],
        [8, 9, 11],
        [7, 10]
    ]
    @test all_neighbors(dhgnn; same_side=true) == [
        [2, 4],
        [1, 3, 4, 5],
        [2, 5],
        [1, 2, 6],
        [2, 3],
        [4],
        [10, 11],
        [9, 10],
        [8, 10],
        [7, 8, 9, 11],
        [7, 10]
    ]

    @test all_neighbors(dhgnn, 2) == [3, 4]
    @test all_neighbors(dhgnn, 2; same_side=true) == [1, 3, 4, 5]

    # inneighbors
    @test inneighbors(dhgnn) == [
        [],
        [],
        [2, 5],
        [1, 2],
        [],
        [4],
        [],
        [10],
        [10],
        [],
        [7, 10]
    ]
    @test inneighbors(dhgnn; same_side=true) == [
        [],
        [],
        [2, 5],
        [1, 2],
        [],
        [4],
        [],
        [9, 10],
        [8, 10],
        [],
        [7, 10]
    ]

    @test inneighbors(dhgnn, 9) == [10]
    @test inneighbors(dhgnn, 9; same_side=true) == [8, 10]

    # outneighbors
    @test outneighbors(dhgnn) == [
        [4],
        [3, 4],
        [],
        [6],
        [3],
        [],
        [11],
        [],
        [],
        [8, 9, 11],
        []
    ]
    @test outneighbors(dhgnn; same_side=true) == [
        [2, 4],
        [1, 3, 4, 5],
        [],
        [6],
        [2, 3],
        [],
        [10, 11],
        [],
        [],
        [7, 8, 9, 11],
        []
    ]

    @test outneighbors(dhgnn, 2) == [3, 4]
    @test outneighbors(dhgnn, 2; same_side=true) == [1, 3, 4, 5]

    # hyperedge_neighbors
    @test hyperedge_neighbors(hgnn) == [[2, 3], [1], [1], [5], [4]]
    @test hyperedge_neighbors(hgnn, 4) == [5]

    # isolated_vertices
    @test length(isolated_vertices(hgnn)) == 0
    @test isolated_vertices(HGNNHypergraph(Hypergraph(5,0))) == [1,2,3,4,5]

    # incidence_matrix
    @test incidence_matrix(hgnn) == [
        1.0  0.0  0.0  0.0  0.0
        1.0  1.0  0.0  0.0  0.0
        0.0  1.0  0.0  0.0  0.0
        1.0  0.0  1.0  0.0  0.0
        0.0  1.0  0.0  0.0  0.0
        0.0  0.0  1.0  0.0  0.0
        0.0  0.0  0.0  1.0  0.0
        0.0  0.0  0.0  0.0  1.0
        0.0  0.0  0.0  0.0  1.0
        0.0  0.0  0.0  1.0  1.0
        0.0  0.0  0.0  1.0  0.0
    ]

    inc = incidence_matrix(dhgnn)
    @test inc[1] == [
        1.0  0.0  0.0  0.0  0.0
        1.0  1.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  1.0  0.0  0.0
        0.0  1.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  1.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  1.0  1.0
        0.0  0.0  0.0  0.0  0.0
    ]
    @test inc[2] == [
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  1.0  0.0  0.0  0.0
        1.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  1.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  1.0
        0.0  0.0  0.0  0.0  1.0
        0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  1.0  0.0
    ]

    # complex_incidence_matrix
    @test complex_incidence_matrix(dhgnn) == [
        0.0-1.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im
        0.0-1.0im  0.0-1.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im
        0.0-0.0im  1.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im
        1.0-0.0im  0.0-0.0im  0.0-1.0im  0.0-0.0im  0.0-0.0im
        0.0-0.0im  0.0-1.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im
        0.0-0.0im  0.0-0.0im  1.0-0.0im  0.0-0.0im  0.0-0.0im
        0.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-1.0im  0.0-0.0im
        0.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im  1.0-0.0im
        0.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-0.0im  1.0-0.0im
        0.0-0.0im  0.0-0.0im  0.0-0.0im  0.0-1.0im  0.0-1.0im
        0.0-0.0im  0.0-0.0im  0.0-0.0im  1.0-0.0im  0.0-0.0im
    ]
    
    # vertex_weight_matrix
    @test vertex_weight_matrix(hgnn) == Diagonal([1.0, 5.0, 0.0, 5.0, 12.0, 4.0, 3.5, 1.0, 5.0, 8.0, 4.0])
    # Non-standard weighting function
    @test vertex_weight_matrix(hgnn; weighting_function=prod) == Diagonal(zeros(11))

    @test vertex_weight_matrix(dhgnn)[1] == Diagonal([1.0, 5.0, 0.0, 1.0, 12.0, 0.0, 3.5, 0.0, 0.0, 8.0, 0.0])
    @test vertex_weight_matrix(dhgnn; weighting_function=prod)[1] == Diagonal(zeros(11))

    @test vertex_weight_matrix(dhgnn)[2] == Diagonal([0.0, 0.0, 0.0, 4.0, 0.0, 4.0, 0.0, 1.0, 5.0, 0.0, 4.0])
    @test vertex_weight_matrix(dhgnn; weighting_function=prod)[2] == Diagonal(zeros(11))

    # hyperedge_weight_matrix
    @test hyperedge_weight_matrix(hgnn) == Diagonal([7.0, 15.0, 5.0, 8.5, 13.0])
    # Non-standard weighting function
    @test hyperedge_weight_matrix(hgnn; weighting_function=prod) == Diagonal(zeros(5))

    @test hyperedge_weight_matrix(dhgnn)[1] == Diagonal([3.0, 15.0, 1.0, 4.5, 7.0])
    @test hyperedge_weight_matrix(dhgnn; weighting_function=prod)[1] == Diagonal(zeros(5))

    @test hyperedge_weight_matrix(dhgnn)[2] == Diagonal([4.0, 0.0, 4.0, 4.0, 6.0])
    @test hyperedge_weight_matrix(dhgnn; weighting_function=prod)[2] == Diagonal(zeros(5))

    # vertex_degree_matrix
    @test vertex_degree_matrix(hgnn) == Diagonal([1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1])

    @test vertex_degree_matrix(dhgnn)[1] == Diagonal([1, 2, 0, 1, 1, 0, 1, 0, 0, 2, 0])
    @test vertex_degree_matrix(dhgnn)[2] == Diagonal([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1])

    # hyperedge_degree_matrix
    @test hyperedge_degree_matrix(hgnn) == Diagonal([3, 3, 2, 3, 3])

    @test hyperedge_degree_matrix(dhgnn)[1] == Diagonal([2, 2, 1, 2, 1])
    @test hyperedge_degree_matrix(dhgnn)[2] == Diagonal([1, 1, 1, 1, 2])

    # normalized_laplacian
    L = normalized_laplacian_matrix(hgnn)
    @test size(L) == (11,11)
    @test L[1,3] == 0.0
    @test isapprox(L, L'; rtol=1e-5)

    L = normalized_laplacian_matrix(dhgnn)
    @test size(L) == (11, 11)
    @test L[1,3] == 0.0-0.0im
    @test isapprox(L, L'; rtol=1e-5)

    # hypergraph_ids
    @test hypergraph_ids(hgnn) == uid1
    @test hypergraph_ids(dhgnn) == did1

    # Case with no hypergraph_ids
    hgnn2 = HGNNHypergraph(uh1.v2he, uh1.he2v, 11, 5, 1, nothing, DataStore(), DataStore(), DataStore())
    @test hypergraph_ids(hgnn2) == ones(11)

    dhgnn2 = HGNNDiHypergraph(dh1; hypergraph_ids = nothing)
    @test hypergraph_ids(dhgnn2) == ones(11)

    # has_self_loops
    @test !(has_self_loops(hgnn))
    @test !(has_self_loops(dhgnn))

    # has_multi_hyperedges
    @test !(has_multi_hyperedges(hgnn))
    hg3 = Hypergraph{Bool}(2,2)
    hg3[:,:] .= true
    hgnn3 = HGNNHypergraph(hg3)
    @test has_multi_hyperedges(hgnn3)

    @test !(has_multi_hyperedges(dhgnn))
    dhg3 = DirectedHypergraph{Bool}(2, 3)
    dhg3.hg_tail[:,:] .= true
    dhg3.hg_head[:,:] .= true
    dhgnn3 = HGNNDiHypergraph(dhg3)
    @test has_multi_hyperedges(dhgnn3)

end


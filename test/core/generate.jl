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

@testset "HyperGraphNeuralNetworks                             random generation" begin
    # Erdos-Renyi random hypergraphs
    
    # Undirected
    Her_un = erdos_renyi_hypergraph(5, 5, HGNNHypergraph)
    @test nhv(Her_un) == 5
    @test nhe(Her_un) == 5
    @test all(length.(Her_un.v2he) .> 0)
    @test all(length.(Her_un.v2he) .<= 5)

    # With specified seed
    Her_un = erdos_renyi_hypergraph(5, 5, HGNNHypergraph; seed=1)
    @test Matrix(Her_un) == [
        nothing   nothing  1         1         1
        1         1         1          nothing  1
        nothing  1         1         1         1
        nothing  1         1          nothing  1
        nothing  1          nothing   nothing   nothing
    ]

    # Directed
    Her_di = erdos_renyi_hypergraph(5, 5, HGNNDiHypergraph)
    @test nhv(Her_un) == 5
    @test nhe(Her_un) == 5
    @test all(length.(Her_di.hg_tail.v2he) .> 0)
    @test all(length.(Her_di.hg_head.v2he) .> 0)
    @test all(length.(Her_di.hg_tail.v2he) .<= 5)
    @test all(length.(Her_di.hg_head.v2he) .<= 5)

    # With specified seed
    Her_di = erdos_renyi_hypergraph(5, 5, HGNNDiHypergraph; seed=42)
    @test Matrix(Her_di) == [
        (1, 1)              (nothing, 1)  (nothing, 1)        (1, 1)  (1, 1)
        (nothing, nothing)  (1, 1)        (nothing, nothing)  (1, 1)  (1, 1)
        (nothing, 1)        (1, 1)        (1, nothing)        (1, 1)  (1, nothing)
        (nothing, 1)        (nothing, 1)  (nothing, nothing)  (1, 1)  (1, 1)
        (1, 1)              (1, 1)        (1, 1)              (1, 1)  (1, 1)
    ]

    # With no self-loops
    DHr_nsl = erdos_renyi_hypergraph(5, 5, HGNNDiHypergraph; no_self_loops=true)
    for i in 1:5
        @test length(intersect(keys(DHr_nsl.hg_tail.v2he[i]), keys(DHr_nsl.hg_head.v2he[i]))) == 0
    end

    # Random k-uniform hypergraph

    # Undirected
    Hk = random_kuniform_hypergraph(5, 5, 3, HGNNHypergraph)
    @test nhv(Hk) == 5
    @test nhe(Hk) == 5
    @test all(length.(Hk.he2v) .== 3)

    # With specified seed
    Hk = random_kuniform_hypergraph(5, 5, 3, HGNNHypergraph; seed=42)
    @test Matrix(Hk) == [
        1         1         1         1          nothing
        nothing   nothing  1         1         1
        1         1          nothing   nothing  1
        1         1         1         1         1
        nothing   nothing   nothing   nothing   nothing
    ]

    # Directed
    DHk = random_kuniform_hypergraph(5, 5, 3, HGNNDiHypergraph)
    @test nhv(DHk) == 5
    @test nhe(DHk) == 5
    @test all(length.(DHk.hg_tail.he2v) .+ length.(DHk.hg_head.he2v) .== 3)

    # With specified seed
    DHk = random_kuniform_hypergraph(5, 5, 3, HGNNDiHypergraph; seed=42)
    @test Matrix(DHk) == [
        (nothing, nothing)  (1, nothing)        (nothing, nothing)  (nothing, 1)        (1, nothing)
        (1, nothing)        (nothing, nothing)  (1, nothing)        (1, nothing)        (nothing, 1)
        (1, nothing)        (nothing, nothing)  (1, nothing)        (1, nothing)        (nothing, 1)
        (nothing, 1)        (1, nothing)        (nothing, 1)        (nothing, nothing)  (nothing, nothing)
        (nothing, nothing)  (1, nothing)        (nothing, nothing)  (nothing, nothing)  (nothing, nothing)
    ]

    # Random d-regular hypergraph

    # Undirected
    Hd = random_dregular_hypergraph(5, 5, 3, HGNNHypergraph)
    @test nhv(Hd) == 5
    @test nhe(Hd) == 5
    @test all(length.(Hd.v2he) .== 3)

    # With specified seed
    Hd = random_dregular_hypergraph(5, 5, 3, HGNNHypergraph; seed=42)
    @test Matrix(Hd) == [
        1          nothing  1         1  nothing
        1          nothing  1         1  nothing
        1         1          nothing  1  nothing
        1         1          nothing  1  nothing
         nothing  1         1         1  nothing
    ]

    # Directed
    DHd = random_dregular_hypergraph(5, 5, 3, HGNNDiHypergraph)
    @test nhv(DHd) == 5
    @test nhe(DHd) == 5
    @test all(length.(DHd.hg_tail.v2he) .+ length.(DHd.hg_head.v2he) .== 3)

    # With specified seed
    DHd = random_dregular_hypergraph(5, 5, 3, HGNNDiHypergraph; seed=42)
    @test Matrix(DHd) == [
        (nothing, nothing)  (1, nothing)        (1, nothing)        (nothing, 1)        (nothing, nothing)
        (1, nothing)        (nothing, nothing)  (nothing, nothing)  (1, nothing)        (1, nothing)
        (nothing, nothing)  (1, nothing)        (1, nothing)        (nothing, 1)        (nothing, nothing)
        (nothing, 1)        (1, nothing)        (1, nothing)        (nothing, nothing)  (nothing, nothing)
        (1, nothing)        (nothing, 1)        (nothing, 1)        (nothing, nothing)  (nothing, nothing)
    ]

    # Random hypergraph with preferential attachment (undirected only, for now)

    H∂ = random_preferential_hypergraph(20, 0.5, HGNNHypergraph)
    @test nhv(H∂) == 20

    uh2 = Hypergraph{Bool}(5,5)
    uh2[1, 1] = true
    uh2[2, 1] = true
    uh2[4, 1] = true
    uh2[2, 2] = true
    uh2[5, 2] = true
    uh2[4, 3] = true
    uh2[2, 3] = true
    uh2[2, 4] = true
    uh2[4, 4] = true
    uh2[5, 4] = true
    uh2[4, 5] = true
    uh2[5, 5] = true

    # With specified seed
    H∂ = random_preferential_hypergraph(8, 0.5, HGNNHypergraph; seed=42, hg=uh2)
    @test Matrix(H∂) == [
        1          nothing   nothing   nothing   nothing   nothing   nothing  1          nothing   nothing
        1         1         1         1          nothing  1         1         1          nothing   nothing
         nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing
        1          nothing  1         1         1         1         1         1         1          nothing
         nothing  1          nothing  1         1         1         1         1          nothing   nothing
         nothing   nothing   nothing   nothing   nothing   nothing  1          nothing   nothing   nothing
         nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing  1          nothing
         nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing   nothing  1       
    ]
end

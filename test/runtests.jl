# Pkg.add(["SimpleDirectedHypergraphs", "SimpleHypergraphs", "StatsBase", "Random", "DataStructures", "Graphs", "Test"])
using HyperGraphNeuralNetworks 
using SimpleDirectedHypergraphs
using SimpleHypergraphs
using StatsBase
using Random
using DataStructures
using Graphs
using Test
using GNNGraphs

# @testset "Undirected HGNN            " begin
#     h = hg_load("data/test_UndiHGNN"; T=Int, HType=Hypergraph)
#     @test size(h) == (11, 5)
#     @test nhv(h) == 11
#     @test nhe(h) == 5
#     m = Matrix(h)
#     @test m == h
    # 

#     mktemp("data") do path, _
#         println(path)
#         hg_save(path, h)

#         loaded_hg = replace(read(path, String), r"\n*$" => "")

#         @test loaded_hg ==
#             reduce(replace,
#                 ["\r\n"=>"\n",
#                 r"^\"\"\"(?s).*\"\"\"\n"=>"", #remove initial comments
#                 r"\n*$"=>""], #remove final \n*
#                 init=read("data/test_UndiHGNN", String)) #no comments

#         @test loaded_hg ==
#             reduce(replace,
#                 ["\r\n"=>"\n",
#                 r"^\"\"\"(?s).*\"\"\"\n"=>"", #remove initial comments
#                 r"\n*$"=>""], #remove final \n*
#                 init=read("data/singlelinecomment", String)) #single line comment

#         @test loaded_hg ==
#             reduce(replace,
#                 ["\r\n"=>"\n",
#                 r"^\"\"\"(?s).*\"\"\"\n"=>"", #remove initial comments
#                 r"\n*$"=>""], #remove final \n*
#                 init=read("data/multiplelinescomment", String)) #multiple lines comment
#     end

@testset "construction" begin
    h1 = Hypergraph{Float64, Int, String}(11,5)
    #1st graph
    h1[1, 1] = 1.0
    h1[2, 1] = 2.0
    h1[4, 1] = 4.0
    h1[2, 2] = 3.0
    h1[5, 2] = 12.0
    h1[3, 2] = 0.0
    h1[4, 3] = 1.0
    h1[6, 3] = 4.0
    #2nd graph
    h1[7, 4] = 3.5
    h1[10, 4] = 1.0
    h1[11, 4] = 4.0
    h1[8, 5] = 1.0
    h1[9, 5] = 5.0
    h1[10, 5] = 7.0

    id1 = [1,1,1,1,1,1,2,2,2,2,2]
    hedata1 = DataStore(i = ([10, 20, 30, 40, 50]))
    
    #construct using exsiting hypergraph
    HGNN1 = HGNNHypergraph(h1; hypergraph_ids = id1)
    @test HGNN1.hypergraph_ids == id1
    @test size(HGNN1) == (11, 5)
    
    
    
    # @test HGNN1.hedata == hedata1
    #incident matrix
    # m1 = Matrix(h1)
    # @test m1 == h1
    # HGNN2 = HGNNHypergraph(m1; hypergraph_ids = id1)
    # @test HGNN2.hypergraph_ids == id1

end





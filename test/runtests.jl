using Random
using StatsBase

using Graphs
using GNNGraphs

using SimpleHypergraphs
using SimpleDirectedHypergraphs
using HyperGraphNeuralNetworks 

using Test

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

@testset "construction and traits" begin
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
    hedata1 = DataStore(i = ([10, 20, 30, 40, 50])) #cannot input this

    #construct using exsiting hypergraph
    HGNN1 = HGNNHypergraph(h1; hypergraph_ids = id1)
    @test size(HGNN1) == (11, 5)
    @test nhv(HGNN1) == 11
    @test nhe(HGNN1) == 5
    @test HGNN1.hypergraph_ids == id1
    @test HGNN1.hedata == DataStore(5) #this should be nothing instead of empty DataStore
    @test HGNN1.hgdata == DataStore(2)

    #construct using matrix
    m = Matrix(h1)
    @test m == h1
    @test m == [1.0     nothing nothing nothing nothing
                2.0     3.0     nothing nothing nothing
                nothing 0.0     nothing nothing nothing
                4.0     nothing 1.0     nothing nothing
                nothing 12.0    nothing nothing nothing
                nothing nothing 4.0     nothing nothing
                nothing nothing nothing 3.5     nothing
                nothing nothing nothing nothing 1.0
                nothing nothing nothing nothing 5.0
                nothing nothing nothing 1.0     7.0
                nothing nothing nothing 4.0     nothing]
    HGNN2 = HGNNHypergraph(m; hypergraph_ids = id1)
    @test HGNN2 == HGNN1

    #construct with no hypergraph and num_nodes vertices
    HGNN3 = HGNNHypergraph(3)
    @test HGNN3.num_vertices == 3
    @test HGNN3.num_hyperedges == 0

    #construct with minimal information
    HGNN4 = HGNNHypergraph()
    @test HGNN4.num_vertices == 0

    #hasvertexmeta and hashyperedgemeta
    @test hasvertexmeta(HGNN1) == true #all return false
    @test hashyperedgemeta(HGNN1) == true
    @test hasvertexmeta(HGNNHypergraph) == true
    @test hashyperedgemeta(HGNNHypergraph) == true
end

@testset "add/remove vertex/hyperedge" begin
    incident = [1.0     2.0
                1.0     nothing
                nothing 1.0
                nothing nothing]
    HGNN1 = HGNNHypergraph(incident)
    #do i have to test functions with !
    
    @test HGNN1.num_vertices == 4
    features1 = DataStore(1)
    hyperedges1 = Dict(2 => 4.0) #connect the new vertex to hyperedge 2
    HGNN2 = add_vertex(HGNN1, features1; hyperedges = hyperedges1)
    @test HGNN2.num_vertices == 5
    @test HGNN2.v2he[5] == Dict(2 => 4.0)
    @test HGNN2 != HGNN1

    @test_throws "Not implemented! Number of vertices in HGNNHypergraph is fixed." SimpleHypergraphs.add_vertex!(HGNN1)
    @test_throws "Not implemented! Number of vertices in HGNNHypergraph is fixed." SimpleHypergraphs.remove_vertex!(HGNN1, 1)
    @test_throws "Not implemented! Number of hyperedges in HGNNHypergraph is fixed." SimpleHypergraphs.add_hyperedge!(HGNN1)
    @test_throws "Not implemented! Number of hyperedges in HGNNHypergraph is fixed." SimpleHypergraphs.remove_hyperedge!(HGNN1, 1)

    HGNN3 = remove_vertex(HGNN2, 5)
    @test HGNN3.num_vertices == 4
    @test HGNN1 == HGNN3

    features4 = DataStore(1)
    vertices4 = Dict(2 => 4.0, 4 => 5.0) #connect the new hyperedge to vertices 2 and 4
    HGNN4 = add_hyperedge(HGNN3, features4; vertices = vertices4)
    @test HGNN4.num_hyperedges == 3
    @test HGNN4.he2v[3] == Dict(2 => 4.0, 4 => 5.0)
    @test HGNN4 != HGNN3

    HGNN5 = remove_hyperedge(HGNN4, 3)
    @test HGNN5.num_hyperedges == 2
    @test HGNN5 == HGNN3

    HGNN6 = remove_vertices(HGNN5, [1, 3])
    print("num ver", HGNN6.num_vertices)
    print("v2he", HGNN6.v2he)
    # @test HGNN6.num_vertices == 2
    # @test HGNN6.v2he == [Dict(1 => 1.0), Dict()]
    ###the num_vertices is still 4 but the lenght of v2he is 2

end


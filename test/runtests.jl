using Random
using StatsBase
using Test
using Graphs
using GNNGraphs
using MLUtils
using HyperGraphNeuralNetworks
using SimpleHypergraphs
using SimpleDirectedHypergraphs

@testset "HyperGraphNeuralNetworks HGNNHypergraph" begin
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
    hedata1 = [10, 20, 30, 40, 50] 

    # Direct Construction
    HGNN0 = HGNNHypergraph(h1.v2he, h1.he2v, 11, 5, 2, id1, DataStore(), DataStore(), DataStore())
    @test size(HGNN0) == (11, 5)
    @test nhv(HGNN0) == 11
    @test nhe(HGNN0) == 5
    @test HGNN0.hypergraph_ids == id1
    @test HGNN0.vdata == DataStore()
    @test HGNN0.hedata == DataStore() 
    @test HGNN0.hgdata == DataStore()

    # Test type equality
    @test HGNN0 == HGNNHypergraph{Float64, Dict{Int, Float64}}(h1.v2he, h1.he2v, 11, 5, 2, id1, DataStore(), DataStore(), DataStore())

    # Construct using existing hypergraph
    HGNN1 = HGNNHypergraph(h1; hypergraph_ids = id1, hedata = hedata1)
    @test size(HGNN1) == (11, 5)
    @test nhv(HGNN1) == 11
    @test nhe(HGNN1) == 5
    @test HGNN1.hypergraph_ids == id1
    @test HGNN1.hedata == DataStore(e = hedata1) 
    @test HGNN1.hgdata == DataStore(2)

    # Test type equality
    @test HGNN1 == HGNNHypergraph{Float64}(h1; hypergraph_ids = id1, hedata=hedata1)
    @test HGNN1 == HGNNHypergraph{Float64, Dict{Int, Float64}}(h1; hypergraph_ids = id1, hedata=hedata1)

    # Construct using matrix
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
    HGNN2 = HGNNHypergraph(m; hypergraph_ids = id1, hedata = hedata1)
    @test HGNN2 == HGNN1

    # Test type equality
    @test HGNN2 == HGNNHypergraph{Float64}(m; hypergraph_ids = id1, hedata = hedata1)
    @test HGNN2 == HGNNHypergraph{Float64, Dict{Int, Float64}}(m; hypergraph_ids = id1, hedata = hedata1)

    # Construct with no hypergraph and num_nodes vertices
    HGNN3 = HGNNHypergraph(3)
    @test HGNN3.num_vertices == 3
    @test HGNN3.num_hyperedges == 0

    #construct with minimal information
    HGNN4 = HGNNHypergraph()
    @test HGNN4.num_vertices == 0

    #hasvertexmeta and hashyperedgemeta
    @test hasvertexmeta(HGNN1) == true 
    @test hashyperedgemeta(HGNN1) == true
    @test hasvertexmeta(HGNNHypergraph) == true
    @test hashyperedgemeta(HGNNHypergraph) == true
end

@testset "HyperGraphNeuralNetworks HGNNHypergraph modification" begin
    incident = [1.0     2.0
                1.0     nothing
                nothing 1.0
                nothing nothing]
    HGNN1 = HGNNHypergraph(incident)
    @test HGNN1.num_vertices == 4

    #add/remove single vertex or hyperedge
    features2 = DataStore(1)
    hyperedges2 = Dict(2 => 4.0) #connect the new vertex to hyperedge 2
    HGNN2 = add_vertex(HGNN1, features2; hyperedges = hyperedges2)
    @test HGNN2.num_vertices == 5
    @test HGNN2.v2he[5] == Dict(2 => 4.0)
    @test HGNN2 != HGNN1

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

    h = Hypergraph{Float64, Int, String}(7,4)
    h[1, 1] = 1.0
    h[2, 1] = 1.0
    h[3, 1] = 1.0
    h[3, 2] = 1.0
    h[4, 2] = 1.0
    h[4, 3] = 1.0
    h[5, 3] = 1.0
    h[6, 3] = 1.0
    h[7, 4] = 1.0
    vdata6 = (a = [1,2,3,4,5,6,7],)
    hedata6 = (b = [1,2,3,4],)
    HGNN6 = HGNNHypergraph(h; vdata = vdata6, hedata = hedata6)

    #add/remove multiple vertices or hyperedges
    HGNN7 = remove_vertices(HGNN6, [2, 5, 6, 7])
    @test HGNN7.num_vertices == 3
    @test HGNN7.num_hyperedges == 4
    @test HGNN7.v2he == [Dict(1 => 1.0), 
                        Dict(1 => 1.0, 2 => 1.0),  
                        Dict(2 => 1.0, 3 => 1.0)]
    @test HGNN7.he2v == [Dict(1 => 1.0, 2 => 1.0)
                        Dict(2 => 1.0, 3 => 1.0)
                        Dict(3 => 1.0)
                        Dict{Int64, Float64}()]
    
    HGNN8 = remove_hyperedges(HGNN7, [2, 4])
    @test HGNN8.num_vertices == 3
    @test HGNN8.num_hyperedges == 2
    @test HGNN8.v2he == [Dict(1 => 1.0), Dict(1 => 1.0), Dict(2 => 1.0)]
    @test HGNN8.he2v == [Dict(1 => 1.0, 2 => 1.0), Dict(3 => 1.0)]
    
    features9 = DataStore(a = [[8], [9]])
    hyperedges9 = [Dict(1 => 2.0), Dict(2 => 3.0)] #connect the new vertex to hyperedges 1 and 2
    HGNN9 = add_vertices(HGNN8, 2, features9; hyperedges = hyperedges9)
    @test HGNN9.num_vertices == 5
    @test HGNN9.v2he[4] == Dict(1 => 2.0)
    @test HGNN9.v2he[5] == Dict(2 => 3.0)
    @test HGNN9.vdata == DataStore(a = [1,3,4,8,9])

    features10 = DataStore(b = [[5], [6]])
    vertices10 = [Dict(1 => 1.0, 4 => 1.0), Dict(3 => 2.0, 5 => 2.0)]
    HGNN10 = add_hyperedges(HGNN9, 2, features10; vertices = vertices10)
    @test HGNN10.num_hyperedges == 4
    @test HGNN10.he2v[3] == Dict(1 => 1.0, 4 => 1.0)
    @test HGNN10.he2v[4] == Dict(3 => 2.0, 5 => 2.0)
    @test HGNN10.hedata == DataStore(b = [1,3,5,6])

    #These functions are not implemented
    @test_throws "Not implemented! Number of vertices in HGNNHypergraph is fixed." SimpleHypergraphs.add_vertex!(HGNN1)
    @test_throws "Not implemented! Number of vertices in HGNNHypergraph is fixed." SimpleHypergraphs.remove_vertex!(HGNN1, 1)
    @test_throws "Not implemented! Number of hyperedges in HGNNHypergraph is fixed." SimpleHypergraphs.add_hyperedge!(HGNN1)
    @test_throws "Not implemented! Number of hyperedges in HGNNHypergraph is fixed." SimpleHypergraphs.remove_hyperedge!(HGNN1, 1)

end

@testset "HyperGraphNeuralNetworks HGNNHypergraph Base functions" begin
    # Base.zero
    zeroHGNN = zero(HGNNHypergraph)
    @test zeroHGNN.num_vertices == 0
    @test zeroHGNN.num_hyperedges == 0
    @test zeroHGNN.num_hypergraphs == 1

    h = Hypergraph{Float64, Int, String}(2, 1)
    h[1, 1] = 1.0
    h[2, 1] = 2.0
    vdata = (a = [[1,2],[3,4]], b = [1, -1])
    hedata = (b = [1],)
    hgdata = [3]
    HGNN = HGNNHypergraph(h; vdata = vdata, hedata = hedata, hgdata = hgdata)

    # Base.copy
    copyHGNN = copy(HGNN; deep = false)
    @test copyHGNN == HGNN
    @test copyHGNN.v2he === HGNN.v2he
    @test copyHGNN.he2v === HGNN.he2v
    deepcopyHGNN = copy(HGNN; deep = true)
    @test deepcopyHGNN !== HGNN
    @test deepcopyHGNN.v2he !== HGNN.v2he
    @test deepcopyHGNN.he2v !== HGNN.he2v

    # Base.show
    normalize_str(s::AbstractString) = replace(s, r"\s+" => " ") |> strip
    @test normalize_str(sprint(show, HGNN)) == normalize_str("HGNNHypergraph(2, 1, 1) with 
    vertex features: DataStore(2) with 2 elements:
        a = 2-element Vector{Vector{Int64}}
        b = 2-element Vector{Int64}, 
    hyperedge features: DataStore(1) with 1 element:
        b = 1-element Vector{Int64}, 
    hypergraph features: DataStore() with 1 element: 
    u = 1-element Vector{Int64} data")
    @test normalize_str(
        sprint(show, MIME("text/plain"), HGNN; context=IOContext(stdout, :compact=>true))
        ) == normalize_str("HGNNHypergraph(2, 1, 1) with 
        vertex features: DataStore(2) with 2 elements:
            a = 2-element Vector{Vector{Int64}}
            b = 2-element Vector{Int64}, 
        hyperedge features: DataStore(1) with 1 element:
            b = 1-element Vector{Int64}, 
        hypergraph features: DataStore() with 1 element: 
            u = 1-element Vector{Int64} data")
    @test normalize_str(
        sprint(show, MIME("text/plain"), HGNN)
        ) == normalize_str("HGNNHypergraph: 
        num_vertices: 2 
        num_hyperedges: 1 
        vdata (vertex data): a = 2-element Vector{Vector{Int64}} 
                            b = 2-element Vector{Int64} 
        hedata (hyperedge data): b = 1-element Vector{Int64}
        hgdata (hypergraph data): u = 1-element Vector{Int64}")

    # MLUtils.numobs
    #TODO: probably move this elsewhere
    @test numobs(HGNN) == HGNN.num_hypergraphs 

    # Base.hash
    newHGNN = add_vertex(HGNN, DataStore(a = [[1, 2]], b = [3]))
    @test newHGNN.vdata == DataStore(a = [[1,2],[3,4], [1,2]], b = [1, -1, 3])
    @test hash(HGNN) == hash(copyHGNN)
    @test hash(HGNN) != hash(newHGNN)

    # Base.getproperty
    @test getproperty(HGNN, :v2he) == HGNN.v2he
    @test_throws ArgumentError getproperty(HGNN, :b)
    @test getproperty(HGNN, :a) == vdata.a
    @test_throws ArgumentError getproperty(HGNN, :foo) 

end

@testset "HyperGraphNeuralNetworks HGNNDiHypergraph" begin
    h1 = DirectedHypergraph{Float64, Nothing, Nothing}(11,5)
    h1[1,1,1] = 1.0
    h1[1,2,1] = 2.0
    h1[2,4,1] = 4.0
    h1[1,2,2] = 3.0
    h1[1,5,2] = 12.0
    h1[2,3,2] = 0.0
    h1[1,4,3] = 1.0
    h1[2,6,3] = 4.0
    # 2nd hypergraph
    h1[1,7,4] = 3.5
    h1[1,10,4] = 1.0
    h1[2,11,4] = 4.0
    h1[2,8,5] = 1.0
    h1[2,9,5] = 5.0
    h1[1,10,5] = 7.0
    id1 = [1,1,1,1,1,1,2,2,2,2,2]
    hedata1 = [10, 20, 30, 40, 50]

    # Direct construction
    HGNN0 = HGNNDiHypergraph(
        h1.hg_tail,
        h1.hg_head,
        11, 5, 2,
        id1,
        DataStore(),
        DataStore(),
        DataStore()
    )
    @test size(HGNN0) == (11, 5)
    @test nhv(HGNN0) == 11
    @test nhe(HGNN0) == 5
    @test HGNN0.hypergraph_ids == id1
    @test HGNN0.vdata == DataStore()
    @test HGNN0.hedata == DataStore() 
    @test HGNN0.hgdata == DataStore()

    # Construct using exsiting directedhypergraph
    HGNN1 = HGNNDiHypergraph(h1, hypergraph_ids = id1, hedata = hedata1)
    @test size(HGNN1) == (11, 5)
    @test nhv(HGNN1) == 11
    @test nhe(HGNN1) == 5
    @test HGNN1.hypergraph_ids == id1
    @test HGNN1.hedata == DataStore(e = hedata1) 
    @test HGNN1.hgdata == DataStore(2)

    #construct using matrix
    m = Matrix(h1)
    @test m == h1
    tailMatrix = getindex.(m, 1)
    headMatrix = getindex.(m, 2)
    @test tailMatrix == [1.0     nothing nothing nothing nothing 
                         2.0     3.0     nothing nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing 1.0     nothing nothing
                         nothing 12.0    nothing nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing nothing 3.5     nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing nothing 1.0     7.0
                         nothing nothing nothing nothing nothing]
    @test headMatrix == [nothing nothing nothing nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing 0.0     nothing nothing nothing
                         4.0     nothing nothing nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing 4.0     nothing nothing
                         nothing nothing nothing nothing nothing
                         nothing nothing nothing nothing 1.0
                         nothing nothing nothing nothing 5.0
                         nothing nothing nothing nothing nothing
                         nothing nothing nothing 4.0     nothing]
    HGNN2 = HGNNDiHypergraph(tailMatrix, headMatrix; hypergraph_ids = id1, hedata = hedata1)
    @test HGNN2 == HGNN1

    #construct with no hypergraph and num_nodes vertices
    HGNN3 = HGNNDiHypergraph(3)
    @test HGNN3.num_vertices == 3
    @test HGNN3.num_hyperedges == 0

    #construct with minimal information
    HGNN4 = HGNNDiHypergraph()
    @test HGNN4.num_vertices == 0
    
    #hasvertexmeta and hashyperedgemeta
    @test hasvertexmeta(HGNN1) == true 
    @test hashyperedgemeta(HGNN1) == true
    @test hasvertexmeta(HGNNDiHypergraph) == true
    @test hashyperedgemeta(HGNNDiHypergraph) == true

    # Base.zero
    zeroHGNN = zero(HGNNDiHypergraph)
    @test zeroHGNN.num_vertices == 0
    @test zeroHGNN.num_hyperedges == 0
    @test zeroHGNN.num_hypergraphs == 1
end

@testset "HGNNDiHypergraph modification functions" begin
    tailMatrix = [1.0     nothing
                  1.0     nothing
                  nothing nothing
                  nothing 1.0]
    headMatrix = [nothing nothing
                  nothing 1.0
                  1.0     1.0
                  nothing nothing]
    vdata1 = (a = [1, 2, 3, 4], b = [1, -1, 1, -1])
    hedata1 = (c = [2.0, 4.0],)
    HGNN1 = HGNNDiHypergraph(tailMatrix, headMatrix; vdata = vdata1, hedata = hedata1)
    
    #add_vertices, add_vertex, remove_vertex, remove_hyperedge
    @test HGNN1.num_vertices == 4
    features1 = DataStore(a = [[5], [6]], b = [[1], [-1]])
    hyperedges_tail1 = [Dict(2 => 2.0), Dict{Int64, Float64}()]
    hyperedges_head1 = [Dict{Int64, Float64}(), Dict(1=>3.0)]
    HGNN2 = add_vertices(HGNN1, 2, features1; hyperedges_tail = hyperedges_tail1,
                            hyperedges_head = hyperedges_head1)
    @test HGNN2.hg_tail.he2v == [Dict(1 => 1.0, 2 => 1.0), Dict(4 => 1.0, 5 => 2.0)]
    @test HGNN2.hg_head.he2v == [Dict(3 => 1.0, 6 => 3.0), Dict(2 => 1.0, 3 => 1.0)]
    @test HGNN2.vdata == DataStore(a = [1, 2, 3, 4, 5, 6], b = [1, -1, 1, -1, 1, -1])

    HGNN3 = remove_vertex(HGNN2, 5)
    @test HGNN3.num_vertices == 5
    @test HGNN3.num_hyperedges == 2
    @test HGNN3.hg_tail.he2v == [Dict(1 => 1.0, 2 => 1.0), Dict(4 => 1.0)]
    @test HGNN3.hg_head.he2v == [Dict(3 => 1.0, 5=> 3.0), Dict(2 => 1.0, 3 => 1.0)]

    features4 = DataStore(c = [[1.0], [2.0]])
    vertices_tail4 = [Dict(3 => 2.0), Dict(5 => 3.0)]
    vertices_head4 = [Dict(2 => 3.0), Dict(4 => 6.0)]
    HGNN4 = add_hyperedges(HGNN3, 2, features4; vertices_tail = vertices_tail4, 
                            vertices_head = vertices_head4)
    @test HGNN4.num_hyperedges == 4
    @test HGNN4.num_vertices == 5
    @test HGNN4.hg_tail.he2v[3] == Dict(3 => 2.0)
    @test HGNN4.hg_tail.he2v[4] == Dict(5 => 3.0)
    @test HGNN4.hg_head.he2v[3] == Dict(2 => 3.0)
    @test HGNN4.hg_head.he2v[4] == Dict(4 => 6.0)
    @test HGNN4.hedata == DataStore(c = [2.0, 4.0, 1.0, 2.0])
    
    HGNN5 = remove_hyperedge(HGNN4, 2)
    @test HGNN5.num_hyperedges == 3
    @test HGNN5.num_vertices == 5
    @test HGNN5.hg_tail.v2he == [Dict(1 => 1.0), Dict(1 => 1.0), Dict(2 => 2.0), 
                                Dict{Int64, Float64}(), Dict(2 => 3.0)]
    @test HGNN5.hg_head.v2he == [Dict{Int64, Float64}(), Dict(2 => 3.0), Dict(1 => 1.0), 
                                Dict(2 => 6.0), Dict(1 => 3.0)]

    h = DirectedHypergraph{Float64, Int, String}(7,4)
    h[1, 1, 1] = 1.0
    h[2, 2, 1] = 1.0
    h[2, 3, 1] = 1.0
    h[1, 3, 2] = 1.0
    h[2, 4, 2] = 1.0
    h[1, 4, 3] = 1.0
    h[1, 5, 3] = 1.0
    h[2, 6, 3] = 1.0
    h[1, 7, 4] = 1.0
    HGNN6 = HGNNDiHypergraph(h)

    #remove_hyperedges
    HGNN7 = remove_vertices(HGNN6, [2, 5, 6, 7])
    @test HGNN7.num_vertices == 3
    @test HGNN7.num_hyperedges == 4
    @test HGNN7.hg_tail.v2he == [Dict(1 => 1.0), 
                                Dict(2 => 1.0),
                                Dict(3 => 1.0)]
    @test HGNN7.hg_head.v2he == [Dict{Int64, Float64}(),
                                Dict(1 => 1.0),
                                Dict(2 => 1.0)]
    @test HGNN7.hg_tail.he2v == [Dict(1 => 1.0),
                                Dict(2 => 1.0),
                                Dict(3 => 1.0),
                                Dict{Int64, Float64}()]
    @test HGNN7.hg_head.he2v == [Dict(2 => 1.0),
                                Dict(3 => 1.0),
                                Dict{Int64, Float64}(),
                                Dict{Int64, Float64}()]

    HGNN8 = remove_hyperedges(HGNN7, [2, 4])
    @test HGNN8.num_vertices == 3
    @test HGNN8.num_hyperedges == 2
    @test HGNN8.hg_tail.v2he == [Dict(1 => 1.0),
                                Dict{Int64, Float64}(),
                                Dict(2 => 1.0)]
    @test HGNN8.hg_head.v2he == [Dict{Int64, Float64}(),
                                Dict(1 => 1.0),
                                Dict{Int64, Float64}()]
    @test HGNN8.hg_tail.he2v == [Dict(1 => 1.0),
                                Dict(3 => 1.0)]
    @test HGNN8.hg_head.he2v == [Dict(2 => 1.0),
                                Dict{Int64, Float64}()]
    
    #These functions are not implemented
    @test_throws "Not implemented! Number of vertices in HGNNDiHypergraph is fixed." SimpleHypergraphs.add_vertex!(HGNN1)
    @test_throws "Not implemented! Number of vertices in HGNNDiHypergraph is fixed." SimpleHypergraphs.remove_vertex!(HGNN1, 1)
    @test_throws "Not implemented! Number of hyperedges in HGNNDiHypergraph is fixed." SimpleHypergraphs.add_hyperedge!(HGNN1)
    @test_throws "Not implemented! Number of hyperedges in HGNNDiHypergraph is fixed." SimpleHypergraphs.remove_hyperedge!(HGNN1, 1)
end

@testset "HyperGraphNeuralNetworks HGNNDiHypergraph Base functions" begin
    h = DirectedHypergraph{Float64, Int, String}(2,1)
    h[1, 1, 1] = 1.0
    h[2, 2, 1] = 2.0
    vdata = (a = [[1,2],[3,4]], b = [1, -1])
    hedata = (b = [1],)
    hgdata = [3]
    HGNN = HGNNDiHypergraph(h; vdata = vdata, hedata = hedata, hgdata = hgdata)

    #base.show
    normalize_str(s::AbstractString) = replace(s, r"\s+" => " ") |> strip
    @test normalize_str(sprint(show, HGNN)) == normalize_str("
        HGNNDiHypergraph(2, 1, 1) with 
        vertex features: DataStore(2) with 2 elements:
            a = 2-element Vector{Vector{Int64}}
            b = 2-element Vector{Int64}, 
        hyperedge features: DataStore(1) with 1 element:
            b = 1-element Vector{Int64}, 
        hypergraph features: DataStore() with 1 element:
            u = 1-element Vector{Int64} data")
    @test normalize_str(
        sprint(show, MIME("text/plain"), HGNN; context=IOContext(stdout, :compact=>true))
        ) == normalize_str("HGNNDiHypergraph(2, 1, 1) with 
                vertex features: DataStore(2) with 2 elements: 
                    a = 2-element Vector{Vector{Int64}} 
                    b = 2-element Vector{Int64}, 
                hyperedge features: DataStore(1) with 1 element: 
                    b = 1-element Vector{Int64}, 
                hypergraph features: DataStore() with 1 element: 
                    u = 1-element Vector{Int64} data")
    @test normalize_str(
        sprint(show, MIME("text/plain"), HGNN)
        ) == normalize_str("HGNNDiHypergraph: num_vertices: 2 num_hyperedges: 1 
                            vdata (vertex data): 
                                a = 2-element Vector{Vector{Int64}} 
                                b = 2-element Vector{Int64} 
                            hedata (hyperedge data): 
                                b = 1-element Vector{Int64} 
                            hgdata (hypergraph data): 
                                u = 1-element Vector{Int64}")
    
    #base.copy
    copyHGNN = copy(HGNN; deep = false)
    @test copyHGNN == HGNN
    @test copyHGNN.hg_tail === HGNN.hg_tail
    @test copyHGNN.hg_head === HGNN.hg_head
    deepcopyHGNN = copy(HGNN; deep = true)
    @test deepcopyHGNN !== HGNN
    @test deepcopyHGNN.hg_tail !== HGNN.hg_tail
    @test deepcopyHGNN.hg_head !== HGNN.hg_head

    #MLUtils.numobs
    @test numobs(HGNN) == HGNN.num_hypergraphs 

    #Bese.hash
    newHGNN = add_vertex(HGNN, DataStore(a = [[1, 2]], b = [3]))
    @test newHGNN.vdata == DataStore(a = [[1,2],[3,4], [1,2]], b = [1, -1, 3])
    @test hash(HGNN) == hash(copyHGNN)
    @test hash(HGNN) != hash(newHGNN)

    #Base.getproperty
    @test getproperty(HGNN, :hg_tail) == HGNN.hg_tail
    @test_throws ArgumentError getproperty(HGNN, :b)
    @test getproperty(HGNN, :a) == vdata.a
    @test_throws ArgumentError getproperty(HGNN, :foo) 

end

@testset "HyperGraphNeuralNetworks random generation" begin
    # Erdos-Renyi random hypergraphs
    
    # Undirected
    Her_un = erdos_renyi_hypergraph(5, 5, HGNNHypergraph)
    @test nhv(Her_un) == 5
    @test nhe(Her_un) == 5
    @test  all(length.(Her_un.v2he) .> 0)
    @test  all(length.(Her_un.v2he) .<= 5)

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
    @test  all(length.(Her_di.hg_tail.v2he) .> 0)
    @test  all(length.(Her_di.hg_head.v2he) .> 0)
    @test  all(length.(Her_di.hg_tail.v2he) .<= 5)
    @test  all(length.(Her_di.hg_head.v2he) .<= 5)

    # With specified seed
    Her_di = erdos_renyi_hypergraph(5, 5, HGNNDiHypergraph; seed=42)
    @test Matrix(Her_di) == [
        (1, 1)              (nothing, 1)  (nothing, 1)        (1, 1)  (1, 1)
        (nothing, nothing)  (1, 1)        (nothing, nothing)  (1, 1)  (1, 1)
        (nothing, 1)        (1, 1)        (1, nothing)        (1, 1)  (1, nothing)
        (nothing, 1)        (nothing, 1)  (nothing, nothing)  (1, 1)  (1, 1)
        (1, 1)              (1, 1)        (1, 1)              (1, 1)  (1, 1)
    ]
end

@testset "HyperGraphNeuralNetworks query" begin

end

@testset "HyperGraphNeuralNetworks transforms" begin

end
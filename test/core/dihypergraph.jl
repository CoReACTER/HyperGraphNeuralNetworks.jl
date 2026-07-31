using StatsBase
using LinearAlgebra
using Test
using Graphs
using GNNGraphs
using MLUtils
using SimpleDirectedHypergraphs
using HyperGraphNeuralNetworks

# Example directed hypergraph
dh1 = DirectedHypergraph{Float64, Int, String}(11,5)
dh1[1,1,1] = 1.0
dh1[1,2,1] = 2.0
dh1[2,4,1] = 4.0
dh1[1,2,2] = 3.0
dh1[1,5,2] = 12.0
dh1[2,3,2] = 0.0
dh1[1,4,3] = 1.0
dh1[2,6,3] = 4.0
#2nd graph
dh1[1,7,4] = 3.5
dh1[1,10,4] = 1.0
dh1[2,11,4] = 4.0
dh1[2,8,5] = 1.0
dh1[2,9,5] = 5.0
dh1[1,10,5] = 7.0
did1 = [1,1,1,1,1,1,2,2,2,2,2]
dhedata1 = [10, 20, 30, 40, 50]

@testset "HyperGraphNeuralNetworks                             HGNNDiHypergraph" begin

    @testset "    construction" begin
	#construct using exsiting directedhypergraph
	HGNN1 = HGNNDiHypergraph(dh1, hypergraph_ids = did1, hedata = dhedata1)
	@test size(HGNN1) == (11, 5)
	@test nhv(HGNN1) == 11
	@test nhe(HGNN1) == 5
	@test HGNN1.hypergraph_ids == did1
	@test HGNN1.hedata == DataStore(e = dhedata1) 
	@test HGNN1.hgdata == DataStore(2)

	#construct using matrix
	m = Matrix(dh1)
	@test m == dh1
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
	HGNN2 = HGNNDiHypergraph(tailMatrix, headMatrix; hypergraph_ids = did1, hedata = dhedata1)
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

     @testset "    modification" begin
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

    @testset "    base functions" begin
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

end;


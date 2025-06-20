"""
TODO: add docstrings
"""

# TODO: you are here
function add_self_loops(hg::H; add_repeated_hyperedge::Bool = false) where {H <: AbstractHGNNHypergraph}
    
    vertices = [1:hg.num_vertices]
    if add_repeated_hyperedge
        
    else

    end

end

function remove_self_loops()
end

function remove_hyperedges()
end

function remove_multi_hyperedges()
end

function remove_vertices()
end

function add_hyperedges()
end

function add_vertices()
end

function rewire_hyperedges()
end

function to_undirected()
end

function set_hyperedge_weight()
end

# MLUtils.batch
function batch()
end

# MLUtils.unbatch
function unbatch()
end

function get_hypergraph()
end

function negative_sample()
end

function random_split_vertices()
end

function random_split_hyperedges()
end

# PageRank diffusion?
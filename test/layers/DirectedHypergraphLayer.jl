using Test
using Lux
using Random
using HyperGraphNeuralNetworks

@testset "DirectedHypergraphLayer" begin
    rng = Random.default_rng()

    layer = DirectedHypergraphLayer(
        3,
        0,
        8;
        activation = tanh,
        normalize = true,
    )

    ps, st = Lux.setup(rng, layer)

    X_vertex = Float32[
        1.0 0.0 2.0;
        0.0 1.0 1.0;
        1.0 1.0 0.0;
        2.0 0.0 1.0
    ]

    source_matrix = Float32[
        1 0 1;
        1 0 0;
        0 1 0;
        0 0 0
    ]

    target_matrix = Float32[
        0 0 0;
        0 1 0;
        1 0 0;
        0 0 1
    ]

    output, new_st = layer(
        (
            X_vertex,
            source_matrix,
            target_matrix,
        ),
        ps,
        st,
    )

    @test size(output.updated_vertices) == (4, 8)
    @test size(output.updated_hyperedges) == (3, 8)

    @test all(isfinite, output.updated_vertices)
    @test all(isfinite, output.updated_hyperedges)

    @test new_st == st
end
using Test
using Lux
using Random
using HyperGraphNeuralNetworks

@testset "DirectedHypergraphLayer" begin

    rng = Random.default_rng()

    layer = DirectedHypergraphLayer(3, 8, tanh)

    ps, st = Lux.setup(rng, layer)

    X_species = Float32[
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

    output, st = layer(
        (X_species, source_matrix, target_matrix),
        ps,
        st
    )

    @test size(output.updated_species) == (4, 8)
    @test size(output.reaction_embeddings) == (3, 8)

    @test all(isfinite, output.updated_species)
    @test all(isfinite, output.reaction_embeddings)

end
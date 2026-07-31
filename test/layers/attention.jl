using Test
using Random
using Lux
using HyperGraphNeuralNetworks

@testset "HyperGraphNeuralNetworks                       DirectedAttentionLayer" begin

    @testset "    Constructor validation" begin
        layer = DirectedAttentionLayer(3, 0, 4)

        @test layer isa DirectedAttentionLayer
        @test layer.vertex_in_dim == 3
        @test layer.hyperedge_in_dim == 0
        @test layer.hidden_dim == 4
        @test layer.return_attention == false

        attention_layer = DirectedAttentionLayer(
            3,
            2,
            4;
            activation = identity,
            attention_activation = identity,
            return_attention = true,
        )

        @test attention_layer.activation === identity
        @test attention_layer.attention_activation === identity
        @test attention_layer.return_attention

        @test_throws ArgumentError DirectedAttentionLayer(0, 0, 4)
        @test_throws ArgumentError DirectedAttentionLayer(-1, 0, 4)
        @test_throws ArgumentError DirectedAttentionLayer(3, -1, 4)
        @test_throws ArgumentError DirectedAttentionLayer(3, 0, 0)
        @test_throws ArgumentError DirectedAttentionLayer(3, 0, -1)
    end

    @testset "    Parameter and state initialisation" begin
        rng = Random.Xoshiro(1234)

        layer = DirectedAttentionLayer(3, 2, 4)
        ps, st = Lux.setup(rng, layer)

        @test size(ps.W_vertex) == (3, 4)
        @test size(ps.b_vertex) == (1, 4)
        @test size(ps.a_source) == (4, 1)
        @test size(ps.a_target) == (4, 1)
        @test size(ps.W_hyperedge) == (10, 4)
        @test size(ps.b_hyperedge) == (1, 4)
        @test size(ps.W_vertex_update) == (8, 4)
        @test size(ps.b_vertex_update) == (1, 4)

        @test st == NamedTuple()
        @test Lux.statelength(layer) == 0
        @test Lux.parameterlength(layer) == 104
        @test Lux.parameterlength(layer) == Lux.parameterlength(ps)

        layer2 = DirectedAttentionLayer(3, 0, 4)
        ps2, _ = Lux.setup(Random.Xoshiro(1234), layer2)

        @test size(ps2.W_hyperedge) == (8, 4)
        @test Lux.parameterlength(layer2) == 96
    end

    @testset "    masked_incidence_softmax" begin

        raw_scores = Float32[0,0,0]

        incidence_matrix = Float32[
            1 0 0
            1 1 0
            0 1 0
        ]

        attention = HyperGraphNeuralNetworks.masked_incidence_softmax(
            raw_scores,
            incidence_matrix,
        )

        expected = Float32[
            0.5 0.0 0.0
            0.5 0.5 0.0
            0.0 0.5 0.0
        ]

        @test size(attention) == size(incidence_matrix)
        @test isapprox(attention, expected)

        @test all(attention[incidence_matrix .== 0] .== 0)

        @test isapprox(sum(attention[:,1]),1.0f0)
        @test isapprox(sum(attention[:,2]),1.0f0)
        @test sum(attention[:,3]) == 0.0f0

        @test all(isfinite, attention)

        unequal_scores = Float32[0,1,2]

        unequal_attention = HyperGraphNeuralNetworks.masked_incidence_softmax(
            unequal_scores,
            incidence_matrix,
        )

        @test unequal_attention[2,1] > unequal_attention[1,1]
        @test unequal_attention[3,2] > unequal_attention[2,2]

        @test isapprox(sum(unequal_attention[:,1]),1.0f0)
        @test isapprox(sum(unequal_attention[:,2]),1.0f0)

        weighted_incidence = Float32[
            2 0
            0 4
            3 5
        ]

        weighted_attention = HyperGraphNeuralNetworks.masked_incidence_softmax(
            Float32[0,0,0],
            weighted_incidence,
        )

        @test isapprox(
            weighted_attention,
            Float32[
                0.5 0.0
                0.0 0.5
                0.5 0.5
            ]
        )

        @test_throws DimensionMismatch HyperGraphNeuralNetworks.masked_incidence_softmax(
            Float32[1,2],
            incidence_matrix,
        )
    end

    @testset "Safe row normalisation" begin

        matrix = Float32[
            1 1
            0 0
            1 3
        ]

        normalised = HyperGraphNeuralNetworks._safe_attention_row_normalise(matrix)

        @test isapprox(
            normalised,
            Float32[
                0.5 0.5
                0.0 0.0
                0.25 0.75
            ]
        )

        @test isapprox(sum(normalised[1,:]),1.0f0)
        @test sum(normalised[2,:]) == 0.0f0
        @test isapprox(sum(normalised[3,:]),1.0f0)

        @test all(isfinite, normalised)
    end
        @testset "Forward pass without hyperedge features" begin
        rng = Random.Xoshiro(2026)

        layer = DirectedAttentionLayer(3, 0, 4)
        ps, st = Lux.setup(rng, layer)

        X_vertex = Float32[
            1 0 2
            0 1 1
            2 1 0
            1 1 1
        ]

        source_matrix = Float32[
            1 0
            1 1
            0 1
            0 0
        ]

        target_matrix = Float32[
            0 1
            0 0
            1 0
            0 0
        ]

        output, st_out = layer(
            (X_vertex, source_matrix, target_matrix),
            ps,
            st,
        )

        @test haskey(output, :updated_vertices)
        @test haskey(output, :updated_hyperedges)
        @test !haskey(output, :source_attention)
        @test !haskey(output, :target_attention)

        @test size(output.updated_vertices) == (4, 4)
        @test size(output.updated_hyperedges) == (2, 4)

        @test eltype(output.updated_vertices) <: AbstractFloat
        @test eltype(output.updated_hyperedges) <: AbstractFloat

        @test all(isfinite, output.updated_vertices)
        @test all(isfinite, output.updated_hyperedges)

        @test st_out == st

        X_hyperedge = similar(X_vertex, 2, 0)

        explicit_output, explicit_st = layer(
            (X_vertex, X_hyperedge, source_matrix, target_matrix),
            ps,
            st,
        )

        @test isapprox(
            explicit_output.updated_vertices,
            output.updated_vertices,
        )

        @test isapprox(
            explicit_output.updated_hyperedges,
            output.updated_hyperedges,
        )

        @test explicit_st == st
    end

    @testset "Forward pass with hyperedge features" begin
        rng = Random.Xoshiro(77)

        layer = DirectedAttentionLayer(3, 2, 5)
        ps, st = Lux.setup(rng, layer)

        X_vertex = Float32[
            1 0 2
            0 1 1
            2 1 0
        ]

        X_hyperedge = Float32[
            1 0
            0 1
        ]

        source_matrix = Float32[
            1 0
            1 1
            0 1
        ]

        target_matrix = Float32[
            0 1
            0 0
            1 0
        ]

        output, st_out = layer(
            (
                X_vertex,
                X_hyperedge,
                source_matrix,
                target_matrix,
            ),
            ps,
            st,
        )

        @test size(output.updated_vertices) == (3, 5)
        @test size(output.updated_hyperedges) == (2, 5)

        @test all(isfinite, output.updated_vertices)
        @test all(isfinite, output.updated_hyperedges)

        @test st_out == st
    end

    @testset "Returned attention properties" begin
        rng = Random.Xoshiro(9)

        layer = DirectedAttentionLayer(
            2,
            0,
            3;
            return_attention = true,
        )

        ps, st = Lux.setup(rng, layer)

        X_vertex = Float32[
            1 0
            0 1
            1 1
            2 1
        ]

        source_matrix = Float32[
            1 0 0
            1 1 0
            0 1 0
            0 0 0
        ]

        target_matrix = Float32[
            0 1 0
            0 0 0
            1 0 0
            0 0 0
        ]

        output, _ = layer(
            (X_vertex, source_matrix, target_matrix),
            ps,
            st,
        )

        @test haskey(output, :source_attention)
        @test haskey(output, :target_attention)

        @test size(output.source_attention) == size(source_matrix)
        @test size(output.target_attention) == size(target_matrix)

        @test all(output.source_attention[source_matrix .== 0] .== 0)
        @test all(output.target_attention[target_matrix .== 0] .== 0)

        @test isapprox(sum(output.source_attention[:, 1]), 1.0f0)
        @test isapprox(sum(output.source_attention[:, 2]), 1.0f0)
        @test sum(output.source_attention[:, 3]) == 0.0f0

        @test isapprox(sum(output.target_attention[:, 1]), 1.0f0)
        @test isapprox(sum(output.target_attention[:, 2]), 1.0f0)
        @test sum(output.target_attention[:, 3]) == 0.0f0

        @test all(isfinite, output.source_attention)
        @test all(isfinite, output.target_attention)
    end
        @testset "Deterministic zero-parameter behaviour" begin
        layer = DirectedAttentionLayer(
            2,
            0,
            3;
            return_attention = true,
        )

        ps, st = Lux.setup(Random.Xoshiro(1), layer)

        zero_ps = map(x -> zero.(x), ps)

        X_vertex = Float32[
            1 2
            3 4
            5 6
        ]

        source_matrix = reshape(
            Float32[
                1,
                1,
                0,
            ],
            3,
            1,
        )

        target_matrix = reshape(
            Float32[
                0,
                0,
                1,
            ],
            3,
            1,
        )

        output, _ = layer(
            (X_vertex, source_matrix, target_matrix),
            zero_ps,
            st,
        )

        @test output.updated_vertices == zeros(Float32, 3, 3)
        @test output.updated_hyperedges == zeros(Float32, 1, 3)

        @test isapprox(
            output.source_attention,
            reshape(Float32[0.5, 0.5, 0.0], 3, 1),
        )

        @test isapprox(
            output.target_attention,
            reshape(Float32[0.0, 0.0, 1.0], 3, 1),
        )
    end

    @testset "Input validation" begin
        layer_without_hyperedge_features =
            DirectedAttentionLayer(3, 0, 4)

        ps0, st0 = Lux.setup(
            Random.Xoshiro(4),
            layer_without_hyperedge_features,
        )

        X_vertex = ones(Float32, 3, 3)

        source_matrix = Float32[
            1 0
            1 1
            0 1
        ]

        target_matrix = Float32[
            0 1
            0 0
            1 0
        ]

        @test_throws DimensionMismatch layer_without_hyperedge_features(
            (
                ones(Float32, 4, 3),
                source_matrix,
                target_matrix,
            ),
            ps0,
            st0,
        )

        @test_throws DimensionMismatch layer_without_hyperedge_features(
            (
                ones(Float32, 3, 2),
                source_matrix,
                target_matrix,
            ),
            ps0,
            st0,
        )

        @test_throws DimensionMismatch layer_without_hyperedge_features(
            (
                X_vertex,
                source_matrix,
                ones(Float32, 3, 3),
            ),
            ps0,
            st0,
        )

        @test_throws DimensionMismatch layer_without_hyperedge_features(
            (
                X_vertex,
                ones(Float32, 4, 2),
                ones(Float32, 4, 2),
            ),
            ps0,
            st0,
        )

        @test_throws DimensionMismatch layer_without_hyperedge_features(
            (
                X_vertex,
                ones(Float32, 3, 1),
                source_matrix,
                target_matrix,
            ),
            ps0,
            st0,
        )

        @test_throws ArgumentError layer_without_hyperedge_features(
            (
                X_vertex,
                source_matrix,
            ),
            ps0,
            st0,
        )

        @test_throws ArgumentError layer_without_hyperedge_features(
            (
                X_vertex,
                zeros(Float32, 2, 0),
                source_matrix,
                target_matrix,
                :extra,
            ),
            ps0,
            st0,
        )

        layer_with_hyperedge_features =
            DirectedAttentionLayer(3, 2, 4)

        ps2, st2 = Lux.setup(
            Random.Xoshiro(5),
            layer_with_hyperedge_features,
        )

        X_hyperedge = ones(Float32, 2, 2)

        @test_throws ArgumentError layer_with_hyperedge_features(
            (
                X_vertex,
                source_matrix,
                target_matrix,
            ),
            ps2,
            st2,
        )

        @test_throws DimensionMismatch layer_with_hyperedge_features(
            (
                X_vertex,
                ones(Float32, 3, 2),
                source_matrix,
                target_matrix,
            ),
            ps2,
            st2,
        )

        @test_throws DimensionMismatch layer_with_hyperedge_features(
            (
                X_vertex,
                ones(Float32, 2, 1),
                source_matrix,
                target_matrix,
            ),
            ps2,
            st2,
        )

        valid_output, valid_state = layer_with_hyperedge_features(
            (
                X_vertex,
                X_hyperedge,
                source_matrix,
                target_matrix,
            ),
            ps2,
            st2,
        )

        @test size(valid_output.updated_vertices) == (3, 4)
        @test size(valid_output.updated_hyperedges) == (2, 4)
        @test valid_state == st2
    end
end

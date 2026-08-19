using Test
using Random
using Lux
using HyperGraphNeuralNetworks

const HGNN = HyperGraphNeuralNetworks

# Tests for `DirectedHypergraphAttentionLayer`.
#
# This test suite covers both the original single-head behaviour and the
# configurable multi-head extension.
#
# The tests verify:
#
# - constructor arguments and validation, including `num_heads`;
# - Lux parameter and state initialisation;
# - parameter shapes and parameter counts for single- and multi-head layers;
# - masked incidence softmax behaviour;
# - safe row normalisation used during hyperedge-to-vertex propagation;
# - forward passes with and without initial hyperedge features;
# - preservation of the public `hidden_dim` output shape for multiple heads;
# - returned source-side and target-side attention coefficients;
# - independent attention normalisation for every head;
# - deterministic behaviour when all learnable parameters are zero;
# - independent parameter initialisation across different attention heads; and
# - input-shape and tuple validation for both single- and multi-head execution.
#
# For `num_heads == 1`, the tests also check that the original single-head
# parameter layout and output behaviour are preserved.
#
# For `num_heads > 1`, each attention head is expected to have independent
# vertex-projection, source-attention, target-attention, and hyperedge-update
# parameters. Per-head vertex and hyperedge representations are concatenated
# internally and projected back to `hidden_dim`, so the public output shape does
# not depend on the number of heads.

@testset "DirectedHypergraphAttentionLayer" begin

    # Verify constructor defaults, custom options, and invalid dimensions.
    @testset "Constructor validation" begin
        layer = HGNN.DirectedHypergraphAttentionLayer(3, 0, 4)

        @test layer isa HGNN.DirectedHypergraphAttentionLayer
        @test layer.vertex_in_dim == 3
        @test layer.hyperedge_in_dim == 0
        @test layer.hidden_dim == 4
        @test layer.num_heads == 1
        @test layer.return_attention == false

        attention_layer = HGNN.DirectedHypergraphAttentionLayer(
            3,
            2,
            4;
            num_heads = 3,
            activation = identity,
            attention_activation = identity,
            return_attention = true,
        )

        @test attention_layer.activation === identity
        @test attention_layer.attention_activation === identity
        @test attention_layer.num_heads == 3
        @test attention_layer.return_attention

        @test_throws ArgumentError HGNN.DirectedHypergraphAttentionLayer(0, 0, 4)
        @test_throws ArgumentError HGNN.DirectedHypergraphAttentionLayer(-1, 0, 4)
        @test_throws ArgumentError HGNN.DirectedHypergraphAttentionLayer(3, -1, 4)
        @test_throws ArgumentError HGNN.DirectedHypergraphAttentionLayer(3, 0, 0)
        @test_throws ArgumentError HGNN.DirectedHypergraphAttentionLayer(3, 0, -1)
        @test_throws ArgumentError HGNN.DirectedHypergraphAttentionLayer(
            3,
            0,
            4;
            num_heads = 0,
        )
        @test_throws ArgumentError HGNN.DirectedHypergraphAttentionLayer(
            3,
            0,
            4;
            num_heads = -2,
        )
    end


    # Verify that `num_heads = 1` preserves the original parameter layout.
    @testset "Single-head parameter and state initialisation" begin
        rng = Random.Xoshiro(1234)

        layer = HGNN.DirectedHypergraphAttentionLayer(3, 2, 4)
        ps, st = Lux.setup(rng, layer)

        # num_heads = 1 preserves the original parameter layout.
        @test layer.num_heads == 1
        @test size(ps.W_vertex) == (3, 4)
        @test size(ps.b_vertex) == (1, 4)
        @test size(ps.a_source) == (4, 1)
        @test size(ps.a_target) == (4, 1)
        @test size(ps.W_hyperedge) == (10, 4)
        @test size(ps.b_hyperedge) == (1, 4)
        @test size(ps.W_vertex_update) == (8, 4)
        @test size(ps.b_vertex_update) == (1, 4)

        @test !haskey(ps, :W_head_vertex)
        @test !haskey(ps, :W_head_hyperedge)

        @test st == NamedTuple()
        @test Lux.statelength(layer) == 0
        @test Lux.parameterlength(layer) == 104
        @test Lux.parameterlength(layer) == Lux.parameterlength(ps)

        layer2 = HGNN.DirectedHypergraphAttentionLayer(3, 0, 4)
        ps2, _ = Lux.setup(Random.Xoshiro(1234), layer2)

        @test size(ps2.W_hyperedge) == (8, 4)
        @test Lux.parameterlength(layer2) == 96
        @test Lux.parameterlength(layer2) == Lux.parameterlength(ps2)
    end


    # Verify per-head parameters and the projections used after concatenation.
    @testset "Multi-head parameter and state initialisation" begin
        rng = Random.Xoshiro(4321)

        layer = HGNN.DirectedHypergraphAttentionLayer(
            3,
            2,
            4;
            num_heads = 3,
        )

        ps, st = Lux.setup(rng, layer)

        # Each head has its own projection, source/target attention vectors,
        # and hyperedge-update parameters.
        @test size(ps.W_vertex) == (3, 4, 3)
        @test size(ps.b_vertex) == (1, 4, 3)
        @test size(ps.a_source) == (4, 3)
        @test size(ps.a_target) == (4, 3)
        @test size(ps.W_hyperedge) == (10, 4, 3)
        @test size(ps.b_hyperedge) == (1, 4, 3)

        # Three 4-dimensional heads concatenate to 12 features before
        # projection back to hidden_dim = 4.
        @test size(ps.W_head_vertex) == (12, 4)
        @test size(ps.b_head_vertex) == (1, 4)
        @test size(ps.W_head_hyperedge) == (12, 4)
        @test size(ps.b_head_hyperedge) == (1, 4)

        @test size(ps.W_vertex_update) == (8, 4)
        @test size(ps.b_vertex_update) == (1, 4)

        @test st == NamedTuple()
        @test Lux.statelength(layer) == 0

        # 3 heads:
        # per head = (3*4+4) + (2*4) + (10*4+4) = 68
        # all heads = 204
        # two 12->4 projections = 104
        # final vertex update = 36
        # total = 344
        @test Lux.parameterlength(layer) == 344
        @test Lux.parameterlength(layer) == Lux.parameterlength(ps)

        layer_without_hyperedge_features =
            HGNN.DirectedHypergraphAttentionLayer(
                3,
                0,
                4;
                num_heads = 2,
            )

        ps2, _ = Lux.setup(
            Random.Xoshiro(4321),
            layer_without_hyperedge_features,
        )

        @test size(ps2.W_hyperedge) == (8, 4, 2)
        @test size(ps2.W_head_vertex) == (8, 4)
        @test size(ps2.W_head_hyperedge) == (8, 4)
        @test Lux.parameterlength(layer_without_hyperedge_features) == 228
        @test Lux.parameterlength(layer_without_hyperedge_features) ==
              Lux.parameterlength(ps2)
    end


    # Verify incidence masking, per-hyperedge softmax normalisation, and empty columns.
    @testset "masked_incidence_softmax" begin
        raw_scores = Float32[0, 0, 0]

        incidence_matrix = Float32[
            1 0 0
            1 1 0
            0 1 0
        ]

        attention = HGNN.masked_incidence_softmax(
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

        @test isapprox(sum(attention[:, 1]), 1.0f0)
        @test isapprox(sum(attention[:, 2]), 1.0f0)
        @test sum(attention[:, 3]) == 0.0f0
        @test all(isfinite, attention)

        unequal_scores = Float32[0, 1, 2]

        unequal_attention = HGNN.masked_incidence_softmax(
            unequal_scores,
            incidence_matrix,
        )

        @test unequal_attention[2, 1] > unequal_attention[1, 1]
        @test unequal_attention[3, 2] > unequal_attention[2, 2]

        @test isapprox(sum(unequal_attention[:, 1]), 1.0f0)
        @test isapprox(sum(unequal_attention[:, 2]), 1.0f0)

        weighted_incidence = Float32[
            2 0
            0 4
            3 5
        ]

        weighted_attention = HGNN.masked_incidence_softmax(
            Float32[0, 0, 0],
            weighted_incidence,
        )

        @test isapprox(
            weighted_attention,
            Float32[
                0.5 0.0
                0.0 0.5
                0.5 0.5
            ],
        )

        @test_throws DimensionMismatch HGNN.masked_incidence_softmax(
            Float32[1, 2],
            incidence_matrix,
        )
    end


    # Verify safe row normalisation, including rows with no memberships.
    @testset "Safe row normalisation" begin
        matrix = Float32[
            1 1
            0 0
            1 3
        ]

        normalised = HGNN._safe_attention_row_normalise(matrix)

        @test isapprox(
            normalised,
            Float32[
                0.5 0.5
                0.0 0.0
                0.25 0.75
            ],
        )

        @test isapprox(sum(normalised[1, :]), 1.0f0)
        @test sum(normalised[2, :]) == 0.0f0
        @test isapprox(sum(normalised[3, :]), 1.0f0)
        @test all(isfinite, normalised)
    end


    # Verify the original single-head forward path when no hyperedge features are supplied.
    @testset "Single-head forward pass without hyperedge features" begin
        rng = Random.Xoshiro(2026)

        layer = HGNN.DirectedHypergraphAttentionLayer(3, 0, 4)
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


    # Verify the single-head forward path when initial hyperedge features are supplied.
    @testset "Single-head forward pass with hyperedge features" begin
        rng = Random.Xoshiro(77)

        layer = HGNN.DirectedHypergraphAttentionLayer(3, 2, 5)
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


    # Verify multi-head concatenation/projection while keeping the public output dimension fixed.
    @testset "Multi-head forward pass without hyperedge features" begin
        rng = Random.Xoshiro(2027)

        layer = HGNN.DirectedHypergraphAttentionLayer(
            3,
            0,
            4;
            num_heads = 3,
        )

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

        # The public output dimension is independent of num_heads.
        @test size(output.updated_vertices) == (4, 4)
        @test size(output.updated_hyperedges) == (2, 4)

        @test !haskey(output, :source_attention)
        @test !haskey(output, :target_attention)

        @test all(isfinite, output.updated_vertices)
        @test all(isfinite, output.updated_hyperedges)
        @test st_out == st

        X_hyperedge = similar(X_vertex, 2, 0)

        explicit_output, explicit_st = layer(
            (
                X_vertex,
                X_hyperedge,
                source_matrix,
                target_matrix,
            ),
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


    # Verify multi-head execution with explicit initial hyperedge features.
    @testset "Multi-head forward pass with hyperedge features" begin
        rng = Random.Xoshiro(78)

        layer = HGNN.DirectedHypergraphAttentionLayer(
            3,
            2,
            5;
            num_heads = 4,
        )

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


    # Verify masking and per-hyperedge normalisation of returned single-head attention.
    @testset "Single-head returned attention properties" begin
        rng = Random.Xoshiro(9)

        layer = HGNN.DirectedHypergraphAttentionLayer(
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


    # Verify returned attention shape and independent normalisation for every attention head.
    @testset "Multi-head returned attention properties" begin
        rng = Random.Xoshiro(10)

        num_heads = 3

        layer = HGNN.DirectedHypergraphAttentionLayer(
            2,
            0,
            3;
            num_heads = num_heads,
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

        @test size(output.source_attention) == (4, 3, num_heads)
        @test size(output.target_attention) == (4, 3, num_heads)

        @test all(isfinite, output.source_attention)
        @test all(isfinite, output.target_attention)

        for head in 1:num_heads
            source_attention =
                output.source_attention[:, :, head]

            target_attention =
                output.target_attention[:, :, head]

            # Non-incidences must remain exactly zero in every head.
            @test all(
                source_attention[source_matrix .== 0] .== 0,
            )

            @test all(
                target_attention[target_matrix .== 0] .== 0,
            )

            # Attention is normalised independently for every head
            # and every non-empty hyperedge.
            @test isapprox(sum(source_attention[:, 1]), 1.0f0)
            @test isapprox(sum(source_attention[:, 2]), 1.0f0)
            @test sum(source_attention[:, 3]) == 0.0f0

            @test isapprox(sum(target_attention[:, 1]), 1.0f0)
            @test isapprox(sum(target_attention[:, 2]), 1.0f0)
            @test sum(target_attention[:, 3]) == 0.0f0
        end
    end


    # With zero parameters, embeddings remain zero and attention depends only on incidence membership.
    @testset "Deterministic zero-parameter single-head behaviour" begin
        layer = HGNN.DirectedHypergraphAttentionLayer(
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


    # The same deterministic zero-parameter behaviour should hold independently for every head.
    @testset "Deterministic zero-parameter multi-head behaviour" begin
        num_heads = 3

        layer = HGNN.DirectedHypergraphAttentionLayer(
            2,
            0,
            3;
            num_heads = num_heads,
            return_attention = true,
        )

        ps, st = Lux.setup(Random.Xoshiro(2), layer)
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

        @test size(output.source_attention) == (3, 1, num_heads)
        @test size(output.target_attention) == (3, 1, num_heads)

        expected_source =
            reshape(Float32[0.5, 0.5, 0.0], 3, 1)

        expected_target =
            reshape(Float32[0.0, 0.0, 1.0], 3, 1)

        for head in 1:num_heads
            @test isapprox(
                output.source_attention[:, :, head],
                expected_source,
            )

            @test isapprox(
                output.target_attention[:, :, head],
                expected_target,
            )
        end
    end


    # Verify that separate heads are initialised with independent learnable parameters.
    @testset "Different heads use independent parameters" begin
        layer = HGNN.DirectedHypergraphAttentionLayer(
            2,
            0,
            3;
            num_heads = 3,
        )

        ps, _ = Lux.setup(Random.Xoshiro(42), layer)

        @test !isapprox(
            ps.W_vertex[:, :, 1],
            ps.W_vertex[:, :, 2],
        )

        @test !isapprox(
            ps.a_source[:, 1],
            ps.a_source[:, 2],
        )

        @test !isapprox(
            ps.a_target[:, 1],
            ps.a_target[:, 2],
        )
    end


    # Verify tuple lengths, feature dimensions, and incidence-matrix dimensions.
    @testset "Input validation" begin
        layer_without_hyperedge_features =
            HGNN.DirectedHypergraphAttentionLayer(3, 0, 4)

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
            HGNN.DirectedHypergraphAttentionLayer(3, 2, 4)

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


    # Verify that multi-head execution enforces the same public input contract.
    @testset "Multi-head input validation" begin
        layer = HGNN.DirectedHypergraphAttentionLayer(
            3,
            2,
            4;
            num_heads = 3,
        )

        ps, st = Lux.setup(Random.Xoshiro(6), layer)

        X_vertex = ones(Float32, 3, 3)
        X_hyperedge = ones(Float32, 2, 2)

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

        @test_throws ArgumentError layer(
            (
                X_vertex,
                source_matrix,
                target_matrix,
            ),
            ps,
            st,
        )

        @test_throws DimensionMismatch layer(
            (
                ones(Float32, 3, 2),
                X_hyperedge,
                source_matrix,
                target_matrix,
            ),
            ps,
            st,
        )

        @test_throws DimensionMismatch layer(
            (
                X_vertex,
                ones(Float32, 3, 2),
                source_matrix,
                target_matrix,
            ),
            ps,
            st,
        )

        @test_throws DimensionMismatch layer(
            (
                X_vertex,
                X_hyperedge,
                source_matrix,
                ones(Float32, 3, 3),
            ),
            ps,
            st,
        )
    end
end



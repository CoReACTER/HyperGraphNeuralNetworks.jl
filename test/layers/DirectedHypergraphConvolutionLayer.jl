using Test
using Lux
using Random
using HyperGraphNeuralNetworks


const CONV_X_VERTEX = Float32[
    1.0 0.0 2.0;
    0.0 1.0 1.0;
    1.0 1.0 0.0;
    2.0 0.0 1.0
]


const CONV_SOURCE_MATRIX = Float32[
    1 0 1;
    1 0 0;
    0 1 0;
    0 0 0
]


const CONV_TARGET_MATRIX = Float32[
    0 0 0;
    0 1 0;
    1 0 0;
    0 0 1
]


@testset "DirectedHypergraphConvolutionLayer" begin

    @testset "Constructor validation" begin
        @test_throws ArgumentError begin
            DirectedHypergraphConvolutionLayer(
                0,
                8,
            )
        end

        @test_throws ArgumentError begin
            DirectedHypergraphConvolutionLayer(
                -1,
                8,
            )
        end

        @test_throws ArgumentError begin
            DirectedHypergraphConvolutionLayer(
                3,
                0,
            )
        end

        @test_throws ArgumentError begin
            DirectedHypergraphConvolutionLayer(
                3,
                -1,
            )
        end
    end


    @testset "Parameter and state initialisation" begin
        rng = MersenneTwister(123)

        layer =
            DirectedHypergraphConvolutionLayer(
                3,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        @test size(ps.W_self) == (3, 8)

        @test size(
            ps.W_source_to_target,
        ) == (3, 8)

        @test size(
            ps.W_target_to_source,
        ) == (3, 8)

        @test size(ps.b_vertex) == (1, 8)

        @test isempty(st)

        @test Lux.statelength(layer) == 0

        @test Lux.parameterlength(layer) ==
              3 * 3 * 8 + 8
    end


    @testset "Basic forward pass" begin
        rng = MersenneTwister(123)

        layer =
            DirectedHypergraphConvolutionLayer(
                3,
                8;
                activation = tanh,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        output, new_st =
            layer(
                (
                    CONV_X_VERTEX,
                    CONV_SOURCE_MATRIX,
                    CONV_TARGET_MATRIX,
                ),
                ps,
                st,
            )

        @test size(
            output.updated_vertices,
        ) == (4, 8)

        @test size(
            output.updated_hyperedges,
        ) == (3, 8)

        @test all(
            isfinite,
            output.updated_vertices,
        )

        @test all(
            isfinite,
            output.updated_hyperedges,
        )

        @test new_st == st
    end


    @testset "Direction changes the result" begin
        rng = MersenneTwister(123)

        layer =
            DirectedHypergraphConvolutionLayer(
                3,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        original_output, _ =
            layer(
                (
                    CONV_X_VERTEX,
                    CONV_SOURCE_MATRIX,
                    CONV_TARGET_MATRIX,
                ),
                ps,
                st,
            )

        reversed_output, _ =
            layer(
                (
                    CONV_X_VERTEX,
                    CONV_TARGET_MATRIX,
                    CONV_SOURCE_MATRIX,
                ),
                ps,
                st,
            )

        @test !isapprox(
            original_output.updated_vertices,
            reversed_output.updated_vertices,
        )
    end


    @testset "Zero-degree vertices and hyperedges remain finite" begin
        rng = MersenneTwister(123)

        source_matrix = Float32[
            1 0 0;
            0 0 0;
            0 1 0;
            0 0 0
        ]

        target_matrix = Float32[
            0 0 0;
            1 0 0;
            0 0 0;
            0 1 0
        ]

        layer =
            DirectedHypergraphConvolutionLayer(
                3,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        output, _ =
            layer(
                (
                    CONV_X_VERTEX,
                    source_matrix,
                    target_matrix,
                ),
                ps,
                st,
            )

        @test all(
            isfinite,
            output.updated_vertices,
        )

        @test all(
            isfinite,
            output.updated_hyperedges,
        )
    end


    @testset "Nonzero incidence values represent membership" begin
        weighted_source =
            Float32[
                2 0 4;
                7 0 0;
                0 3 0;
                0 0 0
            ]

        weighted_target =
            Float32[
                0 0 0;
                0 6 0;
                5 0 0;
                0 0 9
            ]

        binary_source_membership =
            HyperGraphNeuralNetworks.
            _convolution_membership_matrix(
                CONV_SOURCE_MATRIX,
            )

        weighted_source_membership =
            HyperGraphNeuralNetworks.
            _convolution_membership_matrix(
                weighted_source,
            )

        binary_target_membership =
            HyperGraphNeuralNetworks.
            _convolution_membership_matrix(
                CONV_TARGET_MATRIX,
            )

        weighted_target_membership =
            HyperGraphNeuralNetworks.
            _convolution_membership_matrix(
                weighted_target,
            )

        @test binary_source_membership ==
              weighted_source_membership

        @test binary_target_membership ==
              weighted_target_membership
    end


    @testset "Safe inverse" begin
        values =
            Float32[
                0 1 2 4
            ]

        inverse_values =
            HyperGraphNeuralNetworks.
            _safe_inverse(
                values,
            )

        @test inverse_values ==
              Float32[
                  0 1 0.5 0.25
              ]

        @test all(
            isfinite,
            inverse_values,
        )
    end


    @testset "Directional normalisation" begin
        normalisation =
            HyperGraphNeuralNetworks.
            _normalised_directional_incidence(
                CONV_SOURCE_MATRIX,
            )

        @test size(
            normalisation.vertex_to_hyperedge,
        ) == size(CONV_SOURCE_MATRIX)

        @test size(
            normalisation.hyperedge_to_vertex,
        ) == size(CONV_SOURCE_MATRIX)

        @test all(
            isfinite,
            normalisation.vertex_to_hyperedge,
        )

        @test all(
            isfinite,
            normalisation.hyperedge_to_vertex,
        )

        column_sums =
            sum(
                normalisation.vertex_to_hyperedge;
                dims = 1,
            )

        for hyperedge in axes(
            CONV_SOURCE_MATRIX,
            2,
        )
            if any(
                .!iszero.(
                    CONV_SOURCE_MATRIX[:, hyperedge],
                ),
            )
                @test isapprox(
                    column_sums[1, hyperedge],
                    1.0f0;
                    atol = 1.0f-6,
                )
            end
        end
    end


    @testset "Incorrect incidence shapes" begin
        rng = MersenneTwister(123)

        layer =
            DirectedHypergraphConvolutionLayer(
                3,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        wrong_target =
            zeros(Float32, 4, 2)

        @test_throws DimensionMismatch begin
            layer(
                (
                    CONV_X_VERTEX,
                    CONV_SOURCE_MATRIX,
                    wrong_target,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect number of vertex rows" begin
        rng = MersenneTwister(123)

        layer =
            DirectedHypergraphConvolutionLayer(
                3,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        wrong_X_vertex =
            rand(Float32, 5, 3)

        @test_throws DimensionMismatch begin
            layer(
                (
                    wrong_X_vertex,
                    CONV_SOURCE_MATRIX,
                    CONV_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect vertex feature dimension" begin
        rng = MersenneTwister(123)

        layer =
            DirectedHypergraphConvolutionLayer(
                3,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        wrong_X_vertex =
            rand(Float32, 4, 2)

        @test_throws DimensionMismatch begin
            layer(
                (
                    wrong_X_vertex,
                    CONV_SOURCE_MATRIX,
                    CONV_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect tuple length" begin
        rng = MersenneTwister(123)

        layer =
            DirectedHypergraphConvolutionLayer(
                3,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        @test_throws ArgumentError begin
            layer(
                (
                    CONV_X_VERTEX,
                    CONV_SOURCE_MATRIX,
                ),
                ps,
                st,
            )
        end

        @test_throws ArgumentError begin
            layer(
                (
                    CONV_X_VERTEX,
                    CONV_SOURCE_MATRIX,
                    CONV_TARGET_MATRIX,
                    CONV_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Deterministic zero parameters" begin
        rng = MersenneTwister(123)

        layer =
            DirectedHypergraphConvolutionLayer(
                3,
                8;
                activation = tanh,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        zero_ps = (
            W_self =
                zero(ps.W_self),

            W_source_to_target =
                zero(ps.W_source_to_target),

            W_target_to_source =
                zero(ps.W_target_to_source),

            b_vertex =
                zero(ps.b_vertex),
        )

        output, _ =
            layer(
                (
                    CONV_X_VERTEX,
                    CONV_SOURCE_MATRIX,
                    CONV_TARGET_MATRIX,
                ),
                zero_ps,
                st,
            )

        @test output.updated_vertices ==
              zeros(Float32, 4, 8)

        @test output.updated_hyperedges ==
              zeros(Float32, 3, 8)
    end
end
using Test
using Lux
using Random
using HyperGraphNeuralNetworks


const RES_X_VERTEX = Float32[
    1.0 0.0 2.0;
    0.0 1.0 1.0;
    1.0 1.0 0.0;
    2.0 0.0 1.0
]


const RES_SOURCE_MATRIX = Float32[
    1 0 1;
    1 0 0;
    0 1 0;
    0 0 0
]


const RES_TARGET_MATRIX = Float32[
    0 0 0;
    0 1 0;
    1 0 0;
    0 0 1
]


@testset "ResidualDirectedHypergraphLayer" begin

    @testset "Constructor validation" begin
        @test_throws ArgumentError begin
            ResidualDirectedHypergraphLayer(
                0,
                0,
                8,
            )
        end

        @test_throws ArgumentError begin
            ResidualDirectedHypergraphLayer(
                3,
                -1,
                8,
            )
        end

        @test_throws ArgumentError begin
            ResidualDirectedHypergraphLayer(
                3,
                0,
                0,
            )
        end
    end


    @testset "Parameter and state initialisation with projection" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                2,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        @test size(ps.W_vertex) == (3, 8)
        @test size(ps.b_vertex) == (1, 8)

        @test size(ps.W_hyperedge) == (18, 8)
        @test size(ps.b_hyperedge) == (1, 8)

        @test size(ps.W_message_update) == (16, 8)
        @test size(ps.b_message_update) == (1, 8)

        @test size(ps.W_residual) == (3, 8)

        @test isempty(st)
        @test Lux.statelength(layer) == 0
        @test Lux.parameterlength(layer) == 344
    end


    @testset "Parameter initialisation without projection" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                8,
                0,
                8,
            )

        ps, _ =
            Lux.setup(
                rng,
                layer,
            )

        @test size(ps.W_residual) == (0, 0)

        @test Lux.parameterlength(layer) == 344
    end


    @testset "Basic forward pass with residual projection" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                0,
                8;
                activation = tanh,
                normalize = true,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        output, new_st =
            layer(
                (
                    RES_X_VERTEX,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
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


    @testset "Identity residual pathway" begin
        rng = MersenneTwister(123)

        X_vertex =
            rand(
                Float32,
                4,
                8,
            )

        layer =
            ResidualDirectedHypergraphLayer(
                8,
                0,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        residual =
            HyperGraphNeuralNetworks.
            _residual_representation(
                layer,
                X_vertex,
                ps,
            )

        @test residual == X_vertex
    end


    @testset "Projected residual pathway" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                0,
                8,
            )

        ps, _ =
            Lux.setup(
                rng,
                layer,
            )

        residual =
            HyperGraphNeuralNetworks.
            _residual_representation(
                layer,
                RES_X_VERTEX,
                ps,
            )

        @test size(residual) == (4, 8)

        @test residual ==
              RES_X_VERTEX * ps.W_residual
    end


    @testset "Hyperedge input features" begin
        rng = MersenneTwister(123)

        X_hyperedge = Float32[
            1.0 0.0;
            0.0 1.0;
            1.0 1.0
        ]

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                2,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        output, new_st =
            layer(
                (
                    RES_X_VERTEX,
                    X_hyperedge,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
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


    @testset "Direction affects hyperedge representation" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                0,
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
                    RES_X_VERTEX,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
                ),
                ps,
                st,
            )

        reversed_output, _ =
            layer(
                (
                    RES_X_VERTEX,
                    RES_TARGET_MATRIX,
                    RES_SOURCE_MATRIX,
                ),
                ps,
                st,
            )

        @test !isapprox(
            original_output.updated_hyperedges,
            reversed_output.updated_hyperedges,
        )

        @test !isapprox(
            original_output.updated_vertices,
            reversed_output.updated_vertices,
        )
    end


    @testset "Normalisation enabled and disabled" begin
        rng_normalised =
            MersenneTwister(123)

        rng_unnormalised =
            MersenneTwister(123)

        layer_normalised =
            ResidualDirectedHypergraphLayer(
                3,
                0,
                8;
                normalize = true,
            )

        layer_unnormalised =
            ResidualDirectedHypergraphLayer(
                3,
                0,
                8;
                normalize = false,
            )

        ps_normalised, st_normalised =
            Lux.setup(
                rng_normalised,
                layer_normalised,
            )

        ps_unnormalised, st_unnormalised =
            Lux.setup(
                rng_unnormalised,
                layer_unnormalised,
            )

        output_normalised, _ =
            layer_normalised(
                (
                    RES_X_VERTEX,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
                ),
                ps_normalised,
                st_normalised,
            )

        output_unnormalised, _ =
            layer_unnormalised(
                (
                    RES_X_VERTEX,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
                ),
                ps_unnormalised,
                st_unnormalised,
            )

        @test all(
            isfinite,
            output_normalised.updated_vertices,
        )

        @test all(
            isfinite,
            output_unnormalised.updated_vertices,
        )

        @test !isapprox(
            output_normalised.updated_vertices,
            output_unnormalised.updated_vertices,
        )
    end


    @testset "Zero incidence remains finite" begin
        rng = MersenneTwister(123)

        source_matrix =
            zeros(
                Float32,
                4,
                3,
            )

        target_matrix =
            zeros(
                Float32,
                4,
                3,
            )

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                0,
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
                    RES_X_VERTEX,
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


    @testset "Mismatched incidence matrices" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                0,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        wrong_target_matrix =
            zeros(
                Float32,
                4,
                2,
            )

        @test_throws DimensionMismatch begin
            layer(
                (
                    RES_X_VERTEX,
                    RES_SOURCE_MATRIX,
                    wrong_target_matrix,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect number of vertex rows" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                0,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        wrong_X_vertex =
            rand(
                Float32,
                5,
                3,
            )

        @test_throws DimensionMismatch begin
            layer(
                (
                    wrong_X_vertex,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect vertex feature dimension" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                0,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        wrong_X_vertex =
            rand(
                Float32,
                4,
                2,
            )

        @test_throws DimensionMismatch begin
            layer(
                (
                    wrong_X_vertex,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Missing required hyperedge features" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                2,
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
                    RES_X_VERTEX,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect hyperedge rows" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                2,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        wrong_X_hyperedge =
            rand(
                Float32,
                2,
                2,
            )

        @test_throws DimensionMismatch begin
            layer(
                (
                    RES_X_VERTEX,
                    wrong_X_hyperedge,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect hyperedge feature dimension" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                2,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        wrong_X_hyperedge =
            rand(
                Float32,
                3,
                3,
            )

        @test_throws DimensionMismatch begin
            layer(
                (
                    RES_X_VERTEX,
                    wrong_X_hyperedge,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect tuple length" begin
        rng = MersenneTwister(123)

        layer =
            ResidualDirectedHypergraphLayer(
                3,
                0,
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
                    RES_X_VERTEX,
                    RES_SOURCE_MATRIX,
                ),
                ps,
                st,
            )
        end

        @test_throws ArgumentError begin
            layer(
                (
                    RES_X_VERTEX,
                    RES_SOURCE_MATRIX,
                    RES_TARGET_MATRIX,
                    RES_TARGET_MATRIX,
                    RES_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end
end
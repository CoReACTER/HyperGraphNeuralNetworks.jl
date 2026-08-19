using Test
using Lux
using Random
using HyperGraphNeuralNetworks


const GATED_X_VERTEX = Float32[
    1.0 0.0 2.0;
    0.0 1.0 1.0;
    1.0 1.0 0.0;
    2.0 0.0 1.0
]


const GATED_SOURCE_MATRIX = Float32[
    1 0 1;
    1 0 0;
    0 1 0;
    0 0 0
]


const GATED_TARGET_MATRIX = Float32[
    0 0 0;
    0 1 0;
    1 0 0;
    0 0 1
]


@testset "GatedDirectedHypergraphLayer" begin

    @testset "Constructor validation" begin
        @test_throws ArgumentError begin
            GatedDirectedHypergraphLayer(
                0,
                0,
                8,
            )
        end

        @test_throws ArgumentError begin
            GatedDirectedHypergraphLayer(
                3,
                -1,
                8,
            )
        end

        @test_throws ArgumentError begin
            GatedDirectedHypergraphLayer(
                3,
                0,
                0,
            )
        end
    end


    @testset "Parameter and state initialisation" begin
        rng = MersenneTwister(123)

        layer =
            GatedDirectedHypergraphLayer(
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

        @test size(ps.W_message) == (8, 8)
        @test size(ps.b_message) == (1, 8)

        @test size(ps.W_gate) == (16, 8)
        @test size(ps.b_gate) == (1, 8)

        @test isempty(st)
        @test Lux.statelength(layer) == 0
        @test Lux.parameterlength(layer) == 392
    end


    @testset "Basic forward pass" begin
        rng = MersenneTwister(123)

        layer =
            GatedDirectedHypergraphLayer(
                3,
                0,
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
                    GATED_X_VERTEX,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps,
                st,
            )

        @test size(output.updated_vertices) == (4, 8)
        @test size(output.updated_hyperedges) == (3, 8)
        @test size(output.gates) == (4, 8)

        @test all(isfinite, output.updated_vertices)
        @test all(isfinite, output.updated_hyperedges)
        @test all(isfinite, output.gates)

        @test new_st == st
    end


    @testset "Gate values are probabilities" begin
        rng = MersenneTwister(123)

        layer =
            GatedDirectedHypergraphLayer(
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
                    GATED_X_VERTEX,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps,
                st,
            )

        @test all(output.gates .>= 0)
        @test all(output.gates .<= 1)

        @test any(output.gates .> 0)
        @test any(output.gates .< 1)
    end


    @testset "Hyperedge input features" begin
        rng = MersenneTwister(123)

        X_hyperedge = Float32[
            1.0 0.0;
            0.0 1.0;
            1.0 1.0
        ]

        layer =
            GatedDirectedHypergraphLayer(
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
                    GATED_X_VERTEX,
                    X_hyperedge,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps,
                st,
            )

        @test size(output.updated_vertices) == (4, 8)
        @test size(output.updated_hyperedges) == (3, 8)
        @test size(output.gates) == (4, 8)

        @test all(isfinite, output.updated_vertices)
        @test all(isfinite, output.updated_hyperedges)
        @test all(isfinite, output.gates)

        @test new_st == st
    end


    @testset "Direction affects the representation" begin
        rng = MersenneTwister(123)

        layer =
            GatedDirectedHypergraphLayer(
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
                    GATED_X_VERTEX,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps,
                st,
            )

        reversed_output, _ =
            layer(
                (
                    GATED_X_VERTEX,
                    GATED_TARGET_MATRIX,
                    GATED_SOURCE_MATRIX,
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

        normalised_layer =
            GatedDirectedHypergraphLayer(
                3,
                0,
                8;
                normalize = true,
            )

        unnormalised_layer =
            GatedDirectedHypergraphLayer(
                3,
                0,
                8;
                normalize = false,
            )

        ps_normalised,
        st_normalised =
            Lux.setup(
                rng_normalised,
                normalised_layer,
            )

        ps_unnormalised,
        st_unnormalised =
            Lux.setup(
                rng_unnormalised,
                unnormalised_layer,
            )

        normalised_output, _ =
            normalised_layer(
                (
                    GATED_X_VERTEX,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps_normalised,
                st_normalised,
            )

        unnormalised_output, _ =
            unnormalised_layer(
                (
                    GATED_X_VERTEX,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps_unnormalised,
                st_unnormalised,
            )

        @test all(
            isfinite,
            normalised_output.updated_vertices,
        )

        @test all(
            isfinite,
            unnormalised_output.updated_vertices,
        )

        @test !isapprox(
            normalised_output.updated_vertices,
            unnormalised_output.updated_vertices,
        )
    end


    @testset "Zero incidence remains finite" begin
        rng = MersenneTwister(123)

        source_matrix =
            zeros(Float32, 4, 3)

        target_matrix =
            zeros(Float32, 4, 3)

        layer =
            GatedDirectedHypergraphLayer(
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
                    GATED_X_VERTEX,
                    source_matrix,
                    target_matrix,
                ),
                ps,
                st,
            )

        @test all(isfinite, output.updated_vertices)
        @test all(isfinite, output.updated_hyperedges)
        @test all(isfinite, output.gates)
    end


    @testset "Mismatched incidence matrices" begin
        rng = MersenneTwister(123)

        layer =
            GatedDirectedHypergraphLayer(
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
            zeros(Float32, 4, 2)

        @test_throws DimensionMismatch begin
            layer(
                (
                    GATED_X_VERTEX,
                    GATED_SOURCE_MATRIX,
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
            GatedDirectedHypergraphLayer(
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
            rand(Float32, 5, 3)

        @test_throws DimensionMismatch begin
            layer(
                (
                    wrong_X_vertex,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect vertex feature dimension" begin
        rng = MersenneTwister(123)

        layer =
            GatedDirectedHypergraphLayer(
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
            rand(Float32, 4, 2)

        @test_throws DimensionMismatch begin
            layer(
                (
                    wrong_X_vertex,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Missing required hyperedge features" begin
        rng = MersenneTwister(123)

        layer =
            GatedDirectedHypergraphLayer(
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
                    GATED_X_VERTEX,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect number of hyperedge rows" begin
        rng = MersenneTwister(123)

        layer =
            GatedDirectedHypergraphLayer(
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
            rand(Float32, 2, 2)

        @test_throws DimensionMismatch begin
            layer(
                (
                    GATED_X_VERTEX,
                    wrong_X_hyperedge,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect hyperedge feature dimension" begin
        rng = MersenneTwister(123)

        layer =
            GatedDirectedHypergraphLayer(
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
            rand(Float32, 3, 3)

        @test_throws DimensionMismatch begin
            layer(
                (
                    GATED_X_VERTEX,
                    wrong_X_hyperedge,
                    GATED_SOURCE_MATRIX,
                    GATED_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect tuple length" begin
        rng = MersenneTwister(123)

        layer =
            GatedDirectedHypergraphLayer(
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
                    GATED_X_VERTEX,
                    GATED_SOURCE_MATRIX,
                ),
                ps,
                st,
            )
        end
    end
end
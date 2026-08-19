using Test
using Lux
using Random
using HyperGraphNeuralNetworks


const ASYM_X_VERTEX = Float32[
    1.0 0.0 2.0;
    0.0 1.0 1.0;
    1.0 1.0 0.0;
    2.0 0.0 1.0
]


const ASYM_SOURCE_MATRIX = Float32[
    1 0 1;
    1 0 0;
    0 1 0;
    0 0 0
]


const ASYM_TARGET_MATRIX = Float32[
    0 0 0;
    0 1 0;
    1 0 0;
    0 0 1
]


@testset "AsymmetricDirectedHypergraphLayer" begin

    @testset "Constructor validation" begin
        @test_throws ArgumentError begin
            AsymmetricDirectedHypergraphLayer(
                0,
                0,
                8,
            )
        end

        @test_throws ArgumentError begin
            AsymmetricDirectedHypergraphLayer(
                3,
                -1,
                8,
            )
        end

        @test_throws ArgumentError begin
            AsymmetricDirectedHypergraphLayer(
                3,
                0,
                0,
            )
        end
    end


    @testset "Parameter and state initialisation" begin
        rng = MersenneTwister(123)

        layer =
            AsymmetricDirectedHypergraphLayer(
                3,
                2,
                8,
            )

        ps, st =
            Lux.setup(
                rng,
                layer,
            )

        @test size(
            ps.W_source_vertex,
        ) == (3, 8)

        @test size(
            ps.W_target_vertex,
        ) == (3, 8)

        @test size(
            ps.b_source_vertex,
        ) == (1, 8)

        @test size(
            ps.b_target_vertex,
        ) == (1, 8)

        @test size(
            ps.W_hyperedge,
        ) == (18, 8)

        @test size(
            ps.b_hyperedge,
        ) == (1, 8)

        @test size(
            ps.W_hyperedge_to_source,
        ) == (8, 8)

        @test size(
            ps.W_hyperedge_to_target,
        ) == (8, 8)

        @test size(
            ps.W_self,
        ) == (3, 8)

        @test size(
            ps.W_vertex_update,
        ) == (24, 8)

        @test size(
            ps.b_vertex_update,
        ) == (1, 8)

        @test isempty(st)

        @test Lux.statelength(layer) == 0

        @test Lux.parameterlength(layer) == 568
    end


    @testset "Basic forward pass" begin
        rng = MersenneTwister(123)

        layer =
            AsymmetricDirectedHypergraphLayer(
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
                    ASYM_X_VERTEX,
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
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


    @testset "Hyperedge input features" begin
        rng = MersenneTwister(123)

        X_hyperedge = Float32[
            1.0 0.0;
            0.0 1.0;
            1.0 1.0
        ]

        layer =
            AsymmetricDirectedHypergraphLayer(
                3,
                2,
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
                    ASYM_X_VERTEX,
                    X_hyperedge,
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
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


    @testset "Normalisation enabled and disabled" begin
        rng_normalised =
            MersenneTwister(123)

        rng_unnormalised =
            MersenneTwister(123)

        layer_normalised =
            AsymmetricDirectedHypergraphLayer(
                3,
                0,
                8;
                normalize = true,
            )

        layer_unnormalised =
            AsymmetricDirectedHypergraphLayer(
                3,
                0,
                8;
                normalize = false,
            )

        ps_normalised,
        st_normalised =
            Lux.setup(
                rng_normalised,
                layer_normalised,
            )

        ps_unnormalised,
        st_unnormalised =
            Lux.setup(
                rng_unnormalised,
                layer_unnormalised,
            )

        output_normalised, _ =
            layer_normalised(
                (
                    ASYM_X_VERTEX,
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
                ),
                ps_normalised,
                st_normalised,
            )

        output_unnormalised, _ =
            layer_unnormalised(
                (
                    ASYM_X_VERTEX,
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
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


    @testset "Source and target transformations are independent" begin
        rng = MersenneTwister(123)

        layer =
            AsymmetricDirectedHypergraphLayer(
                3,
                0,
                8,
            )

        ps, _ =
            Lux.setup(
                rng,
                layer,
            )

        @test ps.W_source_vertex !==
              ps.W_target_vertex

        @test ps.W_hyperedge_to_source !==
              ps.W_hyperedge_to_target

        @test size(ps.W_source_vertex) ==
              size(ps.W_target_vertex)

        @test size(ps.W_hyperedge_to_source) ==
              size(ps.W_hyperedge_to_target)
    end


    @testset "Direction affects the output" begin
        rng = MersenneTwister(123)

        layer =
            AsymmetricDirectedHypergraphLayer(
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
                    ASYM_X_VERTEX,
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
                ),
                ps,
                st,
            )

        reversed_output, _ =
            layer(
                (
                    ASYM_X_VERTEX,
                    ASYM_TARGET_MATRIX,
                    ASYM_SOURCE_MATRIX,
                ),
                ps,
                st,
            )

        @test !isapprox(
            original_output.updated_vertices,
            reversed_output.updated_vertices,
        )

        @test !isapprox(
            original_output.updated_hyperedges,
            reversed_output.updated_hyperedges,
        )
    end


    @testset "Zero incidence remains finite" begin
        rng = MersenneTwister(123)

        source_matrix =
            zeros(Float32, 4, 3)

        target_matrix =
            zeros(Float32, 4, 3)

        layer =
            AsymmetricDirectedHypergraphLayer(
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
                    ASYM_X_VERTEX,
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
            AsymmetricDirectedHypergraphLayer(
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
                    ASYM_X_VERTEX,
                    ASYM_SOURCE_MATRIX,
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
            AsymmetricDirectedHypergraphLayer(
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
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect vertex feature dimension" begin
        rng = MersenneTwister(123)

        layer =
            AsymmetricDirectedHypergraphLayer(
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
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Missing required hyperedge features" begin
        rng = MersenneTwister(123)

        layer =
            AsymmetricDirectedHypergraphLayer(
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
                    ASYM_X_VERTEX,
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect number of hyperedge rows" begin
        rng = MersenneTwister(123)

        layer =
            AsymmetricDirectedHypergraphLayer(
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
                    ASYM_X_VERTEX,
                    wrong_X_hyperedge,
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect hyperedge feature dimension" begin
        rng = MersenneTwister(123)

        layer =
            AsymmetricDirectedHypergraphLayer(
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
                    ASYM_X_VERTEX,
                    wrong_X_hyperedge,
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect tuple length" begin
        rng = MersenneTwister(123)

        layer =
            AsymmetricDirectedHypergraphLayer(
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
                    ASYM_X_VERTEX,
                    ASYM_SOURCE_MATRIX,
                ),
                ps,
                st,
            )
        end

        @test_throws ArgumentError begin
            layer(
                (
                    ASYM_X_VERTEX,
                    ASYM_SOURCE_MATRIX,
                    ASYM_TARGET_MATRIX,
                    ASYM_TARGET_MATRIX,
                    ASYM_TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end
end
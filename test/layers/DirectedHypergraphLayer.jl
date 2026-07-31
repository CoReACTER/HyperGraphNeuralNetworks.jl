using Test
using Lux
using Random
using Enzyme
using HyperGraphNeuralNetworks


const X_VERTEX = Float32[
    1.0 0.0 2.0;
    0.0 1.0 1.0;
    1.0 1.0 0.0;
    2.0 0.0 1.0
]

const SOURCE_MATRIX = Float32[
    1 0 1;
    1 0 0;
    0 1 0;
    0 0 0
]

const TARGET_MATRIX = Float32[
    0 0 0;
    0 1 0;
    1 0 0;
    0 0 1
]


@testset "DirectedHypergraphLayer" begin

    @testset "Constructor validation" begin
        @test_throws ArgumentError DirectedHypergraphLayer(
            0,
            0,
            8,
        )

        @test_throws ArgumentError DirectedHypergraphLayer(
            3,
            -1,
            8,
        )

        @test_throws ArgumentError DirectedHypergraphLayer(
            3,
            0,
            0,
        )
    end


    @testset "Basic forward pass" begin
        rng = Random.default_rng()

        layer = DirectedHypergraphLayer(
            3,
            0,
            8;
            activation = tanh,
            normalize = true,
        )

        ps, st = Lux.setup(rng, layer)

        output, new_st = layer(
            (
                X_VERTEX,
                SOURCE_MATRIX,
                TARGET_MATRIX,
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


    @testset "Parameter and state initialisation" begin
        rng = Random.default_rng()

        layer = DirectedHypergraphLayer(
            3,
            2,
            8,
        )

        ps, st = Lux.setup(rng, layer)

        @test size(ps.W_vertex) == (3, 8)
        @test size(ps.b_vertex) == (1, 8)

        @test size(ps.W_hyperedge) == (18, 8)
        @test size(ps.b_hyperedge) == (1, 8)

        @test size(ps.W_vertex_update) == (16, 8)
        @test size(ps.b_vertex_update) == (1, 8)

        @test isempty(st)
        @test Lux.statelength(layer) == 0
        @test Lux.parameterlength(layer) == 320
    end


    @testset "Normalisation enabled and disabled" begin
        rng = Random.default_rng()

        layer_normalised = DirectedHypergraphLayer(
            3,
            0,
            8;
            activation = tanh,
            normalize = true,
        )

        layer_unnormalised = DirectedHypergraphLayer(
            3,
            0,
            8;
            activation = tanh,
            normalize = false,
        )

        ps_normalised, st_normalised =
            Lux.setup(rng, layer_normalised)

        ps_unnormalised, st_unnormalised =
            Lux.setup(rng, layer_unnormalised)

        output_normalised, _ = layer_normalised(
            (
                X_VERTEX,
                SOURCE_MATRIX,
                TARGET_MATRIX,
            ),
            ps_normalised,
            st_normalised,
        )

        output_unnormalised, _ = layer_unnormalised(
            (
                X_VERTEX,
                SOURCE_MATRIX,
                TARGET_MATRIX,
            ),
            ps_unnormalised,
            st_unnormalised,
        )

        @test size(output_normalised.updated_vertices) == (4, 8)
        @test size(output_unnormalised.updated_vertices) == (4, 8)

        @test size(output_normalised.updated_hyperedges) == (3, 8)
        @test size(output_unnormalised.updated_hyperedges) == (3, 8)

        @test all(isfinite, output_normalised.updated_vertices)
        @test all(isfinite, output_unnormalised.updated_vertices)

        @test all(isfinite, output_normalised.updated_hyperedges)
        @test all(isfinite, output_unnormalised.updated_hyperedges)
    end


    @testset "Hyperedge input features" begin
        rng = Random.default_rng()

        X_hyperedge = Float32[
            1.0 0.0;
            0.0 1.0;
            1.0 1.0
        ]

        layer = DirectedHypergraphLayer(
            3,
            2,
            8;
            activation = tanh,
            normalize = true,
        )

        ps, st = Lux.setup(rng, layer)

        output, new_st = layer(
            (
                X_VERTEX,
                X_hyperedge,
                SOURCE_MATRIX,
                TARGET_MATRIX,
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


    @testset "Zero-sum normalisation" begin
        M = Float32[
            0 1 0;
            0 2 0;
            0 3 0
        ]

        column_normalised =
            HyperGraphNeuralNetworks.safe_normalise(
                M;
                dims = 1,
            )

        row_normalised =
            HyperGraphNeuralNetworks.safe_normalise(
                M;
                dims = 2,
            )

        @test all(isfinite, column_normalised)
        @test all(isfinite, row_normalised)

        @test column_normalised[:, 1] ==
              zeros(Float32, 3)

        @test column_normalised[:, 3] ==
              zeros(Float32, 3)

        @test row_normalised[1, :] ==
              Float32[0, 1, 0]

        @test row_normalised[2, :] ==
              Float32[0, 1, 0]

        @test row_normalised[3, :] ==
              Float32[0, 1, 0]
    end


    @testset "Invalid normalisation dimension" begin
        M = ones(Float32, 3, 3)

        @test_throws ArgumentError begin
            HyperGraphNeuralNetworks.safe_normalise(
                M;
                dims = 3,
            )
        end
    end


    @testset "Mismatched incidence matrices" begin
        rng = Random.default_rng()

        layer = DirectedHypergraphLayer(
            3,
            0,
            8,
        )

        ps, st = Lux.setup(rng, layer)

        wrong_target_matrix =
            zeros(Float32, 4, 2)

        @test_throws DimensionMismatch begin
            layer(
                (
                    X_VERTEX,
                    SOURCE_MATRIX,
                    wrong_target_matrix,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect number of vertex rows" begin
        rng = Random.default_rng()

        layer = DirectedHypergraphLayer(
            3,
            0,
            8,
        )

        ps, st = Lux.setup(rng, layer)

        wrong_X_vertex =
            rand(Float32, 5, 3)

        @test_throws DimensionMismatch begin
            layer(
                (
                    wrong_X_vertex,
                    SOURCE_MATRIX,
                    TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect vertex feature dimension" begin
        rng = Random.default_rng()

        layer = DirectedHypergraphLayer(
            3,
            0,
            8,
        )

        ps, st = Lux.setup(rng, layer)

        wrong_X_vertex =
            rand(Float32, 4, 2)

        @test_throws DimensionMismatch begin
            layer(
                (
                    wrong_X_vertex,
                    SOURCE_MATRIX,
                    TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Missing required hyperedge features" begin
        rng = Random.default_rng()

        layer = DirectedHypergraphLayer(
            3,
            2,
            8,
        )

        ps, st = Lux.setup(rng, layer)

        @test_throws ArgumentError begin
            layer(
                (
                    X_VERTEX,
                    SOURCE_MATRIX,
                    TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect number of hyperedge rows" begin
        rng = Random.default_rng()

        layer = DirectedHypergraphLayer(
            3,
            2,
            8,
        )

        ps, st = Lux.setup(rng, layer)

        wrong_X_hyperedge =
            rand(Float32, 2, 2)

        @test_throws DimensionMismatch begin
            layer(
                (
                    X_VERTEX,
                    wrong_X_hyperedge,
                    SOURCE_MATRIX,
                    TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end


    @testset "Incorrect hyperedge feature dimension" begin
        rng = Random.default_rng()

        layer = DirectedHypergraphLayer(
            3,
            2,
            8,
        )

        ps, st = Lux.setup(rng, layer)

        wrong_X_hyperedge =
            rand(Float32, 3, 3)

        @test_throws DimensionMismatch begin
            layer(
                (
                    X_VERTEX,
                    wrong_X_hyperedge,
                    SOURCE_MATRIX,
                    TARGET_MATRIX,
                ),
                ps,
                st,
            )
        end
    end
end
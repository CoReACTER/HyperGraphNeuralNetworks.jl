using Lux
using Random

"""
    AsymmetricDirectedHypergraphLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim;
        activation = tanh,
        normalize = true,
        init_weight = Lux.glorot_uniform,
        init_bias = Lux.zeros32,
    )

A Lux-compatible asymmetric message-passing layer for directed hypergraphs.

The layer uses separate learnable transformations for source-side and
target-side vertex information. It also uses separate transformations when
propagating updated hyperedge information back to source and target vertices.

This allows the two roles in a directed hyperedge to be modelled differently
throughout the complete message-passing step.

# Input

When `hyperedge_in_dim == 0`:

    (X_vertex, source_matrix, target_matrix)

When `hyperedge_in_dim > 0`:

    (X_vertex, X_hyperedge, source_matrix, target_matrix)

Expected shapes:

- `X_vertex`: `number_of_vertices × vertex_in_dim`
- `X_hyperedge`: `number_of_hyperedges × hyperedge_in_dim`
- `source_matrix`: `number_of_vertices × number_of_hyperedges`
- `target_matrix`: `number_of_vertices × number_of_hyperedges`

# Output

A named tuple containing:

- `updated_vertices`
- `updated_hyperedges`
"""
struct AsymmetricDirectedHypergraphLayer{F, IW, IB} <:
       Lux.AbstractLuxLayer
    vertex_in_dim::Int
    hyperedge_in_dim::Int
    hidden_dim::Int
    activation::F
    normalize::Bool
    init_weight::IW
    init_bias::IB
end


function AsymmetricDirectedHypergraphLayer(
    vertex_in_dim::Int,
    hyperedge_in_dim::Int,
    hidden_dim::Int;
    activation = tanh,
    normalize::Bool = true,
    init_weight = Lux.glorot_uniform,
    init_bias = Lux.zeros32,
)
    vertex_in_dim > 0 ||
        throw(
            ArgumentError(
                "`vertex_in_dim` must be positive.",
            ),
        )

    hyperedge_in_dim >= 0 ||
        throw(
            ArgumentError(
                "`hyperedge_in_dim` cannot be negative.",
            ),
        )

    hidden_dim > 0 ||
        throw(
            ArgumentError(
                "`hidden_dim` must be positive.",
            ),
        )

    return AsymmetricDirectedHypergraphLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim,
        activation,
        normalize,
        init_weight,
        init_bias,
    )
end


function _asymmetric_initialise_weight(
    initializer,
    rng::AbstractRNG,
    input_dimension::Int,
    output_dimension::Int,
)
    return permutedims(
        initializer(
            rng,
            output_dimension,
            input_dimension,
        ),
    )
end


function _asymmetric_initialise_bias(
    initializer,
    rng::AbstractRNG,
    output_dimension::Int,
)
    return permutedims(
        initializer(
            rng,
            output_dimension,
            1,
        ),
    )
end


function Lux.initialparameters(
    rng::AbstractRNG,
    layer::AsymmetricDirectedHypergraphLayer,
)
    hyperedge_update_in_dim =
        2 * layer.hidden_dim +
        layer.hyperedge_in_dim

    return (
        W_source_vertex = _asymmetric_initialise_weight(
            layer.init_weight,
            rng,
            layer.vertex_in_dim,
            layer.hidden_dim,
        ),

        W_target_vertex = _asymmetric_initialise_weight(
            layer.init_weight,
            rng,
            layer.vertex_in_dim,
            layer.hidden_dim,
        ),

        b_source_vertex = _asymmetric_initialise_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),

        b_target_vertex = _asymmetric_initialise_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),

        W_hyperedge = _asymmetric_initialise_weight(
            layer.init_weight,
            rng,
            hyperedge_update_in_dim,
            layer.hidden_dim,
        ),

        b_hyperedge = _asymmetric_initialise_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),

        W_hyperedge_to_source = _asymmetric_initialise_weight(
            layer.init_weight,
            rng,
            layer.hidden_dim,
            layer.hidden_dim,
        ),

        W_hyperedge_to_target = _asymmetric_initialise_weight(
            layer.init_weight,
            rng,
            layer.hidden_dim,
            layer.hidden_dim,
        ),

        W_self = _asymmetric_initialise_weight(
            layer.init_weight,
            rng,
            layer.vertex_in_dim,
            layer.hidden_dim,
        ),

        W_vertex_update = _asymmetric_initialise_weight(
            layer.init_weight,
            rng,
            3 * layer.hidden_dim,
            layer.hidden_dim,
        ),

        b_vertex_update = _asymmetric_initialise_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),
    )
end


function Lux.parameterlength(
    layer::AsymmetricDirectedHypergraphLayer,
)
    hyperedge_update_in_dim =
        2 * layer.hidden_dim +
        layer.hyperedge_in_dim

    source_vertex_parameters =
        layer.vertex_in_dim * layer.hidden_dim +
        layer.hidden_dim

    target_vertex_parameters =
        layer.vertex_in_dim * layer.hidden_dim +
        layer.hidden_dim

    hyperedge_parameters =
        hyperedge_update_in_dim * layer.hidden_dim +
        layer.hidden_dim

    return_message_parameters =
        2 * layer.hidden_dim * layer.hidden_dim

    self_parameters =
        layer.vertex_in_dim * layer.hidden_dim

    vertex_update_parameters =
        3 * layer.hidden_dim * layer.hidden_dim +
        layer.hidden_dim

    return (
        source_vertex_parameters +
        target_vertex_parameters +
        hyperedge_parameters +
        return_message_parameters +
        self_parameters +
        vertex_update_parameters
    )
end


Lux.initialstates(
    ::AbstractRNG,
    ::AsymmetricDirectedHypergraphLayer,
) = NamedTuple()


Lux.statelength(
    ::AsymmetricDirectedHypergraphLayer,
) = 0


function _asymmetric_unpack_input(
    layer::AsymmetricDirectedHypergraphLayer,
    input::Tuple{Any, Any, Any},
)
    layer.hyperedge_in_dim == 0 ||
        throw(
            ArgumentError(
                "Hyperedge features are required because " *
                "`hyperedge_in_dim` is $(layer.hyperedge_in_dim).",
            ),
        )

    X_vertex,
    source_matrix,
    target_matrix = input

    X_hyperedge = similar(
        X_vertex,
        size(source_matrix, 2),
        0,
    )

    return (
        X_vertex,
        X_hyperedge,
        source_matrix,
        target_matrix,
    )
end


function _asymmetric_unpack_input(
    ::AsymmetricDirectedHypergraphLayer,
    input::Tuple{Any, Any, Any, Any},
)
    return input
end


function _asymmetric_unpack_input(
    ::AsymmetricDirectedHypergraphLayer,
    input::Tuple,
)
    throw(
        ArgumentError(
            "The layer expects either " *
            "(X_vertex, source_matrix, target_matrix) or " *
            "(X_vertex, X_hyperedge, source_matrix, target_matrix).",
        ),
    )
end


function _validate_asymmetric_inputs(
    layer::AsymmetricDirectedHypergraphLayer,
    X_vertex::AbstractMatrix,
    X_hyperedge::AbstractMatrix,
    source_matrix::AbstractMatrix,
    target_matrix::AbstractMatrix,
)
    size(source_matrix) == size(target_matrix) ||
        throw(
            DimensionMismatch(
                "`source_matrix` and `target_matrix` must have " *
                "the same shape.",
            ),
        )

    number_of_vertices,
    number_of_hyperedges =
        size(source_matrix)

    size(X_vertex, 1) == number_of_vertices ||
        throw(
            DimensionMismatch(
                "The number of rows in `X_vertex` must match " *
                "the number of vertices in the incidence matrices.",
            ),
        )

    size(X_vertex, 2) == layer.vertex_in_dim ||
        throw(
            DimensionMismatch(
                "`X_vertex` has $(size(X_vertex, 2)) features, " *
                "but the layer expects $(layer.vertex_in_dim).",
            ),
        )

    size(X_hyperedge, 1) == number_of_hyperedges ||
        throw(
            DimensionMismatch(
                "The number of rows in `X_hyperedge` must match " *
                "the number of hyperedges in the incidence matrices.",
            ),
        )

    size(X_hyperedge, 2) == layer.hyperedge_in_dim ||
        throw(
            DimensionMismatch(
                "`X_hyperedge` has $(size(X_hyperedge, 2)) features, " *
                "but the layer expects $(layer.hyperedge_in_dim).",
            ),
        )

    return nothing
end


function _asymmetric_normalise(
    M::AbstractMatrix;
    dims::Int,
)
    dims in (1, 2) ||
        throw(
            ArgumentError(
                "`dims` must be either 1 or 2.",
            ),
        )

    matrix_sums =
        sum(
            M;
            dims = dims,
        )

    safe_sums =
        ifelse.(
            iszero.(matrix_sums),
            one(eltype(matrix_sums)),
            matrix_sums,
        )

    return M ./ safe_sums
end


function (layer::AsymmetricDirectedHypergraphLayer)(
    input,
    ps,
    st,
)
    (
        X_vertex,
        X_hyperedge,
        source_matrix,
        target_matrix,
    ) = _asymmetric_unpack_input(
        layer,
        input,
    )

    _validate_asymmetric_inputs(
        layer,
        X_vertex,
        X_hyperedge,
        source_matrix,
        target_matrix,
    )

    if layer.normalize
        source_to_hyperedge =
            _asymmetric_normalise(
                source_matrix;
                dims = 1,
            )

        target_to_hyperedge =
            _asymmetric_normalise(
                target_matrix;
                dims = 1,
            )

        hyperedge_to_source =
            _asymmetric_normalise(
                source_matrix;
                dims = 2,
            )

        hyperedge_to_target =
            _asymmetric_normalise(
                target_matrix;
                dims = 2,
            )
    else
        source_to_hyperedge =
            source_matrix

        target_to_hyperedge =
            target_matrix

        hyperedge_to_source =
            source_matrix

        hyperedge_to_target =
            target_matrix
    end

    # Source and target roles receive independent transformations.
    H_source =
        layer.activation.(
            X_vertex *
            ps.W_source_vertex .+
            ps.b_source_vertex
        )

    H_target =
        layer.activation.(
            X_vertex *
            ps.W_target_vertex .+
            ps.b_target_vertex
        )

    # Aggregate the two roles separately into each hyperedge.
    source_messages =
        transpose(
            source_to_hyperedge,
        ) * H_source

    target_messages =
        transpose(
            target_to_hyperedge,
        ) * H_target

    hyperedge_update_input =
        hcat(
            source_messages,
            target_messages,
            X_hyperedge,
        )

    updated_hyperedges =
        layer.activation.(
            hyperedge_update_input *
            ps.W_hyperedge .+
            ps.b_hyperedge
        )

    # Apply independent transformations to messages returning to each role.
    source_return_representation =
        updated_hyperedges *
        ps.W_hyperedge_to_source

    target_return_representation =
        updated_hyperedges *
        ps.W_hyperedge_to_target

    source_vertex_messages =
        hyperedge_to_source *
        source_return_representation

    target_vertex_messages =
        hyperedge_to_target *
        target_return_representation

    self_representation =
        X_vertex *
        ps.W_self

    vertex_update_input =
        hcat(
            self_representation,
            source_vertex_messages,
            target_vertex_messages,
        )

    updated_vertices =
        layer.activation.(
            vertex_update_input *
            ps.W_vertex_update .+
            ps.b_vertex_update
        )

    output = (
        updated_vertices = updated_vertices,
        updated_hyperedges = updated_hyperedges,
    )

    return output, st
end
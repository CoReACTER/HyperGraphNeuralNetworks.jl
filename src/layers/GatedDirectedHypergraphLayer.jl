using Lux
using Random

"""
    GatedDirectedHypergraphLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim;
        activation = tanh,
        normalize = true,
        init_weight = Lux.glorot_uniform,
        init_bias = Lux.zeros32,
    )

A Lux-compatible gated message-passing layer for directed hypergraphs.

The layer first constructs hyperedge representations from source-side and
target-side vertex information. Updated hyperedge representations are then
propagated back to participating vertices.

A learnable gate controls the balance between the information received
through hypergraph message passing and the transformed representation of the
vertex itself. Each hidden feature has its own gate value between zero and
one.

# Arguments

- `vertex_in_dim`: Number of input features for each vertex.
- `hyperedge_in_dim`: Number of input features for each hyperedge. Use `0`
  when no initial hyperedge features are available.
- `hidden_dim`: Size of the updated vertex and hyperedge representations.
- `activation`: Element-wise activation function.
- `normalize`: Whether incidence matrices are normalised before aggregation.
- `init_weight`: Initialiser used for weight matrices.
- `init_bias`: Initialiser used for bias parameters.

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

- `updated_vertices`: `number_of_vertices × hidden_dim`
- `updated_hyperedges`: `number_of_hyperedges × hidden_dim`
- `gates`: `number_of_vertices × hidden_dim`
"""
struct GatedDirectedHypergraphLayer{F, IW, IB} <:
       Lux.AbstractLuxLayer
    vertex_in_dim::Int
    hyperedge_in_dim::Int
    hidden_dim::Int
    activation::F
    normalize::Bool
    init_weight::IW
    init_bias::IB
end


function GatedDirectedHypergraphLayer(
    vertex_in_dim::Int,
    hyperedge_in_dim::Int,
    hidden_dim::Int;
    activation = tanh,
    normalize::Bool = true,
    init_weight = Lux.glorot_uniform,
    init_bias = Lux.zeros32,
)
    vertex_in_dim > 0 ||
        throw(ArgumentError(
            "`vertex_in_dim` must be positive.",
        ))

    hyperedge_in_dim >= 0 ||
        throw(ArgumentError(
            "`hyperedge_in_dim` cannot be negative.",
        ))

    hidden_dim > 0 ||
        throw(ArgumentError(
            "`hidden_dim` must be positive.",
        ))

    return GatedDirectedHypergraphLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim,
        activation,
        normalize,
        init_weight,
        init_bias,
    )
end


function _gated_initialise_weight(
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


function _gated_initialise_bias(
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
    layer::GatedDirectedHypergraphLayer,
)
    hyperedge_update_in_dim =
        2 * layer.hidden_dim +
        layer.hyperedge_in_dim

    return (
        W_vertex = _gated_initialise_weight(
            layer.init_weight,
            rng,
            layer.vertex_in_dim,
            layer.hidden_dim,
        ),

        b_vertex = _gated_initialise_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),

        W_hyperedge = _gated_initialise_weight(
            layer.init_weight,
            rng,
            hyperedge_update_in_dim,
            layer.hidden_dim,
        ),

        b_hyperedge = _gated_initialise_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),

        W_message = _gated_initialise_weight(
            layer.init_weight,
            rng,
            layer.hidden_dim,
            layer.hidden_dim,
        ),

        b_message = _gated_initialise_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),

        W_gate = _gated_initialise_weight(
            layer.init_weight,
            rng,
            2 * layer.hidden_dim,
            layer.hidden_dim,
        ),

        b_gate = _gated_initialise_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),
    )
end


function Lux.parameterlength(
    layer::GatedDirectedHypergraphLayer,
)
    hyperedge_update_in_dim =
        2 * layer.hidden_dim +
        layer.hyperedge_in_dim

    vertex_parameters =
        layer.vertex_in_dim *
        layer.hidden_dim +
        layer.hidden_dim

    hyperedge_parameters =
        hyperedge_update_in_dim *
        layer.hidden_dim +
        layer.hidden_dim

    message_parameters =
        layer.hidden_dim *
        layer.hidden_dim +
        layer.hidden_dim

    gate_parameters =
        2 *
        layer.hidden_dim *
        layer.hidden_dim +
        layer.hidden_dim

    return (
        vertex_parameters +
        hyperedge_parameters +
        message_parameters +
        gate_parameters
    )
end


Lux.initialstates(
    ::AbstractRNG,
    ::GatedDirectedHypergraphLayer,
) = NamedTuple()


Lux.statelength(
    ::GatedDirectedHypergraphLayer,
) = 0


function _gated_unpack_input(
    layer::GatedDirectedHypergraphLayer,
    input::Tuple{Any, Any, Any},
)
    layer.hyperedge_in_dim == 0 ||
        throw(ArgumentError(
            "Hyperedge features are required because " *
            "`hyperedge_in_dim` is $(layer.hyperedge_in_dim).",
        ))

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


function _gated_unpack_input(
    ::GatedDirectedHypergraphLayer,
    input::Tuple{Any, Any, Any, Any},
)
    return input
end


function _gated_unpack_input(
    ::GatedDirectedHypergraphLayer,
    input::Tuple,
)
    throw(ArgumentError(
        "The layer expects either " *
        "(X_vertex, source_matrix, target_matrix) or " *
        "(X_vertex, X_hyperedge, source_matrix, target_matrix).",
    ))
end


function _validate_gated_inputs(
    layer::GatedDirectedHypergraphLayer,
    X_vertex::AbstractMatrix,
    X_hyperedge::AbstractMatrix,
    source_matrix::AbstractMatrix,
    target_matrix::AbstractMatrix,
)
    size(source_matrix) == size(target_matrix) ||
        throw(DimensionMismatch(
            "`source_matrix` and `target_matrix` must have " *
            "the same shape.",
        ))

    number_of_vertices,
    number_of_hyperedges = size(source_matrix)

    size(X_vertex, 1) == number_of_vertices ||
        throw(DimensionMismatch(
            "The number of rows in `X_vertex` must match " *
            "the number of vertices in the incidence matrices.",
        ))

    size(X_vertex, 2) == layer.vertex_in_dim ||
        throw(DimensionMismatch(
            "`X_vertex` has $(size(X_vertex, 2)) features, " *
            "but the layer expects $(layer.vertex_in_dim).",
        ))

    size(X_hyperedge, 1) == number_of_hyperedges ||
        throw(DimensionMismatch(
            "The number of rows in `X_hyperedge` must match " *
            "the number of hyperedges in the incidence matrices.",
        ))

    size(X_hyperedge, 2) == layer.hyperedge_in_dim ||
        throw(DimensionMismatch(
            "`X_hyperedge` has $(size(X_hyperedge, 2)) features, " *
            "but the layer expects $(layer.hyperedge_in_dim).",
        ))

    return nothing
end


function _gated_safe_normalise(
    M::AbstractMatrix;
    dims::Int,
)
    dims in (1, 2) ||
        throw(ArgumentError(
            "`dims` must be either 1 or 2.",
        ))

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


function (layer::GatedDirectedHypergraphLayer)(
    input,
    ps,
    st,
)
    (
        X_vertex,
        X_hyperedge,
        source_matrix,
        target_matrix,
    ) = _gated_unpack_input(
        layer,
        input,
    )

    _validate_gated_inputs(
        layer,
        X_vertex,
        X_hyperedge,
        source_matrix,
        target_matrix,
    )

    membership_matrix =
        source_matrix .+
        target_matrix

    if layer.normalize
        source_used =
            _gated_safe_normalise(
                source_matrix;
                dims = 1,
            )

        target_used =
            _gated_safe_normalise(
                target_matrix;
                dims = 1,
            )

        membership_used =
            _gated_safe_normalise(
                membership_matrix;
                dims = 2,
            )
    else
        source_used = source_matrix
        target_used = target_matrix
        membership_used = membership_matrix
    end

    # Transform the original vertex features.
    H_vertex =
        layer.activation.(
            X_vertex *
            ps.W_vertex .+
            ps.b_vertex
        )

    # Aggregate source-side and target-side information separately.
    source_messages =
        transpose(
            source_used,
        ) * H_vertex

    target_messages =
        transpose(
            target_used,
        ) * H_vertex

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

    # Propagate updated hyperedge representations back to vertices.
    aggregated_messages =
        membership_used *
        updated_hyperedges

    message_representation =
        layer.activation.(
            aggregated_messages *
            ps.W_message .+
            ps.b_message
        )

    # Learn a feature-wise gate from the self and message representations.
    gate_input =
        hcat(
            H_vertex,
            message_representation,
        )

    gates =
        sigmoid.(
            gate_input *
            ps.W_gate .+
            ps.b_gate
        )

    # Interpolate between existing vertex information and new messages.
    updated_vertices =
        gates .*
        message_representation .+
        (one(eltype(gates)) .- gates) .*
        H_vertex

    output = (
        updated_vertices = updated_vertices,
        updated_hyperedges = updated_hyperedges,
        gates = gates,
    )

    return output, st
end
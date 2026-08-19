using Lux
using Random

"""
    DirectedHypergraphConvolutionLayer(
        vertex_in_dim,
        hidden_dim;
        activation = tanh,
        init_weight = Lux.glorot_uniform,
        init_bias = Lux.zeros32,
    )

A Lux-compatible convolution layer for directed hypergraphs.

The layer preserves the distinction between source and target vertices by
performing propagation in both directions:

1. source vertices -> hyperedges -> target vertices;
2. target vertices -> hyperedges -> source vertices.

The two directional messages use separate learnable transformations and are
combined with a transformed representation of each vertex.

Any nonzero entry in an incidence matrix is treated as membership.

# Input

    (X_vertex, source_matrix, target_matrix)

Expected shapes:

- `X_vertex`: `number_of_vertices × vertex_in_dim`
- `source_matrix`: `number_of_vertices × number_of_hyperedges`
- `target_matrix`: `number_of_vertices × number_of_hyperedges`

# Output

The layer returns a named tuple containing:

- `updated_vertices`: `number_of_vertices × hidden_dim`
- `updated_hyperedges`: `number_of_hyperedges × hidden_dim`
"""
struct DirectedHypergraphConvolutionLayer{F, IW, IB} <:
       Lux.AbstractLuxLayer
    vertex_in_dim::Int
    hidden_dim::Int
    activation::F
    init_weight::IW
    init_bias::IB
end


function DirectedHypergraphConvolutionLayer(
    vertex_in_dim::Int,
    hidden_dim::Int;
    activation = tanh,
    init_weight = Lux.glorot_uniform,
    init_bias = Lux.zeros32,
)
    vertex_in_dim > 0 ||
        throw(
            ArgumentError(
                "`vertex_in_dim` must be positive.",
            ),
        )

    hidden_dim > 0 ||
        throw(
            ArgumentError(
                "`hidden_dim` must be positive.",
            ),
        )

    return DirectedHypergraphConvolutionLayer(
        vertex_in_dim,
        hidden_dim,
        activation,
        init_weight,
        init_bias,
    )
end


function _convolution_row_major_weight(
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


function _convolution_row_major_bias(
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
    layer::DirectedHypergraphConvolutionLayer,
)
    return (
        W_self = _convolution_row_major_weight(
            layer.init_weight,
            rng,
            layer.vertex_in_dim,
            layer.hidden_dim,
        ),

        W_source_to_target = _convolution_row_major_weight(
            layer.init_weight,
            rng,
            layer.vertex_in_dim,
            layer.hidden_dim,
        ),

        W_target_to_source = _convolution_row_major_weight(
            layer.init_weight,
            rng,
            layer.vertex_in_dim,
            layer.hidden_dim,
        ),

        b_vertex = _convolution_row_major_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),
    )
end


function Lux.parameterlength(
    layer::DirectedHypergraphConvolutionLayer,
)
    weight_parameters =
        3 * layer.vertex_in_dim * layer.hidden_dim

    bias_parameters =
        layer.hidden_dim

    return weight_parameters + bias_parameters
end


Lux.initialstates(
    ::AbstractRNG,
    ::DirectedHypergraphConvolutionLayer,
) = NamedTuple()


Lux.statelength(
    ::DirectedHypergraphConvolutionLayer,
) = 0


"""
    _convolution_membership_matrix(M)

Convert an incidence matrix into a binary membership matrix.

Any nonzero incidence value is treated as membership.
"""
function _convolution_membership_matrix(
    M::AbstractMatrix,
)
    return convert.(
        eltype(M),
        .!iszero.(M),
    )
end


"""
    _safe_inverse(values)

Compute element-wise inverse values while safely handling zeros.

A zero value is mapped to zero instead of producing Inf.
"""
function _safe_inverse(
    values::AbstractArray,
)
    result = similar(values)

    for index in eachindex(values)
        if iszero(values[index])
            result[index] = zero(eltype(values))
        else
            result[index] =
                one(eltype(values)) / values[index]
        end
    end

    return result
end


"""
    _normalised_directional_incidence(M)

Construct normalised vertex-to-hyperedge and hyperedge-to-vertex incidence
matrices for one side of a directed hypergraph.

Vertex-to-hyperedge propagation is normalised over the vertices belonging to
each hyperedge. Hyperedge-to-vertex propagation is normalised over the
hyperedges incident to each vertex.
"""
function _normalised_directional_incidence(
    M::AbstractMatrix,
)
    membership =
        _convolution_membership_matrix(M)

    hyperedge_degrees =
        sum(
            membership;
            dims = 1,
        )

    vertex_degrees =
        sum(
            membership;
            dims = 2,
        )

    inverse_hyperedge_degrees =
        _safe_inverse(
            hyperedge_degrees,
        )

    inverse_vertex_degrees =
        _safe_inverse(
            vertex_degrees,
        )

    vertex_to_hyperedge =
        membership .* inverse_hyperedge_degrees

    hyperedge_to_vertex =
        membership .* inverse_vertex_degrees

    return (
        vertex_to_hyperedge = vertex_to_hyperedge,
        hyperedge_to_vertex = hyperedge_to_vertex,
    )
end


function _validate_convolution_inputs(
    layer::DirectedHypergraphConvolutionLayer,
    X_vertex::AbstractMatrix,
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

    number_of_vertices =
        size(
            source_matrix,
            1,
        )

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

    return nothing
end


function _unpack_convolution_input(
    ::DirectedHypergraphConvolutionLayer,
    input::Tuple{Any, Any, Any},
)
    return input
end


function _unpack_convolution_input(
    ::DirectedHypergraphConvolutionLayer,
    input::Tuple,
)
    throw(
        ArgumentError(
            "The layer expects a 3-tuple " *
            "(X_vertex, source_matrix, target_matrix).",
        ),
    )
end


function (layer::DirectedHypergraphConvolutionLayer)(
    input::Tuple,
    ps,
    st,
)
    X_vertex, source_matrix, target_matrix =
        _unpack_convolution_input(
            layer,
            input,
        )

    _validate_convolution_inputs(
        layer,
        X_vertex,
        source_matrix,
        target_matrix,
    )

    source_normalisation =
        _normalised_directional_incidence(
            source_matrix,
        )

    target_normalisation =
        _normalised_directional_incidence(
            target_matrix,
        )

    # Aggregate source vertices into hyperedges.
    source_hyperedge_messages =
        transpose(
            source_normalisation.vertex_to_hyperedge,
        ) * X_vertex

    # Aggregate target vertices into hyperedges.
    target_hyperedge_messages =
        transpose(
            target_normalisation.vertex_to_hyperedge,
        ) * X_vertex

    # Transform the source-side hyperedge information.
    transformed_source_messages =
        source_hyperedge_messages *
        ps.W_source_to_target

    # Transform the target-side hyperedge information.
    transformed_target_messages =
        target_hyperedge_messages *
        ps.W_target_to_source

    # Combine both sides to obtain the hyperedge representation.
    updated_hyperedges =
        layer.activation.(
            transformed_source_messages .+
            transformed_target_messages
        )

    # Propagate source-side information to target vertices.
    source_to_target_messages =
        target_normalisation.hyperedge_to_vertex *
        transformed_source_messages

    # Propagate target-side information to source vertices.
    target_to_source_messages =
        source_normalisation.hyperedge_to_vertex *
        transformed_target_messages

    # Transform each vertex's own features.
    self_representation =
        X_vertex *
        ps.W_self

    # Combine the original vertex information with both directional messages.
    updated_vertices =
        layer.activation.(
            self_representation .+
            source_to_target_messages .+
            target_to_source_messages .+
            ps.b_vertex
        )

    output = (
        updated_vertices = updated_vertices,
        updated_hyperedges = updated_hyperedges,
    )

    return output, st
end
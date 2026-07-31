using Lux
using Random
using NNlib: leakyrelu

"""
    DirectedAttentionLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim;
        activation = tanh,
        attention_activation = leakyrelu,
        init_weight = Lux.glorot_uniform,
        init_bias = Lux.zeros32,
        return_attention = false,
    )

A single-head, Lux-compatible attention layer for directed hypergraphs.

The layer learns separate source-side and target-side attention coefficients
when aggregating vertex information into directed hyperedges. Updated
hyperedge representations are then propagated back to participating vertices.

# Arguments

- `vertex_in_dim`: Number of input features associated with each vertex.
- `hyperedge_in_dim`: Number of input features associated with each hyperedge.
  Set this to `0` if no initial hyperedge features are available.
- `hidden_dim`: Size of the learned vertex and hyperedge representations.
- `activation`: Activation applied to updated representations.
- `attention_activation`: Activation applied to raw attention scores.
- `init_weight`: Initialiser used for weight parameters.
- `init_bias`: Initialiser used for bias parameters.
- `return_attention`: Whether the output should include the learned source and
  target attention matrices.

# Input

When `hyperedge_in_dim == 0`, the layer accepts:

    (X_vertex, source_matrix, target_matrix)

When `hyperedge_in_dim > 0`, the layer accepts:

    (X_vertex, X_hyperedge, source_matrix, target_matrix)

Expected shapes:

- `X_vertex`: `number_of_vertices √ó vertex_in_dim`
- `X_hyperedge`: `number_of_hyperedges √ó hyperedge_in_dim`
- `source_matrix`: `number_of_vertices √ó number_of_hyperedges`
- `target_matrix`: `number_of_vertices √ó number_of_hyperedges`

# Output

The layer returns a named tuple containing:

- `updated_vertices`
- `updated_hyperedges`

When `return_attention = true`, it additionally returns:

- `source_attention`
- `target_attention`
"""
struct DirectedAttentionLayer{F, A, IW, IB} <:
       Lux.AbstractLuxLayer
    vertex_in_dim::Int
    hyperedge_in_dim::Int
    hidden_dim::Int
    activation::F
    attention_activation::A
    init_weight::IW
    init_bias::IB
    return_attention::Bool
end


function DirectedAttentionLayer(
    vertex_in_dim::Int,
    hyperedge_in_dim::Int,
    hidden_dim::Int;
    activation = tanh,
    attention_activation = leakyrelu,
    init_weight = Lux.glorot_uniform,
    init_bias = Lux.zeros32,
    return_attention::Bool = false,
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

    return DirectedAttentionLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim,
        activation,
        attention_activation,
        init_weight,
        init_bias,
        return_attention,
    )
end


function _attention_row_major_weight(
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


function _attention_row_major_bias(
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
    layer::DirectedAttentionLayer,
)
    hyperedge_update_in_dim =
        2 * layer.hidden_dim + layer.hyperedge_in_dim

    return (
        # Transform input vertex features.
        W_vertex = _attention_row_major_weight(
            layer.init_weight,
            rng,
            layer.vertex_in_dim,
            layer.hidden_dim,
        ),

        b_vertex = _attention_row_major_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),

        # Compute separate source-side and target-side attention scores.
        a_source = _attention_row_major_weight(
            layer.init_weight,
            rng,
            layer.hidden_dim,
            1,
        ),

        a_target = _attention_row_major_weight(
            layer.init_weight,
            rng,
            layer.hidden_dim,
            1,
        ),

        # Update hyperedges from source messages, target messages,
        # and optional hyperedge features.
        W_hyperedge = _attention_row_major_weight(
            layer.init_weight,
            rng,
            hyperedge_update_in_dim,
            layer.hidden_dim,
        ),

        b_hyperedge = _attention_row_major_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),

        # Update vertices using their existing hidden representation
        # and messages received from hyperedges.
        W_vertex_update = _attention_row_major_weight(
            layer.init_weight,
            rng,
            2 * layer.hidden_dim,
            layer.hidden_dim,
        ),

        b_vertex_update = _attention_row_major_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),
    )
end


function Lux.parameterlength(
    layer::DirectedAttentionLayer,
)
    hyperedge_update_in_dim =
        2 * layer.hidden_dim + layer.hyperedge_in_dim

    vertex_transform_parameters =
        layer.vertex_in_dim * layer.hidden_dim +
        layer.hidden_dim

    attention_parameters =
        2 * layer.hidden_dim

    hyperedge_update_parameters =
        hyperedge_update_in_dim * layer.hidden_dim +
        layer.hidden_dim

    vertex_update_parameters =
        2 * layer.hidden_dim * layer.hidden_dim +
        layer.hidden_dim

    return (
        vertex_transform_parameters +
        attention_parameters +
        hyperedge_update_parameters +
        vertex_update_parameters
    )
end


# The layer has no mutable non-trainable state.
Lux.initialstates(
    ::AbstractRNG,
    ::DirectedAttentionLayer,
) = NamedTuple()

Lux.statelength(::DirectedAttentionLayer) = 0


"""
    masked_incidence_softmax(raw_scores, incidence_matrix)

Convert one raw score per vertex into attention coefficients for each
vertex--hyperedge incidence.

For every hyperedge column:

- vertices outside the hyperedge receive attention weight zero;
- participating vertices receive softmax-normalised weights;
- the weights of participating vertices sum to one;
- an empty hyperedge column remains zero.
"""
function masked_incidence_softmax(
    raw_scores::AbstractVector,
    incidence_matrix::AbstractMatrix,
)
    number_of_vertices, number_of_hyperedges =
        size(incidence_matrix)

    length(raw_scores) == number_of_vertices ||
        throw(
            DimensionMismatch(
                "The number of raw attention scores must match " *
                "the number of vertices in the incidence matrix.",
            ),
        )

    score_matrix =
        reshape(raw_scores, number_of_vertices, 1) .+
        zeros(
            eltype(raw_scores),
            number_of_vertices,
            number_of_hyperedges,
        )

    membership_mask =
        .!iszero.(incidence_matrix)

    negative_infinity =
        convert(eltype(score_matrix), -Inf)

    masked_scores =
        ifelse.(
            membership_mask,
            score_matrix,
            negative_infinity,
        )

    column_has_members =
        any(membership_mask, dims = 1)

    column_maximum =
        maximum(masked_scores, dims = 1)

    safe_column_maximum =
        ifelse.(
            column_has_members,
            column_maximum,
            zero(eltype(column_maximum)),
        )

    exponentials =
        ifelse.(
            membership_mask,
            exp.(score_matrix .- safe_column_maximum),
            zero(eltype(score_matrix)),
        )

    denominators =
        sum(exponentials, dims = 1)

    safe_denominators =
        ifelse.(
            iszero.(denominators),
            one(eltype(denominators)),
            denominators,
        )

    return exponentials ./ safe_denominators
end


"""
    _safe_attention_row_normalise(M)

Safely normalise each row of `M`.

Rows with a sum of zero use a denominator of one, preventing division by zero
and leaving those rows unchanged.
"""
function _safe_attention_row_normalise(
    M::AbstractMatrix,
)
    row_sums =
        sum(M, dims = 2)

    safe_row_sums =
        ifelse.(
            iszero.(row_sums),
            one(eltype(row_sums)),
            row_sums,
        )

    return M ./ safe_row_sums
end


function _unpack_attention_input(
    layer::DirectedAttentionLayer,
    input::Tuple{Any, Any, Any},
)
    layer.hyperedge_in_dim == 0 ||
        throw(
            ArgumentError(
                "Hyperedge features are required because " *
                "`hyperedge_in_dim` is $(layer.hyperedge_in_dim).",
            ),
        )

    X_vertex, source_matrix, target_matrix =
        input

    number_of_hyperedges =
        size(source_matrix, 2)

    X_hyperedge =
        similar(
            X_vertex,
            number_of_hyperedges,
            0,
        )

    return (
        X_vertex,
        X_hyperedge,
        source_matrix,
        target_matrix,
    )
end


function _unpack_attention_input(
    ::DirectedAttentionLayer,
    input::Tuple{Any, Any, Any, Any},
)
    return input
end


function _unpack_attention_input(
    ::DirectedAttentionLayer,
    input::Tuple,
)
    throw(
        ArgumentError(
            "The layer expects either a 3-tuple " *
            "(X_vertex, source_matrix, target_matrix) or a 4-tuple " *
            "(X_vertex, X_hyperedge, source_matrix, target_matrix).",
        ),
    )
end


function _validate_attention_inputs(
    layer::DirectedAttentionLayer,
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

    number_of_vertices, number_of_hyperedges =
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


"""
    (layer::DirectedAttentionLayer)(input, ps, st)

Apply one single-head attention-based directed-hypergraph message-passing
step.

Source-side and target-side attention coefficients are calculated separately.
The updated hyperedge representations are then propagated back to the
participating vertices.
"""
function (layer::DirectedAttentionLayer)(input, ps, st)
    (
        X_vertex,
        X_hyperedge,
        source_matrix,
        target_matrix,
    ) = _unpack_attention_input(
        layer,
        input,
    )

    _validate_attention_inputs(
        layer,
        X_vertex,
        X_hyperedge,
        source_matrix,
        target_matrix,
    )

    # Transform the original vertex features.
    H_vertex =
        layer.activation.(
            X_vertex * ps.W_vertex .+
            ps.b_vertex
        )

    # Compute one source-side and one target-side score per vertex.
    raw_source_scores =
        vec(
            layer.attention_activation.(
                H_vertex * ps.a_source
            ),
        )

    raw_target_scores =
        vec(
            layer.attention_activation.(
                H_vertex * ps.a_target
            ),
        )

    # Normalise scores only across vertices participating in each hyperedge.
    source_attention =
        masked_incidence_softmax(
            raw_source_scores,
            source_matrix,
        )

    target_attention =
        masked_incidence_softmax(
            raw_target_scores,
            target_matrix,
        )

    # Attention-weighted vertex-to-hyperedge aggregation.
    source_messages =
        transpose(source_attention) *
        H_vertex

    target_messages =
        transpose(target_attention) *
        H_vertex

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

    # Propagate updated hyperedge information back to vertices.
    # A vertex participates in a hyperedge if it occurs on either side.
    # Using a logical union avoids double-counting a vertex that is present
    # in both the source and target incidence matrices.
    membership_matrix =
        convert.(
            promote_type(
                eltype(source_matrix),
                eltype(target_matrix),
            ),
            (.!iszero.(source_matrix)) .|
            (.!iszero.(target_matrix)),
        )

    membership_weights =
        _safe_attention_row_normalise(
            membership_matrix,
        )

    vertex_messages =
        membership_weights *
        updated_hyperedges

    vertex_update_input =
        hcat(
            H_vertex,
            vertex_messages,
        )

    updated_vertices =
        layer.activation.(
            vertex_update_input *
            ps.W_vertex_update .+
            ps.b_vertex_update
        )

    basic_output = (
        updated_vertices = updated_vertices,
        updated_hyperedges = updated_hyperedges,
    )

    output =
        if layer.return_attention
            merge(
                basic_output,
                (
                    source_attention = source_attention,
                    target_attention = target_attention,
                ),
            )
        else
            basic_output
        end

    return output, st
end

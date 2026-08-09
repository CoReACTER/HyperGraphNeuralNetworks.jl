using Lux
using Random
using NNlib: leakyrelu

"""
    DirectedHypergraphAttentionLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim;
        num_heads = 1,
        activation = tanh,
        attention_activation = leakyrelu,
        init_weight = Lux.glorot_uniform,
        init_bias = Lux.zeros32,
        return_attention = false,
    )

A Lux-compatible attention-based message-passing layer for directed hypergraphs.

The layer preserves the distinction between the source and target sides of each
directed hyperedge by learning separate source-side and target-side attention
coefficients.

For each attention head, the layer:

1. projects the input vertex features into a hidden representation;
2. computes separate source-side and target-side attention scores;
3. normalises those scores over vertices incident to each hyperedge;
4. forms attention-weighted source and target messages;
5. combines those messages with optional input hyperedge features to update
   hyperedge representations.

The updated hyperedge representations are then propagated back to participating
vertices and used to compute updated vertex representations.

When `num_heads == 1`, the original single-head computation is used.

When `num_heads > 1`, each head has independent vertex-projection,
source-attention, target-attention, and hyperedge-update parameters. The
per-head vertex representations are concatenated and projected back to
`hidden_dim`. The per-head hyperedge representations are also concatenated and
projected back to `hidden_dim`. The public output dimensions therefore remain
independent of the number of attention heads.

# Arguments

- `vertex_in_dim::Int`: Number of input features associated with each vertex.
  Must be positive.
- `hyperedge_in_dim::Int`: Number of input features associated with each
  hyperedge. Use `0` when no initial hyperedge features are supplied.
- `hidden_dim::Int`: Number of features in the returned vertex and hyperedge
  representations. Must be positive.
- `num_heads::Int = 1`: Number of independent attention heads. Must be positive.
  Setting `num_heads = 1` preserves the single-head behaviour.
- `activation = tanh`: Activation applied to hidden vertex, hyperedge, and
  updated vertex representations.
- `attention_activation = leakyrelu`: Activation applied to raw source-side and
  target-side attention scores before normalisation.
- `init_weight = Lux.glorot_uniform`: Initialiser used for learnable weight
  parameters.
- `init_bias = Lux.zeros32`: Initialiser used for learnable bias parameters.
- `return_attention::Bool = false`: If `true`, include the learned source-side
  and target-side attention coefficients in the output.

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

The source and target incidence matrices must have identical shapes. Any
nonzero entry indicates that a vertex is incident to the corresponding
hyperedge on that side.

# Attention

Source-side and target-side attention are computed independently.

Within each hyperedge, attention is normalised only over vertices that are
incident to the hyperedge on the relevant side. Vertices outside the hyperedge
receive attention weight zero. An empty source or target side therefore
produces an all-zero attention column.

For multi-head attention, this normalisation is performed independently for
every head.

# Hyperedge-to-vertex propagation

After hyperedge representations are updated, they are propagated back to
participating vertices.

A vertex is treated as participating in a hyperedge when it has a nonzero entry
in either the source or target incidence matrix. The resulting
vertex--hyperedge membership matrix is row-normalised before propagation.

# Output

The layer returns a named tuple containing:

- `updated_vertices`
- `updated_hyperedges`

with shapes:

- `updated_vertices`: `number_of_vertices √ó hidden_dim`
- `updated_hyperedges`: `number_of_hyperedges √ó hidden_dim`

These output shapes are unchanged when `num_heads` changes.

When `return_attention = true`, the output additionally contains:

- `source_attention`
- `target_attention`

For `num_heads == 1`, each attention output has shape:

    number_of_vertices √ó number_of_hyperedges

For `num_heads > 1`, each attention output has shape:

    number_of_vertices √ó number_of_hyperedges √ó num_heads

where the third dimension indexes the attention head.
"""

struct DirectedHypergraphAttentionLayer{F, A, IW, IB} <:
       Lux.AbstractLuxLayer
    vertex_in_dim::Int
    hyperedge_in_dim::Int
    hidden_dim::Int
    num_heads::Int
    activation::F
    attention_activation::A
    init_weight::IW
    init_bias::IB
    return_attention::Bool
end


function DirectedHypergraphAttentionLayer(
    vertex_in_dim::Int,
    hyperedge_in_dim::Int,
    hidden_dim::Int;
    num_heads::Int = 1,
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

    num_heads > 0 ||
        throw(
            ArgumentError(
                "`num_heads` must be positive.",
            ),
        )

    return DirectedHypergraphAttentionLayer(
        vertex_in_dim,
        hyperedge_in_dim,
        hidden_dim,
        num_heads,
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


function _attention_weight_stack(
    initializer,
    rng::AbstractRNG,
    input_dimension::Int,
    output_dimension::Int,
    num_heads::Int,
)
    weights =
        ntuple(
            _ ->
                _attention_row_major_weight(
                    initializer,
                    rng,
                    input_dimension,
                    output_dimension,
                ),
            num_heads,
        )

    return cat(weights...; dims = 3)
end


function _attention_bias_stack(
    initializer,
    rng::AbstractRNG,
    output_dimension::Int,
    num_heads::Int,
)
    biases =
        ntuple(
            _ ->
                _attention_row_major_bias(
                    initializer,
                    rng,
                    output_dimension,
                ),
            num_heads,
        )

    return cat(biases...; dims = 3)
end


function _attention_vector_stack(
    initializer,
    rng::AbstractRNG,
    hidden_dim::Int,
    num_heads::Int,
)
    vectors =
        ntuple(
            _ ->
                _attention_row_major_weight(
                    initializer,
                    rng,
                    hidden_dim,
                    1,
                ),
            num_heads,
        )

    return hcat(vectors...)
end


function Lux.initialparameters(
    rng::AbstractRNG,
    layer::DirectedHypergraphAttentionLayer,
)
    hyperedge_update_in_dim =
        2 * layer.hidden_dim + layer.hyperedge_in_dim

    if layer.num_heads == 1
        # Keep the original single-head parameter structure unchanged.
        return (
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

    concatenated_dim =
        layer.num_heads * layer.hidden_dim

    return (
        # Each head learns its own vertex projection.
        W_vertex = _attention_weight_stack(
            layer.init_weight,
            rng,
            layer.vertex_in_dim,
            layer.hidden_dim,
            layer.num_heads,
        ),

        b_vertex = _attention_bias_stack(
            layer.init_bias,
            rng,
            layer.hidden_dim,
            layer.num_heads,
        ),

        # Each head learns independent source and target attention vectors.
        a_source = _attention_vector_stack(
            layer.init_weight,
            rng,
            layer.hidden_dim,
            layer.num_heads,
        ),

        a_target = _attention_vector_stack(
            layer.init_weight,
            rng,
            layer.hidden_dim,
            layer.num_heads,
        ),

        # Each head independently updates hyperedge representations.
        W_hyperedge = _attention_weight_stack(
            layer.init_weight,
            rng,
            hyperedge_update_in_dim,
            layer.hidden_dim,
            layer.num_heads,
        ),

        b_hyperedge = _attention_bias_stack(
            layer.init_bias,
            rng,
            layer.hidden_dim,
            layer.num_heads,
        ),

        # Concatenated head representations are projected back to hidden_dim.
        W_head_vertex = _attention_row_major_weight(
            layer.init_weight,
            rng,
            concatenated_dim,
            layer.hidden_dim,
        ),

        b_head_vertex = _attention_row_major_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),

        W_head_hyperedge = _attention_row_major_weight(
            layer.init_weight,
            rng,
            concatenated_dim,
            layer.hidden_dim,
        ),

        b_head_hyperedge = _attention_row_major_bias(
            layer.init_bias,
            rng,
            layer.hidden_dim,
        ),

        # The final vertex update uses the projected multi-head vertex
        # representation and the propagated hyperedge message.
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
    layer::DirectedHypergraphAttentionLayer,
)
    hyperedge_update_in_dim =
        2 * layer.hidden_dim + layer.hyperedge_in_dim

    if layer.num_heads == 1
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

    per_head_vertex_transform_parameters =
        layer.vertex_in_dim * layer.hidden_dim +
        layer.hidden_dim

    per_head_attention_parameters =
        2 * layer.hidden_dim

    per_head_hyperedge_update_parameters =
        hyperedge_update_in_dim * layer.hidden_dim +
        layer.hidden_dim

    all_head_parameters =
        layer.num_heads * (
            per_head_vertex_transform_parameters +
            per_head_attention_parameters +
            per_head_hyperedge_update_parameters
        )

    concatenated_dim =
        layer.num_heads * layer.hidden_dim

    head_projection_parameters =
        2 * (
            concatenated_dim * layer.hidden_dim +
            layer.hidden_dim
        )

    vertex_update_parameters =
        2 * layer.hidden_dim * layer.hidden_dim +
        layer.hidden_dim

    return (
        all_head_parameters +
        head_projection_parameters +
        vertex_update_parameters
    )
end


# The layer has no mutable non-trainable state.
Lux.initialstates(
    ::AbstractRNG,
    ::DirectedHypergraphAttentionLayer,
) = NamedTuple()

Lux.statelength(::DirectedHypergraphAttentionLayer) = 0


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
    layer::DirectedHypergraphAttentionLayer,
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
    ::DirectedHypergraphAttentionLayer,
    input::Tuple{Any, Any, Any, Any},
)
    return input
end


function _unpack_attention_input(
    ::DirectedHypergraphAttentionLayer,
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
    layer::DirectedHypergraphAttentionLayer,
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


function _membership_weights(
    source_matrix::AbstractMatrix,
    target_matrix::AbstractMatrix,
)
    membership_matrix =
        convert.(
            promote_type(
                eltype(source_matrix),
                eltype(target_matrix),
            ),
            (.!iszero.(source_matrix)) .|
            (.!iszero.(target_matrix)),
        )

    return _safe_attention_row_normalise(
        membership_matrix,
    )
end


function _single_attention_head(
    layer::DirectedHypergraphAttentionLayer,
    X_vertex::AbstractMatrix,
    X_hyperedge::AbstractMatrix,
    source_matrix::AbstractMatrix,
    target_matrix::AbstractMatrix,
    W_vertex::AbstractMatrix,
    b_vertex::AbstractMatrix,
    a_source::AbstractMatrix,
    a_target::AbstractMatrix,
    W_hyperedge::AbstractMatrix,
    b_hyperedge::AbstractMatrix,
)
    H_vertex =
        layer.activation.(
            X_vertex * W_vertex .+
            b_vertex
        )

    raw_source_scores =
        vec(
            layer.attention_activation.(
                H_vertex * a_source
            ),
        )

    raw_target_scores =
        vec(
            layer.attention_activation.(
                H_vertex * a_target
            ),
        )

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
            W_hyperedge .+
            b_hyperedge
        )

    return (
        H_vertex = H_vertex,
        updated_hyperedges = updated_hyperedges,
        source_attention = source_attention,
        target_attention = target_attention,
    )
end


function _single_head_forward(
    layer::DirectedHypergraphAttentionLayer,
    X_vertex::AbstractMatrix,
    X_hyperedge::AbstractMatrix,
    source_matrix::AbstractMatrix,
    target_matrix::AbstractMatrix,
    ps,
    st,
)
    head_output =
        _single_attention_head(
            layer,
            X_vertex,
            X_hyperedge,
            source_matrix,
            target_matrix,
            ps.W_vertex,
            ps.b_vertex,
            ps.a_source,
            ps.a_target,
            ps.W_hyperedge,
            ps.b_hyperedge,
        )

    membership_weights =
        _membership_weights(
            source_matrix,
            target_matrix,
        )

    vertex_messages =
        membership_weights *
        head_output.updated_hyperedges

    vertex_update_input =
        hcat(
            head_output.H_vertex,
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
        updated_hyperedges = head_output.updated_hyperedges,
    )

    output =
        if layer.return_attention
            merge(
                basic_output,
                (
                    source_attention = head_output.source_attention,
                    target_attention = head_output.target_attention,
                ),
            )
        else
            basic_output
        end

    return output, st
end


function _multi_head_forward(
    layer::DirectedHypergraphAttentionLayer,
    X_vertex::AbstractMatrix,
    X_hyperedge::AbstractMatrix,
    source_matrix::AbstractMatrix,
    target_matrix::AbstractMatrix,
    ps,
    st,
)
    head_outputs =
        ntuple(
            head ->
                _single_attention_head(
                    layer,
                    X_vertex,
                    X_hyperedge,
                    source_matrix,
                    target_matrix,
                    ps.W_vertex[:, :, head],
                    ps.b_vertex[:, :, head],
                    ps.a_source[:, head:head],
                    ps.a_target[:, head:head],
                    ps.W_hyperedge[:, :, head],
                    ps.b_hyperedge[:, :, head],
                ),
            layer.num_heads,
        )

    concatenated_vertex_heads =
        hcat(
            map(
                head_output -> head_output.H_vertex,
                head_outputs,
            )...,
        )

    concatenated_hyperedge_heads =
        hcat(
            map(
                head_output -> head_output.updated_hyperedges,
                head_outputs,
            )...,
        )

    H_vertex =
        layer.activation.(
            concatenated_vertex_heads *
            ps.W_head_vertex .+
            ps.b_head_vertex
        )

    updated_hyperedges =
        layer.activation.(
            concatenated_hyperedge_heads *
            ps.W_head_hyperedge .+
            ps.b_head_hyperedge
        )

    membership_weights =
        _membership_weights(
            source_matrix,
            target_matrix,
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
            source_attentions =
                map(
                    head_output -> head_output.source_attention,
                    head_outputs,
                )

            target_attentions =
                map(
                    head_output -> head_output.target_attention,
                    head_outputs,
                )

            merge(
                basic_output,
                (
                    source_attention =
                        cat(
                            source_attentions...;
                            dims = 3,
                        ),
                    target_attention =
                        cat(
                            target_attentions...;
                            dims = 3,
                        ),
                ),
            )
        else
            basic_output
        end

    return output, st
end


"""
    (layer::DirectedHypergraphAttentionLayer)(input, ps, st)

Apply one attention-based directed-hypergraph message-passing step.

The input is unpacked and validated before applying either the single-head or
multi-head computation according to `layer.num_heads`.

For a single head, vertex features are projected once, separate source-side and
target-side attention coefficients are computed, and the resulting messages are
used to update hyperedge representations. Updated hyperedge information is then
propagated back to vertices.

For multiple heads, each head independently performs vertex projection,
source-side attention, target-side attention, and hyperedge updating. The
per-head vertex representations and per-head hyperedge representations are
concatenated separately and projected back to `layer.hidden_dim` before the
final hyperedge-to-vertex propagation and vertex update.

Returns `(output, st)`. The layer has no mutable non-trainable state, so the
returned state is unchanged.
"""

function (layer::DirectedHypergraphAttentionLayer)(input, ps, st)
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

    if layer.num_heads == 1
        return _single_head_forward(
            layer,
            X_vertex,
            X_hyperedge,
            source_matrix,
            target_matrix,
            ps,
            st,
        )
    end

    return _multi_head_forward(
        layer,
        X_vertex,
        X_hyperedge,
        source_matrix,
        target_matrix,
        ps,
        st,
    )
end

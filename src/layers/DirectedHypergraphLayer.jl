using Lux
using Random

"""
    safe_column_normalise(M)

Normalise each column of `M` by its column sum.

Columns with a sum of zero use a denominator of one, preventing division by
zero while leaving those columns unchanged.
"""

function safe_column_normalise(M::AbstractMatrix)
    col_sums = sum(M, dims = 1)
    safe_sums = similar(col_sums)

    for i in eachindex(col_sums)
        safe_sums[i] = col_sums[i] == 0 ? one(eltype(col_sums)) : col_sums[i]
    end

    return M ./ safe_sums
end
"""
    safe_row_normalise(M)

Normalise each row of `M` by its row sum.

Rows with a sum of zero use a denominator of one, preventing division by
zero while leaving those rows unchanged.
"""


function safe_row_normalise(M::AbstractMatrix)
    row_sums = sum(M, dims = 2)
    safe_sums = similar(row_sums)

    for i in eachindex(row_sums)
        safe_sums[i] = row_sums[i] == 0 ? one(eltype(row_sums)) : row_sums[i]
    end

    return M ./ safe_sums
end

"""
    DirectedHypergraphLayer(species_in_dim, hidden_dim, activation)

A Lux-compatible message-passing layer for directed hypergraphs.

The layer accepts a species-feature matrix together with source and target
incidence matrices. It performs:

1. A learnable transformation of species features.
2. Separate aggregation of source and target species into reaction embeddings.
3. A learnable transformation of reaction embeddings.
4. Propagation of reaction messages back to participating species.
5. A learnable update of the species embeddings.

# Arguments

- `species_in_dim`: Number of input features associated with each species.
- `hidden_dim`: Size of the hidden species and reaction embeddings.
- `activation`: Element-wise activation function.

# Input

A tuple `(X_species, source_matrix, target_matrix)` where:

- `X_species` has shape `number_of_species × species_in_dim`.
- `source_matrix` has shape `number_of_species × number_of_reactions`.
- `target_matrix` has shape `number_of_species × number_of_reactions`.

# Output

A named tuple containing:

- `updated_species`: Updated species embeddings.
- `reaction_embeddings`: Learned reaction embeddings.
"""

struct DirectedHypergraphLayer{F} <: Lux.AbstractLuxLayer
    species_in_dim::Int
    hidden_dim::Int
    activation::F
end

function Lux.initialparameters(
    rng::AbstractRNG,
    layer::DirectedHypergraphLayer
)
    return (
        W_species = randn(
            rng,
            Float32,
            layer.species_in_dim,
            layer.hidden_dim
        ) .* 0.1f0,

        b_species = zeros(
            Float32,
            1,
            layer.hidden_dim
        ),

        W_reaction = randn(
            rng,
            Float32,
            2 * layer.hidden_dim,
            layer.hidden_dim
        ) .* 0.1f0,

        b_reaction = zeros(
            Float32,
            1,
            layer.hidden_dim
        ),

        W_update = randn(
            rng,
            Float32,
            2 * layer.hidden_dim,
            layer.hidden_dim
        ) .* 0.1f0,

        b_update = zeros(
            Float32,
            1,
            layer.hidden_dim
        )
    )
end

Lux.initialstates(
    ::AbstractRNG,
    ::DirectedHypergraphLayer
) = NamedTuple()

"""
    (layer::DirectedHypergraphLayer)(input, ps, st)

Apply one directed-hypergraph message-passing step.

`ps` contains the learnable Lux parameters and `st` contains the layer state.
The returned state is unchanged because this layer currently has no mutable
state.
"""

function (layer::DirectedHypergraphLayer)(input, ps, st)
    X_species, source_matrix, target_matrix = input

    membership_matrix = source_matrix .+ target_matrix

    source_norm = safe_column_normalise(source_matrix)
    target_norm = safe_column_normalise(target_matrix)
    membership_norm = safe_row_normalise(membership_matrix)

    H_species = layer.activation.(
        X_species * ps.W_species .+ ps.b_species
    )

    reactant_messages =
        transpose(source_norm) * H_species

    product_messages =
        transpose(target_norm) * H_species

    directed_reaction_input =
        hcat(reactant_messages, product_messages)

    H_reaction = layer.activation.(
        directed_reaction_input * ps.W_reaction .+ ps.b_reaction
    )

    species_messages =
        membership_norm * H_reaction

    species_update_input =
        hcat(H_species, species_messages)

    updated_species = layer.activation.(
        species_update_input * ps.W_update .+ ps.b_update
    )

    output = (
        updated_species = updated_species,
        reaction_embeddings = H_reaction
    )

    return output, st
end
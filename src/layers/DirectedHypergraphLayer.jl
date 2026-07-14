using Lux
using Random


function safe_column_normalise(M::AbstractMatrix)
    col_sums = sum(M, dims = 1)
    safe_sums = similar(col_sums)

    for i in eachindex(col_sums)
        safe_sums[i] = col_sums[i] == 0 ? one(eltype(col_sums)) : col_sums[i]
    end

    return M ./ safe_sums
end

function safe_row_normalise(M::AbstractMatrix)
    row_sums = sum(M, dims = 2)
    safe_sums = similar(row_sums)

    for i in eachindex(row_sums)
        safe_sums[i] = row_sums[i] == 0 ? one(eltype(row_sums)) : row_sums[i]
    end

    return M ./ safe_sums
end

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
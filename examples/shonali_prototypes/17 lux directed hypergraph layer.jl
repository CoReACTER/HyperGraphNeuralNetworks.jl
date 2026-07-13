#17 lux directed hypergraph layer

#purpose:
#first Lux-compatible directed hypergraph message-passing layer.
#this version uses Lux v1.31.4 syntax.

using Lux
using Random
using LinearAlgebra
using Statistics

rng = Random.default_rng()
Random.seed!(rng, 42)

println("Lux Directed Hypergraph Layer")


#1.toy species features and incidence matrices

X_species = Float32[
    1.0 0.0 2.0;
    0.0 1.0 1.0;
    1.0 1.0 0.0;
    2.0 0.0 1.0
]

source_matrix = Float32[
    1 0 1;
    1 0 0;
    0 1 0;
    0 0 0
]

target_matrix = Float32[
    0 0 0;
    0 1 0;
    1 0 0;
    0 0 1
]

println("\nInput species feature size:")
println(size(X_species))

println("\nSource matrix size:")
println(size(source_matrix))

println("\nTarget matrix size:")
println(size(target_matrix))


#2.normalisation helpers

function safe_column_normalise(M::AbstractMatrix)
    col_sums = sum(M, dims = 1)
    safe_sums = similar(col_sums)

    for i in eachindex(col_sums)
        safe_sums[i] = col_sums[i] == 0 ? 1 : col_sums[i]
    end

    return Float32.(M ./ safe_sums)
end


function safe_row_normalise(M::AbstractMatrix)
    row_sums = sum(M, dims = 2)
    safe_sums = similar(row_sums)

    for i in eachindex(row_sums)
        safe_sums[i] = row_sums[i] == 0 ? 1 : row_sums[i]
    end

    return Float32.(M ./ safe_sums)
end


#3.custom Lux layer

struct DirectedHypergraphLayer <: Lux.AbstractLuxLayer
    species_in_dim::Int
    hidden_dim::Int
    activation
end


#4.parameter initialisation

function Lux.initialparameters(rng::AbstractRNG, layer::DirectedHypergraphLayer)
    return (
        W_species = randn(rng, Float32, layer.species_in_dim, layer.hidden_dim) .* 0.1f0,
        b_species = zeros(Float32, 1, layer.hidden_dim),

        W_reaction = randn(rng, Float32, 2 * layer.hidden_dim, layer.hidden_dim) .* 0.1f0,
        b_reaction = zeros(Float32, 1, layer.hidden_dim),

        W_update = randn(rng, Float32, 2 * layer.hidden_dim, layer.hidden_dim) .* 0.1f0,
        b_update = zeros(Float32, 1, layer.hidden_dim)
    )
end


function Lux.initialstates(rng::AbstractRNG, layer::DirectedHypergraphLayer)
    return NamedTuple()
end


#5.forward pass

function (layer::DirectedHypergraphLayer)(input, ps, st)

    X_species, source_matrix, target_matrix = input

    membership_matrix = source_matrix .+ target_matrix

    source_norm = safe_column_normalise(source_matrix)
    target_norm = safe_column_normalise(target_matrix)
    membership_norm = safe_row_normalise(membership_matrix)

    #learnable species transformation

    H_species =
        layer.activation.(
            X_species * ps.W_species .+ ps.b_species
        )

    #species -> reaction aggregation

    reactant_messages =
        transpose(source_norm) * H_species

    product_messages =
        transpose(target_norm) * H_species

    directed_reaction_input =
        hcat(
            reactant_messages,
            product_messages
        )

    #learnable reaction transformation

    H_reaction =
        layer.activation.(
            directed_reaction_input * ps.W_reaction .+ ps.b_reaction
        )

    #reaction -> species aggregation

    species_messages =
        membership_norm * H_reaction

    #species update

    species_update_input =
        hcat(
            H_species,
            species_messages
        )

    updated_species =
        layer.activation.(
            species_update_input * ps.W_update .+ ps.b_update
        )

    output = (
        updated_species = updated_species,
        reaction_embeddings = H_reaction
    )

    return output, st

end


#6.create layer and initialise with Lux.setup

layer = DirectedHypergraphLayer(
    size(X_species, 2),
    8,
    tanh
)

ps, st = Lux.setup(rng, layer)


#7.forward pass through directed hypergraph layer

output, st = layer(
    (
        X_species,
        source_matrix,
        target_matrix
    ),
    ps,
    st
)

println("\nUpdated species embeddings:")
println(output.updated_species)

println("\nReaction embeddings:")
println(output.reaction_embeddings)

println("\nUpdated species embedding size:")
println(size(output.updated_species))

println("\nReaction embedding size:")
println(size(output.reaction_embeddings))


#8.simple regression head

regression_head = Lux.Dense(8 => 1)

ps_head, st_head = Lux.setup(rng, regression_head)

#Lux Dense expects features × batch
#reaction_embeddings is reactions × features
#so transpose is used

predictions_matrix, st_head =
    regression_head(
        transpose(output.reaction_embeddings),
        ps_head,
        st_head
    )

predictions = vec(predictions_matrix)

println("\nReaction-level predictions:")
println(predictions)


#9.toy target and loss

toy_energy_barriers = Float32[
    12.5,
    18.2,
    7.4
]

function mse_loss(y_pred, y_true)
    return mean((y_pred .- y_true).^2)
end

loss = mse_loss(predictions, toy_energy_barriers)

println("\nToy MSE loss:")
println(loss)


#10.summary

println("\nSummary")
println("This file implements a Lux-compatible directed hypergraph layer.")
println("It contains learnable species, reaction and update transformations.")
println("It performs species-to-reaction and reaction-to-species message passing.")
println("It returns updated species embeddings and reaction embeddings.")
println("The reaction embeddings are passed into a Lux Dense regression head.")
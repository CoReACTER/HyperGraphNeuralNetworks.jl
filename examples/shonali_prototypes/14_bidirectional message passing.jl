#bidirectional message passing

#purpose:
#comparing simple directed hypergraph message-passing strategies for CRNs.

#this file connects to the thesis question:
#"What message-passing strategy is most appropriate for directed hypergraphs representing chemical reaction networks?"

#tt compares:
#1. Species -> Reaction message passing only
#2. Species -> Reaction -> Species -> Reaction bidirectional message passing

using Statistics
using LinearAlgebra
using Random

Random.seed!(42)


println("Bidirectional Message Passing")



#1.defineing a small Formose-inspired CRN


raw_reactions = [
    "CH2O + CH2O -> Glycolaldehyde",
    "Glycolaldehyde + CH2O -> Glyceraldehyde",
    "Glyceraldehyde -> Dihydroxyacetone",
    "Dihydroxyacetone + CH2O -> Tetrose",
    "Tetrose -> Glycolaldehyde + Glycolaldehyde",
    "Glyceraldehyde + Glycolaldehyde -> Pentose",
    "Pentose -> Dihydroxyacetone + Glycolaldehyde",
    "Tetrose + CH2O -> Hexose",
    "Hexose -> Glyceraldehyde + Glyceraldehyde"
]

println("Raw reactions:")
for r in raw_reactions
    println(r)
end

#2.parser


function parse_side(side::AbstractString)
    species = split(strip(side), "+")
    return strip.(species)
end

function parse_reaction(reaction::AbstractString)
    sides = split(reaction, "->")

    if length(sides) != 2
        error("Reaction must contain exactly one -> symbol: $reaction")
    end

    reactants = parse_side(sides[1])
    products = parse_side(sides[2])

    return reactants, products
end

parsed_reactions = [parse_reaction(r) for r in raw_reactions]

println("\nParsed reactions:")
for (i, (reactants, products)) in enumerate(parsed_reactions)
    println("r$i: ", join(reactants, " + "), " -> ", join(products, " + "))
end

#3.species mappings


all_species = String[]

for (reactants, products) in parsed_reactions
    append!(all_species, String.(reactants))
    append!(all_species, String.(products))
end

species = sort(unique(all_species))

node_to_id = Dict(s => i for (i, s) in enumerate(species))
id_to_node = Dict(i => s for (s, i) in node_to_id)

num_species = length(species)
num_reactions = length(parsed_reactions)

println("\nSpecies:")
println(species)

println("\nNode to ID mapping:")
println(node_to_id)


#4.source and target matrices

#source_matrix[i, j] gives the stoichiometric count of species i as a reactant in reaction j.

#target_matrix[i, j] gives the stoichiometric count of species i as a product in reaction j.

source_matrix = zeros(Float32, num_species, num_reactions)
target_matrix = zeros(Float32, num_species, num_reactions)

for (j, (reactants, products)) in enumerate(parsed_reactions)
    for r in reactants
        source_matrix[node_to_id[String(r)], j] += 1.0f0
    end

    for p in products
        target_matrix[node_to_id[String(p)], j] += 1.0f0
    end
end

membership_matrix = source_matrix .+ target_matrix

println("\nSource matrix:")
println(source_matrix)

println("\nTarget matrix:")
println(target_matrix)

println("\nMembership matrix:")
println(membership_matrix)


#5.building species features

#these are structural baseline features.
#they can later be replaced or extended using chemistry-informed features.

source_count = vec(sum(source_matrix .> 0, dims = 2))
target_count = vec(sum(target_matrix .> 0, dims = 2))
participation_count = source_count .+ target_count

source_stoich_sum = vec(sum(source_matrix, dims = 2))
target_stoich_sum = vec(sum(target_matrix, dims = 2))
total_stoich_sum = source_stoich_sum .+ target_stoich_sum

species_features = hcat(
    source_count,
    target_count,
    participation_count,
    source_stoich_sum,
    target_stoich_sum,
    total_stoich_sum
)

species_features = Float32.(species_features)

species_feature_names = [
    "source_count",
    "target_count",
    "participation_count",
    "source_stoich_sum",
    "target_stoich_sum",
    "total_stoich_sum"
]

println("\nSpecies feature names:")
println(species_feature_names)

println("\nSpecies features:")
println(species_features)


#6.normalise features


function standardise_features(X::AbstractMatrix)
    μ = mean(X, dims = 1)
    σ = std(X, dims = 1)

    σ_safe = similar(σ)

    for i in eachindex(σ)
        σ_safe[i] = σ[i] == 0 ? 1 : σ[i]
    end

    X_norm = (X .- μ) ./ σ_safe

    return Float32.(X_norm)
end

species_features_norm = standardise_features(species_features)

println("\nNormalised species features:")
println(species_features_norm)


#7.helper functions for aggregation


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


#8.species -> reaction message passing

#this aggregates species features into reaction embeddings.

#reactants and products are treated separately, then concatenated.
#this allows the model representation to preserve directionality.

function species_to_reaction(
    X_species::AbstractMatrix,
    source_matrix::AbstractMatrix,
    target_matrix::AbstractMatrix
)
    source_norm = safe_column_normalise(source_matrix)
    target_norm = safe_column_normalise(target_matrix)

    source_reaction_features = transpose(source_norm) * X_species
    target_reaction_features = transpose(target_norm) * X_species

    reaction_features = hcat(source_reaction_features, target_reaction_features)

    return Float32.(reaction_features)
end

reaction_embeddings_oneway =
    species_to_reaction(species_features_norm, source_matrix, target_matrix)


println("Architecture A: Species -> Reaction")


println("\nReaction embeddings from one-way message passing:")
println(reaction_embeddings_oneway)

println("\nReaction embedding size:")
println(size(reaction_embeddings_oneway))


#9.reaction -> species message passing


#this propagates reaction information back to species.

#it allows species representations to be updated using information from the reactions they participate in.

function reaction_to_species(
    X_reaction::AbstractMatrix,
    membership_matrix::AbstractMatrix
)
    membership_norm = safe_row_normalise(membership_matrix)

    updated_species = membership_norm * X_reaction

    return Float32.(updated_species)
end

updated_species_from_reactions =
    reaction_to_species(reaction_embeddings_oneway, membership_matrix)

println("\nUpdated species embeddings from Reaction -> Species:")
println(updated_species_from_reactions)

println("\nUpdated species embedding size:")
println(size(updated_species_from_reactions))


#10.bidirectional message passing


# Architecture B:
#
# Species features
#       ↓
# Species -> Reaction
#       ↓
# Reaction embeddings
#       ↓
# Reaction -> Species
#       ↓
# Updated species embeddings
#       ↓
# Species -> Reaction again
#
#this gives a second reaction representation after information has moved in both directions.

reaction_embeddings_bidirectional =
    species_to_reaction(
        updated_species_from_reactions,
        source_matrix,
        target_matrix
    )


println("Architecture B: Species -> Reaction -> Species -> Reaction")


println("\nReaction embeddings from bidirectional message passing:")
println(reaction_embeddings_bidirectional)

println("\nBidirectional reaction embedding size:")
println(size(reaction_embeddings_bidirectional))


#11.toy regression target


#placeholder energy barriers.
#in the final project, these would come from a real chemical dataset.

toy_energy_barriers = Float32[
    12.5,
    18.2,
    7.4,
    21.0,
    15.6,
    25.3,
    10.2,
    30.1,
    13.8
]

println("\nToy energy barrier targets:")
println(toy_energy_barriers)


#12.simple ridge regression helper


#this is not the final neural network model.
#it is a simple baseline to check whether reaction embeddings can be used for reaction-level regression.

# β = (X'X + λI)^(-1) X'y

function add_intercept(X::AbstractMatrix)
    ones_col = ones(Float32, size(X, 1), 1)
    return hcat(ones_col, Float32.(X))
end

function fit_ridge_regression(X::AbstractMatrix, y::AbstractVector; λ = 1.0f-3)
    X_aug = add_intercept(X)

    I_reg = Matrix{Float32}(I, size(X_aug, 2), size(X_aug, 2))
    I_reg[1, 1] = 0.0f0  # do not regularise intercept

    β = (transpose(X_aug) * X_aug + λ * I_reg) \ (transpose(X_aug) * y)

    return Float32.(β)
end

function predict_ridge(X::AbstractMatrix, β::AbstractVector)
    X_aug = add_intercept(X)
    return Float32.(X_aug * β)
end

function mae(y_true::AbstractVector, y_pred::AbstractVector)
    return mean(abs.(y_true .- y_pred))
end

function rmse(y_true::AbstractVector, y_pred::AbstractVector)
    return sqrt(mean((y_true .- y_pred).^2))
end


#13.comparing one-way vs bidirectional embeddings


β_oneway = fit_ridge_regression(reaction_embeddings_oneway, toy_energy_barriers)
pred_oneway = predict_ridge(reaction_embeddings_oneway, β_oneway)

β_bidirectional = fit_ridge_regression(reaction_embeddings_bidirectional, toy_energy_barriers)
pred_bidirectional = predict_ridge(reaction_embeddings_bidirectional, β_bidirectional)


println("Toy Regression Comparison")


println("\nOne-way predictions:")
println(pred_oneway)

println("\nBidirectional predictions:")
println(pred_bidirectional)

mae_oneway = mae(toy_energy_barriers, pred_oneway)
rmse_oneway = rmse(toy_energy_barriers, pred_oneway)

mae_bidirectional = mae(toy_energy_barriers, pred_bidirectional)
rmse_bidirectional = rmse(toy_energy_barriers, pred_bidirectional)

println("\nOne-way message passing MAE: ", mae_oneway)
println("One-way message passing RMSE: ", rmse_oneway)

println("\nBidirectional message passing MAE: ", mae_bidirectional)
println("Bidirectional message passing RMSE: ", rmse_bidirectional)


#14.reaction-level output by reaction



println("Reaction-level comparison")


for i in 1:num_reactions
    reactants, products = parsed_reactions[i]

    println("\nr$i: ", join(reactants, " + "), " -> ", join(products, " + "))
    println("true energy barrier: ", toy_energy_barriers[i])
    println("one-way prediction: ", pred_oneway[i])
    println("bidirectional prediction: ", pred_bidirectional[i])
end


#15.summary


println("\n====================================")
println("Summary")
println("====================================")

println("Number of species: ", num_species)
println("Number of reactions: ", num_reactions)
println("Original species feature size: ", size(species_features_norm))
println("One-way reaction embedding size: ", size(reaction_embeddings_oneway))
println("Updated species embedding size: ", size(updated_species_from_reactions))
println("Bidirectional reaction embedding size: ", size(reaction_embeddings_bidirectional))

println("\nThis file compares one-way and bidirectional message passing")
println("for reaction-level regression using toy energy barrier targets.")
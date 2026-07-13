#reaction level regression

#purpose:
#this file evaluates whether reaction embeddings learned from the directed hypergraph neural network can predict continuous reaction
#properties such as energy barriers, reaction rates and reaction yields.

#this file connects directly to the thesis question:
#"Which directed hypergraph neural network architecture is most suitable
#for reaction-level regression tasks involving chemical reaction networks?"

#overall workflow:
#
#Chemical Reaction Network
#        ↓
#Species Features
#        ↓
#Species → Reaction Message Passing
#        ↓
#Reaction Embeddings
#        ↓
#Regression Model
#        ↓
#Predicted Reaction Property
#        ↓
#Model Evaluation

using Statistics
using LinearAlgebra
using Random

Random.seed!(42)

println("Reaction Level Regression")


#1.defining a small Formose-inspired CRN


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

println("\nRaw reactions:")

for reaction in raw_reactions
    println(reaction)
end


#2.parser


function parse_side(side::AbstractString)

    species = split(strip(side), "+")

    return strip.(species)

end


function parse_reaction(reaction::AbstractString)

    sides = split(reaction, "->")

    if length(sides) != 2
        error("Reaction must contain exactly one -> symbol.")
    end

    reactants = parse_side(sides[1])
    products = parse_side(sides[2])

    return reactants, products

end


parsed_reactions = [parse_reaction(r) for r in raw_reactions]


println("\nParsed reactions:")

for (i, (reactants, products)) in enumerate(parsed_reactions)

    println(
        "r$i: ",
        join(reactants, " + "),
        " -> ",
        join(products, " + ")
    )

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

#source_matrix[i, j] gives the stoichiometric count of species i
#appearing as a reactant in reaction j.

#target_matrix[i, j] gives the stoichiometric count of species i
#appearing as a product in reaction j.

source_matrix = zeros(Float32, num_species, num_reactions)
target_matrix = zeros(Float32, num_species, num_reactions)

for (j, (reactants, products)) in enumerate(parsed_reactions)

    for reactant in reactants
        source_matrix[node_to_id[String(reactant)], j] += 1.0f0
    end

    for product in products
        target_matrix[node_to_id[String(product)], j] += 1.0f0
    end

end

membership_matrix = source_matrix .+ target_matrix

println("\nSource matrix:")
println(source_matrix)

println("\nTarget matrix:")
println(target_matrix)

println("\nMembership matrix:")
println(membership_matrix)


#5.building structural species features


#these are simple topology-based features.

source_count = vec(sum(source_matrix .> 0, dims = 2))
target_count = vec(sum(target_matrix .> 0, dims = 2))
participation_count = source_count .+ target_count

source_stoich_sum = vec(sum(source_matrix, dims = 2))
target_stoich_sum = vec(sum(target_matrix, dims = 2))
total_stoich_sum = source_stoich_sum .+ target_stoich_sum

structural_features = hcat(

    source_count,
    target_count,
    participation_count,
    source_stoich_sum,
    target_stoich_sum,
    total_stoich_sum

)

structural_features = Float32.(structural_features)

println("\nStructural species features:")
println(structural_features)


#6.building chemistry-informed species features


#toy chemistry descriptors.

#these can later be replaced by real molecular descriptors.

carbon_atoms = Float32[
    1,
    2,
    3,
    3,
    4,
    5,
    6
]

oxygen_atoms = Float32[
    1,
    2,
    3,
    3,
    4,
    5,
    6
]

molecular_weight = Float32[
    30.0,
    60.0,
    90.0,
    90.0,
    120.0,
    150.0,
    180.0
]

chemistry_features = hcat(

    carbon_atoms,
    oxygen_atoms,
    molecular_weight

)

println("\nChemistry-informed features:")
println(chemistry_features)


#7.combining structural and chemistry features


combined_features = hcat(

    structural_features,
    chemistry_features

)

combined_features = Float32.(combined_features)

println("\nCombined species features:")
println(combined_features)


#8.standardising feature matrices


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

structural_features_norm = standardise_features(structural_features)

chemistry_features_norm = standardise_features(chemistry_features)

combined_features_norm = standardise_features(combined_features)

println("\nNormalised structural features:")
println(structural_features_norm)

println("\nNormalised chemistry features:")
println(chemistry_features_norm)

println("\nNormalised combined features:")
println(combined_features_norm)
#9.helper functions for message passing


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


#10.species -> reaction message passing


#this aggregates species features into reaction embeddings.

#reactant features and product features are aggregated
#separately before being concatenated.

function species_to_reaction(

    X_species::AbstractMatrix,
    source_matrix::AbstractMatrix,
    target_matrix::AbstractMatrix

)

    source_norm = safe_column_normalise(source_matrix)
    target_norm = safe_column_normalise(target_matrix)

    reactant_embeddings = transpose(source_norm) * X_species
    product_embeddings = transpose(target_norm) * X_species

    reaction_embeddings = hcat(

        reactant_embeddings,
        product_embeddings

    )

    return Float32.(reaction_embeddings)

end


#11.select feature representation


#this allows easy comparison of different
#species feature representations.

feature_set = "combined"

species_features = combined_features_norm

if feature_set == "structural"

    species_features = structural_features_norm

elseif feature_set == "chemistry"

    species_features = chemistry_features_norm

elseif feature_set == "combined"

    species_features = combined_features_norm

else

    error("Unknown feature set.")

end


println("\nSelected feature set:")
println(feature_set)


#12.construct reaction embeddings


reaction_embeddings = species_to_reaction(

    species_features,
    source_matrix,
    target_matrix

)

println("\nReaction embeddings:")
println(reaction_embeddings)

println("\nReaction embedding size:")
println(size(reaction_embeddings))


#13.select prediction task


#currently toy targets are used.

#these can later be replaced by
#real reaction datasets.

prediction_task = "Energy Barrier"

reaction_targets = Float32[

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

println("\nPrediction task:")
println(prediction_task)

println("\nReaction targets:")
println(reaction_targets)


#14.build regression dataset


#X contains the reaction embeddings.

#rows represent reactions.

#columns represent embedding dimensions.

X = reaction_embeddings

y = reaction_targets

println("\nRegression dataset")

println("Feature matrix size:")
println(size(X))

println("Target vector size:")
println(size(y))


#15.shuffle dataset


indices = collect(1:num_reactions)

Random.shuffle!(indices)

X = X[indices, :]

y = y[indices]

println("\nShuffled reaction indices:")
println(indices)


#16.training and testing split


train_ratio = 0.80

num_train = floor(Int, train_ratio * num_reactions)

train_indices = 1:num_train
test_indices = (num_train + 1):num_reactions

X_train = X[train_indices, :]
X_test = X[test_indices, :]

y_train = y[train_indices]
y_test = y[test_indices]

println("\nTraining samples:")
println(length(y_train))

println("Testing samples:")
println(length(y_test))

#17.helper functions for ridge regression


#an intercept term is added to the regression model.

function add_intercept(X::AbstractMatrix)

    ones_column = ones(Float32, size(X, 1), 1)

    return hcat(ones_column, Float32.(X))

end


#ridge regression solution:
#
#β = (X'X + λI)^(-1)X'y

function fit_ridge_regression(

    X::AbstractMatrix,
    y::AbstractVector;

    λ = 1.0f-3

)

    X_aug = add_intercept(X)

    I_reg = Matrix{Float32}(I, size(X_aug, 2), size(X_aug, 2))

    I_reg[1,1] = 0.0f0

    β =

        (transpose(X_aug) * X_aug + λ * I_reg) \

        (transpose(X_aug) * y)

    return Float32.(β)

end


function predict_ridge(

    X::AbstractMatrix,
    β::AbstractVector

)

    X_aug = add_intercept(X)

    return Float32.(X_aug * β)

end


#18.regression evaluation metrics


function mse(

    y_true::AbstractVector,
    y_pred::AbstractVector

)

    return mean((y_true .- y_pred).^2)

end


function rmse(

    y_true::AbstractVector,
    y_pred::AbstractVector

)

    return sqrt(

        mean(

            (y_true .- y_pred).^2

        )

    )

end


function mae(

    y_true::AbstractVector,
    y_pred::AbstractVector

)

    return mean(

        abs.(y_true .- y_pred)

    )

end


function r_squared(

    y_true::AbstractVector,
    y_pred::AbstractVector

)

    ss_res =

        sum(

            (y_true .- y_pred).^2

        )

    ss_tot =

        sum(

            (y_true .- mean(y_true)).^2

        )

    return 1.0 - ss_res / ss_tot

end


#19.training the regression model


println("\nTraining ridge regression model...")

β = fit_ridge_regression(

    X_train,
    y_train

)

println("\nRegression coefficients:")

println(β)


#20.generate predictions


train_predictions =

    predict_ridge(

        X_train,
        β

    )

test_predictions =

    predict_ridge(

        X_test,
        β

    )


println("\nTraining predictions:")

println(train_predictions)

println("\nTesting predictions:")

println(test_predictions)


#21.evaluate training performance


train_mse =

    mse(

        y_train,
        train_predictions

    )

train_rmse =

    rmse(

        y_train,
        train_predictions

    )

train_mae =

    mae(

        y_train,
        train_predictions

    )

train_r2 =

    r_squared(

        y_train,
        train_predictions

    )


println("\nTraining Performance")

println("MSE: ", train_mse)

println("RMSE: ", train_rmse)

println("MAE: ", train_mae)

println("R²: ", train_r2)


#22.evaluate testing performance


test_mse =

    mse(

        y_test,
        test_predictions

    )

test_rmse =

    rmse(

        y_test,
        test_predictions

    )

test_mae =

    mae(

        y_test,
        test_predictions

    )

test_r2 =

    r_squared(

        y_test,
        test_predictions

    )


println("\nTesting Performance")

println("MSE: ", test_mse)

println("RMSE: ", test_rmse)

println("MAE: ", test_mae)

println("R²: ", test_r2)

#23.display reaction-level predictions


println("\n========================================")
println("Reaction-Level Predictions")
println("========================================")

for i in 1:length(y_test)

    absolute_error = abs(y_test[i] - test_predictions[i])

    println("\nReaction ", i)

    println("True ", prediction_task, ": ", y_test[i])

    println("Predicted: ", test_predictions[i])

    println("Absolute Error: ", absolute_error)

end


#24.compare different feature sets


println("\n========================================")
println("Feature Set Comparison")
println("========================================")


feature_sets = Dict(

    "Structural" => structural_features_norm,

    "Chemistry" => chemistry_features_norm,

    "Combined" => combined_features_norm

)


comparison_results = Dict{String, Dict{String, Float32}}()


for (feature_name, feature_matrix) in feature_sets

    println("\nRunning experiment using ", feature_name, " features...")


    reaction_embeddings = species_to_reaction(

        feature_matrix,

        source_matrix,

        target_matrix

    )


    X = reaction_embeddings

    y = reaction_targets


    X = X[indices, :]
    y = y[indices]


    X_train = X[train_indices, :]
    X_test = X[test_indices, :]

    y_train = y[train_indices]
    y_test = y[test_indices]


    β = fit_ridge_regression(

        X_train,

        y_train

    )


    predictions = predict_ridge(

        X_test,

        β

    )


    current_mse = mse(

        y_test,

        predictions

    )

    current_rmse = rmse(

        y_test,

        predictions

    )

    current_mae = mae(

        y_test,

        predictions

    )

    current_r2 = r_squared(

        y_test,

        predictions

    )


    comparison_results[feature_name] = Dict(

        "MSE" => Float32(current_mse),

        "RMSE" => Float32(current_rmse),

        "MAE" => Float32(current_mae),

        "R2" => Float32(current_r2)

    )


    println("MSE : ", current_mse)
    println("RMSE: ", current_rmse)
    println("MAE : ", current_mae)
    println("R²  : ", current_r2)

end


#25.determine best feature representation


best_feature_set = ""

best_rmse = Inf

for (feature_name, metrics) in comparison_results

    global best_rmse
    global best_feature_set

    if metrics["RMSE"] < best_rmse

        best_rmse = metrics["RMSE"]
        best_feature_set = feature_name

    end

end


println("\nBest feature representation:")

println(best_feature_set)


#26.save experiment results


experiment_results = Dict(

    "prediction_task" => prediction_task,

    "selected_feature_set" => feature_set,

    "training_samples" => length(y_train),

    "testing_samples" => length(y_test),

    "mse" => test_mse,

    "rmse" => test_rmse,

    "mae" => test_mae,

    "r2" => test_r2,

    "predictions" => test_predictions,

    "ground_truth" => y_test,

    "best_feature_set" => best_feature_set,

    "comparison_results" => comparison_results

)


println("\nExperiment results saved.")


#27.final summary


println("\n====================================")
println("Summary")
println("====================================")

println("Prediction Task: ", prediction_task)

println("Feature Set: ", feature_set)

println("Training Samples: ", length(y_train))

println("Testing Samples: ", length(y_test))

println("MSE : ", test_mse)

println("RMSE: ", test_rmse)

println("MAE : ", test_mae)

println("R²  : ", test_r2)

println("Best Feature Set: ", best_feature_set)

println("\nThis file demonstrates reaction-level regression")

println("using directed hypergraph reaction embeddings.")

println("The learned reaction embeddings are evaluated")

println("using a ridge regression baseline and compared")

println("across structural, chemistry-informed and")

println("combined feature representations.")

println("\nThe outputs from this file will be used")

println("in File 16 for architecture comparison.")

println("\nReaction-level regression completed successfully.")
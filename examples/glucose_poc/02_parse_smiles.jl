using CSV
using DataFrames

data_path = joinpath(
    @__DIR__,
    "..",
    "..",
    "data",
    "glucose",
    "glucose_network.csv"
)

df = CSV.read(data_path, DataFrame)

function split_smiles_side(smiles::AbstractString)
    return split(smiles, ".")
end

all_species = Set{String}()

reactant_counts = Int[]
product_counts = Int[]

for row in eachrow(df)

    reactants = split_smiles_side(row.Rsmiles)
    products = split_smiles_side(row.Psmiles)

    push!(reactant_counts, length(reactants))
    push!(product_counts, length(products))

    for molecule in reactants
        push!(all_species, molecule)
    end

    for molecule in products
        push!(all_species, molecule)
    end
end

println("Number of reactions: ", nrow(df))
println("Unique raw molecular species: ", length(all_species))

println("\nMaximum reactants in one reaction: ", maximum(reactant_counts))
println("Maximum products in one reaction: ", maximum(product_counts))

println("\nFirst reaction:")
println("Reactants:")
println(split_smiles_side(df.Rsmiles[1]))

println("\nProducts:")
println(split_smiles_side(df.Psmiles[1]))
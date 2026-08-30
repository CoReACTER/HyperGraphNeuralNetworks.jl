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

function remove_atom_maps(smiles::AbstractString)
    return replace(smiles, r":\d+" => "")
end

raw_species = Set{String}()
clean_species = Set{String}()

for row in eachrow(df)

    reactants = split_smiles_side(row.Rsmiles)
    products = split_smiles_side(row.Psmiles)

    for molecule in vcat(reactants, products)

        push!(raw_species, String(molecule))

        cleaned = remove_atom_maps(molecule)
        push!(clean_species, String(cleaned))
    end
end

println("Unique raw species: ", length(raw_species))
println("Unique species after removing atom maps: ", length(clean_species))

println("\nExample:")
example = first(raw_species)

println("Raw:")
println(example)

println("\nWithout atom maps:")
println(remove_atom_maps(example))
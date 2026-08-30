using CSV
using DataFrames

# Path to glucose dataset
data_path = joinpath(
    @__DIR__,
    "..",
    "..",
    "data",
    "glucose",
    "glucose_network.csv"
)

# Load dataset
df = CSV.read(data_path, DataFrame)

println("Number of reactions: ", nrow(df))

println("\nColumns:")
println(names(df))

println("\nFirst 5 reactions:")
println(first(df, 5))

println("\nColumn types:")
for name in names(df)
    println(name, " => ", eltype(df[!, name]))
end

println("\nMissing values:")
for name in names(df)
    println(name, " => ", count(ismissing, df[!, name]))
end
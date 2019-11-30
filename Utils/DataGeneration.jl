using Random

function generate_random_classification_data(
    n::Int64,
    d::Int64,
    p::Int64;
    σ=1,
    seed=42
)::Tuple{Array{
    Float64, 2}, Array{Int64, 1}, Array{Int64, 1}}
    rng = MersenneTwister(seed)
    group_assignments = rand(rng, 1:p, n)
    y = rand(rng, [-1, 1], n)
    X = σ*randn(rng, n, d) .+ y
    return X, y, group_assignments
end


function generate_random_regression_data(
    n::Int64,
    d::Int64,
    p::Int64;
    σ=0.1,
    seed=42
)::Tuple{Array{
    Float64, 2}, Array{Float64, 1}, Array{Int64, 1}}
    rng = MersenneTwister(seed)
    group_assignments = rand(rng, 1:p, n)
    X = rand(rng, n, d) .* [group_assignments[i] for i=1:n,j=1:d]
    β_true = randn(rng, d) .* rand(rng, [0, 1], d)
    y = X*β_true + σ*randn(rng, n)
    return X, y, group_assignments
end


function generate_random_regression_data(
    n::Int64,
    d::Int64,
    p::Int64;
    σ=0.1,
    seed=42
)::Tuple{Array{
    Float64, 2}, Array{Float64, 1}, Array{Int64, 1}}
    rng = MersenneTwister(seed)
    group_assignments = rand(rng, 1:p, n)
    X = rand(rng, n, d) .* [group_assignments[i] for i=1:n,j=1:d]
    β_true = randn(rng, d) .* rand(rng, [0, 1], d)
    y = X*β_true + σ*randn(rng, n)
    return X, y, group_assignments
end

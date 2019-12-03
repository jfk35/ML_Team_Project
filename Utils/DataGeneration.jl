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

function normalize_data(
        X::Array{Float64, 2})::Array{Float64, 2}
    normX = copy(X)
    for i =1:size(normX)[2]
        col = X[:,i]
        colMean = mean(col)
        stdev = std(col)

        normX[:, i] = (col .- colMean)./stdev
    end
    return normX
end

function load_classification_data( filepath::String
        )::Tuple{Array{
    Float64, 2}, Array{Float64, 1}, Array{Int64, 1}}
    dataset = CSV.read(filepath, missingstring="NA")
    #print(dataset)
    X = dataset[:,[2:7;9:22]]
    y = dataset[:,8]
    groupAssignments = dataset[:,23]
    lnrknnnew = IAI.OptKNNImputationLearner()
    imputedX = IAI.fit_transform!(lnrknnnew, X)

    imputedX = convert(Array{Float64, 2}, imputedX)
    y = convert(Array{Int64,1}, y)
    group_assignments = convert(Array{Int64,1},groupAssignments)

    normX = normalize_data(imputedX)

    return normX, y, groupAssignments
end

function load_regression_data( filepath::String
        )::Tuple{Array{
    Float64, 2}, Array{Float64, 1}, Array{Int64, 1}}
    dataset = CSV.read(filepath)
    X = convert(Matrix, dataset[:,2:12])
    y = dataset[15]
    groupAssignments = dataset[13]
    return X, y, groupAssignments

end

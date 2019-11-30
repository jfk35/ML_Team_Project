using Pkg
Pkg.build("FairML")
using FairML, Random, PyPlot, Dates



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

# Generate low dimensional random data
n = 100
d = 2
p = 3
X, y, group_assignments = generate_random_classification_data(n, d, p, σ=2);
w, b = FairML.svm_classifier(X, y)
#w_fair, b_fair = FairML.fair_svm_classifier(X, y, group_assignments);
# markers = ["x", "o", "v"]
# colors = ["blue", "orange", "red"]
# for k=1:p
#     mask_positive = (y .== 1) .& (group_assignments .== k)
#     mask_negative = (y .== -1) .& (group_assignments .== k)
#     plt.scatter(X[mask_positive, 1], X[mask_positive, 2], label=string("y=1,  ", "k=", k), marker=markers[k],
#         color=colors[1])
#     plt.scatter(X[mask_negative, 1], X[mask_negative, 2], label=string("y=-1, ", "k=", k), marker=markers[k],
#         color=colors[2])
# end
#
# X1_grid = minimum(X[:,1]):0.1:maximum(X[:,1])
# X2_grid = minimum(X[:,2]):0.1:maximum(X[:,2])
# plt.plot(X1_grid, (b .- w[1].*X1_grid)/w[2], color="k", linestyle="--", label="Generic SVM")
# plt.plot(X1_grid, (b_fair .- w_fair[1].*X1_grid)/w_fair[2], color="g", linestyle="--", label="Fair SVM")
# plt.legend();

#FairML.svm_fairness_summary(X, y, group_assignments, w, b)

#FairML.svm_fairness_summary(X, y, group_assignments, w_fair, b_fair)

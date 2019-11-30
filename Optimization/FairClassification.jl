using JuMP, Gurobi, Random, PyPlot, Dates, CSV
GUROBI_ENV = Gurobi.Env()

function get_group_indicator_matrix(
    group_assignments::Array{Int64, 1}
)::Array{Int64, 2}
    n = size(group_assignments, 1)
    p = size(unique(group_assignments), 1)
    A = zeros(n, p)
    for i=1:n
        for k=1:p
            A[i,k] = Int(group_assignments[i] .== k)
        end
    end
    return A
end

function svm_classifier(
    X::Array{Float64, 2},
    y::Array{Int64, 1};
    Γ::Float64=0.01,
    return_objective_value::Bool = false,
    solver_time_limit::Int64=60
)::Union{Tuple{Array{Float64, 1}, Float64}, Float64}

    # Initialize model
    model = Model(solver=GurobiSolver(OutputFlag=0, TimeLimit=solver_time_limit, GUROBI_ENV))

    # Define variables
    n, d = size(X)
    @variable(model, w[1:d])
    @variable(model, b)
    @variable(model, ξ[1:n] >= 0)
    @variable(model, t)

    # Set constraints
    @constraint(model, y .* (X*w .- b) .>= 1 - ξ)
    @constraint(model, t == (ones(n)'*ξ)/n  + Γ*(w'*w + b*b))

    # Set objective
    @objective(model, Min, t)

    # Solve and return optimal objective value or regression coefficients
    solve(model)
    if return_objective_value
        return getobjectivevalue(model)
    else
        return getvalue(w), getvalue(b)
    end
end

function fair_svm_classifier(
    X::Array{Float64, 2},
    y::Array{Int64, 1},
    group_assignments::Array{Int64, 1};
    Γ::Float64=0.01, # regularization penalty
    δ::Float64=0.1, # max percent optimality decrease
    λ::Float64=1.0, # FNR weight
    μ::Float64=1.0, # PR weight
    return_objective_value::Bool = false,
    solver_time_limit::Int64=60,
)::Union{Tuple{Array{Float64, 1}, Float64}, Float64}

    # Process inputs
    A = get_group_indicator_matrix(group_assignments)
    p = size(A, 2) # number of groups
    q = Int(p*(p - 1)/2) # number of pairs of groups
    n, d = size(X) # number of data points and number of features
    group_size = [sum(A[:,k]) for k=1:p]
    positive_group_size = [sum(A[i,k] for i=1:n if (y[i] == 1)) for k=1:p]
    negative_group_size = [sum(A[i,k] for i=1:n if (y[i] == -1)) for k=1:p]
    min_loss = svm_classifier(X, y; Γ=Γ, return_objective_value=true) # Min loss of problem w/o fairness constraints

    # Initialize model
    model = Model(solver=GurobiSolver(OutputFlag=0, TimeLimit=solver_time_limit, GUROBI_ENV))

    # Define variables
    @variable(model, w[1:d])
    @variable(model, b)
    @variable(model, ξ[1:n] >= 0)
    @variable(model, positive_indicator[1:n], Bin)
    @variable(model, error_indicator[1:n], Bin)
    @variable(model, FPR[1:p] >= 0)
    @variable(model, FNR[1:p] >= 0)
    @variable(model, PR[1:p] >= 0)
    @variable(model, Δ_FPR[1:q] >= 0)
    @variable(model, Δ_FNR[1:q] >= 0)
    @variable(model, Δ_PR[1:q] >= 0)
    @variable(model, t >= 0)

    # Set SVM performance constraints
    @constraint(model, y.*(X*w .- b) .>= 1 - ξ)
    @constraint(model, (ones(n)'*ξ)/n + Γ*(w'*w + b*b) <= (1 + δ)*min_loss)

    # Set constraints for positive prediction and error indicator variables
    M = 1e6
    @constraint(model, X*w .- b .<= M*positive_indicator)
    @constraint(model, X*w .- b .>= -M*(1 .- positive_indicator))
    @constraint(model, y.*(X*w .- b) .>= -M*error_indicator)
    @constraint(model, y.*(X*w .- b) .<= M*(1 .- error_indicator))

    # Set constraints for FPR, FNR and PR calculation
    @constraint(model, FPR .== (A'*(error_indicator .* (y .== -1))) ./ negative_group_size)
    @constraint(model, FNR .== (A'*(error_indicator .* (y .== 1))) ./ positive_group_size)
    @constraint(model, PR .== (A'*positive_indicator) ./ group_size)

    # Set fairness constraints
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_FPR[m] >= FPR[k] - FPR[l])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_FPR[m] >= FPR[l] - FPR[k])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_FNR[m] >= FNR[k] - FNR[l])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_FNR[m] >= FNR[l] - FNR[k])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_PR[m] >= PR[k] - PR[l])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_PR[m] >= PR[l] - PR[k])
    @constraint(model, t .== ones(q)'*(Δ_FPR .+ λ*Δ_FNR .+ μ*Δ_PR))

    # Set objective
    @objective(model, Min, t)

    # Solve and logger if desired
    solve(model)

    # Return optimal objective value or regression coefficients
    if return_objective_value
        return getobjectivevalue(model)
    else
        return getvalue(w), getvalue(b)
    end
end

function calculate_false_positive_rate(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1}
)::Float64
    return sum(y_hat[y .== -1] .== 1) / sum(y .== -1)
end


function calculate_false_negative_rate(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1}
)::Float64
    return sum(y_hat[y .== 1] .== -1) / sum(y .== 1)
end


function calculate_positive_rate(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1}
)::Float64
    return sum(y_hat .== 1) / size(y_hat, 1)
end


function calculate_fairness_score_for_metric(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1},
    group_assignments::Array{Int64, 1},
    metric::Function
)::Float64
    group_metrics = [metric(y[group_assignments .== k], y_hat[group_assignments .== k]) for k=1:p]
    return sum([sum([abs(group_metrics[k] - group_metrics[l]) for l=k+1:p]) for k=1:p]) / (p*(p - 1)/2)
end


function calculate_fairness_scores(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1},
    group_assignments::Array{Int64, 1}
)::Tuple{Float64, Float64, Float64}
    Δ_FPR = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_false_positive_rate)
    Δ_FNR = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_false_negative_rate)
    Δ_PR = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_positive_rate)
    return Δ_FPR, Δ_FNR, Δ_PR
end


function svm_predict(
    X::Array{Float64, 2},
    w::Array{Float64, 1},
    b::Float64
)::Array{Int64, 1}
    return 2*Int.((X*w .- b) .> 0) .- 1
end


function svm_fairness_summary(
    X::Array{Float64, 2},
    y::Array{Int64, 1},
    group_assignments::Array{Int64, 1},
    w::Array{Float64, 1},
    b::Float64
)
    y_hat = svm_predict(X, w, b)
    Δ_FPR, Δ_FNR, Δ_PR = calculate_fairness_scores(y, y_hat, group_assignments)
    println("False positive rate discrepency: ", round(Δ_FPR; digits=3))
    println("False negative rate discrepency: ", round(Δ_FNR; digits=3))
    println("Positive rate discrepency: ", round(Δ_PR; digits=3))
end

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

function load_classification_data(
        )::Tuple{Array{
    Float64, 2}, Array{Float64, 1}, Array{Int64, 1}}
    dataset = CSV.read("../fairClassificationData.csv", missingstring="NA")
    #print(dataset)
    X = dataset[:,[2:7;9:22]]
    y = dataset[:,8]
    groupAssignments = dataset[:,23]
    lnrknnnew = IAI.OptKNNImputationLearner()
    imputedX = IAI.fit_transform!(lnrknnnew, X)
    return imputedX, y, groupAssignments
end


X,y,groupAssignments = load_classification_data()
X = convert(Matrix, X)
y = convert(Array{Int64,1}, y)
group_assignments = convert(Array{Int64,1},groupAssignments)
w, b = svm_classifier(X, y)
w_fair, b_fair = fair_svm_classifier(X, y, group_assignments);

markers = ["x", "o", "v"]
colors = ["blue", "orange", "red"]
for k=1:p
    mask_positive = (y .== 1) .& (group_assignments .== k)
    mask_negative = (y .== -1) .& (group_assignments .== k)
    plt.scatter(X[mask_positive, 1], X[mask_positive, 2], label=string("y=1,  ", "k=", k), marker=markers[k],
        color=colors[1])
    plt.scatter(X[mask_negative, 1], X[mask_negative, 2], label=string("y=-1, ", "k=", k), marker=markers[k],
        color=colors[2])
end

X1_grid = minimum(X[:,1]):0.1:maximum(X[:,1])
X2_grid = minimum(X[:,2]):0.1:maximum(X[:,2])
plt.plot(X1_grid, (b .- w[1].*X1_grid)/w[2], color="k", linestyle="--", label="Generic SVM")
plt.plot(X1_grid, (b_fair .- w_fair[1].*X1_grid)/w_fair[2], color="g", linestyle="--", label="Fair SVM")
plt.legend();

svm_fairness_summary(X, y, group_assignments, w, b)
svm_fairness_summary(X, y, group_assignments, w_fair, b_fair)

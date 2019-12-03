using JuMP, Gurobi, Dates
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
    Γ::Float64 = 0.01,
    return_objective_value::Bool = false,
    solver_time_limit::Int64 = 60
)::Union{Tuple{Array{Float64, 1}, Float64}, Tuple{Array{Float64, 1}, Float64, Float64}}

    # Initialize model
    model = Model(
        solver=GurobiSolver(
            OutputFlag=0,
            TimeLimit=solver_time_limit,
            GUROBI_ENV
        )
    )

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
        return getvalue(w), getvalue(b), getobjectivevalue(model)
    else
        return getvalue(w), getvalue(b)
    end
end


function fair_convex_svm_classifier(
    X::Array{Float64, 2},
    y::Array{Int64, 1},
    group_assignments::Array{Int64, 1};
    Γ::Float64 = 0.01, # regularization penalty
    δ::Float64 = 0.1, # max percent optimality decrease
    θ::Float64 = 0.5, # FPR (vs FNR) weight
    η::Float64 = 0.5, # Disparate mistreatment (vs disparate impact) weight
    return_objective_value::Bool = false,
    solver_time_limit::Int64 = 60,
)::Union{Tuple{Array{Float64, 1}, Float64}, Tuple{Array{Float64, 1}, Float64, Float64}}

    # Process inputs
    A = get_group_indicator_matrix(group_assignments)
    p = size(A, 2) # number of groups
    q = Int(p*(p - 1)/2) # number of pairs of groups
    n, d = size(X) # number of data points and number of features
    group_size = [sum(A[:,k]) for k=1:p]
    positive_group_size = [sum(A[i,k] for i=1:n if (y[i] == 1)) for k=1:p]
    negative_group_size = [sum(A[i,k] for i=1:n if (y[i] == -1)) for k=1:p]

    # Initialize model
    model = Model(
        solver=GurobiSolver(
            OutputFlag=0,
            TimeLimit=solver_time_limit,
            GUROBI_ENV
        )
    )

    # Define variables
    @variable(model, w[1:d])
    @variable(model, b)
    @variable(model, ξ[1:n] >= 0)
    @variable(model, FPR[1:p] >= 0)
    @variable(model, FNR[1:p] >= 0)
    @variable(model, PR[1:p] >= 0)
    @variable(model, Δ_FPR[1:q] >= 0)
    @variable(model, Δ_FNR[1:q] >= 0)
    @variable(model, Δ_PR[1:q])
    @variable(model, t >= 0)

    # Get objective value for nominal solution
    _, _, min_loss = svm_classifier(X, y; Γ=Γ, return_objective_value=true)

    # Set SVM performance constraints
    @constraint(model, (ones(n)'*ξ)/n + Γ*(w'*w + b*b) <= (1 + δ)*min_loss)
    @constraint(model, y.*(X*w .- b) .>= 1 - ξ)

    # Set constraints for FPR, FNR and PR calculation
    @constraint(model, FPR .== (A'*(ξ .* (y .== -1))) ./ negative_group_size)
    @constraint(model, FNR .== (A'*(ξ .* (y .== 1))) ./ positive_group_size)
    @constraint(model, PR .== (A'* (X*w .- b)) ./ group_size)

    # Set fairness constraints
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_FPR[m] >= FPR[k] - FPR[l])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_FPR[m] >= FPR[l] - FPR[k])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_FNR[m] >= FNR[k] - FNR[l])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_FNR[m] >= FNR[l] - FNR[k])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_PR[m] >= PR[k] - PR[l])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_PR[m] >= PR[l] - PR[k])
    @constraint(model, t .== ones(q)'*(η*(θ*Δ_FPR .+ (1-θ)Δ_FNR) .+ (1-η)*Δ_PR))

    # Set objective
    @objective(model, Min, t)

    # Solve and return optimal objective value or regression coefficients
    solve(model)
    if return_objective_value
        return getvalue(w), getvalue(b), getobjectivevalue(model)
    else
        return getvalue(w), getvalue(b)
    end
end


function fair_svm_classifier(
    X::Array{Float64, 2},
    y::Array{Int64, 1},
    group_assignments::Array{Int64, 1};
    Γ::Float64 = 0.01, # regularization penalty
    δ::Float64 = 0.1, # max percent optimality decrease
    θ::Float64 = 0.5, # FPR (vs FNR) weight
    η::Float64 = 0.75, # Disparate mistreatment (vs disparate impact) weight
    M::Float64 = 1e3, # big M constant
    use_warm_start::Bool = true,
    cap_error_rates::Bool = false,
    return_objective_value::Bool = false,
    logger::Bool = false,
    mip_gap_abs::Float64 = 1e-3,
    solver_time_limit::Int64 = 300,
)::Union{Tuple{Array{Float64, 1}, Float64}, Tuple{Array{Float64, 1}, Float64, Float64}}

    if logger
        start_time = now()
    end

    # Process inputs
    A = get_group_indicator_matrix(group_assignments)
    p = size(A, 2) # number of groups
    q = Int(p*(p - 1)/2) # number of pairs of groups
    n, d = size(X) # number of data points and number of features
    group_size = [sum(A[:,k]) for k=1:p]
    positive_group_size = [sum(A[i,k] for i=1:n if (y[i] == 1)) for k=1:p]
    negative_group_size = [sum(A[i,k] for i=1:n if (y[i] == -1)) for k=1:p]

    # Initialize model
    model = Model(
        solver=GurobiSolver(
            OutputFlag=0,
            TimeLimit=solver_time_limit,
            MIPGapAbs=mip_gap_abs,
            GUROBI_ENV
        )
    )

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
    if cap_error_rates
        @variable(model, FPR_incease[1:p] >= 0)
        @variable(model, FNR_incease[1:p] >= 0)
        @variable(model, error_rate_penalty >= 0)
    end

    # Get parameters and objective value from nominal solution
    w_nominal, b_nominal, min_loss = svm_classifier(X, y; Γ=Γ, return_objective_value=true)

    # Warm start
    if use_warm_start
        w_convex, b_convex = fair_convex_svm_classifier(X, y, group_assignments;
            Γ=Γ, δ=δ, θ=θ, η=η)
        y_hat_convex = svm_predict(X, w_convex, b_convex)
        setvalue.(w, w_convex)
        setvalue.(b, b_convex)
        setvalue.(positive_indicator, (y_hat_convex .== 1))
        setvalue.(error_indicator, (y_hat_convex .== -y))
    end

    # Set SVM performance constraints
    @constraint(model, (ones(n)'*ξ)/n + Γ*(w'*w + b*b) <= (1 + δ)*min_loss)
    @constraint(model, y.*(X*w .- b) .>= 1 - ξ)

    # Set constraints for positive prediction and error indicator variables
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
    @constraint(model, t .== ones(q)'*(η*(θ*Δ_FPR .+ (1-θ)Δ_FNR) .+ (1-η)*Δ_PR))

    # Set error rate cap constraints
    if cap_error_rates
        y_hat_nominal = svm_predict(X, w_nominal, b_nominal)
        FPR_nominal, FNR_nominal, FDR_nominal, FOR_nominal, PR_star = calculate_classification_metrics_by_group(
            y, y_hat_nominal, group_assignments)
        @constraint(model, FPR_incease .>= FPR .- (1 + δ)*FPR_nominal)
        @constraint(model, FNR_incease .>= FNR .- (1 + δ)*FNR_nominal)
        @constraint(model, error_rate_penalty .== 1e6*ones(p)'*(FPR_incease + FNR_incease))
    end

    # Set objective
    if cap_error_rates
        @objective(model, Min, t + error_rate_penalty)
    else
        @objective(model, Min, t)
    end

    # Solve model and log performance if desired
    solve(model)
    if logger
        stop_time = now()
        objective_value = getobjectivevalue(model)
        objective_bound = getobjectivebound(model)
        println("Objective value: ", round(objective_value; digits=2))
        println("Objective bound: ", round(objective_bound; digits=2))
        println("Absolute MIP gap: ", round(objective_value - objective_bound; digits=2))
        println("Relative MIP gap: ", round((objective_value - objective_bound)/objective_bound * 100; digits=2), "%")
        println("Time elapsed: ", round((stop_time - start_time).value/1e3; digits=2), "s")
    end

    # Return optimal objective value or regression coefficients
    if return_objective_value
        return getvalue(w), getvalue(b), getobjectivevalue(model)
    else
        return getvalue(w), getvalue(b)
    end
end

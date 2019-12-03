using JuMP, Gurobi, Dates, LinearAlgebra
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


function lasso_regressor(
    X::Array{Float64, 2},
    y::Array{Float64, 1};
    Γ::Float64=0.01,
    return_objective_value::Bool = false,
    solver_time_limit::Int64=60
)::Union{Array{Float64, 1}, Tuple{Array{Float64, 1}, Float64}}

    # Initialize model
    model = Model(solver=GurobiSolver(OutputFlag=0, TimeLimit=solver_time_limit, GUROBI_ENV))

    # Define variables
    n, d = size(X)
    @variable(model, β[1:d])
    @variable(model, β_abs[1:d])
    @variable(model, t)

    # Set constraints
    @constraint(model, β_abs .>= β)
    @constraint(model, β_abs .>= -β)
    @constraint(model, t .== (y - X*β)'*(y - X*β)/n + Γ*ones(d)'*β_abs)

    # Set objective
    @objective(model, Min, t);

    # Solve and return optimal objective value or regression coefficients
    solve(model)
    if return_objective_value
        return getvalue(β), getobjectivevalue(model)
    else
        return getvalue(β)
    end
end


function fair_lasso_regressor(
    X::Array{Float64, 2},
    y::Array{Float64, 1},
    group_assignments::Array{Int64, 1};
    Γ::Float64=0.01, # regularization penalty
    δ::Float64=0.1, # max percent optimality decrease
    θ::Float64=0.5, # overshoot (vs undershoot) weighting
    λ::Float64=0.01, # disparate impact penalty
    return_objective_value::Bool = false,
    solver_time_limit::Int64=60,
    log::Bool = false
)::Array{Float64, 1}

    start_time = now()

    # Process inputs
    A = get_group_indicator_matrix(group_assignments)
    p = size(A, 2) # number of groups
    q = Int(p*(p - 1)/2) # number of pairs of groups
    n, d = size(X) # number of data points and number of features
    group_size = [sum(A[k,:]) for k=1:p]

    # Get parameters and metrics from nominal solution
    β_nominal, min_loss = lasso_regressor(X, y; Γ=Γ, return_objective_value=true)
    y_hat_nominal = X*β_nominal
    MU_nominal, MO_nominal, MV_nominal = calculate_regression_metrics_by_group(
        y, y_hat_nominal, group_assignments)

    # Initialize model
    model = Model(solver=GurobiSolver(OutputFlag=0, TimeLimit=solver_time_limit, GUROBI_ENV))

    # Define variables
    @variable(model, β[1:d])
    @variable(model, β_abs[1:d])
    @variable(model, undershoot[1:n] >= 0)
    @variable(model, overshoot[1:n] >= 0)
    @variable(model, MU[1:p] >= 0)
    @variable(model, MO[1:p] >= 0)
    @variable(model, MV[1:p])
    @variable(model, Δ_MO[1:q] >= 0)
    @variable(model, Δ_MU[1:q] >= 0)
    @variable(model, Δ_MV[1:q] >= 0)
    @variable(model, t >= 0)

    # Set LASSO regression performance constraints
    @constraint(model, β_abs .>= β)
    @constraint(model, β_abs .>= -β)
    @constraint(model, (y - X*β)'*(y - X*β)/n + Γ*ones(d)'*β_abs .<= (1 + δ)*min_loss)

    # Set constraints for overshoot and undershoot variables
    @constraint(model, overshoot .>= X*β - y)
    @constraint(model, undershoot .>= y - X*β)

    # Set constraints for mean overshoot, mean undershoot and mean value calculation
    @constraint(model, MU .== (A'*undershoot)./group_size)
    @constraint(model, MO .== (A'*overshoot)./group_size)
    @constraint(model, MV .== (A'*X*β)./group_size)

    # Set fairness constraints
    #@constraint(model, MU .<= (1 + δ)*MU_nominal)
    #@constraint(model, MO .<= (1 + δ)*MO_nominal)
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_MU[m] >= MU[k] - MU[l])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_MU[m] >= MU[l] - MU[k])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_MO[m] >= MO[k] - MO[l])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_MO[m] >= MO[l] - MO[k])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_MV[m] >= MV[k] - MV[l])
    @constraint(model, [k=1:p, l=k+1:p, m=Int(l-k+(p-k/2)*(k-1))], Δ_MV[m] >= MV[l] - MV[k])
    @constraint(model, t .== ones(q)'*(θ*Δ_MU .+ (1-θ)*Δ_MO .+ λ*Δ_MV))

    # Set objective
    @objective(model, Min, t)

    # Solve model
    solve(model)

    # Return optimal objective value or regression coefficients
    if return_objective_value
        return getobjectivevalue(model)
    else
        return getvalue(β)
    end
end

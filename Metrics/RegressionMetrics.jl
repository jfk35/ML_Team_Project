using Statistics

function calculate_mean_overshoot(
    y::Array{Float64, 1},
    y_hat::Array{Float64, 1}
)::Float64
    return sum(max.(y_hat .- y, 0)) / size(y, 1)
end


function calculate_mean_undershoot(
    y::Array{Float64, 1},
    y_hat::Array{Float64, 1}
)::Float64
    return sum(max.(y .- y_hat, 0)) / size(y, 1)
end


function calculate_mean_value(
    y::Array{Float64, 1},
    y_hat::Array{Float64, 1}
)::Float64
    return sum(y_hat) / size(y, 1)
end


function calculate_MAE(
    y::Array{Float64, 1},
    y_hat::Array{Float64, 1}
)::Float64
    return sum(abs.(y .- y_hat)) / size(y, 1)
end


function calculate_R2(
    y::Array{Float64, 1},
    y_hat::Array{Float64, 1},
)::Float64
    SSE = sum((y .- y_hat).^2)
    SST = sum((y .- mean(y)).^2)
    return  1 - SSE/SST
end


function calculate_fairness_score_for_metric(
    y::Array{Float64, 1},
    y_hat::Array{Float64, 1},
    group_assignments::Array{Int64, 1},
    metric::Function
)::Float64
    group_metrics = [metric(y[group_assignments .== k], y_hat[group_assignments .== k]) for k=1:p]
    return sum([sum([abs(group_metrics[k] - group_metrics[l]) for l=k+1:p]) for k=1:p]) / (p*(p - 1)/2)
end


function calculate_fairness_scores(
    y::Array{Float64, 1},
    y_hat::Array{Float64, 1},
    group_assignments::Array{Int64, 1}
)::Tuple{Float64, Float64, Float64}
    Δ_MU = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_mean_undershoot)
    Δ_MO = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_mean_overshoot)
    Δ_MV = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_mean_value)
    return Δ_MU, Δ_MO, Δ_MV
end


function regressor_fairness_summary(
    X::Array{Float64, 2},
    y::Array{Float64, 1},
    group_assignments::Array{Int64, 1},
    β::Array{Float64, 1};
    logger::Bool = false
)::Dict{Symbol, Float64}
    y_hat = X*β
    Δ_MU, Δ_MO, Δ_MV = calculate_fairness_scores(y, y_hat, group_assignments)
    if logger
        println("Mean undershoot discrepency: ", round(Δ_MU; digits=3))
        println("Mean overshoot discrepency: ", round(Δ_MO; digits=3))
        println("Mean value discrepency: ", round(Δ_MV; digits=3))
    end
    return Dict(
        :undershoot_discrepency => Δ_MU,
        :over_shoot_discrepency => Δ_MO,
        :mean_value_discrepency => Δ_MV
    )
end


function regressor_performance_summary(
    X::Array{Float64, 2},
    y::Array{Float64, 1},
    group_assignments::Array{Int64, 1},
    β::Array{Float64, 1};
    logger::Bool = false
)::Dict{Symbol, Float64}
    y_hat = X*β
    MAE = calculate_MAE(y, y_hat)
    R2 = calculate_R2(y, y_hat)
    if logger
        println("MAE: ", round(MAE; digits=3))
        println("R2: ", round(R2; digits=3))
    end
    return Dict(
        :MAE => MAE,
        :R2 => R2
    )
end

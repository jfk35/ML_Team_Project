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


function calculate_false_discovery_rate(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1}
)::Float64
    return sum(y[y_hat .== 1] .== -1) / sum(y_hat .== 1)
end


function calculate_false_ommission_rate(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1}
)::Float64
    return sum(y[y_hat .== -1] .== 1) / sum(y_hat .== -1)
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
)::Tuple{Float64, Float64, Float64, Float64, Float64}
    Δ_FPR = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_false_positive_rate)
    Δ_FNR = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_false_negative_rate)
    Δ_FDR = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_false_discovery_rate)
    Δ_FOR = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_false_ommission_rate)
    Δ_PR = calculate_fairness_score_for_metric(y, y_hat, group_assignments, calculate_positive_rate)
    return Δ_FPR, Δ_FNR, Δ_FDR, Δ_FOR, Δ_PR
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
    Δ_FPR, Δ_FNR, Δ_FDR, Δ_FOR, Δ_PR = calculate_fairness_scores(y, y_hat, group_assignments)
    println("False positive rate discrepency: ", round(Δ_FPR; digits=3))
    println("False negative rate discrepency: ", round(Δ_FNR; digits=3))
    println("False discovery rate discrepency: ", round(Δ_FDR; digits=3))
    println("False omission rate discrepency: ", round(Δ_FOR; digits=3))
    println("Positive rate discrepency: ", round(Δ_PR; digits=3))
end

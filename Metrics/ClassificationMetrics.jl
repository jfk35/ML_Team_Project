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


function calculate_svm_tpr_and_fpr(
    X::Array{Float64, 2},
    y::Array{Int64, 1},
    w::Array{Float64, 1},
    b::Float64
)::Tuple{Array{Float64, 1}, Array{Float64, 1}}
    scores = X*w .- b
    thresholds = sort(unique(scores), rev=true)
    n = size(thresholds, 1)
    TPR = zeros(n)
    FPR = zeros(n)
    for i=1:n
        y_hat =  2*(scores .>= thresholds[i]) .- 1
        TPR[i] = sum((y_hat .== 1) .& (y .== 1))/ sum(y .== 1)
        FPR[i] = 1 - sum((y_hat .== -1) .& (y .== -1))/ sum(y .== -1)
    end
    return TPR, FPR
end


function calculate_svm_auc(
    X::Array{Float64, 2},
    y::Array{Int64, 1},
    w::Array{Float64, 1},
    b::Float64
)::Float64
    TPR, FPR = calculate_svm_tpr_and_fpr(X, y, w, b)
    return 0.5 * sum((TPR[2:end] .+ TPR[1:end-1]) .* (FPR[2:end] .- FPR[1:end-1]))
end


function calculate_accuracy(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1}
)::Float64
    return sum(y .== y_hat) / size(y, 1)
end


function calculate_metric_by_group(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1},
    group_assignments::Array{Int64, 1},
    metric::Function
)::Array{Float64, 1}
    p = size(unique(group_assignments), 1)
    return [metric(y[group_assignments .== k], y_hat[group_assignments .== k]) for k=1:p]
end


function calculate_fairness_score_for_metric(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1},
    group_assignments::Array{Int64, 1},
    metric::Function
)::Float64
    group_metrics = calculate_metric_by_group(y, y_hat, group_assignments, metric)
    return sum([sum([abs(group_metrics[k] - group_metrics[l]) for l=k+1:p]) for k=1:p]) / (p*(p - 1)/2)
end


function calculate_classification_metrics_by_group(
    y::Array{Int64, 1},
    y_hat::Array{Int64, 1},
    group_assignments::Array{Int64, 1}
)::Tuple{Array{Float64, 1}, Array{Float64, 1}, Array{Float64, 1}, Array{Float64, 1}, Array{Float64, 1}}
    FPR = calculate_metric_by_group(y, y_hat, group_assignments, calculate_false_positive_rate)
    FNR = calculate_metric_by_group(y, y_hat, group_assignments, calculate_false_negative_rate)
    FDR = calculate_metric_by_group(y, y_hat, group_assignments, calculate_false_discovery_rate)
    FOR = calculate_metric_by_group(y, y_hat, group_assignments, calculate_false_ommission_rate)
    PR = calculate_metric_by_group(y, y_hat, group_assignments, calculate_false_positive_rate)
    return FPR, FNR, FDR, FOR, PR
end


function calculate_classification_fairness_scores(
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
    b::Float64;
    logger::Bool = false
)::Dict{Symbol, Float64}
    y_hat = svm_predict(X, w, b)
    Δ_FPR, Δ_FNR, Δ_FDR, Δ_FOR, Δ_PR = calculate_classification_fairness_scores(y, y_hat, group_assignments)
    if logger
        println("False positive rate discrepency: ", round(Δ_FPR; digits=3))
        println("False negative rate discrepency: ", round(Δ_FNR; digits=3))
        println("False discovery rate discrepency: ", round(Δ_FDR; digits=3))
        println("False omission rate discrepency: ", round(Δ_FOR; digits=3))
        println("Positive rate discrepency: ", round(Δ_PR; digits=3))
    end
    return Dict(
        :FPR_discrepency => Δ_FPR,
        :TPR_discrepency => Δ_FNR,
        :FDR_discrepency => Δ_FDR,
        :FOR_discrepency => Δ_FOR,
        :PR_discrepency => Δ_PR
    )
end


function svm_performance_summary(
    X::Array{Float64, 2},
    y::Array{Int64, 1},
    group_assignments::Array{Int64, 1},
    w::Array{Float64, 1},
    b::Float64;
    logger::Bool = false
)::Dict{Symbol, Float64}
    accuracy = calculate_accuracy(y, svm_predict(X, w, b))
    AUC = calculate_svm_auc(X, y, w, b)
    if logger
        println("Accuracy: ", round(accuracy; digits=3))
        println("AUC: ", round(AUC; digits=3))
    end
    return Dict(
        :accuracy => accuracy,
        :AUC => AUC
    )
end

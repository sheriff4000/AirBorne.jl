module ARIMA

using LinearAlgebra
using ToeplitzMatrices
using Random
using Distributions
using .Combine

"""
This function applies differencing to a time series

Arguments:
- s::Vector{Real}: The time series to difference
- d::int: The order of differencing

Returns:
- differenced_series::Vector{Real}: The differenced time series
- init_vals::Vector{Real}: The initial values used for differencing
"""
function apply_differencing(s, d=1)
    differenced_series = s
    init_vals = []
    for i in 1:d
        push!(init_vals, differenced_series[1])
        differenced_series = diff(differenced_series)
    end
    return differenced_series, init_vals
end

"""
This function undoes differencing to a time series

Arguments:
- s::Vector{Real}: The time series to restore
- inits::Vector{Real}: The initial values used for differencing
- d::int: The order of differencing

Returns:
- undifferenced_series::Vector{Real}: The restored time series
"""
function undo_differencing(s, inits, d=1)
    # assert(d == length(inits), "The number of initial values should be equal to the differencing order")
    undifferenced_series = s
    # println(cumsum(undifferenced_series) .+ 6)
    for i in 1:d
        current_init = pop!(inits) # pop the last element
        undifferenced_series = [current_init; cumsum(undifferenced_series) .+ current_init]
    end
    return undifferenced_series
end

"""
This function creates an arima prediction of F steps into the future

Arguments:
- s::Vector{Real}: The time series to base the ARIMA model on
- p::int: The order of the autoregressive part (p >= 0)
- d::int: The order of differencing
- q::int: The order of the moving average part (q = 0 -> AR model, q > 0 -> ARMA model)

Returns:
- autocorr::Vector{Real}: The autocorrelation of the time series
"""

function arima(series, p::Int, d::Int, q::Int; F::Int=1)
    # Apply differencing
    differenced_series, inits = apply_differencing(series, d)
    ar_predictions = zeros(F)
    # Calculate the autocorrelation matrix
    autocorr = autocor(differenced_series)
    # Create the Toeplitz matrix
    R_matrix = Toeplitz(autocorr[1:p], autocorr[1:p])
    r_vector = autocorr[2:(p + 1)]
    # ar_coeffs = []

    # # Solve the Yule-Walker equations
    ar_coeffs = R_matrix \ r_vector
    if p > 0
        for i in 1:F
            coeffs = R_matrix \ autocorr[(1 + i):(p + i)]
            ar_predictions[i] = sum(
                differenced_series[(end - p + 1):end] .* reverse(coeffs)
            )
        end
    end

    ma_predictions = zeros(F)
    if q > 0
        sample_var = var(diff(differenced_series))
        # println("sample_var differenced: ", var(diff(differenced_series)))
        # println("sample_var: ", sample_var)
        ma_var =
            (1 / (length(differenced_series) - p - 1)) * sum(
                (
                    differenced_series[i + p + 1] -
                    sum(differenced_series[(i + 1):(i + p)] .* reverse(ar_coeffs))
                )^2 for i in 1:(length(differenced_series) - p - 1)
            )
        # ma_var = sample_var * ma_coeff
        ma_samples = rand(Normal(0, sqrt(ma_var / (length(differenced_series) - p - 1))), F)
        ma_predictions = ma_samples
    end
    # Forecast
    forecast = ar_predictions .+ ma_predictions
    # Undo differencing
    restored_series = undo_differencing([differenced_series; forecast], inits, d)
    return restored_series[(end - F + 1):end]
end

"""
This function returns a forecaster object that can be used with the Combine module

Arguments:
- p::int: The number of lags to consider
- d::int: The order of differencing
- q::int: Moving average order

Returns:
- Combined::Combine.Forecaster: The forecaster
"""

function ArimaForecaster(p::Int, d::Int, q::Int)
    return Combine.Forecaster(arima, [p; d; q])
end

end

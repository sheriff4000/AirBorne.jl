module ARIMA

using LinearAlgebra
using ToeplitzMatrices
using Random
using Distributions
using StatsBase

using ..Forecast: Forecaster

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

function arima(series, p::Int, d::Int, q::Int, reparameterise_window::Int=0; F::Int=1)
    # Apply differencing
    if reparameterise_window > length(series)
        reparameterise_window = length(series)
    end
    all_data =
        reparameterise_window == 0 ? series : series[(end - reparameterise_window + 1):end]
    differenced_series, inits = apply_differencing(all_data, d)
    ar_predictions = zeros(F)
    # Calculate the autocorrelation matrix
    autocorr = autocor(differenced_series)
    # Create the Toeplitz matrix]
    R_matrix = SymmetricToeplitz(autocorr[1:p])
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
        dist = Normal(0, sqrt(ma_var / (length(differenced_series) - p - 1)))
        ma_samples = rand(dist, F)
        ma_predictions = ma_samples
    end
    # Forecast
    forecast = ar_predictions .+ ma_predictions
    # Undo differencing
    restored_series = undo_differencing([differenced_series; forecast], inits, d)
    return restored_series[(end - F + 1):end]
end

struct ArimaForecaster <: Forecaster
    p::Int
    d::Int
    q::Int
    reparameterise_window::Int
    function ArimaForecaster(p::Int, d::Int, q::Int; reparameterise_window::Int=0)
        return new(p, d, q, reparameterise_window)
    end
end

function applyForecast(forecaster::ArimaForecaster, series::Vector{<:Real}; F::Int=1)
    return arima(
        series,
        forecaster.p,
        forecaster.d,
        forecaster.q,
        forecaster.reparameterise_window;
        F=F,
    )
end

export ArimaForecaster

end

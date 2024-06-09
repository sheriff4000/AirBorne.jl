"""
This module centralizes the forecast functions used in the Backtest process.
"""
module Forecast
"""
This struct represents a general forecaster that can be combined with others

Can apply forecast to data with the following signature:
forecastFunction(data::Vector{Real}, params::Real...; F::Int=1) -> Vector{Real}
"""
struct Forecaster
	forecastFunction::Function
	params::Vector{Any}
end

"""
This function applies a forecast to a dataset

Arguments:
- forecaster::Forecaster: The forecaster to use
- data::Vector{Real}: The data to forecast
- F::Int: The number of future values to forecast

Returns:
- forecast::Vector{Real}: The forecast
"""
function applyForecast(forecaster::Forecaster, data; F = 1)
	return forecaster.forecastFunction(data, forecaster.params...; F = F)
end

include("./forecast/Combine.jl") # Combine module must come first
include("./forecast/Linear.jl")
include("./forecast/ARIMA.jl")
end

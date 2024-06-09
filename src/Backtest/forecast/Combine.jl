module Combine

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
This struct represents a combined forecaster that is a linear combination of other forecasters
"""
struct CombinedForecaster <: Forecaster
	forecasters::Vector{<:Forecaster}
	weights::Vector{<:Real}
	CombinedForecaster(forecasters::Vector{Forecaster}, weights::Vector{<:Real}) = length(forecasters) == length(weights) ? new(forecasters, weights) : throw(ArgumentError("Length of forecasters and weights must be equal"))
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

"""
This function applies a combined forecaster to a dataset
"""

function applyForecast(forecaster::CombinedForecaster, data; F = 1)
	forecast = zeros(F)
	for (forecaster, weight) in zip(forecaster.forecasters, forecaster.weights)
		forecast = forecast .+ applyForecast(forecaster, data; F = F) .* weight
	end
	return forecast
end

end

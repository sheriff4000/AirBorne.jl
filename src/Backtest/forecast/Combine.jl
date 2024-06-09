module Combine

"""
This struct represents a combined forecaster that is a linear combination of other forecasters
"""
struct CombinedForecaster <: Forecaster
	forecasters::Vector{<:Forecaster}
	weights::Vector{<:Real}
	CombinedForecaster(forecasters::Vector{Forecaster}, weights::Vector{<:Real}) = length(forecasters) == length(weights) ? new(forecasters, weights) : throw(ArgumentError("Length of forecasters and weights must be equal"))
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

module Combine

"""
This struct represents a forecaster that can be combined with others
"""
struct Forecaster
	forecastFunction::Function
	params::Vector{Real}
end

"""
This function combines the forecasts of multiple forecasters

Arguments:
- forecasters::Vector{(Forecaster, Real)}: The forecasters to combine and their associated weights
- data::Vector{Real}: The data to forecast
- F::Int: The number of future values to forecast

Returns:
- combinedForecast::Vector{Real}: The combined forecast
"""
function combineForecasts(forecasters::Vector{(Forecaster, Real)}, data::Vector{Real}; F::Int = 1)
	combinedForecast = zeros(F)
	for (forecaster, weight) in forecasters
		forecast = forecaster.forecastFunction(data, forecaster.params...; F = F)
		combinedForecast += forecast .* weight
	end
	return combinedForecast
end

end

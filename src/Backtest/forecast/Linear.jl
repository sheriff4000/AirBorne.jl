module Linear

using DataFrames
using RollingFunctions
using ..Combine

"""
This function returns the parameters of a linear regression model that predicts 
the next `lookahead` values from a given dataset

Arguments:
- data::Vector{Real}: The dataset to use
- lookback::int The number of previous values to consider
- lookahead::int The number of future values to predict

Returns:
- params::AbstractArray{Real, 2} A vector of parameters for the linear regression model of shape (lookback, lookahead)
"""

function AutoRegression(data::Vector{<:Real}, lookback::Int; F::Int = 1)
	num_points = length(data) - lookback - F + 1
	inputs = zeros(num_points, lookback)
	outputs = zeros(num_points, F)
	for i in 1:num_points
		inputs[i, :] = (data[i:(i+lookback-1)])
		outputs[i, :] = data[(i+lookback):(i+lookback+F-1)]
	end
	params = inputs \ outputs
	return params
end

"""
This function returns the forecasted values of a linear regression model that predicts
the next `lookahead` values from a given dataset

Arguments:
- data::Vector{Real}: The dataset to use
- lookback::int The number of previous values to consider
- lookahead::int The number of future values to predict
- reparameterise_window::int The number of points to use for reparameterisation

Returns:
- forecast::Vector{Real} The forecasted values
"""

function AutoRegressionForecast(
	data::Vector{<:Real}, lookback::Int, reparameterise_window::Int = 0; F::Int = 1,
)
	if reparameterise_window > length(data)
		println(
			"data must have more points than lookback + lookahead, reparameterise_window set to maximum value",
		)
		reparameterise_window = length(data)
	end
	all_data =
		reparameterise_window == 0 ? data : data[(end-reparameterise_window+1):end]
	params = AutoRegression(all_data, lookback; F = F)
	return data[(end-lookback+1):end]' * params
end

"""
This struct represents a linear forecaster that uses the AutoRegressionForecast function
"""

struct LinearForecaster <: Forecaster
	lookback::Int
	reparameterise_window::Int
	LinearForecaster(lookback::Int; reparameterise_window::Int = 0) = new(lookback, reparameterise_window)
end

function applyForecast(forecaster::LinearForecaster, data::Vector{<:Real}; F::Int = 1)
	return AutoRegressionForecast(data, forecaster.lookback, forecaster.reparameterise_window; F = F)
end


end

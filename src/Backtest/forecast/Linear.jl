using DataFrames
using RollingFunctions

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

function AutoRegression(data::Vector{<:Real}, lookback::Int, lookahead::Int)
    num_points = length(data) - lookback - lookahead + 1
    inputs = zeros(num_points, lookback)
    outputs = zeros(num_points, lookahead)
    for i  in 1:num_points
        inputs[i,:] = (data[i:i+lookback-1])
        outputs[i,:] = data[i+lookback:i+lookback+lookahead-1]
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

function AutoRegressionForecast(data::Vector{<:Real}, lookback::Int, lookahead::Int; reparameterise_window::Int = 0)
    @assert length(data) > reparameterise_window "data must have more points than lookback + lookahead"
    all_data = reparameterise_window == 0 ? data : data[end-reparameterise_window+1:end]
    params = AutoRegression(all_data, lookback, lookahead)
    return data[end-lookback+1:end]' * params
end



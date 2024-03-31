using DataFrames

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
    forecast = reshape(last(data, lookback), (1, lookback)) * params    
    return forecast
end

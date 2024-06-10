module Combine
using AirBorne.Forecast: Forecaster, applyForecast as applyForecaster

"""
This struct represents a combined forecaster that is a linear combination of other forecasters
"""
struct CombinedForecaster <: Forecaster
    forecasters::Vector{T} where {T<:Forecaster}
    weights::Vector{U} where {U<:Real}
    function CombinedForecaster(forecasters::Vector{Forecaster}, weights::Vector{<:Real})
        return if length(forecasters) == length(weights)
            new(forecasters, weights)
        else
            throw(ArgumentError("Length of forecasters and weights must be equal"))
        end
    end
end

"""
This function applies a combined forecaster to a dataset
"""
function applyForecast(forecaster::CombinedForecaster, data::Vector{<:Real}; F=1)
    forecast = zeros(1, F)
    for (forecaster, weight) in zip(forecaster.forecasters, forecaster.weights)
        tmp = applyForecaster(forecaster, data; F=F) .* weight
        forecast = forecast .+ reshape(tmp, (1, F))
    end
    return forecast
end

export CombinedForecaster

end

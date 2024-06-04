module Combine

struct Forecaster
    forecastFunction::Function
    params::Vector{Real}
end

function combineForecasts(forecasters::Vector{(Forecaster, Real)}, data::Vector{Real}; F::Int = 1)
    combinedForecast = zeros(F)
    for (forecaster, weight) in forecasters
        forecast = forecaster.forecastFunction(data, forecaster.params...; F = F)
        combinedForecast += forecast .* weight
    end
    return combinedForecast
end

end
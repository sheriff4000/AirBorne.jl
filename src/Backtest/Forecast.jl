"""
This module centralizes the forecast functions used in the Backtest process.
"""
module Forecast
"""
This struct represents a general forecaster that can be combined with others

Can apply forecast to data with the following signature:
forecastFunction(data::Vector{Real}, params::Real...; F::Int=1) -> Vector{Real}
"""
abstract type Forecaster end

struct BaseForecaster <: Forecaster
    forecastFunction::Function
    params::Vector{Any}
end

export Forecaster, BaseForecaster
"""
This function applies a forecast to a dataset

Arguments:
- forecaster::BaseForecaster: The forecaster to use
- data::Vector{Real}: The data to forecast
- F::Int: The number of future values to forecast

Returns:
- forecast::Vector{Real}: The forecast
"""
function applyForecast(forecaster::BaseForecaster, data; F=1)
    return forecaster.forecastFunction(data, forecaster.params...; F=F)
end

################################################################################
# PURE FORECASTERS #
################################################################################

include("./forecast/Linear.jl")
include("./forecast/ARIMA.jl")

using .Linear: LinearForecaster, applyForecast as applyLinear
using .ARIMA: ArimaForecaster, applyForecast as applyARIMA

"""
These are wrapper functions for the applyForecast function that allow for the use of the specific forecasters
"""

function applyForecast(forecaster::LinearForecaster, data::Vector{<:Real}; F::Int=1)
    return applyLinear(forecaster, data; F=F)
end

function applyForecast(forecaster::ArimaForecaster, data::Vector{<:Real}; F::Int=1)
    return applyARIMA(forecaster, data; F=F)
end

export applyForecast, LinearForecaster, ArimaForecaster

################################################################################
# COMBINED FORECASTERS - those which require definitions of pure forecasters
################################################################################

include("./forecast/Combine.jl")
using .Combine: CombinedForecaster, applyForecast as applyCombine

function applyForecast(forecaster::CombinedForecaster, data::Vector{<:Real}; F::Int=1)
    return applyCombine(forecaster, data; F=F)
end

export applyForecast, CombinedForecaster

end

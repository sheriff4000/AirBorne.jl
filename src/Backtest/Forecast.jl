"""
This module centralizes the forecast functions used in the Backtest process.
"""
module Forecast
include("./forecast/Linear.jl")
include("./forecast/ARIMA.jl")
include("./forecast/Combine.jl")
end
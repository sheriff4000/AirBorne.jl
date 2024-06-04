"""
This module centralizes the forecast functions used in the Backtest process.
"""
module Forecast
include("./forecast/Combine.jl") # Combine module must come first
include("./forecast/Linear.jl")
include("./forecast/ARIMA.jl")
end
module AirBorne
# Order of includes is important. Submodules that have dependencies 
# should be included first.
include("./utils/utils.jl") # Common functions
include("./utils/FM.jl") # Financial Models and Types

include("./ETL/ETL.jl") # Retrieval and pre-processing of data

include("./Backtest/Structures.jl") # Structures for Backtesting, may use Financial Models.
include("./Backtest/Engines.jl")
include("./Backtest/Markets.jl")
include("./Backtest/Strategies.jl")
include("./Backtest/Forecast.jl")
end

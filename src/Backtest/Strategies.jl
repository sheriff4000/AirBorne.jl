"""
    Strategies

    This module centralizes access to different trading strategy templates.

"""
module Strategies
include("./strategies/MeanVarianceMPC.jl")
include("./strategies/Markowitz.jl")
include("./strategies/SMA.jl")
include("./strategies/FALM.jl")
end

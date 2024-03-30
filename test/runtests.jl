using Test
using TestSetExtensions

@testset ExtendedTestSet "All the tests" begin
    include("./AssetValuation.jl")
    include("./backtest_Basic.jl")
    include("./backtest_Advanced.jl")
    include("./backtest_MPC.jl")
    include("./cache.jl")
    include("./FM.jl")
    include("./helloWorlds.jl")
    include("./sources.jl")
    include("./StaticMarket.jl")
    include("./utils.jl")
end

using Test
using AirBorne.Forecast: AutoRegression

@testset "Forecast" begin

    @testset "AutoRegression" begin
        data = [1,2,3,4,5,6,7,8,9,10]
        lookback = 2
        lookahead = 2
        @test AutoRegression(data, 2, 2) ≈ [11.0 12.0]
    end
end

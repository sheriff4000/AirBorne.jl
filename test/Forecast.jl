using Test
using AirBorne.Forecast.Linear: AutoRegressionForecast, LinearForecaster
using AirBorne.Forecast.ARIMA: arima, ArimaForecaster
using AirBorne.Forecast.Combine: combineForecasts, applyForecast
using Random

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

@testset "Forecast" begin
    @testset "AutoRegression" begin
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        lookback = 2
        lookahead = 2
        @test AutoRegressionForecast(data, 2, 0; F=2) ≈ [11.0 12.0]
        @test applyForecast(LinearForecaster(2; reparameterise_window = 0), data; F=2) ≈ [11.0 12.0]
    end
    @testset "ARIMA" begin
        data = rand(100)
        p = 2
        d = 1
        q = 1
        @test !isnothing(arima(data, p, d, q, 0; F=2))
        @test !isnothing(applyForecast(ArimaForecaster(2, 1, 1; reparameterise_window = 0), data; F=2))
    end
    @testset "Combine" begin
        data = rand(100)
        linear = LinearForecaster(2; reparameterise_window = 0)
        arima = ArimaForecaster(2, 1, 1; reparameterise_window = 0)
        forecaster = combineForecasts([(linear, 0.5), (arima, 0.5)])
        @test !isnothing(applyForecast(forecaster, data; F=2))
    end
end

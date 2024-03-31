include("./AirBorne.jl")
using AirBorne.Forecast: AutoRegression

data = [1,2,3,4,5,6,7,8,9,10]
lookback = 3
lookahead = 2
print(AutoRegression(data, lookback, lookahead))
# Expected output:

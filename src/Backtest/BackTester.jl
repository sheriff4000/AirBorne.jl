using Structure: summarizePerformance
using AirBorne.ETL.YFinance: get_interday_data, get_chart_data, parse_intraday_raw_data
using Dates
using Dates: Date, DateTime


struct BackTester
    data::Vector{<:Real}
    tickers::Vector{String}
    assetIDs::Vector{String}
end

function getdata(tickers, start, finish, freq)
    if freq == "1d" 
        return get_interday_data(tickers, start, finish)
    end
    stocks = DataFrames.DataFrame()
    for t in tickers
        data = parse_intraday_raw_data(get_chart_data(t, start, finish, freq))
        stocks = DataFrames.vcat(stocks, data)
    end
    return stocks
end

"""
This function initializes a backtester object

Arguments:
- tickers::Vector{String}: The tickers to use
- start_date::String: The start date of the backtest - dates in the form: "YYYY-MM-DD"
- end_date::String: The end date of the backtest
- freq::String: The frequency of the data

Returns:
- backtester::BackTester: The backtester object
"""
function BackTester(tickers::Vector{String}, start_date::String, end_date::String, freq::String="1d")
    unix(x) = string(round(Int, datetime2unix(DateTime(x))))
    stocks = getdata(tickers, unix(start_date), unix(end_date), freq)
    assetIDs = unique(stocks.assetID)
    return BackTester(data, tickers, assetIDs)
end

"""
    Structures
    
    This module provides data structures defined and used in the Backtest process.

    Facilitating contract enforcement between Engines, Markets and Strategies. 
"""
module Structures

using Base: Base
using DataFrames: DataFrame, leftjoin, innerjoin
using Dates: DateTime, dayofweek, Day
import DotMaps.DotMap as DM
using Statistics: std, mean
using ...AirBorne: Wallet, Portfolio
using ..Utils: lagFill, sortedStructInsert!
using ..ETL.AssetValuation: stockValuation, sharpe, valuePortfolio, returns
using RollingFunctions: rollmean, rollstd, runstd, runmean
using Plots: plot, plot!, mm
using Plots

"""
    TimeEvent

    This structure is used to define events in the simulation. 
    The date attribute is used to determine the order of events in the simulation.
    The type attribute is used to determine the type of event, for example "data_transfer" or "trading_logic".
"""

struct TimeEvent
    date::DateTime
    type::String
end

function Base.length(::TimeEvent)
    return 1
end

"""
    Context definitions for contract between Engines, Markets and Strategies to operate with.
"""
mutable struct ContextTypeA
    eventList::Vector{TimeEvent} # List of events defined by the user
    activeOrders::Vector{Any} # List of orders (keep as Any for agnosticity)
    current_event::TimeEvent
    portfolio::Dict # The market determines the portfolio
    accounts::DM # The market determines the account
    ledger::Vector{Any} # List of transactions
    audit::DM
    extra::DM
    parameters::DM # Parameters that can be fed into an optimization engine
end
# Constructors
function ContextTypeA(event::TimeEvent)
    return ContextTypeA([], [], event, Dict(), DM(), [], DM(), DM(), DM())
end

""" c_get(context::ContextTypeA,key::Union{Symbol,String},default::Any; paramFirst::Bool=true) 

    Returns a value from within the extra or parameters hashmaps of the context object. Use the paramFirst attribute to determine 
    whether the parameters is looked into first or not. 

    This function is particularly useful when a datapoint may come from either "extra" or "parameters", "parameters" tend to contain dynamic hyperparameters
    that can be modified after an optimization routing whilst "extra" contains static hyperparameters, some strategy templates will fetch hyperparameters from either.
"""
function c_get(
    context::ContextTypeA, key::Union{Symbol,String}, default::Any; paramFirst::Bool=true
)
    if paramFirst
        return get(context.parameters, key, get(context.extra, key, default))
    else
        return get(context.extra, key, get(context.parameters, key, default))
    end
end
"""
    nextDay!(context::ContextTypeA;days::Day=Day(1))
"""
function nextDay!(context::ContextTypeA; days::Day=Day(1))
    next_event_date = context.current_event.date + Day(1)
    new_event = TimeEvent(next_event_date, "data_transfer")
    return sortedStructInsert!(context.eventList, new_event, :date)
end

"""
    This context allows for arithmetic operations of portfolio and accounts, such as adding 
    securities by doing portfolio+=security, and or removing Money from an account by doing
    account+=money.
    
    A way to initialize this this context type can look like:

    ```julia 
    using AirBorne.Structures: ContextTypeB, TimeEvent
    using Dates: now
    start_event = TimeEvent(now(),"hi")
    new_context = ContextTypeB(start_event)
    ```

    """
mutable struct ContextTypeB
    eventList::Vector{TimeEvent} # List of events defined by the user
    activeOrders::Vector{Any} # List of orders (keep as Any for agnosticity)
    current_event::TimeEvent
    portfolio::Portfolio # The market determines the portfolio
    accounts::Wallet # The market determines the account
    ledger::Vector{Any} # List of transactions
    audit::DM
    extra::DM
end

# Constructors
function ContextTypeB(event::TimeEvent)
    return ContextTypeB([], [], event, Portfolio(), Wallet(), [], DM(), DM())
end

"""
    summarizePerformance(data::DataFrame, context::ContextTypeA;
    valuationFun::Function=stockValuation,
    removeWeekend::Bool=false,
    keepDaysWithoutData::Bool=true,
    windowSize::Int=5,
    riskFreeRate::Real=0.0
    )

    Given an audit of the portfolio, account, events and OHLCV data it returns a summary of the performance 
    of the portfolio over time.
    
    # Arguments
    - `data::DataFrame`: A dataframe with the data OHLCV_V1 data.
    - `context::ContextTypeA`: Result from running the simulation in DEDS Engine.
    ### Optional keyword arguments
    -`valuationFun::Function=stockValuation`: Stock needs to be valued to establish a notion of returns. This allows to pass custom functions for asset valuation.
    -`removeWeekend::Bool=false`: Many markets close over weekend. If events are kept with lack of activity returns of 0 may be observed.
    -`keepDaysWithoutData::Bool=true`: Without data the most recent  market provided will be used to estimate the value of assets. If events are kept with lack of activity returns of 0 may be observed.
    -`windowSize::Int=5`: Many statistical figures are observed over sliding time windows, this allows to select the size of the timewindows by setting the number of consecutive events considered.
    -`riskFreeRate::Real=0.0`: Sharpe and other metrics rely on a definition of a risk free rate. 
    -`includeAccounts::Bool=true`: By default we assume that the account is not reflected in the portfolio, if accounts are included in the portfolio set *includeAccounts* to false to avoid double counting the value of money in the account.
    
"""
function summarizePerformance(
    OHLCV_data::DataFrame,
    context::ContextTypeA;
    valuationFun::Function=stockValuation,
    removeWeekend::Bool=false,
    keepDaysWithoutData::Bool=true,
    windowSize::Int=5,
    riskFreeRate::Real=0.0,
    includeAccounts::Bool=true,
    plotSummary::Bool=false,
)
    println("Summarizing performance")
    summary = DataFrame(
        "date" => [e.date for e in context.audit.eventHistory],
        "type" => [e.type for e in context.audit.eventHistory],
        "portfolio" => context.audit.portfolioHistory,
        "account" => context.audit.accountHistory,
    )

    joinoperation = keepDaysWithoutData ? leftjoin : innerjoin
    summary = joinoperation(
        summary, valuationFun(OHLCV_data)[!, ["date", "stockValue"]]; on=:date
    )

    sort!(summary, :date)
    if removeWeekend
        summary = summary[dayofweek.(summary.date) .< 6, :]
    end
    summary.stockValue = lagFill(summary.stockValue)
    summary[!, "dollarValue"] = [
        valuePortfolio(r.portfolio, r.stockValue) +
        (includeAccounts ? r.account.usd.balance : 0) for r in eachrow(summary)
    ]
    summary[!, "return"] = returns(summary.dollarValue)
    summary[!, "mean_return"] = runmean(summary[!, "return"], windowSize)
    summary[!, "std_return"] = runstd(summary[!, "return"], windowSize)
    summary[!, "sharpe"] = (summary.mean_return .- riskFreeRate) ./ summary.std_return
    summary[!, "drawdown"] = [
        100 * (summary.dollarValue[i] - maximum(summary.dollarValue[1:i])) /
        maximum(summary.dollarValue[1:i]) for i in 1:length(summary.dollarValue)
    ]
    summary[!, "annual_return"] .=
        (
            summary.dollarValue[end] / summary.dollarValue[1]
        )^(1 / (length(summary.dollarValue) / 252)) - 1
    summary[!, "total_vol"] .= std(summary.return)
    summary[!, "total_sharpe"] .=
        (summary.annual_return .- riskFreeRate) ./ summary.total_vol
    if plotSummary
        plot_summary(summary)
    end
    return summary
end
"""
    print_summary(summary::DataFrame)

    Given a summary of the performance of the portfolio over time, returns plots of the performance 
    summary.
    
    # Arguments
    - `summary::DataFrame`: A dataframe with the summary of the performance of the portfolio over time.
        Must be same format as the output of `summarizePerformance` function.
"""
function plot_summary(summary)
    sharpe = plot(
        summary.date,
        summary.sharpe;
        title="Sharpe Ratio",
        label="value",
        linewidth=1,
        xlabel="date",
        ylabel="Sharpe Ratio",
    )
    rets = plot(
        summary.date,
        summary.return;
        title="Returns",
        label="value",
        linewidth=1,
        xlabel="date",
        ylabel="Returns",
    )
    vol = plot(
        summary.date,
        summary.std_return;
        title="Volatility",
        label="value",
        linewidth=1,
        xlabel="date",
        ylabel="Volatility",
    )
    drawdown = plot(
        summary.date,
        summary.drawdown;
        title="Drawdown",
        label="value",
        linewidth=1,
        xlabel="date",
        ylabel="Drawdown (%)",
    )
    p = plot(
        sharpe,
        rets,
        vol,
        drawdown;
        layout=(2, 2),
        legend=false,
        show=true,
        size=(1000, 800),
        top_margin=10Plots.mm,
        bottom_margin=10Plots.mm,
        left_margin=10Plots.mm,
        right_margin=10Plots.mm,
    )
    println("Annual Return: ", summary.annual_return[end])
    println("Total Volatility: ", summary.total_vol[end])
    println("Total Sharpe Ratio: ", summary.total_sharpe[end])
    return p
end

end

module FALM

using ...Markets.StaticMarket: Order, place_order!, ordersForPortfolioRedistribution
using AirBorne.ETL.AssetValuation: stockValuation
using ...Forecast.Combine: Forecaster, applyForecast
using ...Forecast.Linear: LinearForecaster
using ...Structures: ContextTypeA
using DataFrames: DataFrame
using DirectSearch
using Suppressor

"""
    initialize!

    Template for the initialization procedure, before being passed onto an engine like DEDS a preloaded
    function must be defined so that the initialization function meets the engine requirements.
    
    ```julia
    # Specify custom arguments to tune the behaviour of FALM
    my_initialize!(context,data) = FALM.initialize!(context;...)
    # Or just run with the default parameters
    my_initialize!(context,data) = FALM.trading_logic!(context)
    ```
"""

function falm_initialize!(
    context::ContextTypeA;
    initialCapital::Real=10^5,
    nextEventFun::Union{Function,Nothing}=nothing,
    lookahead::Int=1, # Number of days to forecast
    lpm_order::Float64=2.0, # Order of the LPM
    max_lookback::Int=100, # minimum data points to consider
    tickers::Vector{String}=["^GSPC"], # Tickers to consider
    assetIDs::Union{Vector{String}}=nothing, # Associated AssetIDs for the tickers
    transactionCost::Real=0.02, # Transaction cost
    currency::String="FEX/USD", # Currency to use
    forecastFun::Forecaster=LinearForecaster(1, 0), # Forecasting function
    httype::Int=1, # 1: Weighted Average holding time, 2: Minimum holding time
    min_alloc_threshold::Float64=0.7,
    min_returns_threshold::Float64=0.0002,
)
    context.extra.lookahead = lookahead
    context.extra.lpm_order = lpm_order
    context.extra.max_lookback = max_lookback
    context.extra.htcounter = 0
    context.extra.tickers = sort(tickers)
    context.extra.timecounter = 0
    context.extra.transactionCost = transactionCost
    context.extra.current_prices = Dict()
    context.extra.currentValue = DataFrame()
    context.extra.currency_symbol = currency
    context.extra.assetIDs = assetIDs
    context.extra.forecastFun = forecastFun
    context.extra.httype = httype
    context.extra.min_alloc_threshold = min_alloc_threshold
    context.extra.min_returns_threshold = min_returns_threshold

    ###################################
    ####  Initialise Portfolio  ####
    ###################################
    # context.extra.weights = Dict(t => 0.0 for t in tickers)
    context.extra.desired_weights = Dict(t => 0.0 for t in tickers)

    if !isnothing(assetIDs)
        push!(context.extra.assetIDs, currency)
        [setindex!(context.portfolio, 0.0, n) for n in context.extra.assetIDs] # Initialize an empty portfolio
    end
    ###################################
    ####  Specify Account Balance  ####
    ###################################
    context.accounts.usd = DotMap(Dict())
    context.accounts.usd.balance = initialCapital
    context.accounts.usd.currency = currency

    context.portfolio["FEX/USD"] = initialCapital
    #########################################
    ####  Define first simulation event  ####
    #########################################
    if !(isnothing(nextEventFun))
        nextEventFun(context)
    end
    return nothing
end

"""
    compute_portfolio!

    This function computes the portfolio weights based on the LPM matrix and the forecasted returns

    Arguments:
    - context::ContextTypeA: The context object
    - data::DataFrame: The data to use

    Returns:
    - Nothing
"""

function compute_portfolio!(context::ContextTypeA; data=DataFrame())
    returns = Dict()
    for t in context.extra.tickers
        prices = data[data.symbol .== t, :close][(end - context.extra.max_lookback + 1):end]
        returns[t] = diff(prices) ./ prices[1:(end - 1)]
    end

    #Compute the LPM matrix
    lpm_matrix = zeros(length(context.extra.tickers), length(context.extra.tickers))
    semi_deviations = Dict(
        t => mean(abs.(min.(returns[t], 0)) .^ context.extra.lpm_order)^(1 / 2) for
        t in context.extra.tickers
    )
    for (i, ti) in enumerate(context.extra.tickers)
        for (j, tj) in enumerate(context.extra.tickers)
            dev = semi_deviations[ti] * semi_deviations[tj]
            corr = cor(returns[ti], returns[tj])
            lpm_matrix[i, j] = dev * corr
        end
    end

    #Compute holding time
    holding_times = Dict()
    best_returns = Dict()

    for t in context.extra.tickers
        prices = data[data.symbol .== t, :close]
        # CUSTOM FORECAST ##
        forecast = applyForecast(
            context.extra.forecastFun, prices; F=context.extra.lookahead
        )
        relative_returns = log.(forecast ./ prices[end])
        relative_returns = [relative_returns[i] / i for i in 1:(context.extra.lookahead)]
        holding_times[t] = argmax(collect(Iterators.flatten(relative_returns)))
        best_returns[t] = maximum(relative_returns)
        # CUSTOM FORECAST ##
    end
    s = sum(
        best_returns[t] * context.extra.desired_weights[t] for t in context.extra.tickers
    )
    if s == 0
        holding_time = 0
    else
        holding_time = sum(
            (holding_times[t] * best_returns[t] * context.extra.desired_weights[t]) / s for
            t in context.extra.tickers
        )
    end

    if context.extra.httype == 1
        # Weighted Average Holding time
        context.extra.htcounter = round(Int, holding_time)
    else
        # minimum holding time
        context.extra.htcounter = minimum(values(holding_times))
    end

    #Compute the weights
    dim = length(context.extra.tickers)
    obj(x) = x' * lpm_matrix * x

    ############################
    # constraints #
    ############################
    min_alloc_threshold = 0.7
    max_alloc(x) = sum(x) <= 1.00 #Bound between something and 1
    long_only(x) = all(x .>= 0)
    min_alloc(x) = sum(x) >= context.extra.min_alloc_threshold
    function min_returns(x)
        return sum(best_returns[t] * x[i] for (i, t) in enumerate(context.extra.tickers)) >=
               context.extra.min_returns_threshold
    end
    ############################
    # constraints #
    ###########################

    init_point = [context.extra.desired_weights[t] for t in context.extra.tickers]
    # if init_point is not feasible, use the following
    if !max_alloc(init_point) ||
        !long_only(init_point) ||
        !min_alloc(init_point) ||
        !min_returns(init_point)
        init_point = [
            round(1 / length(context.extra.tickers); digits=3) for
            t in context.extra.tickers
        ]
    end

    weights_problem = DSProblem(
        dim;
        objective=obj,
        granularity=[0.001 for _ in context.extra.tickers],
        initial_point=init_point,
    )

    AddExtremeConstraint(weights_problem, max_alloc)
    AddProgressiveConstraint(weights_problem, min_returns)
    AddProgressiveConstraint(weights_problem, min_alloc)
    AddExtremeConstraint(weights_problem, long_only)
    @suppress Optimize!(weights_problem)

    # Favour the infeasible solution
    if !isnothing(weights_problem.i)
        solution = Dict(
            t => weights_problem.i[j] for (j, t) in enumerate(context.extra.tickers)
        )
    else
        solution = Dict(
            t => weights_problem.x[i] for (i, t) in enumerate(context.extra.tickers)
        )
    end
    context.extra.desired_weights = solution
    return nothing
end

function algo_trading_logic!(
    context::ContextTypeA, data::DataFrame; nextEventFun::Union{Function,Nothing}=nothing
)
    # Only trade when there is enough data
    if context.extra.timecounter < context.extra.max_lookback
        context.extra.timecounter += 1
        return nothing
    end
    # Only trade when the holding time is 0
    if context.extra.htcounter == 0
        compute_portfolio!(context; data=data)

        #Generate orders
        context.extra.currentValue = stockValuation(data)
        assetPricing = context.extra.currentValue[1, "stockValue"]
        assetPricing[context.extra.currency_symbol] = 1.0
        for t in context.extra.tickers
            context.portfolio[data[data.symbol .== t, :assetID][1]] = get(
                context.portfolio, data[data.symbol .== t, :assetID][1], 0.0
            )
        end
        orders = my_ordersForPortfolioRedistribution(
            convert(Dict{String,Float64}, context.portfolio),
            Dict(
                data[data.symbol .== t, :assetID][1] => context.extra.desired_weights[t] for
                t in context.extra.tickers
            ),
            assetPricing;
            account=context.accounts.usd,
            costPropFactor=context.extra.transactionCost,
            costPerTransactionFactor=0.0,
            min_shares_threshold=10^-5,
        )

        [place_order!(context, order) for order in orders]
        return nothing
    else
        context.extra.htcounter -= 1
        return nothing
    end
end

end

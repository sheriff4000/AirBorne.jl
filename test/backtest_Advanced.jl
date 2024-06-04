using AirBorne: AirBorne, Money, Portfolio, Wallet, Security, get_symbol
using AirBorne.ETL.Cache: load_bundle
using AirBorne.Structures: ContextTypeB, TimeEvent, summarizePerformance, nextDay!, c_get
using AirBorne.Markets.StaticMarket:
    addMoneyToAccount!, addSecurityToPortfolio!, execute_orders!, expose_data, keyJE
using AirBorne.Engines.DEDS: run
using AirBorne.Strategies.SMA: interday_initialize!, interday_trading_logic!
using AirBorne.Strategies.Markowitz: Markowitz
# initialize!, trading_logic!

function f(acc, mon)
    return acc[get_symbol(a)] += mon.value * -1
end;
using Dates: DateTime, Day
using Test
using Logging
# Test more sophisticated data structures and strategy templates 
@testset "Backtest: Advanced" begin

    ######################
    ###  ContextTypeB  ###
    ######################
    dummy_event = TimeEvent(DateTime(2018, 1, 1), "example")
    contextB = ContextTypeB(dummy_event) # Initialize ContextTypeB

    @test length(dummy_event) == 1 # Events are atomic, thus have length 1.
    @test typeof(contextB.portfolio) == Portfolio
    @test typeof(contextB.accounts) == Wallet

    # addSecurityToPortfolio!(context.B)
    contextB.accounts += Money(10^5, :USD)
    @test contextB.accounts[:USD] == 10^5
    journal_entry = Dict(
        "exchangeName" => "NMS",
        "ticker" => "AAPL",
        "shares" => 100.0,
        "price" => Money(78.5, :USD),
        "amount" => 100.0 * Money(78.5, :USD),
        "date" => dummy_event.date,
    )
    journal_entry["assetID"] = keyJE(journal_entry)
    addMoneyToAccount!(contextB.accounts, journal_entry)
    addSecurityToPortfolio!(contextB.portfolio, journal_entry)
    vector_portfolio = []
    addSecurityToPortfolio!(vector_portfolio, journal_entry)
    @test size(vector_portfolio) == (1,)
    @test contextB.accounts[:USD] == 92150.0
    @test contextB.portfolio[Symbol(journal_entry["assetID"])] == 100.0

    cache_dir = joinpath(@__DIR__, "assets", "cache")
    data = load_bundle("demo"; cache_dir=cache_dir)
    evaluationEvents = [
        TimeEvent(t, "data_transfer") for t in sort(unique(data.date); rev=true)
    ]
    ######################
    ###  SMA Strategy  ###
    ######################
    simulate_until = DateTime(2019, 2, 1)
    function sma_initialize!(context)
        return interday_initialize!(
            context; longHorizon=20, shortHorizon=5, nextEventFun=nextDay!
        )
    end
    sma_trading_logic!(ctx, dat) = interday_trading_logic!(ctx, dat; nextEventFun=nextDay!)
    contextSMA = run(
        data,
        sma_initialize!,
        sma_trading_logic!,
        execute_orders!,
        expose_data;
        audit=true,
        max_date=DateTime(2019, 2, 1),
    )
    @test size(contextSMA.audit.portfolioHistory) == (760,)
    @test size(
        summarizePerformance(data, contextSMA; removeWeekend=true, plotSummary=true), 1
    ) == 544

    # Thorough testing on c_get
    contextSMA.extra.testFieldA = 20
    contextSMA.parameters.testFieldA = 10

    @test c_get(contextSMA, "testFieldA", nothing; paramFirst=false) == 20 # Look First in Extra
    @test c_get(contextSMA, "testFieldA", nothing) == 10 # Look First in Param  

    contextSMA.parameters.onlyExtraField = 10

    @test c_get(contextSMA, "onlyExtraField", nothing; paramFirst=false) == 10 # Look First in Extra
    @test c_get(contextSMA, "onlyExtraField", nothing) == 10 # Look First in Param  

    contextSMA.parameters.onlyParamField = 10
    @test c_get(contextSMA, "onlyParamField", nothing; paramFirst=false) == 10 # Look First in Extra
    @test c_get(contextSMA, "onlyParamField", nothing) == 10 # Look First in Param  

    @test isnothing(c_get(contextSMA, "made_up_field", nothing))

    ############################
    ###  Markowitz Strategy  ###
    ############################
    my_expose_data(context, data) = expose_data(context, data; historical=false)
    contextMK1 = run(
        data,
        Markowitz.initialize!,
        Markowitz.trading_logic!,
        execute_orders!,
        my_expose_data;
        audit=true,
        max_iter=50,
        initialEvents=evaluationEvents,
    )
    @test size(contextMK1.audit.portfolioHistory) == (51,)
    @test size(summarizePerformance(data, contextMK1; removeWeekend=true), 1) == 51

    c2 = deepcopy(contextMK1) # Make a copy of the context to modify and play with
    c2.extra.returnHistory[!, "NMS/AAPL"] =
        -collect(1:size(c2.extra.returnHistory, 1)) ./ size(c2.extra.returnHistory, 1)
    c2.extra.returnHistory[!, "NMS/GOOG"] =
        -collect(1:size(c2.extra.returnHistory, 1)) ./ size(c2.extra.returnHistory, 1)
    c2.current_event = TimeEvent(c2.current_event.date + Day(1), "test") # Advance Time 
    Markowitz.trading_logic!(c2, my_expose_data(c2, data)) # Test what happens if market is going down
    @test all(c2.extra.idealPortfolioDistribution .== 0)

    MkI2(ctx) = Markowitz.initialize!(ctx; nextEventFun=nextDay!)
    MkTL2(ctx, dat) = Markowitz.trading_logic!(ctx, dat; nextEventFun=nextDay!)
    contextMK2 = run(
        data, MkI2, MkTL2, execute_orders!, my_expose_data; audit=true, max_iter=50
    )

    # Since the data is sensitive to an hour change, there is, purposefully, a mismatch between the portfolio
    # And the data, therefore when summarizing the performance if there is not a perfect match between the
    # time of the data and the evaluation time event, the mismatched events won't be included, 14 in this case.

    @test size(contextMK2.audit.portfolioHistory) == (51,)
    @test size(summarizePerformance(data, contextMK2; removeWeekend=true), 1) == 37
end

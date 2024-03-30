module Utils
export hello_world
using DataFrames: DataFrame, groupby, combine, missing, SubDataFrame
using SparseArrays: sparse, I, blockdiag
"""
	hello_world()

	Returns a string saying "Hello World!".
	"""
function hello_world()
	return "Hello World!"
end

""" 
	rvcat(v): Recursive vcat.

	vcat has limited support for sparce matrices, this function allows to vertically concatenate 
	sparse matrices.
"""
function rvcat(v) # It may not be needed?
	v2 = length(v) > 2 ? rvcat(v[2:end]) : v[2]
	return vcat(v[1], v2)
end

" Recursive block diagonal "
function rblockdiag(v::Union{Vector{Matrix{Union{Float64}}}, Vector{Matrix{Union{Int64}}}})
	v2 = length(v) > 2 ? rblockdiag(v[2:end]) : sparse(v[2])
	return blockdiag(sparse(v[1]), v2)
end

"Kronecker delta"
δ(x, y) = (x == y ? 1 : 0)
δ(x) = (x == 0 ? 1 : 0)

"""
	sortedStructInsert!(v::Vector, x,symbol;rev=true) 

	Inserts a struct into a **sorted** Vector of structs in the right place to keep the array sorted.
	"""
function sortedStructInsert!(v::Vector, x, symbol; rev = true)
	return (splice!(v, searchsorted(v, x; by = v -> getproperty(v, symbol), rev = rev), [x]); v)
end

"""
	sortStruct!(v::Vector, symbol;rev=true) 

	Sorts a struct by a given symbol.
	"""
function sortStruct!(v::Vector, symbol; rev = true)
	return sort!(v; by = v -> getproperty(v, symbol), rev = rev)
end

"""
	deepPush!(list,element)

	Inserts the deepcopy of an element into a collection 
	"""
function deepPush!(list, element)
	return push!(list, deepcopy(element))
end

"""
	get_latest(df,id_symbols,sort_symbol)

	Retrieves last record from a dataframe, sortying by sort_symbol and grouping by id_symbols.

	```julia
	get_latest(past_data,[:exchangeName,:symbol],:date)
	```
"""
function get_latest(df, id_symbols, sort_symbol)
	return combine(groupby(df, id_symbols)) do sdf
		sdf[argmax(sdf[!, sort_symbol]), :]
	end
end

"""
	get_latest_N(sdf::Union{SubDataFrame,DataFrame},by::Symbol,N::Int64; rev=false,fields::Vector=[])

	This function returns a DataFrame with the first N rows of the input dataframe sorted by the column *by* amd the columns specified by *fields*. 
	
	Using the additional parameter, *rev* the sort order gets reversed. 

	Example: get the 5 largest companies in the NASDAQ screener, per sector.
	```julia
		using AirBorne.ETL.NASDAQ: screener
		tickers_df = screener()
		filtered_df =tickers_df[[   x!="" ? parse(Int64, x)<2017 : false for x in tickers_df.ipoyear],["symbol","marketCap","sector"]]
		grouped_df = groupby(filtered_df,"sector")
		f(sdf)= get_latest_N(sdf,:marketCap,5;rev=true, fields = ["symbol", "marketCap"])
		result = combine(gdf,f)
	```
	Another example
	```
	a="A";b="B"

	f2(sdf)= get_latest_N(sdf,:val,3)
	df =DataFrame(Dict(
			"cat"=>[a,a,a,a,a,b,b,b,b,b],
			"val"=>[5,3,2,6,1,4,3,8,6,2],
			 "ix"=>[1,2,3,4,5,6,7,8,9,10],
	))
	combine(groupby(df,"cat"),f2)
	# 6×3 DataFrame
	# Row	cat	    ix	    val
	#       String	Int64	Int64
	# 1     A	    5	    1
	# 2     A	    3	    2
	# 3     A	    2	    3
	# 4     B	    10	    2
	# 5     B	    7	    3
	# 6     B	    6	    4
	```
"""
function get_latest_N(
	sdf::Union{SubDataFrame, DataFrame}, by::Symbol, N::Int64; rev = false, fields::Vector = [],
)
	fields = fields == [] ? names(sdf) : fields
	sorted = sort(sdf, by; rev = rev)
	return DataFrame(Dict([x => first(sorted[!, x], N) for x in fields]))
end

"""
	lagFill(df::DataFrame,col::Symbol; fill::Vector=[missing,nothing]))

	Replaces all missing values in a column of a dataframe for the previous 
	non-missing value.
	### Arguments
	-`inV::Vector`: Input Vector
	-`fill::Vector`: Vector with elements to be filled. I.e., nothing, NaN, missing. By default: [missing,nothing].
"""
function lagFill(inV::Vector; fill::Vector = [missing, nothing])
	out = inV
	trigger_funs = []
	if any(ismissing.(fill))
		deleteat!(fill, findall(x -> ismissing(x), fill))
		push!(trigger_funs, ismissing)
	end
	if any(isnothing.(fill))
		deleteat!(fill, findall(x -> isnothing(x), fill))
		push!(trigger_funs, isnothing)
	end

	for i in 2:length(out)
		if any([f(out[i]) for f in trigger_funs]) || (out[i] ∈ fill)
			out[i] = out[i-1]
		end
	end
	return out
end

"""
	More efficient implementation of a moving average (mean running).
"""
function movingAverage(array::Vector; windowSize::Int = 1, startFrom::Int = 1)
	out = Array{Union{Float64, Nothing}}(undef, length(array)) # Preallocate memory
	start(i) = max(i - windowSize, 1)
	factor(i) = min(windowSize, i + 1 - startFrom)
	sum_value = 0
	for i in startFrom:length(array)
		sum_value += array[i]
		sum_value -= i + 1 - startFrom > windowSize ? array[start(i)] : 0
		out[i] = sum_value / factor(i)
	end
	return out
end

"""
	Given a function (mean, variance, sharpe,...) from an 1-D array to a single element
	It creates an array with same size of original with the function applied from the 
	beginning of the array to the index of the output. 

	!!! Tip "Performance"
		This function is not meant to be highly performant. If a function is used function is
		advised to have a specialized function to calculate its running counterpart.

	# Optional Arguments
	- `windowSize::Union{Int,Nothing}`: If its desired to truncate the number of past elements to be considered, this field can be set to the maximum number of past elements to take into account. This can be used for Moving Averages for example. 
	- `ignoreFirst::Int`: Indicates for how many elements the operation should not be applied. In those elements "nothing" will be placed instead.
"""
function makeRunning(
	array::Vector, fun::Function; windowSize::Union{Int64, Nothing} = nothing, startFrom::Int = 1,
)
	out = Array{Union{Float64, Nothing}}(undef, length(array)) # Preallocate memory
	start(i) = windowSize === nothing ? startFrom : max(i - windowSize + 1, startFrom)
	for i in startFrom:length(array)
		out[i] = fun(array[start(i):i])
	end
	return out
end


end

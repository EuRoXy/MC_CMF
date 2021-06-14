using NetCDF, Statistics, StatsBase, LinearAlgebra
using Plots, Plots.PlotMeasures, StatsPlots
using DataFrames, Distributions

getNCvar(fn, var::String) = dropdims(ncread(fn, var); dims=(1,2,3));

dropnan(ar) = filter(ar -> !isnan(ar), ar);

rd(a::Float64, d) = round(a; digits=d)

#### Data
function getCMF(fn)
    ghi = getNCvar(fn, "GHI")
    ghiCS = getNCvar(fn, "CLEAR_SKY_GHI");

    cmf = ghi ./ ghiCS
    cmf_train = cmf[1:523007]
    cmf_test = cmf[523008:end];
    return dropnan(cmf_train), dropnan(cmf_test)
end

function classify(data, n, op="ep") # divide into n bins
    if op == "el" # bins of equal length
        min = minimum(data)
        max = maximum(data)
        binWidth = (max - min) / n
        binStarts = [min .+ (i-1) * binWidth for i = 1:n]
        state = floor.((data .- min) ./ binWidth)    
        state = Int64.(state)
    elseif op == "ep" # equal population
        binStarts = quantile(data, 0:(1/n):1)
        binWidth = [binStarts[i+1] - binStarts[i] for i in 1:n]
        state = [findlast(data[i] .>= binStarts) for i in 1:length(data)]
    end
    state[state .> 30] .= 30        #.< 1] .= 1
    df = DataFrame(:data=>data, :cls=>state)
    binMean = [mean(groupby(df, :cls)[i].data) for i in 1:n]
    return state, binStarts, binMean 
end

#### Markov Chain
function MCFit(state, od, n)  
    len = length(state)
    T = zeros(n^od, n); # transition maxtrix
    if od == 1
        for i in 2:len
            T[ state[i-1], state[i] ] += 1 # count occurrence
        end
    elseif od == 2
        for i in 2:len-1 
            T[ n*(state[i-1]-1) + state[i], state[i+1] ] += 1
        end
    elseif od == 3
        for i in 2:len-2
            T[ (n^2*(state[i-1]-1) + n*(state[i]-1)) + state[i+1], state[i+2] ] += 1
        end
    end
    # normalize T
    rowSums = sum(T; dims=2)
    T_norm = T ./ rowSums # normalized transition matrix
    return T_norm
end

function MCPredict(T, binStarts, binMean, data_test, od, n; step=1)    # inherit binStarts & binMean from data_train 
    len = length(data_test)
    cls = zeros(Int64, len)
    pred = zeros(len);
    for i in 1:(len-od+1)
        obs = data_test[i:i+od-1]
        if od == 1            
            state = findlast(obs .> binStarts)
        elseif od == 2
            bin1 = findlast(obs[1] .> binStarts)
            bin2 = findlast(obs[2] .> binStarts)
            state = n * (bin1-1) + bin2
        elseif od == 3
            bin1 = findlast(obs[1] .> binStarts)
            bin2 = findlast(obs[2] .> binStarts)
            bin3 = findlast(obs[3] .> binStarts)
            state = n^2 * (bin1-1) + n * (bin2-1) + bin3
        end
        transProbs = T[state, :]
        trans = replace(transProbs, NaN=>0)
        predi = dot(binMean, trans)
        clas = findlast(predi .> binStarts)      
        clas == 0 ? clas = 1 : nothing
        cls[i] = clas
        pred[i] = predi
    end
    return cls, pred
end
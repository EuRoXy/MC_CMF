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
    cmf_train = cmf[1:523007] # 2004 - 2018
    cmf_test = cmf[523008:end]; # 2019
    return dropnan(cmf_train), dropnan(cmf_test)
end

function classify(arr, binStarts)
    min = binStarts[1]
    len = length(arr)
    cls = zeros(Int64, len)
    for i in 1:len
        arr[i] ≤ min ? 
            cls[i] = 1 : 
            cls[i] = findlast(arr[i] .> binStarts)
    end
    return cls
end  

function getBins(data, n; op="ep") # divide into n bins
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
function mcFit(state, od, n)  
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
    T_norm = replace(T_norm, NaN=>0)
    return T_norm
end

function predict_od(obs, od, n, T) # input obs of od 1, 2, 3
    bin1 = classify(obs[1], binStarts)[1]
    state = bin1
    if od > 1           
        bin2 = classify(obs[2], binStarts)[1]
        state = n * (bin1-1) + bin2
        if od > 2
            bin3 = classify(obs[3], binStarts)[1]
            state = n^2 * (bin1-1) + n * (bin2-1) + bin3
        end
    end
    prob = T[state, :]
    pred = dot(binMean, prob)
    return pred
end

function mcPredict(data_test, od, n, T, binStarts) # normal pred
    len = length(data_test)
    pred = zeros(len-od+1)
    for i in 1:(len-od+1)
        obs = data_test[i:i+od-1]
        pred[i] = predict_od(obs, od, n, T)
    end
    cls = classify(pred, binStarts)
#     cls = replace(cls, nothing=>1)
    return pred, cls
end

#### predict from previous prediction
function predict_steps_1d(T, binStarts, data_test, od, n; steps=1)     
    len = length(data_test)
    pred = zeros(len-od-steps+1) 
    T, binStarts = T, binStarts
    for i in 1:length(pred) 
        obs1 = data_test[i] # 9:00
        pred1 = predict_od(obs1, od, n, T) # pred 9:15
        pred[i] = pred1
        if steps > 1 # +2
            pred2 = predict_od(pred1, od, n, T) # pred 9:30
            pred[i] = pred2
            if steps > 2 # +3 
                pred3 = predict_od(pred2, od, n, T) # pred 9:45
                pred[i] = pred3
                if steps > 3 # +4 
                    pred4 = predict_od(pred3, od, n, T) # pred 10:00
                    pred[i] = pred4          
                end
            end
        end
    end
    return pred 
end

function predict_steps_2d(T, binStarts, data_test, od, n; steps=1)     
    len = length(data_test)
    pred = zeros(len-od-steps+1) 
    T, binStarts = T, binStarts
    for i in 1:length(pred) #-steps+1)
        obs1 = data_test[i:i+od-1] # 9:00 & 9:15
        pred1 = predict_od(obs1, od, n, T) # pred 9:30
        pred[i] = pred1
        if steps > 1 # +2
            obs2 = [obs1[2], pred1] # 9:15 & 9:30
            pred2 = predict_od(obs2, od, n, T) # pred 9:45
            pred[i] = pred2
            if steps > 2 # +3 
                obs3 = [pred1, pred2] # 9:30 & 9:45
                pred3 = predict_od(obs3, od, n, T) # pred 10:00
                pred[i] = pred3
                if steps > 3 # +4 
                    obs4 = [pred2, pred3] # 9:45 & 10:00
                    pred4 = predict_od(obs4, od, n, T) # pred 10:15
                    pred[i] = pred4          
                end
            end
        end
    end
    return pred 
end

function predict_steps_3d(T, binStarts, data_test, od, n; steps=1)     
    len = length(data_test)
    pred = zeros(len-od-steps+1) 
    T, binStarts = T, binStarts
    for i in 1:length(pred) #-steps+1)
        obs1 = data_test[i:i+od-1] # 9:00 & 9:15 & 9:30 # true obs
        pred1 = predict_od(obs1, od, n, T) # pred 9:45
        pred[i] = pred1
        if steps > 1 # +2
            obs2 = [obs1[2], obs1[3], pred1] # 9:15 & 9:30 & 9:45
            pred2 = predict_od(obs2, od, n, T) # pred 10:00
            pred[i] = pred2
            if steps > 2 # +3 
                obs3 = [obs2[2], pred1, pred2] # 9:30 & 9:45 & 10:00
                pred3 = predict_od(obs3, od, n, T) # pred 10:15
                pred[i] = pred3
                if steps > 3 # +4 
                    obs4 = [pred1, pred2, pred3] # 9:45 & 10:00 & 10:15
                    pred4 = predict_od(obs4, od, n, T) # pred 10:30
                    pred[i] = pred4          
                end
            end
        end
    end
    return pred 
end

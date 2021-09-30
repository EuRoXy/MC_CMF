using NetCDF # read nc 
using LinearAlgebra, Statistics, Dates # shipped with JL
#using StatsBase #, Distributions # core stats 
using DataFrames # basic data
using Distances

getNCvar(fn::String, var::String) = dropdims(ncread(fn, var); dims=(1,2,3));

dropnan(ar) = filter(ar -> !isnan(ar), ar);

rd(a::Float64, d) = round(a; digits=d)

#### Data
function getCMF0(fn; raw=0)
    fn_ = joinpath("data", fn)
    ghi = getNCvar(fn_, "GHI")
    ghiCS = getNCvar(fn_, "CLEAR_SKY_GHI");
    cmf = ghi ./ ghiCS
    cmf_train = cmf[1:523007] # 2004 - 2018
    cmf_test = cmf[523008:end] # 2019
    if raw == 0
        tr, te = dropnan(cmf_train), dropnan(cmf_test)
    else
        tr, te = cmf_train, cmf_test
    end
    return tr, te
end

function getCMF1(fn) # just for 2020
    fn_ = joinpath("data", fn)
    ghi = getNCvar(fn_, "GHI")
    ghiCS = getNCvar(fn_, "CLEAR_SKY_GHI");
    cmf = ghi ./ ghiCS
    return cmf
end

function calCMF(df1)
    df2 = filter(:ghi => g -> (!iszero(g) & !isnan(g)), df1)
    df2.real = df2.ghi ./ df2.ghiCS
    return df2
end

function classify(arr, binStarts)
    len = length(arr)
    cls = zeros(Int64, len)
    for i in 1:len
        arr[i] ≤ binStarts[1] ?
            cls[i] = 1 : 
            cls[i] = findlast(arr[i] .> binStarts)
        if arr[i] > binStarts[end] 
            cls[i] = N 
        end
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
#         binWidth = [binStarts[i+1] - binStarts[i] for i in 1:n]
        state = [findlast(data[i] .>= binStarts) for i in 1:length(data)]
    end
    state[state .> n] .= n        
    state[state .< 1] .= 1
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
    rowSums = sum(T; dims=2) # normalize T
    T_norm = T ./ rowSums # normalized transition matrix
    T_norm = replace(T_norm, NaN=>0)
    return T_norm
end

function predict_od(obs, od, n, T, binStarts, binMean) # input obs of od 1, 2, 3
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

function mcPredict(data_test, od, n, T, binStarts, binMean) # normal pred
    len = length(data_test)
    pred = zeros(len-od+1)
    for i in 1:(len-od+1)
        obs = data_test[i:i+od-1]
        pred[i] = predict_od(obs, od, n, T, binStarts, binMean)
    end
#     cls = classify(pred, binStarts)
    return pred #, cls
end

#### predict from previous prediction
function predict_steps_1d(T, binStarts, binMean, data_test, od, n; steps=1)     
    len = length(data_test)
    pred = zeros(len-od-steps+1) 
    T, binStarts = T, binStarts
    for i in 1:length(pred) 
        obs1 = data_test[i] # 9:00
        pred1 = predict_od(obs1, od, n, T, binStarts, binMean) # pred 9:15
        pred[i] = pred1
        if steps > 1 # +2
            pred2 = predict_od(pred1, od, n, T, binStarts, binMean) # pred 9:30
            pred[i] = pred2
            if steps > 2 # +3 
                pred3 = predict_od(pred2, od, n, T, binStarts, binMean) # pred 9:45
                pred[i] = pred3
                if steps > 3 # +4 
                    pred4 = predict_od(pred3, od, n, T, binStarts, binMean) # pred 10:00
                    pred[i] = pred4          
                end
            end
        end
    end
    return pred 
end

function predict_steps_2d(T, binStarts, binMean, data_test, od, n; steps=1) 
    len = length(data_test)
    pred = zeros(len-od-steps+1) 
    T, binStarts = T, binStarts
    for i in 1:length(pred) #-steps+1)
        obs1 = data_test[i:i+od-1] # 9:00 & 9:15
        pred1 = predict_od(obs1, od, n, T, binStarts, binMean) # pred 9:30
        pred[i] = pred1
        if steps > 1 # +2
            obs2 = [obs1[2], pred1] # 9:15 & 9:30
            pred2 = predict_od(obs2, od, n, T, binStarts, binMean) # pred 9:45
            pred[i] = pred2
            if steps > 2 # +3 
                obs3 = [pred1, pred2] # 9:30 & 9:45
                pred3 = predict_od(obs3, od, n, T, binStarts, binMean) # pred 10:00
                pred[i] = pred3
                if steps > 3 # +4 
                    obs4 = [pred2, pred3] # 9:45 & 10:00
                    pred4 = predict_od(obs4, od, n, T, binStarts, binMean) # pred 10:15
                    pred[i] = pred4          
                end
            end
        end
    end
    return pred 
end

function predict_steps_3d(T, binStarts, binMean, data_test, od, n; steps=1)     
    len = length(data_test)
    pred = zeros(len-od-steps+1) 
    T, binStarts = T, binStarts
    for i in 1:length(pred) #-steps+1)
        obs1 = data_test[i:i+od-1] # 9:00 & 9:15 & 9:30 # true obs
        pred1 = predict_od(obs1, od, n, T, binStarts, binMean) # pred 9:45
        pred[i] = pred1
        if steps > 1 # +2
            obs2 = [obs1[2], obs1[3], pred1] # 9:15 & 9:30 & 9:45
            pred2 = predict_od(obs2, od, n, T, binStarts, binMean) # pred 10:00
            pred[i] = pred2
            if steps > 2 # +3 
                obs3 = [obs2[2], pred1, pred2] # 9:30 & 9:45 & 10:00
                pred3 = predict_od(obs3, od, n, T, binStarts, binMean) # pred 10:15
                pred[i] = pred3
                if steps > 3 # +4 
                    obs4 = [pred1, pred2, pred3] # 9:45 & 10:00 & 10:15
                    pred4 = predict_od(obs4, od, n, T, binStarts, binMean) # pred 10:30
                    pred[i] = pred4          
                end
            end
        end
    end
    return pred 
end

#### evaluation
function getDFtm(fn) # include time stamp for GHI
    cols = [:yr, :mo, :d, :hr, :mins, :ghi, :ghiCS]
    vars = ["ut_year", "ut_month", "ut_day", "ut_hour", "ut_minute", "GHI", "CLEAR_SKY_GHI"];
    dateDic = Dict(zip(cols, vars))

    fn1 = joinpath("data", fn)
#     for (c, v) in dateDic
#         @eval $c = getNCvar(fn1, $v)
#     end
    yr = getNCvar(fn1, vars[1])
    mo = getNCvar(fn1, vars[2])
    d = getNCvar(fn1, vars[3])
    hr = getNCvar(fn1, vars[4])
    mins = getNCvar(fn1, vars[5])
    ghi = getNCvar(fn1, vars[6])
    ghiCS = getNCvar(fn1, vars[7])

    df = DataFrame(:year=>yr, :month=>mo, :day=>d, :hour=>hr, :minute=>mins)
    dt = map(df -> DateTime(df.year, df.month, df.day, df.hour, df.minute), eachrow(df))
    df1 = DataFrame(:time=>dt, :month=>mo, :ghi=>ghi, :ghiCS=>ghiCS)
    return df1
end
# function getDF(od, steps, n...) #, data_train_cls, data_test, data_test_cls, binStarts, binMean) 
# function getDF(od, steps, n; test_neib=test_neib, hyb=0) 
function getDF(od, steps; n=N, df_test=df1_test, d_test=test, d_neib=test_neib_w) # df, 1-d array, 1-d array
    df = df_test[(od+steps-1):(end-1), :]
    df.real = d_test[(od+steps-1):(end-1)]
    df.pers = d_test[(od-1):(end-steps-1)]
    df.neib = d_neib[1:end-od-steps+1] 

    T = mcFit(data_train_cls, od, n) # transition matrix
    filter!(:real=> c-> !isnan(c), df) # remove nan in central real series 
    pred = mcPredict(df.real, od, n, T, binStarts, binMean)   
    df.pred = vcat(ones(steps) * NaN, pred[1:end-steps+1])
    if steps > 1 
        if od == 1
            pred_n = predict_steps_1d(T, binStarts, binMean, df.real, od, n; steps=steps)
        elseif od == 2
            pred_n = predict_steps_2d(T, binStarts, binMean, df.real, od, n; steps=steps)
        elseif od == 3
            pred_n = predict_steps_3d(T, binStarts, binMean, df.real, od, n; steps=steps)
        end
        df.pred_n = vcat(ones(steps) * NaN, pred_n[1:end-steps+1], ones(steps) * NaN)
        filter!(:pred_n => p_n -> !isnan(p_n), df)
        df.dif_pred_n = df.pred_n .- df.real 
    end
    
    filter!([:neib, :pers] => (n, p) -> (!isnan(n) & !isnan(p)), df)
    df.real_cls = classify(df.real, binStarts)
    df.dif_pers = df.pers .- df.real
    df.dif_neib = df.neib .- df.real
    df.dif_pred = df.pred .- df.real     
    return df
end

function hybrid(dfA, dfB, steps) # A for eval 2019, B for test 2020
    gb = groupby(dfA, :real_cls)
#     if err == "mae"
    mae_pers = [meanad(g.pers, g.real) for g in gb]
    mae_pred = [meanad(g.pred, g.real) for g in gb]
    mae_neib = [meanad(g.neib, g.real) for g in gb]
    df1 = DataFrame(:mae_pers=>mae_pers, :mae_neib=>mae_neib, :mae_pred=>mae_pred) 
    if steps > 1
        df1.mae_pred_n = [meanad(g.pred_n, g.real) for g in gb]
    end 
    mae_min = Int64[]
    for i in 1:size(df1, 1) 
        row = Array(eachrow(df1)[i])
        id = findfirst(row .== minimum(row))
        push!(mae_min, id)
    end
    if length(mae_min) < 30
        push!(mae_min, mae_min[end])
    end
    dfB.mae_min = Int64[0; mae_min[dfB.real_cls[1:end-1]]]
    dfB.hyb_m = map(eachrow(dfB)) do r
        if r.mae_min ≤ 1
            r.pers
        elseif r.mae_min == 2
            r.neib
        elseif r.mae_min == 3
            r.pred
        else 
            r.pred_n
        end
    end
    rmse_pers = [rmsd(g.pers, g.real) for g in gb]
    rmse_pred = [rmsd(g.pred, g.real) for g in gb]
    rmse_neib = [rmsd(g.neib, g.real) for g in gb]
    df2 = DataFrame(:rmse_pers=>rmse_pers, :rmse_neib=>rmse_neib, :rmse_pred=>rmse_pred)
    if steps > 1
        df2.rmse_pred_n = [rmsd(g.pred_n, g.real) for g in gb]
    end
    rmse_min = Int64[]
    for i in 1:size(df2, 1) 
        row = Array(eachrow(df2)[i])
        id = findfirst(row .== minimum(row))
        push!(rmse_min, id)
    end
    if length(rmse_min) < 30
        push!(rmse_min, rmse_min[end])
    end
    dfB.rmse_min = Int64[0; rmse_min[dfB.real_cls[1:end-1]]]
    dfB.hyb_r = map(eachrow(dfB)) do r
        if r.rmse_min ≤ 1
            r.pers
        elseif r.rmse_min == 2
            r.neib
        elseif r.rmse_min == 3
            r.pred
        else
            r.pred_n
        end
    end
    dfB.dif_hyb_m = dfB.hyb_m .- dfB.real
    dfB.dif_hyb_r = dfB.hyb_r .- dfB.real
    return dfB
end

function getGHI(dff, steps)
    dff.ghi_pers = dff.ghiCS .* dff.pers
    dff.ghi_neib = dff.ghiCS .* dff.neib
    dff.ghi_pred = dff.ghiCS .* dff.pred
    dff.ghi_hyb_m = dff.ghiCS .* dff.hyb_m
    dff.ghi_hyb_r= dff.ghiCS .* dff.hyb_r;
    if steps > 1
        dff.ghi_pred_n = dff.ghiCS .* dff.pred_n
        df = dff[:, [:month, :ghi, :ghi_pers, :ghi_neib, :ghi_pred, :ghi_pred_n, :ghi_hyb_m, :ghi_hyb_r]]
    else
        df = dff[:, [:month, :ghi, :ghi_pers, :ghi_neib, :ghi_pred, :ghi_hyb_m, :ghi_hyb_r]]
    end
    return df
end
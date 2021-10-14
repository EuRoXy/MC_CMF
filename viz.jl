using Plots, StatsPlots, Plots.PlotMeasures # , Measures #
default(fmt=:png, dpi=:120)
#=
viz types
- histogram
- scatter
- heatmap
- stem for ACF
- box

- 1.od dif
- mn & sd for dif methods
=#

function hist_pred_vs_real(df) 
    h11 = histogram(df.real, bin=binStarts, xticks=ticks, label="real", leg=:topleft, 
            xlabel="CMF", ylabel="Counts", tickfontsize=6, xrotation=45, 
            color=3, lw=0.2, fillalpha=0.5)
    histogram!(df.pred, bin=binStarts, xticks=ticks, label="pred",  
            color=1, lw=0.2, fillalpha=0.5)
    return h11
end

function hist_dif_pred_pers(df, od, step)
    mn_dif_pred, sd_dif_pred = rd.( [mean(df.dif_pred), std(df.dif_pred)], 3 ) # wrong coz +/- cancel out
    mn_dif_pers, sd_dif_pers = rd.( [mean(df.dif_pers), std(df.dif_pers)], 3 )

    h12 = histogram(df.dif_pers, normalize=:probability, bin=2*N, label="pers", 
        color=4, lw=0.2, fillalpha=0.5)
#     ylim1 = ceil(mode(df.dif_pred); digits=1)
    histogram!(df.dif_pred, normalize=:probability, bin=2*N, label="pred", 
        ylim=(0,0.4), xlabel="ΔCMF", ylabel="Frequency", title="$(od).order +$(15*step) min", dpi=:100, 
        color=1, lw=0.2, fillalpha=0.5)
    annotate!(0.6,0.25, "pers: $(abs(mn_dif_pers)) ± $(sd_dif_pers)", 7)
    annotate!(0.6,0.23, "pred: $(mn_dif_pred) ± $(sd_dif_pred)", 7)
    return h12
end

function hist_cls_dif_pred_pers(df, od, step)
    mn_dif_cls_pred, sd_dif_cls_pred = rd.( [mean(df.dif_cls_pred), std(df.dif_cls_pred)], 1 )
    mn_dif_cls_pers, sd_dif_cls_pers = rd.( [mean(df.dif_cls_pers), std(df.dif_cls_pers)], 1 )

    h13 = histogram(df.dif_cls_pers, normalize=:probability, bin=N, leg=false, 
        color=4, lw=0.2, fillalpha=0.5)
#     ylim1 = ceil(maximum(df.dif_cls_pred); digits=1)
    histogram!(df.dif_cls_pred, normalize=:probability, bin=N, dpi=:100,
        xlim=(-N,N), ylim=(0,0.5), xlabel="ΔCMF class", ylabel="Frequency", title="$(od).order +$(15*step)min", 
        color=1, lw=0.2, fillalpha=0.5)
    annotate!(20,0.3, "pers: $(abs(mn_dif_cls_pers)) ± $(sd_dif_cls_pers)", 7)
    annotate!(20,0.28, "pred: $(mn_dif_cls_pred) ± $(sd_dif_cls_pred)", 7)
    return h13
end

function mae_rmse(df; tit="")
    gb = groupby(df, :real_cls)

    # mean absolute deviation
    mae_pred = [meanad(g.pred, g.real) for g in gb]
    mae_pers = [meanad(g.pers, g.real) for g in gb]

    # root mean squared error
    rmse_pred = [rmsd(g.pred, g.real) for g in gb]
    rmse_pers = [rmsd(g.pers, g.real) for g in gb];

    ylim1 = ceil(maximum(mae_pred); digits=1)
    ylim2 = ceil(maximum(rmse_pred); digits=1)
    p21 = plot(mae_pred, seriestype=:scatter, marker=(0.7, stroke(0)), leg=:topright, label="pred",
        ylim=(0,ylim1), xlabel="real CMF class", ylabel="Mean absolute error", title=tit)
    plot!(mae_pers, seriestype=:scatter, c, label="pers", color=4)
    
    p22 = plot(rmse_pred, seriestype=:scatter, marker=(0.7, stroke(0)), label="pred", 
        ylim=(0,ylim2), xlabel="real CMF class", ylabel="Root mean square error", title=tit)
    plot!(rmse_pers, seriestype=:scatter, marker=(0.7, stroke(0)), label="pers", color=4)
    return p21, p22
end

function viz_err(df, binMean, xti; tit="2. order +$(15*2) min", err="mae")
    gb = groupby(df, :real_cls)
    if err == "mae"
        ylab = "Mean absolute error"
        tit = tit
        ylim = 0.3
        err_pred = [meanad(g.pred, g.real) for g in gb]
        err_pers = [meanad(g.pers, g.real) for g in gb]
        err_pred_n = [meanad(g.pred_n, g.real) for g in gb]
        err_neib = [meanad(g.neib, g.real) for g in gb]
    elseif err == "rmse"
        ylab = "root mean square error"
        tit = ""
        ylim = 0.4
        err_pred = [rmsd(g.pred, g.real) for g in gb]
        err_pers = [rmsd(g.pers, g.real) for g in gb]
        err_pred_n = [rmsd(g.pred_n, g.real) for g in gb]
        err_neib = [rmsd(g.neib, g.real) for g in gb]
    end   
    (df == df1t && err == "mae") ? leg1=:topleft : leg1 = :none
    if df == df1t 
        errs = [err_pers, err_pred, err_neib]
        clrs = [4 1 5]
        labs = ["pers" "pred_a" "neib_s"]
    else
        errs = [err_pers, err_pred, err_pred_n, err_neib]
        clrs = [4 1 7 5]
        labs = ["pers" "pred_a" "pred_b" "neib_s"]
    end        
    p = plot(binMean, errs, c=clrs, marker=(0.7, stroke(0)), dpi=:120,
        leg=leg1, label=labs,
        xticks=xti, xrotation=45, ylim=(0,ylim), tickfontsize=5,
        xlabel="real CMF", ylabel=ylab, title=tit)
    return p
end

function viz_box(df, steps, tit; pred="pred")
    pred == "new" ? 
        (dif = df.dif_pred_n[1+steps:end]) :  
        (dif = df.dif_pred[1+steps:end])
    pred == "new" ? ylab = "pred_n" : ylab = "pred"
    b = boxplot(df.real_cls[1:end-steps], dif, leg=false, marker=(0.3, stroke(0)), lw=.7,
        xticks=(1:2:30, ticks[1:2:30]), xrotation=45,
        title=tit, xlabel="real CMF class mean at t", ylabel="dif ($(ylab) - real)")
    return b
end

function viz_mn_sd(df; tit="+$(15*2) min")
    gb = groupby(df, :real_cls)
    mns_real = [mean(g.real) for g in gb]
    mns_pers = [mean(g.pers) for g in gb]
    mns_pred = [mean(g.pred) for g in gb]
    mns_neib = [mean(g.neib) for g in gb]

    sds_real = [std(g.real) for g in gb]
    sds_pers = [std(g.pers) for g in gb]
    sds_pred = [std(g.pred) for g in gb]
    sds_neib = [std(g.neib) for g in gb]    
    
    if df != df21
        mns_pred_n = [mean(g.pred_n) for g in gb]
        sds_pred_n = [std(g.pred_n) for g in gb]
        mns = [mns_real, mns_pers, mns_pred, mns_pred_n, mns_neib]
        sds = [sds_real, sds_pers, sds_pred, sds_pred_n, sds_neib]
        lab = ["real" "pers" "pred_a" "pred_b" "neib_w"]
        clr = [3 4 1 7 5]
    else   
        mns = [mns_real, mns_pers, mns_pred, mns_neib]
        sds = [sds_real, sds_pers, sds_pred, sds_neib]
        lab = ["real" "pers" "pred_a" "neib_w"] 
        clr = [3 4 1 5] 
    end
    p_mns = plot(binMean, mns, c=clr, label=lab, leg=:bottomright, marker=(3, 0.7, :o, stroke(0)),     
        ylabel="mean")
    p_sds = plot(binMean, sds, c=clr, leg=false, marker=(3, 0.7, :o, stroke(0)),     
        ylabel="standard deviation")
    p = plot(p_mns, p_sds, title=tit, xlabel="real CMF", xticks=xti, xrotation=45, 
        tickfontsize=7, labelfontsize=10, leftmargin=20px, bottommargin=20px, size=(1200, 500))
    return p
end

function viz_bias(df; tit="+$(15*2) min")
    gb = groupby(df, :real_cls)
    bias_pers = [mean(g.dif_pers) for g in gb]
    bias_pred = [mean(g.dif_pred) for g in gb]

    bias_neib = [mean(g.dif_neib) for g in gb]
    bias_hyb_m = [mean(g.dif_hyb_m) for g in gb]
    bias_hyb_r = [mean(g.dif_hyb_r) for g in gb]

    if df != df21
        bias_pred_n = [mean(g.dif_pred_n) for g in gb]
        biases = [bias_pers bias_pred bias_pred_n bias_neib bias_hyb_m bias_hyb_r]
        lab = ["pers" "pred_a" "pred_b" "neib_w" "hyb_m" "hyb_r"]
        clr = [4 1 7 5 6 2]
    else   
        biases = [bias_pers bias_pred bias_neib bias_hyb_m bias_hyb_r]
        lab = ["pers" "pred_a" "neib_w" "hyb_m" "hyb_r"]
        clr = [4 1 5 6 2]
    end

    bi = plot(binMean, biases, label=lab, c=clr, marker=(2, 0.7, :o, stroke(0)), title=tit)
    return bi
end

function viz_dif(df, steps)
    df.dif_cmf = -df.dif_pers
    max_dif = floor(maximum(df.dif_cmf); digits=1)
    min_dif = floor(minimum(df.dif_cmf); digits=1)    
    difBinStarts = collect(min_dif:0.1:max_dif)
    df.cls_dif_cmf = classify(df.dif_cmf, difBinStarts)
#     df = filter(:dif_neib => d -> !isnan(d), df)    
    gb = groupby(df, :cls_dif_cmf)
    bin_mn = [mean(g.dif_cmf) for g in gb]
    mae_pers = [meanad(g.pers, g.real) for g in gb]
    mae_pred = [meanad(g.pred, g.real) for g in gb]
    mae_neib = [meanad(g.neib, g.real) for g in gb]
    mae_hyb_m = [meanad(g.hyb_m, g.real) for g in gb]    
    if steps != 1
        mae_pred_b = [meanad(g.pred_n, g.real) for g in gb]
        maes = [mae_pers mae_neib mae_pred mae_pred_b mae_hyb_m]        
        clr = [4 5 1 7 2]
    else
        maes = [mae_pers mae_neib mae_pred mae_hyb_m]
        clr = [4 5 1 2]        
    end        
    steps == 2 ? lab = ["pers" "neib" "pred_a" "pred_b" "hyb_m"] : lab = false
    labDic = Dict(1 => "realₜ₋₁", 2 => "realₜ₋₂",
                  3 => "realₜ₋₃", 4 => "realₜ₋₄")
    real = labDic[steps]
    p = plot(bin_mn, maes, c=clr, label=lab, leg=:bottomleft,
        marker=(0.7, stroke(0)), frame=:origin, #aspect_ratio=1, 
        xticks=rd.(bin_mn,2), xrotation=45, tickfontsize=6, 
        xlabel="ΔCMF (realₜ - $(real))", ylabel="MAEₜ", title="+$(15*steps) min") 
    return p
end

function viz_ghi_err(dff, steps; tit="+$(15*2) min", err="mae")
    gb = groupby(dff, :month)
    if err == "mae"
        errs_pers = [meanad(g.ghi, g.ghi_pers) for g in gb]
        errs_neib = [meanad(g.ghi, g.ghi_neib) for g in gb]
        errs_pred = [meanad(g.ghi, g.ghi_pred) for g in gb]

        errs_hyb_m = [meanad(g.ghi, g.ghi_hyb_m) for g in gb]
        errs_hyb_r = [meanad(g.ghi, g.ghi_hyb_r) for g in gb]
    elseif err == "rmse"
        errs_pers = [rmsd(g.ghi, g.ghi_pers) for g in gb]
        errs_neib = [rmsd(g.ghi, g.ghi_neib) for g in gb]
        errs_pred = [rmsd(g.ghi, g.ghi_pred) for g in gb]

        errs_hyb_m = [rmsd(g.ghi, g.ghi_hyb_m) for g in gb]
        errs_hyb_r = [rmsd(g.ghi, g.ghi_hyb_r) for g in gb]
    end
    if steps > 1
        err == "mae" ?
            (errs_pred_n = [meanad(g.ghi, g.ghi_pred_n) for g in gb]) :
            (errs_pred_n = [rmsd(g.ghi, g.ghi_pred_n) for g in gb])
        errs = [errs_pers errs_neib errs_pred errs_pred_n errs_hyb_m errs_hyb_r]
        clr = [4 5 1 7 6 2]
    else
        errs = [errs_pers errs_neib errs_pred errs_hyb_m errs_hyb_r]
        clr = [4 5 1 6 2]
    end
    steps == 3 ? 
        (lab = ["pers" "neib" "pred_a" "pred_b" "hyb_m" "hyb_r"]) : 
        (lab = "")
    p = plot(errs, c=clr, label=lab, fillalpha=0.5, marker=(0.7, stroke(0)), title=tit)
    return p
end

function mae_vs_rmse(df1t, df2t, df3t, df4t; tit="Berlin"*" 2020")
    df1 = df1t[:, [:real, :pers, :neib, :pred, :hyb_m, :hyb_r]] 
    df2 = df2t[:, [:real, :pers, :neib, :pred, :pred_n, :hyb_m, :hyb_r]]
    df3 = df3t[:, [:real, :pers, :neib, :pred, :pred_n, :hyb_m, :hyb_r]]
    df4 = df4t[:, [:real, :pers, :neib, :pred, :pred_n, :hyb_m, :hyb_r]]

    len = size(df2, 2)
    lab = ["pers", "neib", "pred_a", "pred_b", "hyb_m", "hyb_r"];

    df_err = DataFrame(:method => lab)

    mae1 = [meanad(df1[:,1], df1[:,i]) for i in 2:(len-1)]
    insert!(mae1, 4, NaN)
    df_err.mae1 = mae1
    df_err.mae2 = [meanad(df2[:,1], df2[:,i]) for i in 2:len]   
    df_err.mae3 = [meanad(df3[:,1], df3[:,i]) for i in 2:len]   
    df_err.mae4 = [meanad(df4[:,1], df4[:,i]) for i in 2:len]

    rmse1 = [rmsd(df1[:,1], df1[:,i]) for i in 2:(len-1)]
    insert!(rmse1, 4, NaN)
    df_err.rmse1 = rmse1
    df_err.rmse2 = [rmsd(df2[:,1], df2[:,i]) for i in 2:len]   
    df_err.rmse3 = [rmsd(df3[:,1], df3[:,i]) for i in 2:len]   
    df_err.rmse4 = [rmsd(df4[:,1], df4[:,i]) for i in 2:len] 
    @show df_err;

    p = plot(leg=:bottomright, #xlim=(0.05, 0.16), #aspect_ratio=1, #ylim=(0,0.25), 
        xlabel="MAE", ylabel="RMSE", title=tit)
    clrs = [4, 5, 1, 7, 6, 2]
    for i in 1:(len-1)
        plot!(Array(df_err[i,2:5]), Array(df_err[i,6:end]), marker=(3, 0.7, stroke(0)), c=clrs[i], label=lab[i])
    end
    return p
end
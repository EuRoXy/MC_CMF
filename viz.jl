include("MC.jl")
using Plots, Plots.PlotMeasures, StatsPlots

function hist_pred_vs_real(df) 
    h11 = histogram(df.real, bin=binStarts, xticks=ticks, label="real", leg=:topleft, 
            xlabel="CMF", ylabel="Counts", tickfontsize=6, xrotation=45, 
            color=3, lw=0.2, fillalpha=0.5)
    histogram!(df.pred, bin=binStarts, xticks=ticks, label="pred",  
            color=1, lw=0.2, fillalpha=0.5)
    return h11
end

function hist_dif_pred_pers(df, od, step)
    mn_dif_pred, sd_dif_pred = rd.( [mean(df.dif_pred), std(df.dif_pred)], 3 )
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

function viz_err(binMean, df, tit; ticks=ticks, err="mae")
    gb = groupby(df, :real_cls)
    if err == "mae"
        ylab = "mean absolute error"
        err_pred = [meanad(g.pred, g.real) for g in gb]
        err_pers = [meanad(g.pers, g.real) for g in gb]
        err_pred_n = [meanad(g.pred_n, g.real) for g in gb]
    elseif err == "rmse"
        ylab = "root mean square error"
        err_pred = [rmsd(g.pred, g.real) for g in gb]
        err_pers = [rmsd(g.pers, g.real) for g in gb]
        err_pred_n = [rmsd(g.pred_n, g.real) for g in gb]
    end            
    p = plot(binMean, [err_pers, err_pred, err_pred_n], c=[4 1 7], marker=(0.7, stroke(0)), 
        leg=:bottomleft, label=["pers" "pred_a" "pred_b"],
        xticks=(binMean[1:2:26], ticks[1:2:26]), xrotation=45, 
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

function viz_mn_sd(df; tit="2. order +$(15*2) min")
    gb = groupby(df, :real_cls)
    mns_real = [mean(g.real) for g in gb]
    mns_pers = [mean(g.pers) for g in gb]
    mns_pred = [mean(g.pred) for g in gb]
    mns_pred_n = [mean(g.pred_n) for g in gb];

    sds_real = [std(g.real) for g in gb]
    sds_pers = [std(g.pers) for g in gb]
    sds_pred = [std(g.pred) for g in gb]
    sds_pred_n = [std(g.pred_n) for g in gb];

    if df == df22 
        (ylab1, ylab2) = ("mean", "standard deviation") 
        leg1 = :bottomright
    else 
        (ylab1, ylab2) = ("", "")
        leg1 = :none
    end
    mns = plot(binMean, [mns_real, mns_pers, mns_pred, mns_pred_n], c=[3 4 1 7], leg=leg1, 
            ylabel=ylab1, title=tit)
    sds = plot(binMean, [sds_real, sds_pers, sds_pred, sds_pred_n], c=[3 4 1 7], leg=false, 
            xlabel="real CMF", ylabel=ylab2)
    return mns, sds
end
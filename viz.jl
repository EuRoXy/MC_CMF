include("MC.jl")

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

    h12 = histogram(df.dif_pers, normalize=:probability, bin=n, label="pers", 
        color=4, lw=0.2, fillalpha=0.5)

    histogram!(df.dif_pred, normalize=:probability, bin=2*n, label="pred", 
        ylim=(0,0.4), xlabel="ΔCMF", ylabel="Frequency", title="$(od).order pred +$(15*step) min", dpi=:150, 
        color=1, lw=0.2, fillalpha=0.5)

    annotate!(0.6,0.25, "pers: $(abs(mn_dif_pers)) ± $(sd_dif_pers)", 7)
    annotate!(0.6,0.23, "pred: $(mn_dif_pred) ± $(sd_dif_pred)", 7)
    return h12
end

function hist_cls_dif_pred_pers(df)
    mn_dif_cls_pred, sd_dif_cls_pred = rd.( [mean(df.dif_cls_pred), std(df.dif_cls_pred)], 1 )
    mn_dif_cls_pers, sd_dif_cls_pers = rd.( [mean(df.dif_cls_pers), std(df.dif_cls_pers)], 1 )

    h13 = histogram(df.dif_cls_pers, normalize=:probability, bin=n, leg=false, 
        color=4, lw=0.2, fillalpha=0.5)

    histogram!(df.dif_cls_pred, normalize=:probability, bin=n, 
        xlim=(-30,30), ylim=(0,0.5), xlabel="ΔCMF class", ylabel="Frequency", #title="$(od).order pred +$(15*od)min", dpi=:150,
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
    plot!(mae_pers, seriestype=:scatter, marker=(0.7, stroke(0)), label="pers", color=4)
    
    p22 = plot(rmse_pred, seriestype=:scatter, marker=(0.7, stroke(0)), label="pred", 
        ylim=(0,ylim2), xlabel="real CMF class", ylabel="Root mean square error", title=tit)
    plot!(rmse_pers, seriestype=:scatter, marker=(0.7, stroke(0)), label="pers", color=4)
    return p21, p22
end
using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using NeuralArithmetic
using ValueHistories
using DataFrames
using Statistics
using Measurements
using Plots
pyplot()

folder = datadir("rational")
# (_,_,files) = first(walkdir(folder))
# files = map(f->joinpath(folder,f), files)

f(x) = (x^3-2x)/(2(x^2-5))
#xt = Float32.(reshape(-5:0.05:5,1,:))
#xt = Float32.(reshape(-10:0.05:10,1,:))
xt = Float32.(reshape(-10:0.1:10,1,:))
#xt = Float32.(reshape(-15:0.1:15,1,:))
yt = f.(xt)

df = collect_results!(folder, black_list=[:history])

# df = filter(r->(a = r.dim==10;
#                 b = r.layer=="gatednpux" ? r.βpsl1==0 : true;
#                 a && b),
#            df)
df = filter(r->r.uplim==7, df)

df.val = map(m->Flux.mse(m(xt),yt), df.model)
df.output = map(m->vec(m(xt)), df.model)

adf = combine(groupby(df,"layer")) do gdf
    (mse = measurement(mean(gdf.mse), std(gdf.mse)/sqrt(length(gdf.mse))),
     val = measurement(mean(gdf.val), std(gdf.val)/sqrt(length(gdf.val))),
     layer = first(gdf.layer))
end
display(adf)

x = vec(xt)
y = vec(yt)
key = "mse"
ylim = (-5,5)
#ylim = (-10,10)

s1 = plot(ylim=ylim, title="GatedNPU")
for row in eachrow(df)
    d = parse_savename(row.path)[2]
    if d["layer"] == "gatednpux"
        plot!(s1, x, row.output, label="run #$(d["run"])", c=:black)
    end
end
plot!(s1,x,y,label="Truth",lw=2,c=1)
vline!(s1, [-5,5], c=:gray, lw=2, ls=:dash, label="Train range")

s2 = plot(ylim=ylim, lw=2, title="NALU")
for row in eachrow(df)
    d = parse_savename(row.path)[2]
    if d["layer"] == "nalu"
        plot!(s2, x, row.output, label="run #$(d["run"])", c=:black)
    end
end
plot!(s2,x,y,label="Truth",lw=2,c=1)
vline!(s2, [-5,5], c=:gray, lw=2, ls=:dash, label="Train range")

s3 = plot(ylim=ylim, lw=2, title="Dense")
for row in eachrow(df)
    d = parse_savename(row.path)[2]
    if d["layer"] == "dense"
        plot!(s3, x, row.output, label="run #$(d["run"])", c=:black)
    end
end
plot!(s3,x,y,label="Truth",lw=2,c=1)
vline!(s3, [-5,5], c=:gray, lw=2, ls=:dash, label="Train range")


p1 = plot(s1,s2,s3,layout=(1,3),size=(1000,300))
display(p1)


best = combine(groupby(df, "layer")) do gdf
    out = sort!(DataFrame(gdf), key)
    out[1,:]
end
display(best[!,["layer","mse","val"]])

s1 = plot(x, y, label="Truth", ylim=ylim, lw=2)
plot!(s1, x, best[1,"output"], title=best[1,"layer"], label="MSE: $(best[1,"val"])")
display(s1)
s2 = plot(x, y, label="Truth", ylim=ylim, lw=2)
plot!(s2, x, best[2,"output"], title=best[2,"layer"], label="MSE: $(best[2,"val"])")
s3 = plot(x, y, label="Truth", ylim=ylim, lw=2)
plot!(s3, x, best[3,"output"], title=best[3,"layer"], label="MSE: $(best[3,"val"])")
p2 = plot(s1,s2,s3,layout=(1,3),size=(1000,300))
display(p2)

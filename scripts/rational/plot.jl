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
xt = Float32.(reshape(-10:0.1:10,1,:))
yt = f.(xt)

df = collect_results!(folder, black_list=[:history])
df = filter(r->r.dim==20, df)
df.val = map(m->Flux.mse(m(xt),yt), df.model)

adf = combine(groupby(df,"layer")) do gdf
    (mse = measurement(mean(gdf.mse), std(gdf.mse)/sqrt(length(gdf.mse))),
     val = measurement(mean(gdf.val), std(gdf.val)/sqrt(length(gdf.val))),
     layer = first(gdf.layer))
end
display(adf)

df.output = map(m->vec(m(xt)), df.model)
x = vec(xt)
y = vec(yt)

s1 = plot(x, y, label="Truth", ylim=(-5,5), lw=2, title="GatedNPU")
for row in eachrow(df)
    d = parse_savename(row.path)[2]
    if d["layer"] == "gatednpux"
        plot!(s1, x, row.output, label="run #$(d["run"])", c=:gray)
    end
end
#plot!(p1, x, y, label="Truth", ylim=(-5,5), lw=2, title="GatedNPU", c=:blue)

s2 = plot(x, y, label="Truth", ylim=(-5,5), lw=2, title="NALU")
for row in eachrow(df)
    d = parse_savename(row.path)[2]
    if d["layer"] == "nalu"
        plot!(s2, x, row.output, label="run #$(d["run"])", c=:gray)
    end
end

s3 = plot(x, y, label="Truth", ylim=(-5,5), lw=2, title="Dense")
for row in eachrow(df)
    d = parse_savename(row.path)[2]
    if d["layer"] == "dense"
        plot!(s3, x, row.output, label="run #$(d["run"])", c=:gray)
    end
end


p1 = plot(s1,s2,s3,layout=(1,3),size=(1000,300))
display(p1)

best = combine(groupby(df, "layer")) do gdf
    out = sort!(DataFrame(gdf), "mse")
    out[1,:]
end


s1 = plot(x, y, label="Truth", ylim=(-5,5), lw=2)
plot!(s1, x, best[1,"output"], label=best[1,"layer"], title="MSE: $(best[1,"val"])")
s2 = plot(x, y, label="Truth", ylim=(-5,5), lw=2)
plot!(s2, x, best[2,"output"], label=best[2,"layer"], title="MSE: $(best[2,"val"])")
s3 = plot(x, y, label="Truth", ylim=(-5,5), lw=2)
plot!(s3, x, best[3,"output"], label=best[3,"layer"], title="MSE: $(best[3,"val"])")
p2 = plot(s1,s2,s3,layout=(1,3),size=(1000,300))
display(p2)

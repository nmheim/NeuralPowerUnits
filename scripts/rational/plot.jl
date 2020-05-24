using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using NeuralArithmetic
using ValueHistories
using DataFrames

folder = datadir("rational")
# (_,_,files) = first(walkdir(folder))
# files = map(f->joinpath(folder,f), files)

f(x) = (x^3-2x)/(2(x^2-5))
xt = Float32.(reshape(-5:0.05:5,1,:))
yt = f.(xt)

df = collect_results!(folder)
df.output = map(m->vec(m(yt)), df.model)

x = vec(xt)
y = vec(yt)
p1 = plot(x, y, label="Truth", ylim=(-5,5))

for row in eachrow(df)
    if row.model == "gatednpux"
        plot!(p1, vec(xt), vec(row.output))
    end
end
display(p1)
error()

display(df)
error()

for m in df.model
    m(yt)
df.output = df.model
display(df)
error()


for file in files
    (root,d,_) = parse_savename(file)[2]
    if "complicated" == basename(root)

    @unpack model = load(file)
display(files)

for 
error()

pyplot()
p1 = plot(vec(xt), vec(yt),label="Truth", ylim=(-6,6))
plot!(p1, vec(xt), vec(npu(xt)), label="GatedNPU $(Flux.mse(npu(xt),yt))")
plot!(p1, vec(xt), vec(nalu(xt)), label="NALU $(Flux.mse(nalu(xt),yt))")
#plot!(p1, vec(xt), vec(nmu(xt)), label="NMU $(Flux.mse(nmu(xt),yt))")
plot!(p1, vec(xt), vec(dense(xt)), label="Dense $(Flux.mse(dense(xt),yt))")
display(p1)
error()


model = res[:model]
history = res[:history]
c = res[:config]
@unpack lowlim,uplim = c
(x,y,xt,yt) = generate(f,lowlim,uplim)
display(validationplot(xt,yt,model(xt),lowlim,uplim))
error()
println("")
heat(model)

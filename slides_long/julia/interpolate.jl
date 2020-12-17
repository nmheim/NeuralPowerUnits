using DrWatson
using Flux

model = load(datadir("dense-square_hdim=5_lr=0.01.bson"))[:model]
predict(m,x::Real) = m([x])[1]

using Plots
using LaTeXStrings
pgfplotsx()
#pyplot()
theme(:default, lw=2, legend=false)

x = -1:0.1:1
f(x) = x^2
p1 = plot(x, f, label="Truth", color="black", size=(200,200)) 
plot!(p1, x, x->predict(model,x), label="Dense", ls=:dash)
savefig(p1, "x2-interp.tikz")

x = -2:0.1:2
p2 = plot(x, f, label="Truth", color="black", size=(200,200)) 
vline!(p2, [-1,1], color="black", lw=1, alpha=0.5)
plot!(p2, x, x->predict(model,x), label="Dense", ls=:dash, color=2)
savefig(p2, "x2-extrap.tikz")

using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Flux
using LinearAlgebra
using NeuralArithmetic
using UnicodePlots


_sine_taylor(x,n) = (-1)^n * x^(2n+1) / factorial(2n+1)
_sum_taylor(f,x,n;n0=0) = sum(map(i->f(x,i),n0:n))
sine_taylor(x,n) = _sum_taylor(_sine_taylor,x,n)

_exp_taylor(x,n) = x^n/factorial(n)
exp_taylor(x,n) = _sum_taylor(_exp_taylor,x,n)

_log_taylor(x,n) = (-1)^(n+1) * x^n/n
log_taylor(x,n) = _sum_taylor(_log_taylor,x,n,n0=1)

x = -3:0.1:3

plt = lineplot(x,exp_taylor.(x,0))
for n in 1:6
    lineplot!(plt,x,exp_taylor.(x,n))
end


x = 0.1:0.1:3

plt = lineplot(x,log_taylor.(x,1))
for n in 2:6
    lineplot!(plt,x,log_taylor.(x,n))
end
display(plt)


x = Float32.(reshape(-5:0.1:5, 1, :))
f(x) = sine_taylor.(x,4)
#f(x) = exp_taylor.(x,3)
#f(x) = exp.(x)
f(x) = sin.(x)


# x = Float32.(reshape(0.1:0.1:3, 1, :))
# f(x) = log.(1 .+ x)
y = f(x)

dim = 10

data = Iterators.repeated((x,y), 100000)
opt = ADAM(1e-2)
model = Chain(NPU(1,dim), NAU(dim,1))
model = Chain(NPUX(1,dim), NAU(dim,1))
#model = Chain(GatedNPU(1,dim), NAU(dim,1))
#model = Chain(GatedNPUX(1,dim), NAU(dim,1))
model = Chain(NALU(1,dim), NALU(dim,1))
#model = Chain(NMU(1,dim), NAU(dim,1))
#model = Chain(Dense(1,dim,σ),Dense(dim,dim,σ),Dense(dim,1))
ps = params(model)
loss(x,y) = Flux.mse(model(x),y) + 0.001norm(ps,1)


cb = [Flux.throttle(
      ()->(xt = Float32.(reshape(-10:0.1:10,1,:));
           #xt = Float32.(reshape(0.1:0.1:8,1,:));
           yt = f(xt);
           p1 = lineplot(vec(xt), vec(yt));
           lineplot!(p1, vec(xt), vec(model(xt)));
           display(p1);
           @info loss(x,y) loss(xt,yt)
          ), 1)
     ]
Flux.train!(loss, ps, data, opt, cb=cb)

display(UnicodePlots.heatmap(model[1].W, title=summary(model[1])))
display(UnicodePlots.heatmap(reshape(model[2].W,:,1), title=summary(model[2])))

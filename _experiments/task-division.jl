using DrWatson
@quickactivate "arithmetic"

using Flux
using Zygote
using Plots
using GMExtensions
using NeuralArithmetic
using Distributions: Uniform
using Parameters
using ValueHistories
using LinearAlgebra

task = "task-division"
T           = Float32
batch       = 50
inlen       = 4
outlen      = 1
niters      = 30000
r           = Uniform(0.1,3)

function init_nau(T, a::Int, b::Int)
    m = zeros(T,a,b)
    m[diagind(m,0)] .= T(1)
    return m
end

init_nmu(T, a::Int, b::Int) = zeros(T, a, b)

setup = @dict(T, batch, inlen, outlen, niters, α0, β0, lr,
              esamples, r)
 
net = Chain(
            NAU(inlen, inlen, init=(s...)->init_nau(T,s...)),
            ReNMUX(inlen, outlen, init=(s...)->init_nmu(T,s...))
           )
net = Chain(
            NAU(inlen, inlen),
            ReNMUX(inlen, outlen)
           )


function f(x)
    x1 = x[1,:]
    x2 = x[2,:]
    y = x1 ./ x2
    reshape(y, 1, :)
end

function generate()
    x = T.(rand(r, inlen, batch))
    y = f(x)
    (x,y)
end

(x,y) = generate()

train_data = [generate() for _ in 1:niters]
ps = params(net)
loss(x,y) = Flux.mse(net(x), y) #+ sum(norm, ps)
history = MVHistory()

logging = Flux.throttle((ii,train_loss)->(@info "Step: $ii" train_loss), 1)
pushhist = Flux.throttle((ii,train_loss,net)->(
    push!(history, :loss, ii, train_loss);
    push!(history, :ps, ii, Flux.destructure(net)[1]);
), 0.1)
Zygote.@nograd logging
Zygote.@nograd pushhist


opt = Descent(lr)
for (ii,d) in enumerate(train_data)
    gs = Flux.gradient(ps) do
        train_loss = loss(d...)
        logging(ii, train_loss)
        pushhist(ii, train_loss, net)
        train_loss
    end

    Flux.Optimise.update!(opt, ps, gs)
end


pyplot()
p1 = plothistory(history)

(x,y) = generate()
display(net(x))
display(f(x))
display(task)

p2 = plot(
    annotatedheatmap(net[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
    annotatedheatmap(net[2].W[end:-1:1,:], c=:bluesreds, title="ReNMUX", clim=(-1,1)),
    size=(600,300)
)
display(p1)

# savefig(p1, plotsdir("10-param-func-bayes-$task-history.pdf"))
# savefig(p2, plotsdir("10-param-func-bayes-$task.pdf"))

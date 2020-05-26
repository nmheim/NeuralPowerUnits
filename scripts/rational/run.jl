using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Flux
using Parameters
using LinearAlgebra
using NeuralArithmetic
using ValueHistories
using Plots
unicodeplots()

include(srcdir("unicodeheat.jl"))
#pyplot()

# f(x) = x^2
# f(x) = x^4/10 + x^3/10 - (13x^2)/10 - x/10 + 6/5

prefix = "complicated"
f(x) = (x^3-2x)/(2(x^2-5))

# prefix = "1divx2.15"
# f(x) = x/(x-2.15)/(x+1.35)/5

function validationplot(xt,yt,tt,lowlim,uplim)
    p1 = plot(vec(xt), vec(yt), ylim=(-5,5))
    plot!(p1, vec(xt), vec(tt))
    vline!([lowlim,uplim],c=:gray)
end


function generate(f,lowlim,uplim)
    x = Float32.(reshape(lowlim:0.1:uplim, 1, :))
    y = f.(x)
    xt = Float32.(reshape((lowlim*1.5):0.1:(uplim*1.5),1,:))
    yt = f.(xt)
    (x,y,xt,yt)
end

iml1(model::GatedNPUX) = norm(model.Im,1)
iml1(model::NPUX) = norm(model.Im,1)
iml1(model) = 0
iml1(model::Chain) = sum(iml1, model)

#init(a,b) = rand(Float32,a,b)/2
init(a,b) = Flux.glorot_uniform(a,b)

function get_model(layer,dim)
    if layer == "nalu"
        return Chain(NALU(1,dim),
                     NALU(dim,dim),
                     NALU(dim,1))
    elseif layer == "gatednpux"
        return Chain(GatedNPUX(1,dim, initRe=init),
                     NAU(dim,dim,init=init),
                     GatedNPUX(dim,1, initRe=init))
    elseif layer == "npu"
        return Chain(NPU(1,dim),
                     NAU(dim,dim),
                     NPU(dim,1))
    elseif layer == "npux"
        return Chain(NPUX(1,dim),
                     NAU(dim,dim),
                     NPUX(dim,1))
    elseif layer == "dense"
        return Chain(Dense(1,dim,σ),
                     Dense(dim,dim,σ),
                     Dense(dim,1))
    elseif layer == "nmu"
        return Chain(NMU(1,dim), NAU(dim,dim), NMU(dim,1))
    else
        error("unknown layer: $layer")
    end
end

function run(c::Dict, f::Function)
    @unpack dim, lowlim, uplim, niter, lr, βpsl1, βiml1, layer = c
    h = MVHistory()
    (x,y,xt,yt) = generate(f,lowlim,uplim)
    data = Iterators.repeated((x,y), niter)
    opt = RMSProp(lr)

    model = get_model(layer, dim)
    ps = params(model)

    mse(x,y) = sum(abs2, model(x) .- y)
    loss(x,y) = mse(x,y) + βiml1*iml1(model) + βpsl1*norm(ps,1)

    curriter = 1
    minmse = mse(x,y)
    minimizer = Flux.f32(model)
    cb = [Flux.throttle(()->(
               p1 = validationplot(xt,yt,model(xt),lowlim,uplim);
               display(p1);
               println();
               display(heat(model));
               println();
               @info layer curriter mse(x,y) minmse loss(x,y) loss(xt,yt)
              ), 1),
          Flux.throttle(() -> (push!(h, :μz, Flux.destructure(model)[1]);
                               push!(h, :mse, mse(xt,yt))), 0.1),
          () -> (curriter += 1;
                 newmse = mse(x,y);
                 if newmse < minmse
                     minimizer = Flux.f32(model);
                     minmse = newmse;
                 end)
         ]
    Flux.train!(loss, ps, data, opt, cb=cb)
    model = minimizer
    ps = params(minimizer)
    opt = RMSProp(lr/10)
    data = Iterators.repeated((x,y), niter*3)

    Flux.train!(loss, ps, data, opt, cb=cb)
    c[:model] = minimizer
    c[:history] = h
    c[:mse] = minmse
    return c
end


@progress for nr in 1:10
    npures, _ = produce_or_load(datadir("rational"),
                             Dict(:lowlim =>  -7,
                                  :uplim  =>  7,
                                  :dim    =>  10,
                                  :niter  =>  20000,
                                  :lr     =>  5e-4,
                                  :run    =>  nr,
                                  :layer  => "gatednpux",
                                  :βiml1  =>  0.0,
                                  :βpsl1  =>  1.0),
                             c -> run(c, f), prefix=prefix,
                             force=false, digits=8)
    nalures, _ = produce_or_load(datadir("rational"),
                             Dict(:lowlim =>  -5,
                                  :uplim  =>  5,
                                  :dim    =>  20,
                                  :niter  =>  20000,
                                  :lr     =>  5e-4,
                                  :run    =>  nr,
                                  :layer  => "nalu",
                                  :βiml1  =>  0.0,
                                  :βpsl1  =>  0.0),
                             c -> run(c, f), prefix=prefix,
                             force=false, digits=8)
    denseres, _ = produce_or_load(datadir("rational"),
                             Dict(:lowlim =>  -5,
                                  :uplim  =>  5,
                                  :dim    =>  20,
                                  :niter  =>  10000,
                                  :lr     =>  1e-2,
                                  :layer  => "dense",
                                  :βiml1  =>  0.0,
                                  :run    =>  nr,
                                  :βpsl1  =>  0.0),
                             c -> run(c, f), prefix=prefix,
                             force=false, digits=8)
end

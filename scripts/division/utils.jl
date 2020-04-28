function x12_x14(x)
    x1 = x[1,:]
    x2 = x[2,:]
    x3 = x[3,:]
    x4 = x[4,:]
    y = (x1 .+ x2) .* (x1 .+ x2 .+ x3 .+ x4)
    reshape(y, 1, :)
end

function generate(inlen::Int, batch::Int, r::Uniform)
    x = Float32.(rand(r, inlen, batch))
    y = x12_x14(x)
    (x,y)
end

function init_diag(T, a::Int, b::Int)
    m = zeros(T,a,b)
    m[diagind(m,0)] .= 1f0
    return m
end

function initf(s::String)
    if s == "diag"
        return (a,b) -> init_diag(Float32,a,b)
    elseif s == "glorotuniform"
        return (a,b) -> Flux.glorot_uniform(a,b)
    elseif s == "rand"
        return (a,b) -> rand(Float32,a,b)/10
    elseif s == "randn"
        return (a,b) -> randn(Float32,a,b)/10
    elseif s == "zero"
        return (a,b) -> zeros(Float32,a,b)
    elseif s== "one"
        return (a,b) -> ones(Float32,a,b)
    else
        throw(ArgumentError("Unkown init: $s"))
    end
end

function mapping(in, out, init_nau, init_nmu)
    nau = NAU(in, in, init=init_nau)
    nmu = NPU(in, out, init=init_nmu)
    Chain(nau, nmu)
end

function train!(loss, model, data, opt, history=MVHistory())
    ps = params(model)
    train_loss = 0f0
    mse_loss = 0f0

    logging = Flux.throttle((i)->(
        @info "Step $i: $train_loss $mse_loss $([model.v[1],model.σ[1]])";
        W = model.m[1].W;
        p1 = UnicodePlots.heatmap(model.m[1].W[end:-1:1,:],
                                  width=size(W,1), height=size(W,2));
        # p2 = UnicodePlots.heatmap(model.m[2].W[end:-1:1,:],
        #                           width=size(W,1), height=size(W,2));
        display(p1);
        #display(p2);
       ), 2)

    pushhist = Flux.throttle((i)->(
            push!(history, :μz, i, copy(Flux.destructure(model)[1]));
            push!(history, :loss, i, [train_loss]);
       ),
    0.1)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1
    try 
        for d in data
            # if i == 1000
            #     ps = params(model,p0)
            # end
            gs = gradient(ps) do
                mse_loss = Flux.mse(model(d[1]),d[2])
                train_loss = loss(d...)
                return train_loss
            end
            logging(i)
            pushhist(i)
            # @info p0 gs[p0]
            # if any(isnan.(p0)) error("asdf") end
            Flux.Optimise.update!(opt, ps, gs)
            i += 1
        end
    catch e
        println("Error during training!")
        for (exc,bt) in Base.catch_stack()
            showerror(stdout,exc,bt)
            println()
        end
    end

    history
end


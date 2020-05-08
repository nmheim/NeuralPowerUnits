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

function get_model(model::String, inlen::Int, fstinit::String, sndinit::String)
    if model == "npu"
        return Chain(NAU(inlen, inlen, init=initf(fstinit)),
                     NPU(inlen, 1, init=initf(sndinit)))
    elseif model == "nmu"
        return Chain(NAU(inlen, inlen, init=initf(fstinit)),
                     NMU(inlen, 1, init=initf(sndinit)))
    elseif model == "nalu"
        return Chain(NALU(inlen, inlen,
                          initNAC=initf(fstinit),
                          initG=initf(fstinit),
                          initb=initf(fstinit)),
                     NALU(inlen, 1,
                          initNAC=initf(sndinit),
                          initG=initf(sndinit),
                          initb=initf(sndinit)))
    elseif model == "npux"
        return Chain(Flux.fmap(ComplexMatrix, NAU(inlen,inlen,init=initf(fstinit))),
                     NPU(inlen, 1, init=initf(sndinit)))
    elseif model == "gatednpu"
        return Chain(NAU(inlen, inlen, init=initf(fstinit)),
                     GatedNPU(inlen, 1, init=initf(sndinit)))
    else
        error("Unknown model string: $model")
    end
end

function train!(loss, model, data, val_data, opt, sch::Schedule, history=MVHistory())
    ps = params(model)
    trn_loss, val_loss, mse_loss, reg_loss = 0f0, 0f0, 0f0, 0f0

    logging = Flux.throttle((i)->(
            @info("Step $i | β=$(sch.eta)", trn_loss, mse_loss, reg_loss, val_loss);
            m = get_mapping(model);
            (h,w) = size(m[1].W);
            p1 = UnicodePlots.heatmap(m[1].W[end:-1:1,:], height=h, width=w);
            display(p1);
        ),
    5)
    pushhist = Flux.throttle((i)->(
            push!(history, :μz, i, copy(Flux.destructure(model)[1]));
            val_loss = Flux.mse(model(val_data[1]), val_data[2]);
            push!(history, :loss, i, [trn_loss,mse_loss,reg_loss,val_loss]);
       ),
    5)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1
    try 
        @progress for d in data
            # if i == 20000
            #     lr = 1e-2
            #     @info "lr=$lr"
            #     opt == ADAM(lr)
            # end
            factor = step!(sch)
            gs = gradient(ps) do
                trn_loss, mse_loss, reg_loss = loss(d..., factor)
                return trn_loss
            end
            logging(i)
            pushhist(i)
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

get_mapping(m) = m

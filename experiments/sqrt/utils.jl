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
        return (a,b) -> rand(Float32,a,b)
    elseif s == "randn"
        return (a,b) -> randn(Float32,a,b)
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

    logging = Flux.throttle((i)->(@info "Step $i: $train_loss"), 1)
    pushhist = Flux.throttle((i)->(
            push!(history, :μz, i, copy(Flux.destructure(model)[1]));
            push!(history, :loss, i, [train_loss]);
       ),
    0.1)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1
    try 
        for d in data
            gs = gradient(ps) do
                train_loss = loss(d...)
                return train_loss
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


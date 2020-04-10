function mapping(in, out, init_nau, init_nmu)
    nau = NAU(in, in, init=init_nau)
    nmu = ReNMUX(in, out, init=init_nmu)
    Chain(nau, nmu)
end

function train!(loss, model, data, opt, history=MVHistory())
    ps = params(model)
    train_loss = 0f0

    logging = Flux.throttle((i)->(@info "Step $i" [tot llh kld lpλ]), 1)
    pushhist = Flux.throttle((i)->(
            push!(history, :μz, i, Flux.destructure(model)[1]);
            push!(history, :loss, i, [train_loss]);
       ),
    0.5)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1
    try 
        for d in data
            gs = gradient(ps) do
                train_loss = loss(d...)
                return train_loss
            end
            # logging(i)
            #pushhist(i)
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



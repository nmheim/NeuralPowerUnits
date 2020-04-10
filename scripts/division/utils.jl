function mapping(in, out, init_nau, init_nmu)
    nau = NAU(in, in, init=init_nau)
    nmu = ReNMUX(in, out, init=init_nmu)
    Chain(nau, nmu)
end

function ardnet(mapping, α0, β0, out)
    μz = Flux.destructure(mapping)[1]
    zlen = length(μz)
    λz = ones(Float32,zlen)/10
    σz = ones(Float32,zlen)/10
    σx = ones(Float32,1)

    e = Gaussian(μz, σz)
    p = Gaussian(NoGradArray(zeros(Float32,zlen)), λz)
    h = InverseGamma(α0,β0,zlen,true)
    d = CMeanGaussian{ScalarVar}(FluxDecoder(mapping), σx, out)
    ARDNet(h, p, e, d)
end

function train!(loss, model, data, opt, history=MVHistory())
    ps = params(model)
    tot, llh, kld, lpλ = 0f0, 0f0, 0f0, 0f0

    logging = Flux.throttle((i)->(@info "Step $i" [tot llh kld lpλ]), 1)
    pushhist = Flux.throttle((i)->(
            push!(history, :μz, i, copy(mean(model.encoder)));
            push!(history, :σz, i, copy(var(model.encoder)));
            push!(history, :λz, i, copy(var(model.prior)));
            push!(history, :σx, i, copy(var(model.decoder,ones(1))));
            push!(history, :loss, i, [tot, llh, kld, lpλ]);
       ),
    0.5)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1
    try 
        for d in data
            gs = gradient(ps) do
                (tot, llh, kld, lpλ) = loss(d...)
                return tot
            end
            logging(i)
            pushhist(i)
            Flux.Optimise.update!(opt, ps, gs)
            i += 1
        end
    catch e
        println("Error during training: $(e.msg)")
        for (exc,bt) in Base.catch_stack()
            showerror(stdout,exc,bt)
            println()
        end
    end

    history
end

function notelbo(m::ARDNet, x, y; α0=1, β0=0, esamples=1)
    ps = reshape(rand(m.encoder),:)
    llh = sum(logpdf(m.decoder, y, x, ps))
    kld = sum(kl_divergence(m.encoder, m.prior))
    lpλ = sum(logpdf(m.hyperprior, var(m.prior)))

    _elbo = -llh + kld - lpλ
    _elbo, -llh, kld, -lpλ
end

get_mapping(m::ARDNet) = m.decoder.mapping.restructure(mean(m.encoder))

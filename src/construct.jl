function conv_encoder(slen, zlen, mintlen;
                      init_conv=Flux.glorot_uniform,
                      init_dense=Flux.glorot_uniform)
    act = tanh
    conv_zlen = zlen-slen
    dense_zlen = slen

    convnet = Chain(
        x -> reshape(x, slen, size(x,2), 1, size(x,3)), # [slen, tlen,  1, batch]
        Conv((1, 3),  1=>32, act, init=init_conv),      # [slen, tlen, 32, batch]
        Conv((1, 2), 32=>32, act, init=init_conv),
        x -> cat(                                       # [slen, 1, 2*32, batch]
            mean(x, dims=2),
            maximum(x, dims=2), dims=3),
        x -> reshape(x, :, size(x,4)),                  # [slen*2*32, batch]
        Dense(slen*2*32, conv_zlen,                     # [zlen, batch]
              initW=init_conv, initb=init_conv)
       )

    act = tanh
    densenet = Chain(
        x -> reshape(x[:,1:mintlen,:], :, size(x,3)),
        Dense(slen*mintlen, 50, act, initW=init_dense, initb=init_dense),
        Dense(50, 50, act, initW=init_dense, initb=init_dense),
        Dense(50, dense_zlen, initW=init_dense, initb=init_dense)
       )

    encoder = CatLayer(convnet, densenet)
end

function Parameters.reconstruct(m::CMeanGaussian{ScalarVar,FluxODEDecoder},
                     tlength::Int, dt::T, olength::Int) where T
    timesteps = range(T(0), step=dt, length=tlength)
    slen = m.mapping.slength
    ode = m.mapping.model
    H = m.mapping.observe
    dec = reconstruct(m.mapping, timesteps=timesteps)
    reconstruct(m, xlength=olength, mapping=dec)
end

Parameters.reconstruct(m::Rodent, tlen::Int, dt::Real, olen::Int) =
    Rodent(m.hyperprior,m.prior,m.encoder,reconstruct(m.decoder,tlen,dt,olen))

function init_diag(T, a::Int, b::Int)
    m = zeros(T,a,b)
    m[diagind(m,0)] .= T(1)
    return m
end
init_diag(a::Int,b::Int) = init_diag(T,a,b)

function GenerativeModels.elbo(m::Rodent, x::AbstractArray{T,3}; β=1) where T
    xf = reshape(x, :, size(x,3))

    μz = mean(m.encoder, x)
    σ2z = var(m.encoder, xf)
    rz = randn!(similar(μz))
    z = μz .+ sqrt.(σ2z) .* rz

    llh = sum(logpdf(m.decoder, xf, z))
    kld = sum(IPMeasures._kld_gaussian(μz,σ2z,mean_var(m.prior)...))
    lpλ = sum(logpdf(m.hyperprior, var(m.prior)))

    llh - β*(kld - lpλ)
end



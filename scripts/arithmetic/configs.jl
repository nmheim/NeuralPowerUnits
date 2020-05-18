using Parameters

@with_kw struct AddL1SearchConfig
    batch::Int      = 128
    niters::Int     = 1e5
    lr::Real        = 5e-3

    βstart::Real    = 1f-5
    βend::Real      = 1f-2
    βgrowth::Real   = 10f0
    βstep::Int      = 10000

    lowlim::Real    = -1
    uplim::Real     = 1
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0

    inlen::Int      = 100
    fstinit::String = "rand"
    sndinit::String = "rand"
    model::String   = "gatednpu"

    run::Int        = 1
end

@with_kw struct MultL1SearchConfig
    batch::Int      = 128
    niters::Int     = 1e5
    lr::Real        = 5e-3

    βstart::Real    = 1f-5
    βend::Real      = 1f-3
    βgrowth::Real   = 10f0
    βstep::Int      = 10000

    lowlim::Real    = -1
    uplim::Real     = 1
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0

    inlen::Int      = 100
    fstinit::String = "rand"
    sndinit::String = "rand"
    model::String   = "gatednpu"

    run::Int        = 1
end

@with_kw struct DivL1SearchConfig
    batch::Int      = 128
    niters::Int     = 1e5
    lr::Real        = 2e-3

    βstart::Real    = 1f-7
    βend::Real      = 1f-6
    βgrowth::Real   = 10f0
    βstep::Int      = 10000

    lowlim::Real    = 0
    uplim::Real     = 0.5
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0

    inlen::Int      = 100
    fstinit::String = "rand"
    sndinit::String = "rand"
    model::String   = "gatednpu"

    run::Int        = 1
end

@with_kw struct SqrtL1SearchConfig
    batch::Int      = 128
    niters::Int     = 1e5
    lr::Real        = 1e-2

    βstart::Real    = 1f-5
    βend::Real      = 1f-2
    βgrowth::Real   = 10f0
    βstep::Int      = 10000

    lowlim::Real    = 0
    uplim::Real     = 2
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0

    inlen::Int      = 100
    fstinit::String = "rand"
    sndinit::String = "rand"
    model::String   = "gatednpu"

    run::Int        = 1
end


using Distributions: Uniform
using Sobol

function ranges(xlen::Int,subset::Real,overlap::Real)
    len = round(Int, xlen*subset)
    ovl = round(Int, len*overlap)
    ii = 1:len
    jj = (len-ovl+1):(2len-ovl)
    (ii,jj)
end

function ranges(xlen::Int,subset::Real)
    len = round(Int, xlen*subset)
    ii = 1:len
end

function applyoperator(op::Function, X::Array, subset::Real, overlap::Real)
    (ii,jj) = ranges(size(X,1),subset,overlap)
    a = vec(sum(X[ii,:], dims=1))
    b = vec(sum(X[jj,:], dims=1))
    t = reshape(op.(a, b), 1, :)
end

function applyoperator(op::Function, x::Array, subset::Real)
    ii = ranges(size(x,1), subset)
    a = vec(sum(x[ii,:], dims=1))
    t = reshape(op.(a), 1, :)
end

invx(x::Real) = 1/x
invx(X::Array, subset::Real) = applyoperator(invx, X, subset)
add(X::Array, subset::Real, overlap::Real) = applyoperator(+, X, subset, overlap)
mult(X::Array, subset::Real, overlap::Real) = applyoperator(*, X, subset, overlap)
Base.Math.sqrt(X::Array, subset::Real) = applyoperator(sqrt, X, subset)


function arithmetic_invx_dataset(xlen::Int, subset::Real, lowlim::Real, uplim::Real; sampler="sobol")
    r = nothing
    function get_sampler()
        if sampler == "rand"
            return Uniform(0,1)
        elseif sampler == "sobol"
            return SobolSeq(xlen)
        else
            error("unknown sampler: $sampler")
        end
    end

    sample(s::SobolSeq, batch::Int) = reduce(hcat, [next!(s) for i = 1:batch])
    sample(s::Uniform, batch::Int) = rand(d,xlen,batch)

    r = get_sampler()

    function generate(batch::Int)
        X = Float32.(sample(r,batch) .* (uplim-lowlim) .+ lowlim)
        t = invx(X, subset)
        (X,t)
    end
end

function arithmetic_sqrt_dataset(xlen::Int, d::Uniform=Uniform(0,2), subset::Real=0.25)
    len = round(Int, xlen*subset)
    ii = 1:len

    function generate(batch::Int)
        X = Float32.(rand(d, xlen, batch))
        a = vec(sum(X[ii,:], dims=1))
        t = reshape(sqrt.(a), 1, :)
        (X,t)
    end
end

"""
arithmetic_dataset(op::Function, xlen::Int; d::Uniform=Uniform(-2,2)
                            subset::Real=0.25, overlap::Real=0.5)

Creates a function `generate(batchsize::Int)` that, when called, returns a
batch of inputs and labels as defined in the arithmetic task of the *Neural
Arithmetic Units* paper.
"""
function arithmetic_dataset(op::Function, xlen::Int; d::Uniform=Uniform(-2,2),
                            subset::Real=0.25, overlap::Real=0.5)
    len = round(Int, xlen*subset)
    ovl = round(Int, len*overlap)
    ii = 1:len
    jj = (len-ovl+1):(2len-ovl)
    @info "Arithmetic dataset: ($op). Sum indices: a=$ii b=$jj"

    if op == sqrt
        return arithmetic_sqrt_dataset(xlen, d, subset)
    elseif op == invx
        return arithmetic_invx_dataset(xlen, d, subset)
    end

    function generate(batch::Int)
        X = Float32.(rand(d, xlen, batch))
        a = vec(sum(X[ii,:], dims=1))
        b = vec(sum(X[jj,:], dims=1))
        t = reshape(op.(a, b), 1, :)
        (X,t)
    end
end

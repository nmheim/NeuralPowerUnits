using Distributions: Uniform


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
    jj = (len-ovl):(2len-ovl)
    @info "Arithmetic dataset sum indices: a=$ii b=$jj"

    function generate(batch::Int)
        X = Float32.(rand(d, xlen, batch))
        a = vec(sum(X[ii,:], dims=1))
        b = vec(sum(X[jj,:], dims=1))
        t = reshape(op.(a, b), 1, :)
        (X,t)
    end
end

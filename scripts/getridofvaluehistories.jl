using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using ValueHistories
using Flux
using NeuralArithmetic
include(srcdir("history.jl"))
include(srcdir("mvhistory.jl"))
include(joinpath(@__DIR__, "arithmetic", "configs.jl"))

function getdirofvaluehistories(h::ValueHistories.MVHistory)
    newh = MVHistory()
    for k in keys(h)
        idx, vals = get(h,k)
        for (i,v) in zip(idx,vals)
            push!(newh, k, i, v)
        end
    end
    newh
end

folders = ["addition_npu_l1_search"
          ,"mult_npu_l1_search"
          ,"sqrt_npu_l1_search"]
folders = map(datadir, folders)

@progress for folder in folders
    itr = walkdir(folder)
    (_,_,files) = first(itr)
    for bname in files
        fname = joinpath(folder, bname)
        res = load(fname)
        h = res[:history]
        if h isa ValueHistories.MVHistory
            @info "converting $bname"
            newh = getdirofvaluehistories(h)
            newres = copy(res)
            newres[:history] = newh
            backupfname = joinpath(folder, "oldmvhistory-$(bname)")
            mv(fname, backupfname)
            wsave(fname, newres)
        else
            @info "skipping $bname"
        end
    end
end

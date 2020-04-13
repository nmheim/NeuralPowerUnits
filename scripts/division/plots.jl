using Plots
using Parameters
using GMExtensions
pyplot()

include(joinpath(@__DIR__, "ard_utils.jl"))
get_mapping(m::Chain) = identity(m)

patterns = ["ard_xovery", "msel1_xovery", "msel2_xovery"]
frames = map(patterns) do pattern
    runs = load(datadir("pattern=$pattern.bson"))[:runs]
    mean_runs = aggregateruns(runs)
    sort!(mean_runs, :μmse)
end

for (pattern,df) in zip(patterns,frames)
    name = df[1,:name]
    res = load(datadir("division_$(pattern)_run1", name))
    @unpack model, history = res
    p1 = plothistory(history, title=name)
    net = get_mapping(model)
    p2 = plot(
        annotatedheatmap(net[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
        annotatedheatmap(net[2].W[end:-1:1,:], c=:bluesreds, title="ReNMUX", clim=(-1,1)),
        size=(600,300))
    wsave(plotsdir("division_xovery_best", "$(basename(name))-history.svg"), p1)
    wsave(plotsdir("division_xovery_best", "$(basename(name))-mapping.svg"), p2)
end


# config = Config()
# res, fname = produce_or_load(datadir("division_xovery"), config, run)
# 
# m = res[:model]
# h = res[:history]
# 
# pyplot()
# p1 = plothistory(h)
# net = get_mapping(m)
# p2 = plot(
#     annotatedheatmap(net[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
#     annotatedheatmap(net[2].W[end:-1:1,:], c=:bluesreds, title="ReNMUX", clim=(-1,1)),
#     size=(600,300))
# wsave(plotsdir("division_xovery", "$(basename(splitext(fname)[1]))-history.svg"), p1)
# wsave(plotsdir("division_xovery", "$(basename(splitext(fname)[1]))-mapping.svg"), p2)


using Plots
using Parameters
using GMExtensions
pyplot()

function aggregateruns(runs::DataFrame)
   mean_runs = by(runs, :name) do r
       (α0=first(r.α0),
        β0=first(r.β0),
        initnmu=first(r.initnmu),
        initnau=first(r.initnau),
        μmse=mean(r.mse),
        σmse=std(r.mse),
        μL1=mean(r.L1),
        σL1=std(r.L1),
        μL2=mean(r.L2),
        σL2=std(r.L2))
   end
end

function find_best(mean_df::DataFrame, df::DataFrame)
    name = mean_df[1,:name]
    df = filter!(row->row[:name]==name, copy(df))
    sort!(df, :mse)
    run = df[1,:run]
    name = df[1,:name]
    run, name
end


include(joinpath(@__DIR__, "ard_utils.jl"))
get_mapping(m::Chain) = identity(m)

patterns = ["ard_xovery", "msel1_xovery", "msel2_xovery"]

frames = map(patterns) do pattern
    runs = load(datadir("pattern=$pattern.bson"))[:runs]
end

mean_frames = map(frames) do runs
    mean_runs = aggregateruns(runs)
    sort!(mean_runs, :μmse)
end

for (pattern,mean_df,df) in zip(patterns,mean_frames,frames)
    run, name = find_best(mean_df, df)
    res = load(datadir("division_$(pattern)_run$run", name))
    @unpack model, history = res
    p1 = plothistory(history)
    net = get_mapping(model)
    p2 = plot(
        annotatedheatmap(net[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
        annotatedheatmap(net[2].W[end:-1:1,:], c=:bluesreds, title="ReNMUX", clim=(-1,1)),
        size=(600,300))
    wsave(plotsdir("division_xovery_best", "$pattern-$(basename(name))-history.svg"), p1)
    wsave(plotsdir("division_xovery_best", "$pattern-$(basename(name))-mapping.svg"), p2)
end

# Neural Arithmetic Units

This repository contains all the code to reproduce our NIPS submission. It is
written in Julia and all necessary dependencies can be installed from the Julia
REPL.  To start Julia with this project environment run `julia --project` from
this directory.
And then, to install all necessary dependencies, run `]instantiate` from the REPL.
```julia
$ julia --project
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.3.0 (2019-11-26)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> ]
(NIPS_2020_NPU) pkg> instantiate
...
```

All neural arithmetic units are defined in the [src](src) folder. The differnt types of NPUs
can be found [here](src/npu.jl).
Note that the three NPU types are not named as in the paper. The list
below maps paper NPU names to type names used in this repo.
```julia
"NPU"           => "GatedNPUX"
"NaiveNPU"      => "NPUX"
"real NPU"      => "GatedNPU"
```

All scripts that produce our experiments (Sec. 4) are in the [scripts](scripts) folder:

### 4.1 Fractional SIR identification: [fracode](scripts/fracode)

Run experiments and collect resulting data:
```julia
julia> include("scripts/fracode/run.jl")
julia> include("scripts/fracode/collect.jl")
```
All created models are stored in a `data` directory at the root of this repo.
Now you can create our plots by running
```julia
julia> include("scripts/fracode/model-ps.jl")  # note that this does not necessarily plot the best model!
julia> include("scripts/fracode/pareto-sir.jl")
```

### 4.2 Simple arithmetic task: [simple](scripts/simple)

Run experiments and produce the validation table
```julia
julia> include("scripts/simple/run.jl")
julia> include("scripts/simple/results_table.jl")
```
Create plot
```julia
julia> include("scripts/simple/plots.jl")
```


### 4.3 Large scale arithmetic task: [arithmetic](scripts/arithmetic)

Each learning task (addition, multiplication, division, sqrt) is contained
in a seperate script:
```julia
julia> include("scripts/arithmetic/add_l1_runs.jl")
julia> include("scripts/arithmetic/mult_l1_runs.jl")
julia> include("scripts/arithmetic/div_l1_runs.jl")
julia> include("scripts/arithmetic/sqrt_l1_runs.jl")
```
Collect all data and run the testing script
```julia
julia> include("scripts/arithmetic/collect.jl")
julia> include("scripts/arithmetic/revalidate.jl")
```
and finally produce the pareto plot and the results table
```julia
julia> include("scripts/arithmetic/pareto.jl")
julia> include("scripts/arithmetic/results_table.jl")
```

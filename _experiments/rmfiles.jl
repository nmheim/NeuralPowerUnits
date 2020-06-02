using DrWatson
@quickactivate "NIPS_2020_NMUX"

function find_invalid_filenames(folder, checkforzero = ["lr","βgrowth","βstart","βend","βstep"])
    (root,_,files) = first(walkdir(datadir(folder)))
    rmfiles = Set()
    for f in files
        display(savename(delete!(parse_savename(f)[2],"run")))
        display(f)
        error()
        
        config = parse_savename(f)[2]
        for k in checkforzero
            if config[k] == 0
                @info "Key $k of $(f) is zero."
                push!(rmfiles, joinpath(root, f))
            end
        end
    end
    return rmfiles
end


folders = ["addition_npu_l1_search"
          ,"mult_npu_l1_search"
          ,"sqrt_npu_l1_search"
          ,"div_npu_l1_search"]

rmfiles = union([find_invalid_filenames(f) for f in folders]...)

println("About to delete $(length(rmfiles)) files")
println("Really remove? (yes/no)")
answer = readline(stdin)

if answer == "yes"
    rm.(rmfiles)
    println("Removed")
else
    println("Cancelled")
end

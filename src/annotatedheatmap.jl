@userplot AnnotatedHeatmap
@recipe function f(h::AnnotatedHeatmap; annotationtexts=[], annotationargs=(:white,))
    y = h.args[1]              #Get the input arguments, stored in h.args 
                               # - in this case there's only one
    typeof(y) <: AbstractMatrix || error("Pass a Matrix as the arg to heatmap")

    grid := false                      # turn off the background grid
    xaxis := false
    yaxis := false

    @series begin                      # the main series, showing the heatmap
        seriestype := :heatmap
        y
    end

    rows, cols = size(y)

    #horizontal lines
    for i in 0:cols         # each line is added as its own series, for clearer code
        @series begin
            seriestype := :path
            primary := false          # to avoid showing the lines in a legend
            linewidth := 2
            linecolor --> :white
            [i, i] .+ 0.5, [0, rows] .+ 0.5  # x and y values of lines
        end
    end

    for i in 0:rows
        @series begin
            seriestype := :path
            primary := false
            linewidth := 2
            linecolor --> :white
            [0, cols] .+ 0.5, [i,i] .+ 0.5
        end
    end

    annotations = if length(annotationtexts) == 0
        text.(round.(reshape(y,:), digits=2), annotationargs...)
    else
        text.(annotationtexts, annotationargs...)
    end
    @series begin
        seriestype := :scatter
        # make the points transparent - setting marker (or seriestype)
        # to :none doesn't currently work right
        markerstrokecolor := RGBA(0,0,0,0.)
        seriescolor := RGBA(0,0,0,0.)
        series_annotations := annotations
        primary := false
        repeat(1:cols, inner = rows), repeat(1:rows, outer = cols)
    end
end

import Pkg
Pkg.activate(@__DIR__)

using Glob
using ModelComparison
using JSON
using Impute
using Missings
using ProgressMeter
using Random

@show Threads.nthreads()

function fill_missing(data)
    Impute.interp(data) |> Impute.locf() |> Impute.nocb() |> disallowmissing
end

function prepare(results; interval = 240)
    index, data = Int64[], allowmissing(Float64[])
    for result in results
        push!(index, result["timestamp"])
        push!(data, result["min"] > 0 ? result["min"] : missing)
    end
    resample_interval(index, data, 240)
end

function process(file)
    output = "$(file).models.json"
    @info "Processing $(file) => $(output)"

    results = parsefile(Vector{Dict}, file)
    index, data = prepare(results)

    # sklearn mixture models do not support missing observations,
    # so we fill missing data.
    data = fill_missing(data)

    Random.seed!(2020)
    mm = fit(FiniteMixtureModel{Normal}, data, 1:15, BIC, n_init = 3)

    Random.seed!(2020)
    dpmm = fit(InfiniteMixtureModel{Normal}, data, n_components = 15, n_init = 3)

    Random.seed!(2020)
    hmm = fit(FiniteHMM{Normal}, data, 1:15, estimator = fit_map)

    Random.seed!(2020)
    hdphmm = fit(InfiniteHMM, data, n_components = 15, iter = 250)

    results = Dict("MM" => mm, "DPMM" => dpmm, "HMM" => hmm, "HDPHMM" => hdphmm)

    write(output, json(results))
end

function main(args)
    files = glob("*.ndjson", args[1])
    @show length(files)

    p = Progress(length(files))

    #     Threads.@threads for file in files
    for file in files
        try
            # Retry once, then catch exception
            retry(process)(file)
        catch e
            showerror(stderr, e, catch_backtrace())
        end
        next!(p)
    end
end

main(ARGS)

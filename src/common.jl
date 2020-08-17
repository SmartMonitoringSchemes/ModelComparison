abstract type Model end

function fit(
    T::Type{<:Model},
    data::AbstractVector{<:Real},
    Ks::AbstractVector{Int},
    penalty::Function;
    kwargs...,
)
    models = []
    for K in Ks
        try
            model = fit(T, data, n_components = K)
            push!(models, (model, penalty(model, data)))
        catch e
            @error e
        end
    end
    models[argmin([x[2] for x in models])][1]
end

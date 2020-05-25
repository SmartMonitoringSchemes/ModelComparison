abstract type Model end

function fit(
    T::Type{<:Model},
    data::AbstractVector{<:Real},
    Ks::AbstractVector{Int},
    penalty::Function;
    kwargs...,
)
    models = map(Ks) do K
        model = fit(T, data, n_components = K)
        (model, penalty(model, data))
    end
    models[argmin([x[2] for x in models])][1]
end

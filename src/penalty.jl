nparams(d::T) where {T<:Distribution} = fieldcount(T)

nparams(model::MixtureModel) =
    ncomponents(model) - 1 + sum(d -> nparams(d), components(model))

nparams(model::HMM) = size(model, 1)^2 - size(model, 1) + sum(d -> nparams(d), model.B)

function BIC(model, data)
    ll = loglikelihood(model, data)
    k = nparams(model)
    n = length(data)
    k * log(n) - 2 * ll
end

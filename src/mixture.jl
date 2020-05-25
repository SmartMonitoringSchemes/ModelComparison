abstract type FiniteMixtureModel{C} <: Model end
abstract type InfiniteMixtureModel{C} <: Model end

# sklearn to Distributions.jl
function MixtureModel(o::PyObject)
    a = o.weights_
    B = [Normal(μ, sqrt(σ2)) for (μ, σ2) in zip(o.means_, o.covariances_)]
    MixtureModel(B, a)
end

# sklearn.mixture.GaussianMixture wrapper
function fit(::Type{FiniteMixtureModel{Normal}}, data::AbstractVector{<:Real}; kwargs...)
    data_ = reshape(data, :, 1)
    model = GaussianMixture(; kwargs...).fit(data_)
    MixtureModel(model)
end

# sklearn.mixture.BayesianGaussianMixture wrapper
function fit(::Type{InfiniteMixtureModel{Normal}}, data::AbstractVector{<:Real}; kwargs...)
    data_ = reshape(data, :, 1)
    model = BayesianGaussianMixture(; kwargs...).fit(data_)
    MixtureModel(model)
end

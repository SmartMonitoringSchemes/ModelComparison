abstract type FiniteHMM{C} <: Model end
abstract type InfiniteHMM{C} <: Model end

# https://maxmouchet.github.io/HMMBase.jl/stable/examples/fit_map/
function fit_map(::Type{<:Normal}, observations, responsibilities)
    μ = mean(observations, Weights(responsibilities))
    ss = suffstats(NormalKnownMu(μ), observations, responsibilities)
    prior = InverseGamma(0.5, 2)
    posterior = posterior_canon(prior, ss)
    σ2 = mode(posterior)
    Normal(μ, sqrt(σ2))
end

function weak_hdp_prior(data)
    obs_med, obs_var = robuststats(Normal, data)
    tp = TransitionDistributionPrior(Gamma(2, 10), Gamma(100, 10), Beta(500, 1))
    op = DPMMObservationModelPrior{Normal}(
        NormalInverseChisq(obs_med, obs_var, 1, 10),
        Gamma(1, 0.5),
    )
    BlockedSamplerPrior(1.0, tp, op)
end

# HMMBase.jl wrapper
function fit(
    ::Type{FiniteHMM{C}},
    data::AbstractVector{<:Real};
    n_components = 1,
    kwargs...,
) where {C}
    hmm = HMM(randtransmat(n_components), [C() for _ = 1:n_components])
    fit_mle(hmm, data; kwargs...)[1]
end

# HDPHMM.jl wrapper
function fit(
    ::Type{<:InfiniteHMM},
    data::AbstractVector{<:Real};
    n_components = 1,
    prior = weak_hdp_prior(data),
    kwargs...,
)
    segment(data, prior, L = n_components; kwargs...).model
end

module ModelComparison

using CodecZstd
using ConjugatePriors
using ConjugatePriors: NormalKnownMu
using HDPHMM
using PyCall
using StatsBase

# Re-export for convenience
# https://github.com/simonster/Reexport.jl
# https://github.com/JuliaLang/julia/issues/1986
using Reexport
@reexport using Distributions
@reexport using HMMBase

import Distributions: MixtureModel, fit
import HMMBase: nparams

export FiniteMixtureModel,
    FiniteHMM,
    InfiniteMixtureModel,
    InfiniteHMM,
    BIC,
    fit,
    fit_map,
    parsefile,
    weak_hdp_prior

const BayesianGaussianMixture = PyNULL()
const GaussianMixture = PyNULL()

function __init__()
    copy!(BayesianGaussianMixture, pyimport("sklearn.mixture").BayesianGaussianMixture)
    copy!(GaussianMixture, pyimport("sklearn.mixture").GaussianMixture)
end

include("common.jl")
include("penalty.jl")
include("mixture.jl")
include("hmm.jl")
include("io.jl")

end

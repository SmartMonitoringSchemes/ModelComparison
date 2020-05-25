using ModelComparison
using Test

data = rand(1000)

@test fit(FiniteMixtureModel{Normal}, data, n_components = 5, n_init = 3) isa MixtureModel
@test fit(FiniteMixtureModel{Normal}, data, 1:5, BIC) isa MixtureModel

@test fit(FiniteHMM{Normal}, data, n_components = 5, estimator = fit_map) isa HMM
@test fit(FiniteHMM{Normal}, data, 1:5, BIC, estimator = fit_map) isa HMM

@test fit(InfiniteMixtureModel{Normal}, data, n_components = 10, n_init = 3) isa MixtureModel
@test fit(InfiniteHMM, data, n_components = 10) isa HMM

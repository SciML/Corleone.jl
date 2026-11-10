module CorleoneDistributionsExtension

import Corleone
using Distributions: UnivariateDistribution, logpdf

Corleone.log_density(prior::UnivariateDistribution, x) = logpdf(prior, x)
Corleone.validate_prior(::UnivariateDistribution) = nothing

end

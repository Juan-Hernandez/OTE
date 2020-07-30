

# 1. Load packages

# 1.1 Distributions and non-linear solvers
using Distributions
using Statistics
using Roots
using Cubature #To solve integrals numerically.

# 1.2 Differential equations
using DifferentialEquations
using ForwardDiff
# 1.2.1 Extend prevfloat for duals
	Base.nextfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = ForwardDiff.Dual{T}(nextfloat(d.value), d.partials)
	Base.prevfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = ForwardDiff.Dual{T}(prevfloat(d.value), d.partials)
	Base.one(irr::AbstractIrrational) = one(irr*1)
# 1.3 Saving output
using DataFrames
using CSV
# 1.4 Make figures
fig_graphs && using PyPlot

# 2. Type definitions
include("definitions.jl")

# 3. Solving Functions
# 3.1 Initialization
include("initializeOTEmodel!.jl")
# 3.2 Runge-Kutta solver functions
include("fullrungekutta!.jl")
include("entderivatives!.jl")
include("entcontrols!.jl")
include("globalderivatives!.jl")
#include("globalcontrols!.jl")
include("globalcontrols1!.jl")

# 4 Results functions
# 4.0 Function to solve integrals from propositions and taxes.
include("Integrals.jl")
# 4.1 Taxes liabilities and marginal taxes:
include("taxes.jl")
# 4.2 Graphs:
include("graphs.jl")
# 4.3 Propositions:
# include("Propositions.jl")
# 4.4 Dataframes/Results:
include("resultsdata.jl")

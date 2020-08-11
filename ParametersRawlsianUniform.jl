## This file is in .gitignore as not to commit all testing changes in parameters.
## Save with other name to keep particular combinations of parameters

# 1. Economic parameters

ecopar=EconomicParameters(
	# 1.1 Workers' parameters
	2.0192,	# χ::Float64 	# Scale parameter for labor supply
	0.4528,	# ψ::Float64 	# Elasticity of labor supply
	# 1.2 Entrepreneurs' parameters
	0.73,	# α::Float64	# Returns to scale/labor elasticity of output
	0.2135,	# β::Float64	# Scale parameter for evasion
	0.1817,	# σ::Float64	# Elasticity of evasion
	10.0,	# ς::Float64	# Entrepreneurs' self-provision of labor hours
	# 1.3 Labor informality parameters
	false,	# informal::Bool	# Indicator of active-inactive informality
	0.1021,	# κ::Float64	# Workers' informal supply scale parameter
	0.0912,	# ρ::Float64	# Workers' informal supply elasticity
	0.12873,# δ::Float64	# Entrepreners' informal demand scale parameter
	0.7341,	# γ::Float64	# Entrepreneurs' informal demand elasticity
	# 1.4 Planner's parameters
	false,	# utilit::Bool	# Indicator of utilitarian planner
	0.1,	# ϕ::Float64 	# Concave utilitarian parameter
	0.15 	# G::Float64	# Expenditure needs
	)	# Close constructor call

# 2. Distribution parameters
# 2.1 Type of distribution
uniform = true	# ::Bool	# Indicator of uniform distributions.
if uniform # Uniform case
	# 2.2 Distribution moments
	μ_w=10.0	# ::Float64	# Mean of worker's ability
	μ_e=10.0	# ::Float64	# Mean of entrepreneur's ability
	σ2_w=6.0	# ::Float64	# Variance of worker's ability
	σ2_e=6.0	# ::Float64	# Varaince of entrepreneur's ability
	σ_we=0.0	# ::Float64	# Covariance of abilities.
	# 2.3 Distribution support
	θ_w_l=μ_w-((12.0^0.5)/2.0)*(σ2_w^0.5)	# ::Float64	# Lower bound of worker's ability
	θ_w_u=μ_w+((12.0^0.5)/2.0)*(σ2_w^0.5)	# ::Float64	# Upper bound of worker's ability
	θ_e_l=μ_e-((12.0^0.5)/2.0)*(σ2_e^0.5)	# ::Float64	# Lower bound of entrepreneurs's ability
	θ_e_u=μ_e+((12.0^0.5)/2.0)*(σ2_e^0.5)	# ::Float64	# Upper bound of entrepreneurs's ability
else # Log-Normal case
	#Define the bounds for the distributions: we need to limit the normal distribution
	constant_w_l = 5.0e-2  # ::Float64	# Percentage of worker's distribution we are not taking in lower bound
	constant_w_u = 1.0-0.1 # ::Float64	# Percentage of worker's distribution we are not taking in upper bound
    constant_e_l = 5.0e-2  # ::Float64	# Percentage of entrepreneur's distribution we are not taking in lower bound
    constant_e_u = 1.0-0.1 # ::Float64	# Percentage of entrepreneur's distribution we are not taking in upper bound
	# 2.2 Distribution moments
	μ_w  = 1.7626	# ::Float64	# Mean of worker's ability
	μ_e  = 1.2528	# ::Float64	# Mean of entrepreneur's ability
	σ2_w = 1.0921	# ::Float64	# Variance of worker's ability
	σ2_e = 1.1675	# ::Float64	# Varaince of entrepreneur's ability
	σ_we = 0.0		# ::Float64	# Covariance of abilities.
	# 2.3 Distribution support
	log_θ_w_l = quantile(Normal(μ_w,σ2_w^0.5),constant_w_l); # ::Float64 # Lower bound of the logarithmic worker's distribution
	θ_w_l     = exp(log_θ_w_l) 				# ::Float64	# Lower bound of worker's ability
	log_θ_e_l = quantile(Normal(μ_e,σ2_e^0.5),constant_e_l); # ::Float64 # Lower bound of the logarithmic entrepreneur's distribution
	θ_e_l     = exp(log_θ_e_l) 				# ::Float64	# Lower bound of entrepreneur's ability
	log_θ_w_u = quantile(Normal(μ_w,σ2_w^0.5),constant_w_u); # ::Float64 # Lower bound of the logarithmic worker's distribution
	θ_w_u     = exp(log_θ_w_u) 				# ::Float64	# Lower bound of worker's ability
	log_θ_e_u = quantile(Normal(μ_e,σ2_e^0.5),constant_e_u); # ::Float64 # Lower bound of the logarithmic entrepreneur's distribution
	θ_e_u     = exp(log_θ_e_u) 				# ::Float64	# Lower bound of entrepreneur's ability
end # end if
# 2.4 Constructor call
dispar=DistributionParameters(uniform, μ_w, μ_e, σ2_w, σ2_e, σ_we, θ_w_l, θ_w_u, θ_e_l, θ_e_u )

# 3. Computation parameters
compar=ComputationParameters(
	500,	# globalsize::Int64
	500,	# entsize::Int64
	1e-8,	# abstol::Float64
	1e-5,	# reltol::Float64
	false,	# verbosebool::Bool
	false,	# debugbool::Bool
	 Rosenbrock23()	# alg::T where T<:OrdinaryDiffEqAlgorithm
	# Tsit5()		# alg::T where T<:OrdinaryDiffEqAlgorithm
	# RK4()
	) # Close constructor call

# 4. Initial guess for prices and final states/costates
# 4.1 Guess for prices
λ_bar	=   1.0
ω_bar	=   1.34318
priceguess= [λ_bar, ω_bar]
# 4.2 Guess for final states
u_u		=   640.7
ew_u	=   dispar.θ_e_u*(1.0-0.007)
New_State_u	=	NaN # Ve*he @ ew_u (filled after entrepreneurs' problem)
μ_u		=   0.0 # Fixed
L_u		=   0.0	# Fixed
Y_u		=   0.0	# Fixed
finalstateguess=[ew_u, New_State_u, u_u, μ_u, L_u, Y_u]

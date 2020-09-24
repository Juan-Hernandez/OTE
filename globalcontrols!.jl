function globalcontrols!(controls::AbstractArray{T,1}, ss::Array{T,1}, prices::Array{Float64,1}, pa::EconomicParameters, verbosebool::Bool) where T<:Real
	# INPUT: control vector, state vector and parameters
	# MODIFIES: control vector
	debugbool::Bool=false
# 1.0 Extracting
	θ::T   =ss[1]
	e::T   =ss[2]
	New_A::T =ss[3]
	u::T   =ss[4]
	μ::T   =ss[5]
	h_e::T =ss[6]
	h_w::T =ss[7]
	λ::Float64=prices[1]
	ω::Float64=prices[2]

# 1.1 Initialize constants ̇
	z_u::T    = (1.0/pa.β)^(1.0/pa.σ)*(1.0 - 1e-10) #Max possible evasion.
	ln_max::T = 1e10
	n_full_info::T = (λ*pa.α*e/ω)^(1.0/(1.0-pa.α))

# 1.2 Define aux functions:
	nfoc(nvar, zvar) = λ*pa.α*e/(1.0-pa.β*zvar^pa.σ)*h_e - ω*nvar^(1.0-pa.α)/(1.0-pa.β*zvar^pa.σ)*h_e-pa.α*New_A;
	z_opt(nvar)      = pa.σ/λ*nvar^pa.α/h_e*New_A;
	nfoc_at_zopt(nvar) = nfoc( nvar, z_opt(nvar) );
	den_p(nvar,zvar)   = θ*nvar^pa.α*(1.0-pa.β*zvar^pa.σ)

# 2 Define cases and solve
# 2.0 We solve first l, as it doesn't depend on other variables:
den_l = ( λ*pa.χ*h_w - pa.χ/θ*(1.0+pa.ψ)*( μ + New_A) )
den_l <= 0.0 ? (controls[3]=ln_max*h_w) : (controls[3]=(ω*θ*h_w/den_l)^(1.0/pa.ψ) )
verbosebool && println("l = ", controls[3], " den_l = ", den_l)

# 2.1 Case 1: he = 0
	if h_e<1e-10 # h_e=0.0
		verbosebool && println("Case 1: h_e=0.0")
		if New_A < 0.0
			controls[1] = ln_max  # n_foc always positive
			controls[2] = 0.0     # z_foc always negative
		else
			controls[1] = pa.ς # n_foc always negative
			controls[2] = min(e*pa.ς^pa.α, (1.0/pa.β)^(1.0/pa.σ)-1e-10) # zfoc always positive, so keep the higher value
		end #end if
		#Solve for p:
		controls[4] = (pa.χ*controls[3]^(1.0+pa.ψ))/den_p(controls[1],controls[2])

		verbosebool && println("n = ", ForwardDiff.value(controls[1]),
							   ",  z = ", ForwardDiff.value(controls[2]),
							   ",  l = ", ForwardDiff.value(controls[3]),
							   ",  p = ", ForwardDiff.value(controls[4]) )

		return nothing
	end

# 2.2 Case 2: New_A = (Ve*he+ϕe)/̇ue ≦ 0
	# 2.2.1 We evaluate the interior solution for n and z:
	if New_A*controls[3] <= 0.0
		verbosebool && println("Case 2: (Ve*he+ϕe)/̇ue ≦ 0.")
		controls[2] = 0.0 # z_foc always negative
		controls[1] = (λ/ω*pa.α*e - pa.α/ω*New_A/h_e)^(1.0/(1.0-pa.α)); # Interior solution for n in z=0.0
		controls[4] = (pa.χ*controls[3]^(1.0+pa.ψ))/den_p(controls[1],controls[2])

		#Final Output:
			verbosebool && println("n = ", ForwardDiff.value(controls[1]),
									",  z = ", ForwardDiff.value(controls[2]),
									",  l = ", ForwardDiff.value(controls[3]),
									",  p = ", ForwardDiff.value(controls[4]) )

		return nothing
	end

# 2.3 Case 3: New_A = (Ve*he+ϕe)/̇ue > 0
	verbosebool && println("Case 3: (Ve*he+ϕe)/̇ue > 0.")
	# 2.3.1 We find an interior solution for n and z:
	# Using bisection method to find the value of n:
	if nfoc_at_zopt(pa.ς)*nfoc_at_zopt(n_full_info)<0
		controls[1] = find_zero(nfoc_at_zopt, (pa.ς,n_full_info), Bisection());

		z_candidate1 = z_opt(controls[1])
		z_candidate2 = z_u
		z_candidate3 = e*controls[1]^pa.α - ω/λ*(controls[1]-pa.ς)
		controls[2]  = min(z_candidate1, z_candidate2, z_candidate3)
		controls[4] = (pa.χ*controls[3]^(1.0+pa.ψ))/den_p(controls[1],controls[2])
	else
		# 2.3.2 We evaluate the corner solution for n = ̄m:
			controls[1] = pa.ς
			controls[2] = z_opt(controls[1])
			controls[4] = (pa.χ*controls[3]^(1.0+pa.ψ))/den_p(controls[1],controls[2]);
	end

	verbosebool && println("n = ", ForwardDiff.value(controls[1]),
							",  z = ", ForwardDiff.value(controls[2]),
							",  l = ", ForwardDiff.value(controls[3]),
							",  p = ", ForwardDiff.value(controls[4]) )

	return nothing

end # end function

function globalcontrols(ss::Array{T,1}, prices::Array{Float64,1}, pa, verbosebool::Bool=true) where T<:Real
	controls=Array{T}(undef,4)
	globalcontrols!(controls, ss, prices, pa, verbosebool)
	return Tuple(controls)
end

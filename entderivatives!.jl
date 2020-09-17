function entderivatives!(du,u,pa,e)
# 0. Unpack
	uu	= u[1]
	μ 	= u[2]
	λ	= pa.prices[1]
	ω	= pa.prices[2]
# 1. Obtain densities, consider out of domain cases
	if e<pa.dispar.θ_e_l || e>pa.dispar.θ_e_u
		h_e=zero(e)
	else
		h_e=pa.modist.he(pa.dispar.θ_w_u, e)
	end

# 2. Construct state object
	vv = [e, uu, μ, h_e]
	#println("he = ", h_e, ", vv = ", ForwardDiff.value.(vv))
	# println(isa(e,Float64),isa(uu,Float64),isa(μ,Float64),isa(h_e,Float64))
	# println(isa(vv[1],Float64),isa(vv[2],Float64),isa(vv[3],Float64),isa(vv[4],Float64))
# 3. Find optimal controls
	(nn, zz) = entcontrols( vv, pa.prices, pa.ecopar, pa.compar.debugbool)

# 4. Create derivatives and return
	du[1] = nn^pa.ecopar.α*(1.0 - pa.ecopar.β*zz^pa.ecopar.σ)
	du[2] = (λ - pa.ecopar.utilit*pa.ecopar.ϕ*uu^(pa.ecopar.ϕ-1.0))*h_e
	du[3] = -nn*h_e
	du[4] = (e*nn^pa.ecopar.α - pa.ecopar.β/(1.0+pa.ecopar.σ)*zz^(1.0+pa.ecopar.σ) - uu)*h_e
	# println(du)
	return nothing
end

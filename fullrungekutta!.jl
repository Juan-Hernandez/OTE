function fullrungekutta!(model::OTEmodel)
# 0. Unpack and preallocate
# 0.1 Unpack
	globalsize  = model.compar.globalsize
	fullsize	= model.compar.entrepsize+globalsize-1
	α = model.ecopar.α
	β = model.ecopar.β
	σ = model.ecopar.σ
	ψ = model.ecopar.ψ
	χ = model.ecopar.χ
	λ = model.prices[1]
	ω = model.prices[2]
# 0.2 Pre-allocate
	h_e::Float64=NaN
	h_w::Float64=NaN
	e::Float64=NaN
	θ::Float64=NaN
	lenght_sol::Int64=0
# 0.3 Auxiliary functions
	Ve(nvar,zvar,evar,uvar)= ( model.ecopar.utilit*uvar^model.ecopar.ϕ + model.prices[1]*(evar*nvar^α
	 		- β/(1.0+σ)*zvar^(1.0+σ) - uvar) - model.prices[2]*(nvar-model.ecopar.ς) )

# 1. Define parameters and callback setm
# 1.1 Build parameter struct
	rkpar       = RKParameters(model.ecopar, model.compar, model.dispar, model.modist, model.prices)

# 1.2 Callback set
	# 1.2.1 Domain Callback
	tempstate	= ones(6)
	callback_domain=GeneralDomain(statesdomain!, tempstate, save=false)
	# 1.2.2 Θ_e_l callback (terminate integration, functions defined below)
	callback_Θ_e_l=ContinuousCallback(condition_Θ_e_l, affect_Θ_e_l!; save_positions=(false,true) )
	# 1.2.3 Create Callback Set
	cbset=CallbackSet(callback_Θ_e_l, callback_domain)
# 2. Entrepreneurs' problem
# 2.1 Define ODE problem
	entprob  = ODEProblem(entderivatives!, model.states[4:7, fullsize], (model.states[2, fullsize], model.states[2, globalsize]), rkpar)
# 2.2 Solve ODE problem: no callback
	entsolution = solve(entprob, model.compar.alg, abstol=model.compar.abstol, reltol=model.compar.reltol,
						saveat=reverse(model.states[2, globalsize:fullsize]) )
# 2.3 Fill states and controls
	for i = fullsize:-1:globalsize
		# states[1, i]=solution_RKPack.t[i]
		#Update vector of states:
		model.states[4:7, i ] = entsolution[fullsize+1-i]
		e=model.states[2,i]
		h_e = model.modist.he(model.dispar.θ_w_u, e)
		entcontrols!(view(model.controls,1:2, i), [e, model.states[4:5, i]... , h_e],
						model.prices, model.ecopar, model.compar.verbosebool)
		model.compar.verbosebool && println("States: ", model.states[2:7, i])
		model.compar.verbosebool && println("Controls: ", model.controls[1:2,i], " he: ",h_e)
	end
	#Fill for the value of l and p in θ_w_u:
	model.controls[3,globalsize] = ( ω*model.states[1,globalsize]/(λ*χ) )^(1.0/ψ)
	model.controls[4,globalsize] = ( χ/model.states[1,globalsize]*model.controls[3,globalsize]^(1.0+ψ)/(model.controls[1,globalsize]^α*(1.0-β*model.controls[2,globalsize]^σ)) )

# 3. Global problem
# 3.1 Calculate ϕ_e from states and controls at globalsize
	# Recall e and he are set at globalsize by previous for loop.
	#model.states[3,globalsize]= -( Ve(model.controls[1,globalsize], model.controls[2,globalsize],e,model.states[4,globalsize])*h_e
	#			+ model.states[5,globalsize]*model.controls[1,globalsize]^α*(1.0-β*model.controls[2,globalsize]^σ) )
	model.states[3,globalsize]= - model.states[5,globalsize]

# 3.2 Define and solve ODE problem
	globalprob	= ODEProblem(globalderivatives!, model.states[2:7, globalsize],(model.states[1, globalsize], model.states[1, 1]), rkpar)
	globalsolution = solve(globalprob, model.compar.alg, abstol=model.compar.abstol, reltol=model.compar.reltol,
						callback=cbset, saveat=Iterators.reverse(model.states[1, 1:globalsize]))
						#, maxiters = 1e7 )
# 3.3 Fill states and controls
	(nstates, lenght_sol)=size(globalsolution)
	model.states[1, 1:(globalsize-lenght_sol)].=NaN
	firstind=globalsize-lenght_sol+1
	for i = (globalsize-1):-1:firstind
		model.states[1, i]=globalsolution.t[globalsize-i+1] # Must do for first θ
		θ=model.states[1,i]
		#Update vector of states:
		model.states[2:7, i] = globalsolution[globalsize-i+1]
		e=model.states[2,i]
		h_e = model.modist.he(θ, e)
		h_w = model.modist.hw(θ, e)
		globalcontrols!(view(model.controls,:,i), [model.states[1:5,i]..., h_e, h_w], model.prices, model.ecopar, false)
		model.compar.verbosebool && println("States: ", model.states[:,i])
		model.compar.verbosebool && println("[he, hw]: ", [h_e,h_w])
		model.compar.verbosebool && println("Controls: ", model.controls[:,i])
		model.compar.verbosebool && println("Ve*he+phi: ", Ve(model.controls[1:2, i]..., e, model.states[4, i])*h_e + model.states[3, i] )
	end

# 4. Print initial state for boundary conditions
	Ve_first= Ve(model.controls[1:2,firstind]..., e, model.states[4,firstind])
	println("θw = ", model.states[1,firstind], ". Model States: θe = ", model.states[2,firstind], ", ϕe = ", model.states[3,firstind],
			", u = ", model.states[4,firstind], ", μ = ", model.states[5,firstind], ", L = ", model.states[6,firstind],
			", Y = ", model.states[7,firstind] )
	println("Controls: n = ", model.controls[1,firstind], ", z = ", model.controls[2,firstind],
			", l = ", model.controls[3,firstind], ", p = ", model.controls[4,firstind])
	println("Prices: λ = ", model.prices[1], ", ω = ", model.prices[2])
	println("Ve*he+phi = ", Ve_first*h_e+model.states[3,firstind] )
	return lenght_sol
end

function statesdomain!(resid,u,p,t)
	resid[1]=tanh((u[1]-p.dispar.θ_e_l+1e-4)*1e7)-1.0
	resid[2]=tanh((p.dispar.θ_e_u-u[1]-1e-3)*1e7)-1.0
	resid[3]=tanh((u[3]-0.1)*1e7)-1.0
	resid[4:6].=0.0
end

function condition_Θ_e_l(u,t,integrator)
	u[1]-integrator.p.dispar.θ_e_l*(1+integrator.p.compar.reltol*10)
end

function affect_Θ_e_l!(integrator)
	println("hit")
	terminate!(integrator)
end

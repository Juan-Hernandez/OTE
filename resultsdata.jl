function modeldata(model::OTEmodel,lenght_sol::Int64)
# 0 Unpack:
	# 0.1 Prices:
	λ	= model.prices[1]
	ω	= model.prices[2]
	# 0.2 Parameters:
	α = model.ecopar.α
	β = model.ecopar.β
	σ = model.ecopar.σ
	χ = model.ecopar.χ
	ψ = model.ecopar.ψ
	ς = model.ecopar.ς
	ϕ = model.ecopar.ϕ
	utilit = model.ecopar.utilit
	# 0.3 Size of the problem:
	globalsize = model.compar.globalsize
	fullsize   = model.compar.entrepsize+globalsize-1
	first_sol_Global = globalsize - lenght_sol + 1 #The first θw where there is a solution in the global problem
	# 0.4 Elasticities for the propositions:
	ε_l_1Tl′ = 1.0/ψ;
	ε_l_θw   = ε_l_1Tl′;
	ε_z_Tc′  = 1.0/σ;
	# 0.4 Define the matrices we are going to use:
	MarginalTaxes = Array{Float64,2}(undef,fullsize,3) #For marginal taxes
	taxliabilities = Array{Float64,2}(undef,fullsize,6) #For taxes liabilities
	controls_full = Array{Float64,2}(undef,fullsize,2); #The matrix for controls full information
	debug = Array{Float64,2}(undef,fullsize,7); #The matrix for debug
	Propositions = Array{Float64}(undef,fullsize,7) #Order: Proposition 1, Proposition 2, Proposition 3
	FOCs = Array{Float64}(undef,fullsize,4)
	for i in [MarginalTaxes,taxliabilities,controls_full,debug,Propositions,FOCs]
		fill!(i,NaN);
	end #end for

# 1. Define some of the elements for the dataframe:
	for i in first_sol_Global:fullsize
		# 1.1 Productivities and States:
		θw  = model.states[1,i]
		e 	= model.states[2,i]
		u	= model.states[4,i]
		μ	= model.states[5,i]
		L	= model.states[6,i]
		Y	= model.states[7,i]
		# 1.2 Controls:
		n	= model.controls[1,i]
		z	= model.controls[2,i]
		l	= model.controls[3,i]
		p	= model.controls[4,i]
		# 1.3 Marginal taxes:
		MarginalTaxes[i,1] = ( λ*α*e*n^(α-1.0) - ω )/ω #Tn'
		MarginalTaxes[i,2] = β*z^σ #Tc'
		MarginalTaxes[i,3] = ( θw*ω-λ*χ*l^ψ )/(θw*ω) #Tl'
		# 1.4 Values for the planner:
		Vw = (utilit*u^ϕ - λ*u) + ω*l*θw - λ*χ/(1.0+ψ)*l^(1.0+ψ);
        Ve =  utilit*u^ϕ + λ*( e*n^α - β/(1.0+σ)*z^(1.0+σ) - u ) - ω*( n-ς );
		ϕe = model.states[3,i]*( n^α*(1.0-β*z^σ) ) - Ve*model.modist.he(θw, e);
		FOCs[i,4] = model.states[3,i]
		model.states[3,i] = ϕe

		# 1.4 Taxes Liabilities (also includes the tax base):
			# 1.4.1 Tax bases (Recall that for Tc we need to have the value of Tn):
			taxliabilities[i,1] = ω*( n-ς ) #Tn
			taxliabilities[i,3] = θw*ω*l #Tl
			# 1.4.2 Tax liabilities (Recall that for Tc we need to have the value of Tn):
			# We get the value of Tl from the definition of uw, we solve Tn and Tc through integrals
			taxliabilities[i,6] = θw*ω/λ*l - χ/(1.0+ψ)*l^(1.0+ψ) - u #Tl
			if i == first_sol_Global
				#This is the first value for the integral
				taxliabilities[i,4] = 0.0 #Tn
				taxliabilities[i,5] = e*n^α - ω/λ*(n -ς) - β/(1.0+σ)*z^(1.0-σ) - u #Tc
				#We give the value of the base for Tc:
				taxliabilities[i,2] = e*n^α - ω*( n-ς ) - taxliabilities[i,4] - z #Tc
			else
				# Get the values of the integrals:
				for j in [1,2] #We are solving for Tn, and Tc.
					if j == 2 #This is the tax base for Tc
						taxliabilities[i,2] = e*n^α - ω*( n-ς ) - taxliabilities[i,4] - z #Tc
					end #end if
					#We are solving the integrals with the trapezoidal rule.
					previous_sol = taxliabilities[i-1,j+3]
					average_functions = (MarginalTaxes[i-1,j]+MarginalTaxes[i,j])/2.0
					distance_interval = (taxliabilities[i,j]-taxliabilities[i-1,j])
					sol_int      = NaN
					sol_int      = previous_sol + average_functions*distance_interval;
					taxliabilities[i,j+3] = sol_int
				end #end for
			end #end if

		#1.5 Propositions: Here we put the information for the things we don't need to integrate
		# 1.5.1 The first proposition:
			# Right-hand side of the equation:
			Propositions[i,2] = MarginalTaxes[i,3]/(1.0-MarginalTaxes[i,3])*ε_l_1Tl′/(1.0+ε_l_1Tl′)*θw*model.modist.hw(θw, e)
			if i <= globalsize
				Propositions[i,3] = ε_z_Tc′*z/n^α*model.modist.he(θw, e)
			else #This is for the entrepreneurs problem.
				Propositions[i,3] = ε_z_Tc′*z/n^α*model.modist.he(model.dispar.θ_w_u, e) #We keep θ_w_u for the entrepreneurs' problem
			end  #end if
		# 1.5.2 The second proposition
			Propositions[i,4] = ε_z_Tc′*z/n^α
			Propositions[i,5] = MarginalTaxes[i,1]/( (1.0+MarginalTaxes[i,1])*(1.0-MarginalTaxes[i,2]) )*e

		# 1.6 Full information controls
			controls_full[i,1]=(α*λ*e/ω)^(1.0/(1.0-α)) #n full information
			controls_full[i,2]=((θw*ω)/(λ*χ))^(1.0/ψ) #l full information

		# 1.7 Debug variables (marginal utilities, A and max evasion)
		debug[i,1] = model.modist.hw(θw, e) #h_w
		if i <= globalsize
			debug[i,2] = model.modist.he(θw, e) #h_e
		else #This is for the entrepreneurs problem
			debug[i,2] = model.modist.he(model.dispar.θ_w_u, e) #h_e
		end  #end if
		debug[i,3] = χ/θw*l^(1.0+ψ); #u_w'
		debug[i,4] = n^α*(1.0-β*z^σ); #u_e'
		debug[i,5] = ω*ς+(utilit*u^ϕ-λ*u)+ϕe/debug[i,2]; #A
		debug[i,6] = -(1.0-α)/α*ω*controls_full[i,1]; #A if z is 0 and n_full_info
		debug[i,7] = e*controls_full[i,1]^α - ω/λ*(controls_full[i,1]-ς); #Max posible evasion

		#Solving the FOCs:
		FOCs[i,1] = p/n*α*debug[i,2]*( -(1.0-α)/α*ω*n + λ*β/(1.0+σ)*z^(1.0+σ) - debug[i,5] )
		FOCs[i,2] = σ*β*z^(σ-1.0)/(1.0-β*z^σ)*p*debug[i,2]*( λ*e*n^α - ω*n - λ/σ*z*( 1.0 - β/(1.0+σ)*z^σ ) + debug[i,5] )
		FOCs[i,3] = l^ψ*( χ/θw*(1.0+ψ)*μ - λ*χ*debug[i,1] ) + ω*θw*debug[i,1] + (1.0+ψ)*p/l*debug[i,2]*( λ*e*n^α - ω*n - λ*β/(1.0+σ)*z^(1.0+σ) + debug[i,5] )
	end # end for

	#Now we solve the integrals in reverse
	#We now get the value of the integral for the entrepreneurs'problem in proposition 3
	for i in fullsize:-1:globalsize+1
		if i == fullsize
			#This is the first value for the integral
			Propositions[i,7] = 0.0 #Is the value for the right side of the equation of proposition 3
		else
			# Get the values of the integrals:
			#We are solving the integrals with the trapezoidal rule.
			previous_sol = Propositions[i+1,7]
			average_functions = ( (1.0-utilit*ϕ*model.states[4,i]^(ϕ-1.0)/λ)*(model.modist.he(model.dispar.θ_w_u, model.states[2,i])) +
								  (1.0-utilit*ϕ*model.states[4,i+1]^(ϕ-1.0)/λ)*(model.modist.he(model.dispar.θ_w_u, model.states[2,i+1])) )/2.0
			distance_interval = (model.states[2,i]-model.states[2,i+1])
			sol_int      = NaN
			sol_int      = previous_sol + average_functions*distance_interval;
			Propositions[i,7] = sol_int
		end # end if
	end #end for
	#Solve integrals in global problem
	for i in globalsize:-1:first_sol_Global
		if i == globalsize
			#This is the first value for the integral
			Propositions[i,1] = 0.0 #Is the value for μ in the upper bound
			Propositions[i,6] = 0.0 #Is the value for the left side of the equation of proposition 3
			Propositions[i,7] = 0.0 #Is the value for the right side of the equation of proposition 3
		else
			# Get the values of the integrals:
			for j in [1,6,7] #We are solving for the integral.
				#We are solving the integrals with the trapezoidal rule.
				previous_sol = Propositions[i+1,j]
				if j == 1
					average_functions = ( (1.0-utilit*ϕ*model.states[4,i]^(ϕ-1.0)/λ)*(debug[i,1]+debug[i,2]*model.controls[4,i]) +
										  (1.0-utilit*ϕ*model.states[4,i+1]^(ϕ-1.0)/λ)*(debug[i+1,1]+debug[i+1,2]*model.controls[4,i+1]) )/2.0
				elseif j == 6
					#Define Vw and Ve for one of the integrals in proposition 3:
			  		Vw  = (utilit*model.states[4,i]^ϕ - λ*model.states[4,i]) + ω*model.controls[3,i]*model.states[1,i] - λ*χ/(1.0+ψ)*model.controls[3,i]^(1.0+ψ);
			  		Ve  = (utilit*model.states[4,i]^ϕ - λ*model.states[4,i]) + λ*model.states[2,i]*model.controls[1,i]^α - λ*β/(1.0+σ)*model.controls[2,i]^(1.0+σ) - ω*( model.controls[1,i]-ς );
					Vw1 = (utilit*model.states[4,i+1]^ϕ - λ*model.states[4,i+1]) + ω*model.controls[3,i+1]*model.states[1,i+1] - λ*χ/(1.0+ψ)*model.controls[3,i+1]^(1.0+ψ);
			  		Ve1 = (utilit*model.states[4,i+1]^ϕ - λ*model.states[4,i+1]) + λ*model.states[2,i+1]*model.controls[1,i+1]^α - λ*β/(1.0+σ)*model.controls[2,i+1]^(1.0+σ) - ω*( model.controls[1,i+1]-ς );
					average_functions = ( 1.0/λ*(Ve - Vw)*model.modist.gg(model.states[1,i], model.states[2,i])/(model.controls[1,i]^α*(1.0-β*model.controls[2,i]^σ) ) +
										  1.0/λ*(Ve1 - Vw1)*model.modist.gg(model.states[1,i+1], model.states[2,i+1])/(model.controls[1,i+1]^α*(1.0-β*model.controls[2,i+1]^σ) ) )/2.0
				else
					average_functions = ( (1.0-utilit*ϕ*model.states[4,i]^(ϕ-1.0)/λ)*(model.controls[4,i]*model.modist.he(model.states[1,i], model.states[2,i])) +
										  (1.0-utilit*ϕ*model.states[4,i+1]^(ϕ-1.0)/λ)*(model.controls[4,i+1]*model.modist.he(model.states[1,i+1], model.states[2,i+1])) )/2.0
				end # end if
				distance_interval = (model.states[1,i]-model.states[1,i+1])
				sol_int      = NaN
				sol_int      = previous_sol + average_functions*distance_interval;
				Propositions[i,j] = sol_int
			end #end for
		end #end if
	end # end for

# 2. Define our dataframe:
	# Results_DF = DataFrame( hcat(transpose(model.states),transpose(model.controls),MarginalTaxes,
	# 						taxliabilities, Propositions, controls_full, debug),
	# 						[:θw, :θe, :ϕe, :u, :μ, :L, :Y, :n, :z, :l, :p, :Tn′, :Tc′, :Tl′,
	# 						:baseTn, :baseTc, :baseTl, :Tn, :Tc, :Tl, :Prop1LSDebug, :Prop1RS1, :Prop1RS2,
	# 						:Prop2LS1, :Prop2RS1, :Prop3LS2, :Prop3RS1, :nfull, :lfull,
	# 						:hw, :he, :uw′, :ue′, :A, :Afull, :MaxEvasion] )

	Results_DF = DataFrame( hcat(transpose(model.states),transpose(model.controls),MarginalTaxes,
							taxliabilities, Propositions, controls_full, debug, FOCs),
							[:θw, :θe, :ϕe, :u, :μ, :L, :Y, :n, :z, :l, :p, :Tn′, :Tc′, :Tl′,
							:baseTn, :baseTc, :baseTl, :Tn, :Tc, :Tl, :Prop1LSDebug, :Prop1RS1, :Prop1RS2,
							:Prop2LS1, :Prop2RS1, :Prop3LS2, :Prop3RS1, :nfull, :lfull,
							:hw, :he, :uw′, :ue′, :A, :Afull, :MaxEvasion, :FOCn, :FOCz, :FOCl, :NewState] )

	Results_DF = Results_DF[first_sol_Global:fullsize,:]
	return Results_DF

end # end function

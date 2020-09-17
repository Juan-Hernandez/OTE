function graphs_MainGlobal(data::DataFrame, dir::AbstractString)
#This is the function that makes the graphs for the global problem.
# 0 Unpack:
	# 0.1. Define font for graphs
	rc("font", family="serif")
	# 0.2. Directory to save graphs
	original_dir = pwd() #To save the original directory.
	cd(dir)
	# 0.3. Define the sizes of the global and entrepreneur problem for the graphs:
	(fullsize,) = size(data)
	globalsize = fullsize-model.compar.entrepsize

# 1. Graphs for the states:
	# 1.1. Define the number of states:
	namesDF = names(data)
	namesDF_states = namesDF[2:7]
	rows_graph_states = floor(Int64,(size(namesDF_states)[1] + 2)/2)
	#States
	fig, states = plt.subplots(rows_graph_states,2)
	fig.suptitle("Optimal States - Global Problem")
		#u_w:
	states[1,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:u)
	states[1,1].set(ylabel="uw",xlabel = "θw")
		#μ:
	states[1,2].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:μ)
	states[1,2].set(ylabel="μ",xlabel = "θw")
		#e:
	states[2,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:θe)
	states[2,1].plot(data[1:globalsize,:].:θw, repeat([model.dispar.θ_e_l],globalsize), "tab:green")
	states[2,1].set(ylabel="e",xlabel = "θw")
		#ϕe:
	states[2,2].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:ϕe)
	states[2,2].plot(data[1:globalsize,:].:θw, repeat([0],globalsize), "tab:green")
	states[2,2].set(ylabel="φe",xlabel = "θw")
		#Y:
	states[3,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Y)
	states[3,1].plot(data[1:globalsize,:].:θw, repeat([0.0],globalsize), "tab:green")
	states[3,1].set(ylabel="Y",xlabel = "θw")
		#λ:
	states[3,2].plot(data[1:globalsize,:].:θw, repeat([ model.prices[1] ],globalsize))
	states[3,2].set(ylabel="λ",xlabel = "θw")
		#L:
	states[rows_graph_states,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:L)
	states[rows_graph_states,1].plot(data[1:globalsize,:].:θw, repeat([0],globalsize), "tab:green")
	states[rows_graph_states,1].set(ylabel="L",xlabel = "θw")
		#ω:
	states[rows_graph_states,2].plot(data[1:globalsize,:].:θw, repeat([ model.prices[2] ],globalsize))
	states[rows_graph_states,2].set(ylabel="ω",xlabel = "θw")

	savefig("StatesMainGlobal.png")

# 2. Graphs for the controls:
	# 2.1. Define the number of controls:
	namesDF = names(data)
	namesDF_controls = namesDF[8:11]
	rows_graph_controls = floor(Int32,(size(namesDF_controls)[1])/2)
	#Controls
	fig, controls=plt.subplots(rows_graph_controls,2)
	fig.suptitle("Optimal Controls - Global Problem")
		#n:
	controls[1,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:n)
	controls[1,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:nfull, linestyle="--")
	controls[1,1].set(ylabel="n",xlabel = "θw")
	controls[1,1].legend(["n","n full info"],loc="upper left")
		#z:
	controls[1,2].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:z)
	controls[1,2].set(ylabel="z",xlabel = "θw")
		#l:
	controls[2,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:l)
	controls[2,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:lfull, linestyle="--")
	controls[2,1].set(ylabel="l",xlabel = "θw")
	controls[2,1].legend(["l","l full info"],loc="upper left")
		#p:
	controls[2,2].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:p)
	controls[2,2].set(ylabel="p",xlabel = "θw")

	savefig("ControlsMainGlobal.png")

# 3. Graphs for the marginal taxes:
	#Marginal taxes:
	fig, margtax=plt.subplots(2,3)
	fig.suptitle("Marginal Taxes")
		#τ_n_prime:
	margtax[1,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Tn′)
	margtax[1,1].set(ylabel="τ_n'", xlabel="θw")
		#τ_c_prime:
	margtax[1,2].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Tc′)
	margtax[1,2].set(ylabel="τ_c'", xlabel="θw")
		#τ_l_prime:
	margtax[1,3].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Tl′)
	margtax[1,3].set(ylabel="τ_l'", xlabel="θw")
		#τ_n:
	margtax[2,1].plot(data[1:globalsize,:].:baseTn, data[1:globalsize,:].:Tn′)
	margtax[2,1].set(ylabel="τ_n'", xlabel="ω n")
		#τ_c:
	margtax[2,2].plot(data[1:globalsize,:].:baseTc, data[1:globalsize,:].:Tc′)
	margtax[2,2].set(ylabel="τ_c'", xlabel="e*n^α- ω*(n - ̅m) -Tn -z")
		#τ_l:
	margtax[2,3].plot(data[1:globalsize,:].:baseTl, data[1:globalsize,:].:Tl′)
	margtax[2,3].set(ylabel="τ_l'", xlabel="θw l ω")

	savefig("MarginalTaxesMainGlobal.png")

	cd(original_dir) #To get back to the oroginal directory
end #end function

function graphs_MainGloAndEnt(data::DataFrame, dir::AbstractString)
#This is the function that makes the graphs for the global and entrepreneurs problem.
# 0 Unpack:
	# 0.1. Define font for graphs
	rc("font", family="serif")
	# 0.2. Directory to save graphs
	original_dir = pwd() #To save the original directory.
	cd(dir)
	# 0.3. Define the sizes of the global and entrepreneur problem for the graphs:
	(fullsize,) = size(data)
	globalsize = fullsize-model.compar.entrepsize

# 1. Graphs for the states:
	# 1.1. Define the number of states:
	namesDF = names(data)
	namesDF_states = namesDF[2:7]
	rows_graph_states = floor(Int32,(size(namesDF_states)[1])/2)
	#States
	fig, states = plt.subplots(rows_graph_states,2)
	fig.suptitle("Optimal States")
		#u_w:
	states[1,1].plot(data.:θe, data.:u)
	states[1,1].set(ylabel="uw",xlabel = "θe")
		#μ:
	states[1,2].plot(data.:θe, data.:μ)
	states[1,2].set(ylabel="μ",xlabel = "θe")
		#Y:
	states[2,1].plot(data.:θe, data.:Y)
	states[2,1].plot(data.:θe, repeat([0.0],fullsize), "tab:green")
	states[2,1].set(ylabel="Y",xlabel = "θe")
		#λ:
	states[2,2].plot(data.:θe, repeat([ model.prices[1] ],fullsize))
	states[2,2].set(ylabel="λ",xlabel = "θe")
		#L:
	states[rows_graph_states,1].plot(data.:θe, data.:L)
	states[rows_graph_states,1].plot(data.:θe, repeat([0],fullsize), "tab:green")
	states[rows_graph_states,1].set(ylabel="L",xlabel = "θe")
		#ω:
	states[rows_graph_states,2].plot(data.:θe, repeat([ model.prices[2] ],fullsize))
	states[rows_graph_states,2].set(ylabel="ω",xlabel = "θe")

	savefig("StatesMainGloAndEnt.png")

# 2. Graphs for the controls:
	# 2.1. Define the number of controls:
	namesDF = names(data)
	namesDF_controls = namesDF[8:10]
	rows_graph_controls = floor(Int32,(size(namesDF_controls)[1])/2)
	#Controls
	fig, controls=plt.subplots(rows_graph_controls,2)
	fig.suptitle("Optimal Controls")
		#n:
	controls[1,1].plot(data.:θe, data.:n)
	controls[1,1].plot(data.:θe, data.:nfull, linestyle="--")
	controls[1,1].set(ylabel="n",xlabel = "θe")
	controls[1,1].legend(["n","n full info"],loc="upper left")
		#z:
	controls[2,1].plot(data.:θe, data.:z)
	controls[2,1].set(ylabel="z",xlabel = "θe")

	savefig("ControlsMainGloAndEnt.png")

# 3. Graphs for the marginal taxes:
	#Marginal taxes:
	fig, margtax=plt.subplots(2,2)
	fig.suptitle("Marginal Taxes")
		#τ_n_prime:
	margtax[1,1].plot(data.:θe, data.:Tn′)
	margtax[1,1].set(ylabel="τ_n'", xlabel="θe")
		#τ_c_prime:
	margtax[1,2].plot(data.:θe, data.:Tc′)
	margtax[1,2].set(ylabel="τ_c'", xlabel="θe")
		#τ_n:
	margtax[2,1].plot(data.:baseTn, data.:Tn′)
	margtax[2,1].set(ylabel="T_n", xlabel="ω n")
		#τ_c:
	margtax[2,2].plot(data.:baseTc, data.:Tc′)
	margtax[2,2].set(ylabel="T_c", xlabel="e*n^α- ω*(n-̅m) - Tn -z")

	savefig("MarginalTaxesMainGloAndEnt.png")

	cd(original_dir) #To get back to the oroginal directory
end #end function

function graphs_debug(data::DataFrame, dir::AbstractString)
#This is the function that makes the graphs for the debuging.
# 0 Unpack:
	# 0.1. Define font for graphs
	rc("font", family="serif")
	# 0.2. Directory to save graphs
	original_dir = pwd() #To save the original directory.
	cd(dir)
	# 0.3. Define the sizes of the global and entrepreneur problem for the graphs:
	(fullsize,) = size(data)
	globalsize = fullsize-model.compar.entrepsize

# 1. Graphs for the marginal utilities:
	fig, utilities=plt.subplots(1,2)
	fig.suptitle("Change in Utilities")
		#ue_prime:
	utilities[1].plot(data.:θw, data.:uw′)
	utilities[1].set(ylabel="uw' in global",xlabel="θw")
		#ue_prime:
	utilities[2].plot(data.:θe, data.:ue′)
	utilities[2].set(ylabel="ue'",xlabel="θe")

	savefig("UtilitiesDebug.png")

# 2. Graphs for A:
	fig, A_graphs=plt.subplots(1,2)
	fig.suptitle("A Graphs")
		#Value for A:
	A_graphs[1].plot(data.:θw, data.:A)
	A_graphs[1].set(ylabel="A",xlabel="θw")
		#Bound for A:
	A_graphs[2].plot(data.:θw, data.:Afull)
	A_graphs[2].set(ylabel="A Bound",xlabel="θw")

	savefig("A_Debug.png")

# 3. Graphs for Z:
	fig, Z_graphs=plt.subplots(2,3)
	fig.suptitle("Evasion Graphs")

	#Global problem
		#Value for max possible evasion:
		Z_graphs[1,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:MaxEvasion)
		Z_graphs[1,1].set(ylabel="e*n^α - ω/λ*(n-̅m)",xlabel="θw")
		#Evasion in model:
		Z_graphs[1,2].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:z)
		Z_graphs[1,2].set(ylabel="z",xlabel="θw")
		#Combined:
		Z_graphs[1,3].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:MaxEvasion, data[1:globalsize,:].:θw, data[1:globalsize,:].:z)
		Z_graphs[1,3].set(xlabel="θw")
		Z_graphs[1,3].legend(["e*n^α - ω/λ*(n-̅m)","z"],loc="upper left")

	#Entrepreneurs problem
		#Value for max possible evasion:
		Z_graphs[2,1].set(title = "Entrepreneurs' problem")
		Z_graphs[2,1].plot(data[globalsize:fullsize,:].:θe, data[globalsize:fullsize,:].:MaxEvasion)
		Z_graphs[2,1].set(ylabel="θe*n^α - ω/λ*(n-̅m)",xlabel="θe")
		#Evasion in model:
		Z_graphs[2,2].set(title = "Entrepreneurs' problem")
		Z_graphs[2,2].plot(data[globalsize:fullsize,:].:θe, data[globalsize:fullsize,:].:z)
		Z_graphs[2,2].set(ylabel="z",xlabel="θe")
		#Combined:
		Z_graphs[2,3].set(title = "Entrepreneurs' problem")
		Z_graphs[2,3].plot(data[globalsize:fullsize,:].:θe, data[globalsize:fullsize,:].:MaxEvasion,
						   data[globalsize:fullsize,:].:θe, data[globalsize:fullsize,:].:z)
		Z_graphs[2,3].set(ylabel="θe*n^α - ω/λ*(n-̅m) or z", xlabel="θe")
		plt.legend(["θe*n^α - ω/λ*(n-̅m)","z"],loc="upper right")

	savefig("Z_Debug.png")

# 4. Graphs for integrals:
	fig, debug_propositions=plt.subplots(2,2)
	fig.suptitle("Debugs for propositions")
		#4.1. μ - integral proposition 1:
	debug_propositions[1,1].plot(data.:θw, (data[:, :μ] - data[:, :Prop1LSDebug]))
	debug_propositions[1,1].set(ylabel="μ - integral", xlabel="θw")
		#4.2. Difference from right and left side of proposition 1:
	debug_propositions[1,2].plot(data.:θw, (- data[:, :μ] - data[:, :Prop1RS1] - data[:, :Prop1RS2]))
	debug_propositions[1,2].set(ylabel="Difference in sides proposition 1",xlabel="θw")
		#4.3. Difference from right and left side of proposition 2:
	debug_propositions[2,1].plot(data.:θw, (data[:, :Prop2LS1] - data[:, :Prop2RS1]))
	debug_propositions[2,1].set(ylabel="Difference in sides proposition 2",xlabel="θw")
		#4.4. Difference from right and left side of proposition 3:
	debug_propositions[2,2].plot(data.:θw, (data[:, :Prop1RS2] + data[:, :Prop3LS2] - data[:, :Prop3RS1] ) )
	debug_propositions[2,2].set(ylabel="Difference in sides proposition 3",xlabel="θw")

	savefig("Propositions_Debug.png")

	cd(original_dir) #To get back to the original directory
end #end function

function graphs_OtherResults(data::DataFrame, dir::AbstractString)
#This is the function that makes the graphs for the other results (includes the
#propositions and taxes liabilities).
# 0 Unpack:
	# 0.1. Define font for graphs
	rc("font", family="serif")
	# 0.2. Directory to save graphs
	original_dir = pwd() #To save the original directory.
	cd(dir)
	# 0.3. Define the sizes of the global and entrepreneur problem for the graphs:
	(fullsize,) = size(data)
	globalsize = fullsize-model.compar.entrepsize

# 1. Graphs for the propositions:
	#Proposition 1
	fig, prop1=plt.subplots(1)
	fig.suptitle("Proposition 1")
	prop1.plot(data[1:globalsize,:].:θw, -data[1:globalsize,:].:μ)
	prop1.plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Prop1RS1)
	prop1.plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Prop1RS2)
	prop1.legend(["-μ","Tl'/(1-Tl') εl/(1+εl) θw hw","εz z/n^α he"],loc="upper right")
	prop1.set(xlabel="θw")

	#Proposition 2
	fig, prop2=plt.subplots(1)
	fig.suptitle("Proposition 2")
	prop2.plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Prop2LS1)
	prop2.set(xlabel="θw")
	prop2.plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Prop2RS1)
	prop2.legend(["εz z/n^α", "e/(1-Tc') Tn'/(1+Tn')"], loc= "upper right")

	#Proposition 3
	fig, prop3=plt.subplots(1)
	fig.suptitle("Proposition 3")
	prop3.plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Prop1RS2)
	prop3.plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Prop3LS2)
	prop3.plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Prop3RS1)
	prop3.plot(data[1:globalsize,:].:θw, repeat([data[globalsize, :μ]],globalsize) )
	prop3.legend(["εz z/n^α he", "1/λ [Ve-Vw] g 1/ue'", "∫( 1-ϕ/λ uw^ϕ-1 ) he", "μe(θw_u)"],loc="upper right")
	prop3.set(xlabel="θw")

	savefig("Propositions.png")

# 2. Tax path:
	#Taxes liabilities in Global Problem:
	fig, tax=plt.subplots(2,3)
	fig.suptitle("Taxes Global Problem")
		#τ_n:
	tax[1,1].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Tn)
	tax[1,1].set(ylabel="T_c", xlabel="e")
		#τ_c:
	tax[1,2].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Tc)
	tax[1,2].set(ylabel="T_n", xlabel="e")
		#τ_l:
	tax[1,3].plot(data[1:globalsize,:].:θw, data[1:globalsize,:].:Tl)
	tax[1,3].set(ylabel="T_l", xlabel="θw")
		#τ_n:
	tax[2,1].plot(data[1:globalsize,:].:baseTn, data[1:globalsize,:].:Tn)
	tax[2,1].set(ylabel="T_n", xlabel="ω n")
		#τ_c:
	tax[2,2].plot(data[1:globalsize,:].:baseTc, data[1:globalsize,:].:Tc)
	tax[2,2].set(ylabel="T_c", xlabel="e*n^α- ω n-Tn -z")
		#τ_l:
	tax[2,3].plot(data[1:globalsize,:].:baseTl, data[1:globalsize,:].:Tl)
	tax[2,3].set(ylabel="T_l", xlabel="θw l ω")

	#Taxes liabilities in Entrepreneurs Problem:
	fig, tax_ent=plt.subplots(2,2)
	fig.suptitle("Taxes Entrepreneurs Problem")
		#τ_c:
	tax_ent[1,1].plot(data[globalsize:fullsize,:].:θe, data[globalsize:fullsize,:].:Tn)
	tax_ent[1,1].set(ylabel="T_n", xlabel="θe")
		#τ_n:
	tax_ent[1,2].plot(data[globalsize:fullsize,:].:θe, data[globalsize:fullsize,:].:Tc)
	tax_ent[1,2].set(ylabel="T_c", xlabel="θe")
		#τ_n:
	tax_ent[2,1].plot(data[globalsize:fullsize,:].:baseTn, data[globalsize:fullsize,:].:Tn)
	tax_ent[2,1].set(ylabel="T_n", xlabel="ωe n")
		#τ_n:
	tax_ent[2,2].plot(data[globalsize:fullsize,:].:baseTc, data[globalsize:fullsize,:].:Tc)
	tax_ent[2,2].set(ylabel="T_c", xlabel="θe*n^α- ωe n-Tn-z")

	savefig("TaxesLiabilities.png")

	cd(original_dir) #To get back to the oroginal directory
end #end function

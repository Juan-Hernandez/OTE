################################################################################
####																		####
####					MAIN SCRIPT FOR OPTIMAL TAXATION					####
####																		####
################################################################################

# To run inside a module, run OptTaxModule

# 0 Output switches
fig_graphs = true # Indicator to print figures (true is when we print figures).
fig_graphs && (fig_main = true) # Indicator for main figures.
fig_graphs && (fig_GloAndEnt = false) # Indicator for figures of both problems.
fig_graphs && (fig_debug = false) # Indicator for debug figures.
fig_graphs && (fig_others = false) # Indicator for debug figures.
export_results = true #Indicator to export the results dataframe to excel.

# 1 Load packages definitions and functions
include("OTEutils.jl")

# 2 Set parameters and initialize
include("testparameters.jl")
#include("ParametersRawlsianUniform.jl") #Parameters for Rawlsian Uniform.
#include("ParametersUtilitarianUniform.jl") #Parameters for Utilitarian Uniform.
	# This defines:
		# Parameter structures:	ecopar compar dispar
		# Initial guesses: 		priceguess finalguess
		# Specific parameters:	globalsize

# 3 Create and initialize model
model=OTEmodel(ecopar, compar, dispar)
initializeOTEmodel!(model, priceguess, finalstateguess)

# 4 Solve model
# So far this runs the inner routine.
# We need a boundary problem shooting algorithm.
lenght_sol = fullrungekutta!(model)

# 5 Generate results
Results_DF = modeldata(model,lenght_sol)
export_results && CSV.write("Results_DF.csv", Results_DF)

# 6 Plot results
fig_graphs && (fig_main && graphs_MainGlobal(Results_DF,".\\Graphs"))
fig_graphs && (fig_GloAndEnt && graphs_MainGloAndEnt(Results_DF,".\\Graphs"))
fig_graphs && (fig_debug && graphs_debug(Results_DF, ".\\Graphs"))
fig_graphs && (fig_others && graphs_OtherResults(Results_DF, ".\\Graphs"))

function globalcontrols!(controls::AbstractArray{T,1}, ss::Array{T,1}, prices::Array{Float64,1}, pa::EconomicParameters, verbosebool::Bool) where T<:Real
# INPUT: control vector, state vector and parameters
# MODIFIES: control vector
debugbool::Bool=false
# 1.0 Preallocating
nn_z0::T	=NaN
A_cons::T	=NaN
corner_nn::T=NaN
corner_zz::T=NaN
corner_ll::T=NaN
corner_pp::T=NaN
den_l_temp::T=NaN
z_2::T	=NaN
z_int_lwbar::T	=NaN
corner_hamiltonian::T=NaN
max_hamiltonian::T=-Inf
# 1.1 Extracting
θ::T   =ss[1]
e::T   =ss[2]
ϕ_e::T =ss[3]
u::T  =ss[4]
μ::T   =ss[5]
h_e::T =ss[6]
h_w::T =ss[7]
λ::Float64=prices[1]
ω::Float64=prices[2]

# 1.2 Initialize constants
z_lwbar::T    = 0.0
z_u::T    = (1.0/pa.β)^(1.0/pa.σ)*(1.0 - 1e-10) #Max possible evasion.
ln_max::T     = 1e10
n_full_info::T  = (λ*pa.α*e/ω)^(1.0/(1.0-pa.α))
pre_tax_profits_at_nmin::T = λ*e*pa.ς^pa.α # - ω*pa.ς
# The following depend on A and thus on h_e>0
if h_e>0.0
  A_cons	= ω*pa.ς+ pa.utilit*u^pa.ϕ-λ*u+ϕ_e/h_e
  nn_z0	= -A_cons*1.0/ω*pa.α/(1.0-pa.α)
end
# 1.3 Define aux functions:
n_opt(zvar)         = pa.α/ω/(1.0-pa.α)*(λ*pa.β/(1.0+pa.σ)*zvar^(1.0+pa.σ)-A_cons)
zfoc(nvar, zvar)    = λ*e*nvar^pa.α - ω*nvar - λ/pa.σ*zvar*(1.0-pa.β/(1.0+pa.σ)*zvar^pa.σ) + A_cons
zfoc_at_nopt(zvar)  = zfoc(n_opt(zvar), zvar)
zfoc_at_nmin(zvar)  = zfoc(pa.ς, zvar)
den_l(nvar,zvar)    = ( λ*pa.χ*h_w - pa.χ/θ*(1.0+pa.ψ)*( μ + h_e/( nvar^pa.α*(1.0-pa.β*zvar^pa.σ) )*
            ( λ*e*nvar^pa.α - λ*pa.β/(1.0+pa.σ)*zvar^(1.0+pa.σ) -ω*nvar + A_cons) ) )
ll_opt(nvar,zvar)   = (ω*θ*h_w/den_l(nvar,zvar))^(1.0/pa.ψ)
den_p(nvar,zvar)    = θ*nvar^pa.α*(1.0-pa.β*zvar^pa.σ)

objective(lvar, nvar, pvar, zvar) = ( pa.utilit*u^pa.ϕ*(h_w+pvar*h_e) + μ*pa.χ/θ*lvar^(1.0+pa.ψ)
              + λ*pvar*h_e*( e*nvar^pa.α - pa.β/(1.0+pa.σ)*zvar^(1.0+pa.σ) - u )
              - λ*h_w*( u + pa.χ/(1.0+pa.ψ)*lvar^(1.0+pa.ψ) )
              + ω*(θ*lvar*h_w-(nvar-pa.ς)*pvar*h_e) + ϕ_e*pvar ) # Hamiltonian. big parenthesis needed for multline

# 2 Define cases and solve

# 2.0 Check he=zero and solve
if h_e<1e-10
  verbosebool && println("Case 0: h_e=0.0")
  # h_e=0.0
  if ϕ_e<0.0
    controls[1] = ln_max  # n_foc always positive
    controls[2] = 0.0     # z_foc always negative
    den_l_temp = λ*pa.χ*h_w - pa.χ/θ*(1.0+pa.ψ)*(μ+ ϕ_e/ln_max^pa.α)
    if den_l_temp > 0.0
      controls[3] = (ω*θ*h_w/den_l_temp)^(1.0/pa.ψ)
      controls[4] = pa.χ*controls[3]^(1.0+pa.ψ)/(θ*ln_max^pa.α)
    else
      controls[3] = ln_max
      controls[4] = pa.χ*ln_max^(1.0+pa.ψ-pa.α)/θ
    end
  else
    controls[1]=pa.ς # n_foc always negative
    controls[2]=min(e*pa.ς^pa.α, (1.0/pa.β)^(1.0/pa.σ)-1e-10) # zfoc at z_u
    den_l_temp=λ*pa.χ*h_w - pa.χ/θ*(1.0+pa.ψ)*( μ + ϕ_e/( controls[1]^pa.α*(1.0-pa.β*controls[2]^pa.σ) ) )
    if den_l_temp>0.0
      controls[3] = (ω*θ*h_w/den_l_he0)^(1.0/pa.ψ)
      controls[4] = pa.χ*controls[3]^(1.0+pa.ψ)/(θ*controls[1]^pa.α*(1.0-pa.β*controls[2]^pa.σ))
    else
      controls[3] = ln_max
      controls[4] = pa.χ*ln_max^(1.0+pa.ψ)/(θ*controls[1]^pa.α*(1.0-pa.β*controls[2]^pa.σ))
    end
  end
  verbosebool && println("n = ", ForwardDiff.value(controls[1]),
              ",  z = ", ForwardDiff.value(controls[2]),
              ",  l = ", ForwardDiff.value(controls[3]),
              ",  p = ", ForwardDiff.value(controls[4]) )
  return nothing
end

# 2.1 Case 1: n_opt(z=0)>n_full_info.
if (nn_z0>n_full_info)
  verbosebool && println("Case 1: n_opt(z=0)>n_full_info.")
  controls[2] = 0.0 # zz
  controls[1] = nn_z0 # nn
  h_w*den_l(controls[1],controls[2]) < 0.0 ? (controls[3]=ln_max) : (controls[3]=ll_opt(controls[1],controls[2]))
  controls[4] = (pa.χ*controls[3]^(1.0+pa.ψ))/den_p(controls[1],controls[2])
  verbosebool && println("n = ", ForwardDiff.value(controls[1]),
              ",  z = ", ForwardDiff.value(controls[2]),
              ",  l = ", ForwardDiff.value(controls[3]),
              ",  p = ", ForwardDiff.value(controls[4]) )
  return nothing
end

# 2.2 Evaluate potential interior solution:
z_2     = ((1.0+pa.σ)/(λ*pa.β)*(A_cons+(1.0-pa.α)/pa.α*ω*n_full_info))^(1.0/(1.0+pa.σ))
z_u = min(z_u,z_2);
z_int_lwbar = max(0.0, (A_cons + pa.ς*ω*(1.0-pa.α)/pa.α)/(λ*pa.β)*(1.0+pa.σ) )^(1.0/(1.0+pa.σ)) # min z for interior n
weirdinterior = z_int_lwbar<pre_tax_profits_at_nmin/λ && zfoc_at_nmin(z_int_lwbar)>=0 # Boolean for when A is not that big

if nn_z0>=pa.ς || weirdinterior
# 2.2.1 Interior feasible
  verbosebool && println("Case 2: Interior n")
  # Using bisection method:
  if zfoc_at_nopt(z_int_lwbar)*zfoc_at_nopt(z_u)<0
    controls[2] = find_zero(zfoc_at_nopt, (z_int_lwbar,z_u), Bisection());
  else
    println("e: ", e," z_2: ", z_2)
    println("z_int_lwbar: ",z_int_lwbar, " z_u: ", z_u, " zfoc(z_u): ", zfoc_at_nopt(z_u))
    error("Bisection: signs equal --> Cannot solve.")
  end

  controls[1] = n_opt(controls[2])
  h_w*den_l(controls[1],controls[2]) < 0.0 ? (controls[3]=ln_max) : (controls[3]=ll_opt(controls[1],controls[2]))
  debugbool && println("den_l= ", ForwardDiff.value(den_l(controls[1],controls[2])))
  controls[4] = (pa.χ*controls[3]^(1.0+pa.ψ))/den_p(controls[1],controls[2]);

  max_hamiltonian = objective(controls[3],controls[1],controls[4],controls[2])
# 2.2.1.2 Evaluate corner solution:
  corner_zz = min(z_int_lwbar, pre_tax_profits_at_nmin/λ, z_u)
  corner_nn = max(nn_z0,pa.ς)
  h_w*den_l(corner_nn, corner_zz) < 0.0 ? (corner_ll =ln_max) : (corner_ll=ll_opt(corner_nn,corner_zz))
  corner_pp = (pa.χ*corner_ll^(1.0+pa.ψ))/den_p(corner_nn,corner_zz);

  corner_hamiltonian = objective(corner_ll,corner_nn,corner_pp,corner_zz)
# 2.2.1.3 Keep best solution
  if corner_hamiltonian>max_hamiltonian
    verbosebool && println("Case 2.b Corner n and z.")
      controls[1]=corner_nn
      controls[2]=corner_zz
      controls[3]=corner_ll
      controls[4]=corner_pp
      max_hamiltonian=corner_hamiltonian
  end
  verbosebool && println("n = ", ForwardDiff.value(controls[1]),
              ",  z = ", ForwardDiff.value(controls[2]),
              ",  l = ", ForwardDiff.value(controls[3]),
              ",  p = ", ForwardDiff.value(controls[4]) )

return nothing
end

# 2.3 Case 3: n at lower bound
verbosebool && print("Case 3: n at lower bound. ")
controls[1] = pa.ς
# 2.3.1 Check interior z
z_u = min(z_int_lwbar, pre_tax_profits_at_nmin/λ, z_u) # z_u just for the case \bar m > n_full_info
if zfoc_at_nmin(z_lwbar)*zfoc_at_nmin(z_u)<0 # Interior z feasible
  controls[2] = find_zero(zfoc_at_nmin, (z_lwbar,z_u), Bisection())
  # verbosebool && print("Interior z feasible. ")
  h_w*den_l(controls[1],controls[2])<0.0 ? (controls[3]=ln_max) : (controls[3]=ll_opt(controls[1],controls[2]))
  controls[4] = (pa.χ*controls[3]^(1.0+pa.ψ))/den_p(controls[1],controls[2]);
  max_hamiltonian = objective(controls[3], controls[1], controls[4], controls[2])
end
# 2.3.2 True corner
if z_u>0.0
  h_w*den_l(controls[1], z_u)<0.0 ? (corner_ll=ln_max) : (corner_ll=ll_opt(controls[1], z_u))
  corner_pp = (pa.χ*corner_ll^(1.0+pa.ψ))/den_p(controls[1], z_u);
  corner_hamiltonian = objective(corner_ll, controls[1], corner_pp, z_u)
  if corner_hamiltonian>max_hamiltonian
    verbosebool && println("Case 3.b Corner z.")
    controls[2]=z_u
    # controls[1]=controls[1]
    controls[3]=corner_ll
    controls[4]=corner_pp
    max_hamiltonian=corner_hamiltonian
  else
    verbosebool && println("Case 3.a Interior z.")
  end
end

# 2. Evaluating the z=0.0 corner solutions (this shouldn't be the answer)
corner_zz = 0.0
# corner_nn = controls[1];
h_w*den_l(controls[1], corner_zz) <= 0.0 ? corner_ll=ln_max : corner_ll=ll_opt(controls[1], corner_zz)
corner_pp = (pa.χ*corner_ll^(1.0+pa.ψ))/den_p(controls[1], corner_zz);
corner_hamiltonian = objective(corner_ll, controls[1], corner_pp, corner_zz)
if corner_hamiltonian > max_hamiltonian
  controls[2] = corner_zz
  # controls[1] = corner_nn;
  controls[3] = corner_ll
  controls[4] = corner_pp
  max_hamiltonian=corner_hamiltonian
  @warn("Case 3.c Corner n, z=0.0.")
end
#Final Output:
  verbosebool && println("n = ", ForwardDiff.value(controls[1]),
              ",  z = ", ForwardDiff.value(controls[2]),
              ",  l = ", ForwardDiff.value(controls[3]),
              ",  p = ", ForwardDiff.value(controls[4]) )
end

function globalcontrols(ss::Array{T,1}, prices::Array{Float64,1}, pa, verbosebool::Bool=false) where T<:Real
controls=Array{T}(undef,4)
globalcontrols!(controls, ss, prices, pa, verbosebool)
return Tuple(controls)
end

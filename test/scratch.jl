using EquivariantTensors, BenchmarkTools
import EquivariantTensors as ET
O3alt = ET.O3alt

##

nn = (1,1,2)
ll = (2,2,2)
mm = (-2,2,0)
L = 2 
k = 1
basis = O3.B_SpheriCart()

O3.m_generate(nn, ll, L, k, basis)
@btime O3.m_generate(nn, ll, L, k, basis)
@code_warntype O3.m_generate(nn, ll, L, k, basis)

##

nn = (1,1,2)
ll = (2,2,2)
L = 0
basis = O3alt.B_SpheriCart()

O3alt.coupling_coeffs(L, ll, nn; PI=true, basis)


##

# mm = SA[-2,-1,3]
mm = (-2, -1, 3)
@btime ET.O3.m_filter( $mm, 0, ET.O3.B_SpheriCart() )
@code_warntype ET.O3.m_filter(mm, 0, ET.O3.B_SpheriCart() )



mm = [-3,1,2]
MM1 = ET.O3.signed_mmset( mm )[:]
MM2 = ET.O3.new_signed_mmset( mm )
_MM1 = [ [mm...] for mm in MM1 ]
sort(_MM1) == sort(MM2)


@code_warntype new_signed_mmset( [-2,1,3] )
@code_warntype new_signed_mmset2( [-2,1,3] )
@code_warntype new_signed_mmset3( [-2,1,3] )

@btime new_signed_mmset( $([-2,1,3]) )
@btime new_signed_mmset2( $([-2,1,3]) )
@btime new_signed_mmset3( $([-2,1,3]) )


@btime ET.O3.new_signed_mmset1( $([-2,1,3]) )
@btime ET.O3.signed_mmset0( $([-2,1,3]) )

mm = [-2,1,3]

for mask in 1:(2^(length(mm)))
   @show mask-1 
   @show digits(mask-1, base=2, pad=3)
end

@btime digits(6, base=2, pad=3)
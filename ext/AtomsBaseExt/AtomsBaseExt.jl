
module AtomsBaseExt 

include(joinpath(@__DIR__(), "atoms_data.jl"))

import AtomsBase: ChemicalSpecies, atomic_number 
import EquivariantTensors.Atoms: bond_len

# -------------- Bond-length heuristics

bond_len(s::Symbol) = bond_len(ChemicalSpecies(s))
bond_len(s::ChemicalSpecies) = bond_len(atomic_number(s.atomic_number))

function bond_len(z::Integer)   # assume Integer === atomic number 
   if haskey(AtomsData.lengthscales, z)
      return AtomsData.lengthscales[z]["bond_len"][1]
   end
   error("No typical bond length for atomic number $z is known. Please specify manually.")
end

"""
heuristic for bond-length between species... 
"""
bond_len(z1, z2) = (bond_len(z1) + bond_len(z2)) / 2


end 
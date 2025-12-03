
module AtomsBaseExt 


import AtomsBase: ChemicalSpecies, atomic_number 
import EquivariantTensors.Atoms: bond_len

# some data about bond lengths
include("atoms_data.jl")


# -------------- Bond-length heuristics

bond_len(s::Symbol) = bond_len(ChemicalSpecies(s))
bond_len(s::ChemicalSpecies) = bond_len(atomic_number(s.atomic_number))

function bond_len(z::Integer)   # assume Integer === atomic number 
   if haskey(LENGTHSCALES, z)
      return LENGTHSCALES[z]["bond_len"][1]
   end
   error("No default bond length for chemical species $(ChemicalSpecies(z)) (Z = $z).")
end

"""
heuristic for bond-length between species... 
"""
bond_len(z1, z2) = (bond_len(z1) + bond_len(z2)) / 2

# -------------- Utilities

# build an agnesi transform with default parameters 
include("atoms_agnesi.jl")


end 
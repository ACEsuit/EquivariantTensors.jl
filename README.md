# EquivariantTensors

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ACEsuit.github.io/EquivariantTensors.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ACEsuit.github.io/EquivariantTensors.jl/dev/)
[![Build Status](https://github.com/ACEsuit/EquivariantTensors.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ACEsuit/EquivariantTensors.jl/actions/workflows/CI.yml?query=branch%3Amain)

`EquivariantTensors.jl` provides tools to construct equivariant tensor layers to be used in equivariant models, as well as computational kernels to evaluate those layers. 

The plan is that it becomes the backend for several [ACEsuit]() packages such as [ACEpotentials.jl](https://github.com/ACEsuit/ACEpotentials.jl) and [ACEhamiltonians.jl](https://github.com/ACEsuit/ACEhamiltonians.jl). 

The package is a work in progress. It is already usable, but functionality is still incomplete. A key goal is to provide full `ChainRules` and `Lux` integration, including optimized CPU and GPU kernels, and shared interfaces to enable drop-in replacements for different tensor format. Key components from various `ACEsuit` packages that can be shared are slowly being moved into this package. 

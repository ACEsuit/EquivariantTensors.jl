

const _SO3_QUAD_DEGREES = sort(collect(keys(_SO3_QUAD_DATA)))

"""
    QuadSO3{T}

Gauss-type quadrature rule for integration over SO(3) with respect to the
normalized Haar measure. Data from Manuel Gräf's PhD thesis (TU Chemnitz, 2013).

Iterating yields `(weight, node)` pairs where `node` is an `SMatrix{3,3}`.
Calling `q(f)` computes `∫_{SO(3)} f(R) dR ≈ ∑ wᵢ f(Rᵢ)`.
"""
struct QuadSO3{T}
   nodes::Vector{SMatrix{3, 3, T, 9}}
   weights::Vector{T}
   degree::Int
end

function QuadSO3(N::Integer)
   # Find the smallest available degree >= N
   idx = findfirst(d -> d >= N, _SO3_QUAD_DEGREES)
   if idx === nothing
      error("QuadSO3: no quadrature rule available for degree $N " *
            "(maximum available: $(_SO3_QUAD_DEGREES[end]))")
   end
   N_actual = _SO3_QUAD_DEGREES[idx]
   data = _SO3_QUAD_DATA[N_actual]
   nodes = [SMatrix{3, 3, Float64, 9}(
               d[1], d[4], d[7],   # column-major: col1 = row1,row2,row3 of entry
               d[2], d[5], d[8],
               d[3], d[6], d[9]) for d in data]
   weights = [d[10] for d in data]
   # Normalize weights to sum to 1
   wsum = sum(weights)
   weights ./= wsum
   return QuadSO3{Float64}(nodes, weights, Int(N_actual))
end

Base.length(q::QuadSO3) = length(q.weights)
Base.eltype(::QuadSO3{T}) where {T} = Tuple{T, SMatrix{3, 3, T, 9}}

function Base.iterate(q::QuadSO3, i::Int=1)
   i > length(q) && return nothing
   return (q.weights[i], q.nodes[i]), i + 1
end

function (q::QuadSO3)(f)
   return sum(w * f(R) for (w, R) in q)
end

# ---------------------------------------------------------------

"""
    QuadO3{T}

Quadrature rule for integration over O(3) with respect to the normalized Haar
measure. Constructed from a `QuadSO3` rule by averaging over inversion:
for each SO(3) node R, the O(3) rule includes both R and -R with half the
weight.

Iterating yields `(weight, node)` pairs where `node` is an `SMatrix{3,3}`.
Calling `q(f)` computes `∫_{O(3)} f(Q) dQ ≈ ∑ wᵢ f(Qᵢ)`.
"""
struct QuadO3{T}
   nodes::Vector{SMatrix{3, 3, T, 9}}
   weights::Vector{T}
   degree::Int
end

function QuadO3(N::Integer)
   qso3 = QuadSO3(N)
   nodes = vcat(qso3.nodes, [SMatrix{3,3,Float64,9}(-R) for R in qso3.nodes])
   weights = vcat(qso3.weights, qso3.weights) ./ 2
   return QuadO3{Float64}(nodes, weights, qso3.degree)
end

Base.length(q::QuadO3) = length(q.weights)
Base.eltype(::QuadO3{T}) where {T} = Tuple{T, SMatrix{3, 3, T, 9}}

function Base.iterate(q::QuadO3, i::Int=1)
   i > length(q) && return nothing
   return (q.weights[i], q.nodes[i]), i + 1
end

function (q::QuadO3)(f)
   return sum(w * f(R) for (w, R) in q)
end

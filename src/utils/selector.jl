
# A few simple utility functions for mapping 
# a categorical variable to an index, or 
# pairs of categorical variables to indices.
# 
# TODO: provide a sorted categories version? 
#       could be useful if we ever have to deal 
#       with large category sets.


""" 
   cat2idx(categories, a)

maps a -> i such that categories[i] == a. Basically 
the same as `findfirst`, similar performance but seems to 
avoid an allocation. 
"""
function cat2idx(categories, a)
   for (i, c) in enumerate(categories)
       if a == c
           return i
       end
   end 
   # error("Value $a not found in categories")
   return -1 
end

function idx2cat(categories, i)
   return categories[i]
end

"""
   catcat2idx(categories1, categories2, a1, a2)
   catcat2idx(categories, a1, a2)

same as `cat2idx` but for pairs of categorical variables, 
mapping to a linear index.       
"""
function catcat2idx(categories1, categories2, a1, a2)
   i1 = cat2idx(categories1, a1)
   i2 = cat2idx(categories2, a2)
   return (i1 - 1) * length(categories2) + i2
end


catcat2idx(categories, a1, a2) = 
   catcat2idx(categories, categories, a1, a2)


"""
   symidx(j1, j2, n)   

Computes the index in a flattened symmetric matrix of size n x n,
given the row/column indices j1, j2 (1-based). E.g. if n == 3 then 
the mapping is as follows:
```
   [ 1 2 3
     2 4 5
     3 5 6 ]
```
"""
function symidx(j1, j2, n) 
   i1, i2 = minmax(j1, j2)
   return i2 + n * (i1-1) - (i1 * (i1 - 1)) ÷ 2
end

"""
   catcat2idx_sym(categories, a1, a2)

Same as `catcat2idx` but for symmetric pairs, i.e., 
`(a1, a2)` and `(a2, a1)` map to the same index. 
"""   
function catcat2idx_sym(categories, a1, a2)
   i1 = cat2idx(categories, a1)
   i2 = cat2idx(categories, a2)
   return symidx(i1, i2, length(categories))
end

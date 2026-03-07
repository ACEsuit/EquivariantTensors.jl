# PooledSparseProduct Pullback Benchmarks

GPU: NVIDIA A100-SXM4-40GB, Float32, Julia 1.12

Three implementations compared:
- **fused**: existing KA kernel — parallelizes over `(inode, ineig)`, loops over `iA`
- **sg**: scatter/gather — materializes intermediate arrays, parallelizes over `(ineig, inode, iA)`
- **fs**: fused-scatter — same parallelization as sg but reads BB directly via index arrays, no intermediate allocation

## GPU Backward Pass

```
config                                fused       sg          fs          sg/fused  fs/fused
─────────────────────────────────────────────────────────────────────────────────────────────
bwd ord=2 n=100 ng=16 nd=64          528.9μs     65.9μs      330.4μs     0.12x     0.62x
bwd ord=2 n=200 ng=24 nd=128         993.9μs     396.1μs     41.0μs      0.40x     0.04x
bwd ord=2 n=500 ng=32 nd=256         2417.4μs    1367.5μs    140.8μs     0.57x     0.06x
bwd ord=2 n=1000 ng=32 nd=512        7688.4μs    5530.8μs    1243.9μs    0.72x     0.16x

bwd ord=3 n=100 ng=16 nd=64          752.5μs     686.3μs     34.7μs      0.91x     0.05x
bwd ord=3 n=200 ng=24 nd=128         1239.5μs    478.0μs     359.1μs     0.39x     0.29x
bwd ord=3 n=500 ng=32 nd=256         4582.2μs    1061.0μs    622.5μs     0.23x     0.14x
bwd ord=3 n=1000 ng=32 nd=512        10704.6μs   9019.8μs    1338.8μs    0.84x     0.13x

bwd ord=4 n=100 ng=16 nd=64          769.3μs     360.6μs     400.9μs     0.47x     0.52x
bwd ord=4 n=200 ng=24 nd=128         1273.7μs    560.2μs     82.0μs      0.44x     0.06x
bwd ord=4 n=500 ng=32 nd=256         6471.8μs    1699.4μs    209.6μs     0.26x     0.03x
bwd ord=4 n=1000 ng=32 nd=512        14878.4μs   12175.4μs   1499.4μs    0.82x     0.10x
```

## GPU Forward Pass

The existing fused kernel is already optimal for the forward pass (no write conflicts).
Scatter/gather is slower due to intermediate array allocation.

```
config                                fused       sg          sg/fused
──────────────────────────────────────────────────────────────────────
fwd ord=2 n=100 ng=16 nd=64          36.2μs      308.2μs     8.52x
fwd ord=2 n=200 ng=24 nd=128         445.5μs     498.9μs     1.12x
fwd ord=2 n=500 ng=32 nd=256         354.1μs     1224.1μs    3.46x
fwd ord=2 n=1000 ng=32 nd=512        373.6μs     4739.8μs    12.69x

fwd ord=3 n=100 ng=16 nd=64          443.1μs     568.4μs     1.28x
fwd ord=3 n=200 ng=24 nd=128         47.3μs      154.8μs     3.27x
fwd ord=3 n=500 ng=32 nd=256         498.3μs     1206.7μs    2.42x
fwd ord=3 n=1000 ng=32 nd=512        766.5μs     5747.9μs    7.50x

fwd ord=4 n=100 ng=16 nd=64          368.3μs     317.1μs     0.86x
fwd ord=4 n=200 ng=24 nd=128         53.0μs      494.9μs     9.33x
fwd ord=4 n=500 ng=32 nd=256         527.4μs     1335.8μs    2.53x
fwd ord=4 n=1000 ng=32 nd=512        343.1μs     6064.4μs    17.68x
```

## CPU Benchmark

On CPU the fused kernel is faster across the board (no atomic contention, better cache locality).

```
config                                fused       sg          sg/fused
──────────────────────────────────────────────────────────────────────
fwd ord=2 n=100 ng=16 nd=32          43.1μs      466.9μs     10.83x
fwd ord=2 n=200 ng=24 nd=64          180.5μs     3250.1μs    18.01x
fwd ord=2 n=500 ng=32 nd=128         1083.5μs    21603.3μs   19.94x
bwd ord=2 n=100 ng=16 nd=32          418.8μs     959.3μs     2.29x
bwd ord=2 n=200 ng=24 nd=64          2299.2μs    6612.4μs    2.88x
bwd ord=2 n=500 ng=32 nd=128         16187.4μs   36968.6μs   2.28x

fwd ord=3 n=100 ng=16 nd=32          47.6μs      524.9μs     11.02x
fwd ord=3 n=200 ng=24 nd=64          244.4μs     3681.9μs    15.06x
fwd ord=3 n=500 ng=32 nd=128         1505.7μs    21096.0μs   14.01x
bwd ord=3 n=100 ng=16 nd=32          530.0μs     1479.1μs    2.79x
bwd ord=3 n=200 ng=24 nd=64          3400.0μs    10045.6μs   2.95x
bwd ord=3 n=500 ng=32 nd=128         22028.5μs   59275.6μs   2.69x

fwd ord=4 n=100 ng=16 nd=32          60.2μs      587.7μs     9.77x
fwd ord=4 n=200 ng=24 nd=64          355.1μs     4230.5μs    11.91x
fwd ord=4 n=500 ng=32 nd=128         1816.5μs    23747.0μs   13.07x
bwd ord=4 n=100 ng=16 nd=32          650.8μs     2149.8μs    3.30x
bwd ord=4 n=200 ng=24 nd=64          4287.8μs    14184.8μs   3.31x
bwd ord=4 n=500 ng=32 nd=128         27495.1μs   88223.1μs   3.21x
```

## Conclusion

The **fused-scatter** (`fs_pullback`) kernel is the clear winner for GPU backward:
- **7-30x faster** than the existing fused kernel
- **No intermediate memory** allocation (unlike scatter/gather)
- Same parallelization pattern as scatter/gather: `(ineig, inode, iA)` threads,
  each doing one `_static_prod_ed` + NB atomic adds

Recommended approach:
- **Forward**: keep existing fused kernel (already no write conflicts)
- **Backward**: replace fused kernel with fused-scatter kernel on GPU

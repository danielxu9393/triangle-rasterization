import "../../slang_common/large_arrays.slang";

// This kernel mirrors the original CUDA logic exactly.
// Keys are uint64 composed as (tile << 32) | depthBits, where depthBits is the
// bit pattern of a float. We recover tile and depth with 64/32-bit ops and bit-casts.
// Output is [num_tiles, 2] int64: [:,0] = start, [:,1] = end. Start is first idx with
// depth > epsilon; tiles with no such element get start == end.

inline uint  _extractTile(uint64_t key)      { return uint(key >> 32); }
inline float _extractDepth(uint64_t key)     { return asfloat(uint(key & 0xffffffffu));}

[AutoPyBindCUDA]
[CUDAKernel]
void delineate_tiles(
    // 1-D array of int64 keys (bit-packed as above), length >= num_keys
    const TensorView<int64_t> sorted_keys,
    // [num_tiles, 2] int64 output, zero-initialized by the caller
    const TensorView<int64_t> out_tile_boundaries,
    // epsilon threshold (see original CUDA)
    const float epsilon
) {
    let num_keys = sorted_keys.size(0);

    // 64-bit global index
    let idx = (int64_t)cudaBlockIdx().x * (int64_t)cudaBlockDim().x
            + (int64_t)cudaThreadIdx().x;
    if (idx >= num_keys) return;

    // current key -> tile/depth
    let this_key   = reinterpret<uint64_t>(load(sorted_keys, idx));
    let this_tile  = _extractTile(this_key);
    let this_depth = _extractDepth(this_key);

    // For idx > 0, compare to previous key to detect tile boundary and/or
    // behind->front transition within a tile (lastDepth <= eps && thisDepth > eps)
    if (idx > 0) {
        let prev_key   = reinterpret<uint64_t>(load(sorted_keys, idx - 1));
        let last_tile  = _extractTile(prev_key);
        let last_depth = _extractDepth(prev_key);

        if (this_tile != last_tile) {
            // end of the last tile
            store(out_tile_boundaries, uint64_t2((uint64_t)last_tile, 1), idx);

            // start of this tile if front-facing
            if (this_depth > epsilon) {
                store(out_tile_boundaries, uint64_t2((uint64_t)this_tile, 0), idx);
            }

            // if the last tile was all behind (no front entries encountered),
            // set its start to idx (so start == end)
            if (last_depth <= epsilon) {
                store(out_tile_boundaries, uint64_t2((uint64_t)last_tile, 0), idx);
            }
        }
        else if (last_depth <= epsilon && this_depth > epsilon) {
            // transition within the same tile from behind -> front
            store(out_tile_boundaries, uint64_t2((uint64_t)this_tile, 0), idx);
        }
    }

    // Special-case the very last element
    if (idx == num_keys - 1) {
        // end of current tile is N
        store(out_tile_boundaries, uint64_t2((uint64_t)this_tile, 1), num_keys);

        // if the tile never had a front-facing entry, set start to N as well
        if (this_depth <= epsilon) {
            store(out_tile_boundaries, uint64_t2((uint64_t)this_tile, 0), num_keys);
        }
    }
}

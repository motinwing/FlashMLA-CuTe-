#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cooperative_groups.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm90::fwd {

using namespace cute;

template<int D_QK, bool HAVE_TOPK_LENGTH>
class KernelTemplate {
public:

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;

static constexpr int B_H = 64;
static constexpr int B_TOPK = 64;    // TopK block size
static constexpr int NUM_THREADS = 128*3;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits)

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape( 
    // tile_to_shape，输入一个原子布局（通常是一个warp_group可以一次性读取的大小）和一个更大一些的布局（通常是一个数据block的大小），
    // 然后再给定维度顺序Step<_1, _2>{}，输入
    GMMA::Layout_K_SW128_Atom<bf16>{}, // 保证swizzle的原子块，是能够保证以最快速度搬运的布局单元
    Shape<Int<B_H>, Int<64*NUM_TILES>>{}, // tile的尺寸
    Step<_1, _2>{} // 表示第1维连续，也就是沿行维度连续（列优先，沿行维度即一整列），第2维度，也就是列维度是由第1维度排列而成的。cuda的默认排布是列优先，所以每一列都是连续的（即沿行维度连续）
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed = decltype(composition(
	SmemLayoutKTiles<NUM_TILES>{},
	Layout<Shape<Int<64*NUM_TILES>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
));

using SmemLayoutQ = SmemLayoutQTiles<D_Q/64>;
using SmemLayoutO = SmemLayoutOTiles<D_V/64>;
using SmemLayoutK = SmemLayoutKTiles<D_Q/64>;
using SmemLayoutV = SmemLayoutKTilesTransposed<D_V/64>;
using SmemLayoutHalfV = SmemLayoutKTilesTransposed<D_V/64/2>;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    union {
        array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } q_o;
    array_aligned<bf16, cosize_v<SmemLayoutK>> k[2];
    array_aligned<bf16, cosize_v<SmemLayoutS>> s[D_QK == 576 ? 1 : 2];  // For V3.2 (whose D_QK is 576), we overlap sS[0] with k's RoPE part to save shared memory; For MODEL1 (whose D_QK is 512), we allocate two buffers

    bool is_kv_valid[2][B_TOPK];
    float2 sM[32];
    float2 sL[64];   // For reduction across WG0/1 in epilogue
    float final_max_logits[64], final_lse[64];
    transac_bar_t bar_q, bar_k0_free[2], bar_k0_ready[2], bar_k1_free[2], bar_k1_ready[2], bar_is_kv_valid_ready;
};

using TiledMMA_QK = decltype(make_tiled_mma(
    GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
    // 64x64x16: MNK tile大小，F32BF16BF16: MNK 数据类型，SS A存储于SMEM(可以选择S或R) B也存储于SMEM(只能选择S) C默认且只能存储于R, <GMMA::Major::K, GMMA::Major::K> A为K主序布局，B也一样。也可以选择MN主序布局
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
    GMMA::MMA_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
    GMMA::MMA_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{}
));

template<
    typename Shape_Q, typename TMA_Q
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    CUtensorMap tensor_map_O;
};

enum NamedBarriers : uint32_t { // hopper 架构提供16个独立的硬件屏障通道 绝对id从0到15，这里用enum完成映射。
    wg0_bunch_0_ready = 0,
    wg1_bunch_0_ready = 1,
    wg0_s0_ready = 2,
    wg1_s1_ready = 3,
    sL_ready = 4,
    warpgroup0_sync = 5,
    warpgroup1_sync = 6,
    epilogue_sync = 7
};

// Save rPb (64x64, bfloat16) to sP using the stmatrix instruction
template<
    typename Tensor0,
    typename Tensor1
>
static __forceinline__ __device__ void save_rS_to_sS(
    Tensor0 const &rPb,
    Tensor1 const &sP,
    int idx_in_warpgroup
) {
    auto r2s_copy = make_tiled_copy_C(
        Copy_Atom<SM90_U32x4_STSM_N, bf16>{},
        TiledMMA_QK{}
    );
    ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);
    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
}

template<typename TMAParams>
static __device__ __forceinline__ void
devfunc(const SparseAttnFwdParams &params, const TMAParams &tma_params);

static void run(const SparseAttnFwdParams &params);

};


};

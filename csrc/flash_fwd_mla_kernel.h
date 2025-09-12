#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cooperative_groups.h>  //! add for cluster
using namespace cute;

#include "named_barrier.h"
#include "utils.h"
#include "softmax.h"
#include "static_switch.h"
#include "flash_mla.h"


template<typename PrecType, int DIM, int DIM2 = DIM>
constexpr auto getSmemLayoutK() {
    constexpr int headSizeBytes = sizeof(PrecType) * DIM;
    constexpr int headSizeBytes2 = sizeof(PrecType) * DIM2;

    if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
        return GMMA::Layout_K_SW128_Atom<PrecType>{};
    } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
        return GMMA::Layout_K_SW64_Atom<PrecType>{};
    } else {
        return GMMA::Layout_K_SW32_Atom<PrecType>{};
    }
}

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::bfloat16_t, int kHeadDimV_ = 0>
struct Flash_fwd_kernel_traits_mla {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kNWarpsS = 4;
    static constexpr int kNThreadsS = kNWarpsS * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kHeadDimV = kHeadDimV_ != 0 ? kHeadDimV_ : kHeadDim;
    static_assert(kHeadDimV % 32 == 0);
    static_assert(kHeadDimV <= kHeadDim);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = decltype(make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<kNWarpsS / 4>, _1, _1>>{}));

    static constexpr int AtomLayoutNO = kNThreads / kNThreadsS;
    using TiledMmaO = decltype(make_tiled_mma(
            cute::GMMA::rs_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kHeadDimV / AtomLayoutNO>, Int<kBlockN>>,
                    GMMA::Major::K, GMMA::Major::MN>(),
            Layout<Shape<Int<kNWarpsS / 4>, Int<AtomLayoutNO>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim>(),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    using SmemLayoutP = Layout<Shape<Shape<_2, _2>, Int<kNThreadsS>, _1, Int<kBlockN / 8>>>;
    using SmemLayoutRow = Layout<Shape<_2, Int<kNThreadsS>>, Stride<_1, _2>>;

    using SmemLayoutAtomO = decltype(composition(
            Swizzle<kSwizzle, 3, 3>{},
            Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
            SmemLayoutAtomO{},
            Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<SM90_U32x4_STSM_N, Element>;  //! SM90_U32x4_STSM_N  explain: unsinged 32 bit x4 store to shared memory, no transpose
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;  //! 128 bit alignment for float 

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    static constexpr int kNThreadsLoad = kNThreads - kNThreadsS;
    static_assert(kNThreadsLoad % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");

    using GmemLayoutAtom = Layout<
            Shape<Int<kNThreadsLoad / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(make_tiled_copy(
            Copy_Atom<Gmem_copy_struct, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemLayoutAtomO = Layout<
            Shape<Int<kNThreadsS / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtomO{},
            Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static constexpr int kGmemThreadsPerRowAccum = kBlockKSmem / kGmemElemsPerLoadAccum;
    using GmemLayoutAtomOaccum = Layout<
            Shape<Int<kNThreadsS / kGmemThreadsPerRowAccum>, Int<kGmemThreadsPerRowAccum>>,
            Stride<Int<kGmemThreadsPerRowAccum>, _1>>;
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            GmemLayoutAtomOaccum{},
            Layout<Shape<_1, _4>>{}));  // Val layout, 4 vals per store
};

namespace flash {

using namespace cute;

using TMABarrier = cutlass::arch::ClusterTransactionBarrier;

template<typename Kernel_traits>
struct SharedStorageMLA {
    union {
        struct {
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutQ>> smem_q;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutK> * 2> smem_k;  // Double buffer
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutP>> smem_p;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_scale;
        };
        struct {
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_max;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_sum;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutO>> smem_o;  //! swap the order of smem_max and smem_o to support cluster reduce
        };
        struct {  //!
            cute::array_aligned<typename Kernel_traits::Element, cosize_v<typename Kernel_traits::SmemLayoutQ>> smem_sQ;   //!
            cute::array_aligned<typename Kernel_traits::Element, cosize_v<typename Kernel_traits::SmemLayoutK>> smem_sK0;  //!
            cute::array_aligned<typename Kernel_traits::Element, cosize_v<typename Kernel_traits::SmemLayoutK>> smem_sK1;  //!
            cute::array_aligned<typename Kernel_traits::Element, cosize_v<typename Kernel_traits::SmemLayoutP>> smem_sP0;  //!
            TMABarrier smem_sScale0[64];
        };
    };
};


struct SM90_BULK_COPY_S2S
{
  CUTE_DEVICE static void
  copy(void const* smem_ptr,
       void      * dsmem_ptr, int32_t copy_bytes, uint64_t* mbar_ptr)
  {
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    uint32_t dsmem_int_ptr  = cast_smem_ptr_to_uint(dsmem_ptr);
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :
                     : "r"(dsmem_int_ptr), "r"(smem_int_ptr), "r"(copy_bytes), "r"(smem_int_mbar)
                     : "memory");
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Split, typename SharedStorage, typename AccO, typename Softmax>
__forceinline__ __device__ void store(const Flash_fwd_mla_params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx,
                                      SharedStorage &shared_storage, AccO tOrO, Softmax softmax) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    // Epilogue

    const int split_offset = __ldg(params.num_splits_ptr + bidb);  //! totally offset

    // T0_DEBUG_PRINT("split_offset %d,bidb %d  m_block %d n_split_idx %d kNThreadsS %d\n",split_offset, bidb, m_block, n_split_idx, kNThreadsS);

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(tOrO, params.scale_softmax);

    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shared_storage.smem_o.data())), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)  (64, 512)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
            !Split,
            typename Kernel_traits::SmemCopyAtomO,
            typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_o);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(tOrO);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    __syncthreads();

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_v;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    const index_t row_offset_lseaccum = ((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    // T0_DEBUG_PRINT("row_offset_o %ld, params.o_batch_stride %ld params.o_row_stride %ld params.o_head_stride %ld bidh %d bidb %d\n",row_offset_o, params.o_batch_stride, params.o_row_stride, params.o_head_stride,bidh, bidb);
    //! type and format symbols must match
    // T0_DEBUG_PRINT("row_offset_oaccum %ld,row_offset_lse %ld row_offset_lseaccum %ld params.h %d   params.seqlen_q  %d split_offset %d n_split_idx %d\n",row_offset_oaccum,row_offset_lse,row_offset_lseaccum, params.h,  params.seqlen_q ,split_offset, n_split_idx);

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
                                 //! o_row_stride and kHeadDimV is same in this version
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + (Split ? row_offset_lseaccum : row_offset_lse)),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});

    using GmemTiledCopyO = std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO, typename Kernel_traits::GmemTiledCopyOaccum>;
    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    if (tidx >= kNThreadsS) { return; } //! only use 128 threads to deal with epilogue

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // ((MMA=4, X), MMA_M, MMA_K=1)
    Tensor taccOcO_row = taccOcO(make_coord(0, _, 0), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M

    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < params.seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }  //! todo 
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, params.seqlen_q - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_causal,  int CLUSTER_DIM, typename SharedStorage>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mla(const Flash_fwd_mla_params &params,
                                                                   const int bidb, const int bidh, const int m_block,
                                                                   const int n_split_idx, const int seqlen_k,
                                                                   const int n_block_min, const int n_block_max, const bool NoSplit,
                                                                   SharedStorage &shared_storage) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    static_assert(kNThreads == 256 and kNThreadsS == 128);
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;
    int n_block = n_block_max - 1;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutVtransposed{});

    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Kernel_traits::SmemLayoutP{});
    Tensor tPsP = sP(_, tidx % kNThreadsS, _, _);
    Tensor sScale_o = make_tensor(make_smem_ptr(shared_storage.smem_scale.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tScale_osScale_o = sScale_o(_, tidx % kNThreadsS);
    Tensor sRow_max = make_tensor(make_smem_ptr(shared_storage.smem_max.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tRow_maxsRow_max = sRow_max(_, tidx % kNThreadsS);
    Tensor sRow_sum = make_tensor(make_smem_ptr(shared_storage.smem_sum.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tRow_sumsRow_sum = sRow_sum(_, tidx % kNThreadsS);

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt = thr_mma_o.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrO = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
    clear(tOrO);

    flash::Softmax<2 * size<1>(tOrO)> softmax;

    int warp_group_idx = cutlass::canonical_warp_group_idx();
    if (warp_group_idx == 0) {
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tidx);
        Tensor tSrQ = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
        Tensor tSrK = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

        if (n_block % 2 == 1) {
            // Double buffer for sK
            constexpr int sK_offset = size(sK);
            tSrK.data() = tSrK.data() + sK_offset / 8;
            tOrVt.data() = tOrVt.data() + sK_offset / 8;
        }

        // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
        // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
        // We will have at least 1 "masking" iteration.
        // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
        // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
        constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
#pragma unroll 1
        for (int masking_step = n_masking_steps; n_block >= n_block_min; --masking_step, --n_block) {
            __syncthreads();

            Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
            flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma, tSrQ, tSrK, tSrS);

            const bool is_masking_step = masking_step > 0;
            const bool is_first_masking_step = masking_step == n_masking_steps;

            if (is_masking_step) {
                Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
                Tensor tScS = thr_mma.partition_C(cS);
#pragma unroll
                for (int i = 0; i < size(tSrS); ++i) {
                    if constexpr (!Is_causal) {  // Just masking based on col
                        if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) tSrS(i) = -INFINITY;
                    } else {
                        // Ensure seqlen_k - 1 - (n_block * kBlockN + col) >= (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                        // col <= seqlen_k - 1 - n_block * kBlockN - (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                        int row = int(get<0>(tScS(i)));
                        int col_limit_right = seqlen_k - 1 - n_block * kBlockN - (params.seqlen_q - 1 - (m_block * kBlockM + row)) / params.ngroups;
                        if (int(get<1>(tScS(i))) > col_limit_right) tSrS(i) = -INFINITY;
                    }
                }
            }

            // We have key_padding_mask so we'll need to Check_inf
            Tensor scale_o = is_first_masking_step
                             ? softmax.template softmax</*Is_first=*/true,  /*Check_inf=*/Is_causal>(tSrS, params.scale_softmax_log2)
                             : is_masking_step ?
                               softmax.template softmax</*Is_first=*/false, /*Check_inf=*/Is_causal>(tSrS, params.scale_softmax_log2)
                                               : softmax.template softmax</*Is_first=*/false, /*Check_inf=*//*Is_local=*/false>(tSrS, params.scale_softmax_log2);

            Tensor rP = flash::convert_type<Element>(tSrS);
            cute::copy(rP, tPsP);
            cute::copy(scale_o, tScale_osScale_o);

            cutlass::arch::NamedBarrier::arrive(kNThreads, static_cast<int>(NamedBarriers::SReady));

            flash::rescale_o(tOrO, scale_o);

            Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
            flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_o, tOrP, tOrVt, tOrO);

            // Double buffer for sK
            const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
            tSrK.data() = tSrK.data() + sK_offset / 8;
            tOrVt.data() = tOrVt.data() + sK_offset / 8;
        }

        cute::copy(softmax.row_max, tRow_maxsRow_max);
        cute::copy(softmax.row_sum, tRow_sumsRow_sum);
        cutlass::arch::NamedBarrier::arrive(kNThreads, static_cast<int>(NamedBarriers::SoftmaxReady));
    } else {
        const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
        int cur_block_table = __ldg(&block_table[n_block]);

        const index_t row_offset_q = bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
        Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.q_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_Q;
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx - kNThreadsS);
        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                                                              params.seqlen_q - m_block * kBlockM);

        const index_t row_offset_k = (bidh / params.h_h_k_ratio) * params.k_head_stride;
        Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.k_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_K;
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx - kNThreadsS);
        Tensor tKgK = gmem_thr_copy_K.partition_S(gK);
        Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
        Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_K.partition_S(cK);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
        Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));

        if (n_block % 2 == 1) {
            // Double buffer for sK
            constexpr int sK_offset = size(sK);
            tKsK.data() = tKsK.data() + sK_offset;
            tOrVt.data() = tOrVt.data() + sK_offset / 8;
        }

        // We need to clear the sK smem tiles because K is V.
        const index_t offset_k = cur_block_table * params.k_batch_stride;
        tKgK.data() = tKgK.data() + offset_k;
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK,
                                                                                        seqlen_k - n_block * kBlockN);
        tKgK.data() = tKgK.data() + -offset_k;
        cute::cp_async_fence();

        if (n_block - 1 >= n_block_min) {
            cur_block_table = __ldg(&block_table[n_block - 1]);
        }

#pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
            flash::cp_async_wait<0>();
            __syncthreads();

            if (n_block - 1 >= n_block_min) {
                // Double buffer for sK
                const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
                tKsK.data() = tKsK.data() + sK_offset;

                const index_t offset_k = cur_block_table * params.k_batch_stride;
                tKgK.data() = tKgK.data() + offset_k;
                flash::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK);
                tKgK.data() = tKgK.data() + -offset_k;
                cute::cp_async_fence();
            }

            cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<int>(NamedBarriers::SReady));

            if (n_block - 2 >= n_block_min) {
                cur_block_table = __ldg(&block_table[n_block - 2]);
            }

            typename Kernel_traits::TiledMma tiled_mma;
            auto tSrS_layout = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}).layout();
            Tensor rP = make_tensor<Element>(tSrS_layout);
            Tensor scale_o = make_tensor<float>(Shape<_2>{});
            cute::copy(tScale_osScale_o, scale_o);
            cute::copy(tPsP, rP);

            flash::rescale_o(tOrO, scale_o);

            Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
            flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_o, tOrP, tOrVt, tOrO);

            // Double buffer for sK
            const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
            tOrVt.data() = tOrVt.data() + sK_offset / 8;
        }

        cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<int>(NamedBarriers::SoftmaxReady));
        cute::copy(tRow_maxsRow_max, softmax.row_max);
        cute::copy(tRow_sumsRow_sum, softmax.row_sum);
    }
    // T0_DEBUG_PRINT("After main loop: bidb %d bidh %d m_block %d n_split_idx %d\n", bidb, bidh, m_block, n_split_idx);
    // if (NoSplit)
    //     store<Kernel_traits, false>(params, bidb, bidh, m_block, n_split_idx, shared_storage, tOrO, softmax);
    // else
    //     store<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx, shared_storage, tOrO, softmax);
    if (NoSplit)
        store<Kernel_traits, false>(params, bidb, bidh, m_block, n_split_idx & 0xffff, shared_storage, tOrO, softmax);
    else{
        if constexpr(CLUSTER_DIM == 1){
            store<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx & 0xffff, shared_storage, tOrO, softmax);
        }else if(n_split_idx >> 16 == 0){
            store<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx & 0xffff, shared_storage, tOrO, softmax);
        } else{

            using ElementAccum = typename Kernel_traits::ElementAccum;

            const int tidx = threadIdx.x;

            typename Kernel_traits::TiledMmaO tiled_mma_o;
            auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

            // Epilogue

            // const int split_offset = __ldg(params.num_splits_ptr + bidb);  //! totally offset
            int split_idx = params.num_splits_ptr[bidb] + n_split_idx & 0xffff;
            int END_SP_IDX = params.num_splits_ptr[bidb+1];

            bool is_no_split = false;
       
            T0_DEBUG_PRINT("split_offset %d,bidb %d  m_block %d n_split_idx %d kNThreadsS %d\n",split_idx, bidb, m_block, n_split_idx&0xffff, kNThreadsS);

            Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, true>(tOrO, params.scale_softmax);
            // if(threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 0){
            //     print(lse);
            //     printf(" size %d\n",decltype(size(lse))::value);
            // }

            
            // Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum *>(shared_storage.smem_o.data())), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)  (64, 512)  //! 
            Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum *>(shared_storage.smem_sK0.data())), Layout<Shape<_64, _512>, Stride<Int<512>,_1>>{}); // (SMEM_M,SMEM_N)  (64, 512)

            // if(threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 0){
            //     print(sOaccum);
            //     printf("sOaccum size %d\n",decltype(size(sOaccum))::value);
            // }
            // Partition sO to match the accumulator partitioning
            using SmemTiledCopyO =  typename Kernel_traits::SmemCopyAtomOaccum;
            auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_o);
            auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
            Tensor rO = flash::convert_type<ElementAccum>(tOrO);
            Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
            Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

            // if(threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 0){  //! why add some print, and compute result is error?
            //     // print(taccOrOaccum);
            //     // printf("taccOrOaccum size %d\n",decltype(size(taccOrOaccum))::value);  // ! can't print register tensor, otherwise the result of output O is None
            //     print(taccOsOaccum);
            //     printf("taccOsOaccum size %d\n",decltype(size(taccOsOaccum))::value);
            // }
            T0_DEBUG_PRINT("gridDim {%d %d %d} blockdim {%d %d %d} cluster %d bidb %d bidh %d m_block %d n_split_idx %d split_idx %d END_SP_IDX %d\n",gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, CLUSTER_DIM, bidb, bidh, m_block, n_split_idx&0xffff, split_idx, END_SP_IDX);

            __syncthreads();

            cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);  // ! copy register to smem

            const index_t row_offset_o = bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
            const index_t row_offset_oaccum = (((split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_v;
            const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            const index_t row_offset_lseaccum = ((split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            // T0_DEBUG_PRINT("row_offset_o %ld, params.o_batch_stride %ld params.o_row_stride %ld params.o_head_stride %ld bidh %d bidb %d\n",row_offset_o, params.o_batch_stride, params.o_row_stride, params.o_head_stride,bidh, bidb);
            // //! type and format symbols must match
            // T0_DEBUG_PRINT("row_offset_oaccum %ld,row_offset_lse %ld row_offset_lseaccum %ld params.h %d   params.seqlen_q  %d split_idx %d n_split_idx %d\n",row_offset_oaccum,row_offset_lse,row_offset_lseaccum, params.h,  params.seqlen_q ,split_idx, n_split_idx);

            Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) +  row_offset_o),Shape<Int<kBlockM>, Int<kHeadDimV>>{},make_stride( params.o_row_stride, _1{}));
                                        //! o_row_stride and kHeadDimV is same in this version
            Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr)  + row_offset_lseaccum), Shape<Int<kBlockM>>{}, Stride<_1>{});

            if(END_SP_IDX +n_split_idx&0xffff - split_idx == 1){ //! mod-v2 <= -> ==  , CLUSTER_DIM -> 1
                is_no_split = true;
                gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse), Shape<Int<kBlockM>>{}, Stride<_1>{});
            }


            // using GmemTiledCopyO = std::conditional_t<false, typename Kernel_traits::GmemTiledCopyO, typename Kernel_traits::GmemTiledCopyOaccum>;
            // GmemTiledCopyO gmem_tiled_copy_Oaccum;
            // auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
            // Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
            // Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

            __syncthreads();

            //! I need to use 8 warps to deal with epilogue {cluster reduce }
            // if (tidx >= kNThreadsS) { return; } //! only use 128 threads to deal with epilogue

            // ! I cann't need to use registers as bridge between smem and gmem
            // Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
            // cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);   //! copy smem to register
            
            
            namespace cg = cooperative_groups;
            cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
            int rank_id = cluster.block_rank(); // Get the rank of this block in the cluster
            // int CLUSTER_DIM = cluster.size()/blockDim.x;    // Total number of blocks in the cluste

            char* sO_addr = (char*)shared_storage.smem_sK0.data();	// ! it's different with new version , output sO is in sQ position
            char* hl_dsm_buf = (char*)shared_storage.smem_sP0.data(); //!
            ElementAccum* hl_dsm_sO_temp_buf = (ElementAccum*)shared_storage.smem_sQ.data(); // !
            

            Tensor sLSE_buf = make_tensor(make_smem_ptr((float*)hl_dsm_buf), Layout<
                Shape<Int<kBlockM>>,
                Stride<_1>
            >{});

            if( tidx < kNThreadsS) {
                Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
                Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // ((MMA=4, X), MMA_M, MMA_K=1)
                Tensor taccOcO_row = taccOcO(make_coord(0, _, 0), _, 0);
                CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
                if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
                    for (int mi = 0; mi < size(lse); ++mi) {
                        const int row = get<0>(taccOcO_row(mi));
                        if (row < params.seqlen_q - m_block * kBlockM) { sLSE_buf(row) = lse(mi); }  //! todo 
                    }
                }
            }

            constexpr int SPLIT_IN_CLUSTER  = CLUSTER_DIM;

            float* sLSE_buf_temp2_addr = (float*)hl_dsm_buf+kBlockM*sizeof(float);
            Tensor sLSE_buf_temp2 = make_tensor(make_smem_ptr(sLSE_buf_temp2_addr), Layout<
                Shape<Int<kBlockM>>,  //! mod-v
                Stride<_1>	
            >{});

            float * sO_buf_addr = (float*)sO_addr;
            Tensor sOdsmBuf = make_tensor(make_smem_ptr(sO_buf_addr), Layout<
                Shape<_64, _512>,
                Stride<Int<512>, _1>	// We use stride = 520 here to avoid bank conflict
            >{});

            // if(tidx == 0 && blockIdx.z == 0 && blockIdx.x == 0){
            //     // print(sOdsmBuf);
            //     for(int i = 0; i < 10; i++){
            //         printf("en %f %f %f %f %f %f %f %f\n ",sOdsmBuf(i,0),sOdsmBuf(i,10),sOdsmBuf(i,20),sOdsmBuf(i,30),sOdsmBuf(i,40),sOdsmBuf(i,50),sOdsmBuf(i,60), sOdsmBuf(i,70));
            //     }
            //     // printf("sOdsmBuf size %d\n",decltype(size(sOdsmBuf))::value);
            // }

            Tensor sOdsmBuf_temp= make_tensor(make_smem_ptr(hl_dsm_sO_temp_buf), Layout<
                Shape<_32, _512>, 
                Stride<Int<512>, _1>	// We use stride = 520 here to avoid bank conflict //! 520->
            >{});

            int num_valid_seq_q = min(params.seqlen_q - m_block*kBlockM, kBlockM);
            int H_NUM_PER_SM = cute::ceil_div(num_valid_seq_q, SPLIT_IN_CLUSTER);
            int warp_idx = threadIdx.x / 32; // ! warp_idx is the warp idx in the cluster
            int lane_id = threadIdx.x % 32;

            ElementAccum* peer_dsm_sO_buf_addr; // ! peer dsm sO buffer address
            ElementAccum* peer_dsm_sO_buf_temp_addr;  //! every threads have one copy
            ElementAccum* peer_dsm_sLSE_buf_addr;

             if(warp_idx < SPLIT_IN_CLUSTER){
                 if (warp_idx != rank_id) {
                    peer_dsm_sO_buf_addr = cluster.map_shared_rank(sO_buf_addr, warp_idx);
                    peer_dsm_sO_buf_temp_addr = cluster.map_shared_rank(hl_dsm_sO_temp_buf, warp_idx);
                    // peer_dsm_sO_buf_temp_addr[j] = cluster.map_shared_rank(sO_buf_addr, j);
                    peer_dsm_sLSE_buf_addr = cluster.map_shared_rank(sLSE_buf_temp2_addr, warp_idx);
                }
            }

            TMABarrier* barriers_dsm = (TMABarrier*)(shared_storage.smem_sScale0);
            if(tidx < kBlockM){ 
                barriers_dsm[threadIdx.x].init(1);
            }
            cutlass::arch::fence_view_async_shared();  //! to sync barrier
  
            int stage = max(num_valid_seq_q/ 32,1); // ! todo maybe use the T::BLOCK_SIZE_M? constexpr
            int H_NUM_PER_SM_PER_STAGE = H_NUM_PER_SM/ stage; //! mod-v
            // int H_NUM_PER_SM_PER_STAGE = cute::ceil_div(H_NUM_PER_SM, stage);


            cluster.sync(); // ! sync all blocks in the cluster  

            if(warp_idx < SPLIT_IN_CLUSTER){ // every warp copy one rank's data 
                if(warp_idx != rank_id){  
                    // sLSE_buf<j> --> sLSE_buf_temp2< rank>
                    if(lane_id < H_NUM_PER_SM_PER_STAGE){  // 32 // every thread copy one head's data   //! 64 head only copy half
                        // if(SPLIT_IN_CLUSTER*lane_id + warp_idx < num_valid_seq_q){ // ! no need, actually
                        SM90_BULK_COPY_S2S::copy((ElementAccum*)((ElementAccum*)sO_addr + sOdsmBuf.layout()(SPLIT_IN_CLUSTER*lane_id + warp_idx , 0)), (ElementAccum*)(peer_dsm_sO_buf_temp_addr + sOdsmBuf.layout()(SPLIT_IN_CLUSTER*lane_id+rank_id, 0)),  kHeadDimV * sizeof(ElementAccum), (uint64_t*)(&barriers_dsm[lane_id*SPLIT_IN_CLUSTER + warp_idx]));    
                        *(ElementAccum*)(peer_dsm_sLSE_buf_addr + SPLIT_IN_CLUSTER*lane_id + rank_id) = sLSE_buf(SPLIT_IN_CLUSTER*lane_id + warp_idx);
                        // barriers_dsm[lane_id*SPLIT_IN_CLUSTER + warp_idx].arrive_and_expect_tx(T::HEAD_DIM_V * sizeof(ElementAccum));
                        // }
                    }
                }              
            }
            if(stage == 2){
                cluster.sync(); // ! sync all blocks in the cluster  
                if(warp_idx < SPLIT_IN_CLUSTER){ // every warp copy one rank's data
                    if(warp_idx != rank_id){  
                        // sLSE_buf<j> --> sLSE_buf_temp2< rank>
                        if(lane_id < H_NUM_PER_SM_PER_STAGE){  // 32 // every thread copy one head's data   //! 64 head only copy half
                            if(SPLIT_IN_CLUSTER*H_NUM_PER_SM_PER_STAGE + warp_idx < num_valid_seq_q){
                                SM90_BULK_COPY_S2S::copy((ElementAccum*)((ElementAccum*)sO_addr + sOdsmBuf.layout()(SPLIT_IN_CLUSTER*lane_id+ H_NUM_PER_SM_PER_STAGE*SPLIT_IN_CLUSTER + warp_idx, 0)), (ElementAccum*)(peer_dsm_sO_buf_addr + sOdsmBuf.layout()(SPLIT_IN_CLUSTER*lane_id+rank_id, 0)),  kHeadDimV * sizeof(ElementAccum), (uint64_t*)(&barriers_dsm[SPLIT_IN_CLUSTER*lane_id+ H_NUM_PER_SM_PER_STAGE*SPLIT_IN_CLUSTER + warp_idx]));    
                                *(ElementAccum*)(peer_dsm_sLSE_buf_addr + SPLIT_IN_CLUSTER*lane_id + H_NUM_PER_SM_PER_STAGE*SPLIT_IN_CLUSTER+rank_id) = sLSE_buf(SPLIT_IN_CLUSTER*lane_id+ H_NUM_PER_SM_PER_STAGE*SPLIT_IN_CLUSTER + warp_idx);
                                // barriers_dsm[SPLIT_IN_CLUSTER*lane_id+ H_NUM_PER_SM_PER_STAGE*SPLIT_IN_CLUSTER + warp_idx].arrive_and_expect_tx(T::HEAD_DIM_V * sizeof(ElementAccum));
                            }
                        }
                    }              
                }

            }
            constexpr int WARP_DIM = 8;

            int H_NUM_PER_WARP = cute::ceil_div(H_NUM_PER_SM, WARP_DIM);

            float local_lse[8];  // ! local lse in one warp //! mod-v
            constexpr int MAX_H_NUM_PER_WARP = 8/CLUSTER_DIM;
            float max_lse[MAX_H_NUM_PER_WARP];  // ! max lse in one warp  

            cluster.sync(); // ! sync all blocks in the cluster  

            // if(tidx == 0 && blockIdx.z == 0 && blockIdx.x == 0){
            //     // print(sOdsmBuf);
            //     for(int i = 0; i < 10; i++){
            //         printf("ed %f %f %f %f %f %f %f %f\n ",sOdsmBuf_temp(i,0),sOdsmBuf_temp(i,10),sOdsmBuf_temp(i,20),sOdsmBuf_temp(i,30), sOdsmBuf_temp(i,40),sOdsmBuf_temp(i,50),sOdsmBuf_temp(i,60), sOdsmBuf_temp(i,70));
            //     }
            //     // printf("sOdsmBuf size %d\n",decltype(size(sOdsmBuf))::value);
            // }

            CUTLASS_PRAGMA_UNROLL //! mod-v
            for (int i = 0; i < MAX_H_NUM_PER_WARP; ++i) { 
                max_lse[i] =  -INFINITY;
                local_lse[i*SPLIT_IN_CLUSTER +rank_id] = sLSE_buf(SPLIT_IN_CLUSTER*WARP_DIM*i + SPLIT_IN_CLUSTER*warp_idx + rank_id);
                max_lse[i] = max(max_lse[i],local_lse[i*SPLIT_IN_CLUSTER +rank_id]);
                if(SPLIT_IN_CLUSTER*WARP_DIM*i + SPLIT_IN_CLUSTER*warp_idx + rank_id < num_valid_seq_q){
                    CUTLASS_PRAGMA_UNROLL
                    for(int j = 0 ; j < SPLIT_IN_CLUSTER; j++) {  //! need to unroll mode    
                        if (j != rank_id) {
                            local_lse[i*SPLIT_IN_CLUSTER +j] = sLSE_buf_temp2(SPLIT_IN_CLUSTER*WARP_DIM*i + SPLIT_IN_CLUSTER*warp_idx + j);
                            max_lse[i] = max(max_lse[i], local_lse[i*SPLIT_IN_CLUSTER +j]); // ! max lse and sum lse continue with num head per warp
                            // !!! last version is error
                        }
                    }
                }
            }
            
            float sum_lse[MAX_H_NUM_PER_WARP];  // ! sum lse in one warp
            CUTLASS_PRAGMA_UNROLL
            for (int h_idx = 0; h_idx < MAX_H_NUM_PER_WARP; ++h_idx) { //! need to unroll mode
                if(SPLIT_IN_CLUSTER*WARP_DIM*h_idx + SPLIT_IN_CLUSTER*warp_idx + rank_id < num_valid_seq_q){
                    max_lse[h_idx] = max_lse[h_idx] == -INFINITY ? 0.0f : max_lse[h_idx];  // ! hl add In case all local LSEs are -inf
                    sum_lse[h_idx] = 0.0f;
                    CUTLASS_PRAGMA_UNROLL
                    for(int j = 0 ; j < SPLIT_IN_CLUSTER; j++) {
                            sum_lse[h_idx] = sum_lse[h_idx] + expf(local_lse[SPLIT_IN_CLUSTER*h_idx+j] - max_lse[h_idx]);
                    }
                    sum_lse[h_idx] = (sum_lse[h_idx] == 0.f || sum_lse[h_idx] != sum_lse[h_idx]) ? INFINITY : logf(sum_lse[h_idx]) + max_lse[h_idx]; //!  float global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? INFINITY : log2f(sum_lse) + max_lse;
                    if(lane_id == 0){
                        // if(is_no_split){
                        //     // sLSE_buf(rank_id  + warp_idx* SPLIT_IN_CLUSTER + SPLIT_IN_CLUSTER*WARP_DIM*h_idx) = sum_lse[h_idx]/(float)M_LOG2E ; // ! todo  / M_LOG2E  only last write gmail
                        //     gSoftmaxLse(rank_id  + warp_idx* SPLIT_IN_CLUSTER + SPLIT_IN_CLUSTER*WARP_DIM*h_idx) = sum_lse[h_idx]/(float)M_LOG2E ;
                        // }else{
                        //     // sLSE_buf(rank_id  + warp_idx* SPLIT_IN_CLUSTER + SPLIT_IN_CLUSTER*WARP_DIM*h_idx) = sum_lse[h_idx] ; // ! todo  / M_LOG2E  only last write gmail
                        //     gSoftmaxLse(rank_id  + warp_idx* SPLIT_IN_CLUSTER + SPLIT_IN_CLUSTER*WARP_DIM*h_idx) = sum_lse[h_idx] ;
                        // }
                        gLSEaccum(rank_id  + warp_idx* SPLIT_IN_CLUSTER + SPLIT_IN_CLUSTER*WARP_DIM*h_idx) = sum_lse[h_idx] ;
                    }
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int h_idx = 0; h_idx < MAX_H_NUM_PER_WARP; ++h_idx) { //! need to unroll mode
                if(SPLIT_IN_CLUSTER*WARP_DIM*h_idx + SPLIT_IN_CLUSTER*warp_idx + rank_id < num_valid_seq_q){
                    CUTLASS_PRAGMA_UNROLL
                    for(int j = 0 ; j < SPLIT_IN_CLUSTER; j++) {
                        local_lse[h_idx*SPLIT_IN_CLUSTER+j] = expf(local_lse[h_idx*SPLIT_IN_CLUSTER+j] - sum_lse[h_idx]);
                    }   
                }
            }
            constexpr int ELEMS_PER_THREAD = 512 / 32;  // ! elems per thread
            ElementAccum result[ELEMS_PER_THREAD];
            CUTLASS_PRAGMA_UNROLL
            for (int h_idx = 0; h_idx < H_NUM_PER_WARP; ++h_idx) {
                if(SPLIT_IN_CLUSTER*WARP_DIM*h_idx + SPLIT_IN_CLUSTER*warp_idx + rank_id < num_valid_seq_q){  //! todo
                    CUTLASS_PRAGMA_UNROLL
                    for (int ie = 0; ie < ELEMS_PER_THREAD; ++ie)
                        result[ie] = 0.0f;
                    for(int sp = 0; sp < SPLIT_IN_CLUSTER; sp++){
                        if(local_lse[h_idx*SPLIT_IN_CLUSTER+sp] != 0.f){
                            if(sp == rank_id){
                                CUTLASS_PRAGMA_UNROLL
                                for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                                    result[i] += local_lse[h_idx*SPLIT_IN_CLUSTER+sp] * sOdsmBuf(SPLIT_IN_CLUSTER*WARP_DIM*h_idx + SPLIT_IN_CLUSTER*warp_idx + sp, lane_id + i*32);
                                }
                            }
                            else{
                                if(SPLIT_IN_CLUSTER*WARP_DIM*h_idx + SPLIT_IN_CLUSTER*warp_idx + rank_id >=  H_NUM_PER_SM_PER_STAGE*SPLIT_IN_CLUSTER){
                                    CUTLASS_PRAGMA_UNROLL
                                    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                                        result[i] += local_lse[h_idx*SPLIT_IN_CLUSTER+sp] *  sOdsmBuf( SPLIT_IN_CLUSTER*WARP_DIM*h_idx + SPLIT_IN_CLUSTER*warp_idx  -H_NUM_PER_SM_PER_STAGE*SPLIT_IN_CLUSTER + sp,lane_id + i*32);
                                    }
                                }else{
                                    CUTLASS_PRAGMA_UNROLL
                                    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                                        result[i] += local_lse[h_idx*SPLIT_IN_CLUSTER+sp] *  sOdsmBuf_temp( SPLIT_IN_CLUSTER*WARP_DIM*h_idx + SPLIT_IN_CLUSTER*warp_idx + sp,lane_id + i*32);
                                    }
                                }
                            }
                        }
                    }
                    if(is_no_split){
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < ELEMS_PER_THREAD; ++i)  // ! todo merge into global write
                            gO(SPLIT_IN_CLUSTER*WARP_DIM*h_idx + SPLIT_IN_CLUSTER*warp_idx + rank_id, lane_id + i*32) = (Element)result[i];
                    }
                    else{
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < ELEMS_PER_THREAD; ++i)  // ! todo merge into global write
                            sOdsmBuf(SPLIT_IN_CLUSTER*WARP_DIM*h_idx + SPLIT_IN_CLUSTER*warp_idx + rank_id, lane_id + i*32) = result[i];
                    }
                }
            }


            if(!is_no_split){
                Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>( params.oaccum_ptr) +  row_offset_oaccum ),Shape<Int<kBlockM>, Int<kHeadDimV>>{},make_stride(kHeadDimV, _1{}));
                
                cutlass::arch::fence_view_async_shared();
                __syncthreads();

                int row = threadIdx.x;
                if (SPLIT_IN_CLUSTER*row + rank_id < num_valid_seq_q ) {
                    SM90_BULK_COPY_S2G::copy(&sOdsmBuf(SPLIT_IN_CLUSTER*row + rank_id, _0{}), &gOaccum(SPLIT_IN_CLUSTER*row + rank_id, _0{}), kHeadDimV*sizeof(ElementAccum));
                    cute::tma_store_arrive();
                    // gSoftmaxLseAccum(SPLIT_IN_CLUSTER*row + rank_id) = sLSE_buf(SPLIT_IN_CLUSTER*row + rank_id); 
                }
                cute::tma_store_wait<0>();
            }


            
            // if(threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 0){  //! why add some print, and compute result is error?
            //     print(sOdsmBuf_temp);
            //     printf("sOdsmBuf_temp size %d\n",decltype(size(sOdsmBuf_temp))::value);  // ! can't print register tensor, otherwise the result of output O is None
            //     print(sLSE_buf);
            //     printf("sLSE_buf size %d\n",decltype(size(sLSE_buf))::value);  // !
            //     // print(taccOcO);
            //     // printf("taccOcO size %d\n",decltype(size(taccOcO))::value);  // !  ArithTuple(0,0) o ((_2,_2,_32),_1,_1):((_1@1,_8@0,_8@1),_0,_0)taccOcO size 128
            // }

            // // Construct identity layout for sO
            // Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
            // // Repeat the partitioning with identity layouts
            // Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
            // Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
            //  if(threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 0){  //! why add some print, and compute result is error?
            //     // print(taccOrOaccum);
            //     // printf("taccOrOaccum size %d\n",decltype(size(taccOrOaccum))::value);  // ! can't print register tensor, otherwise the result of output O is None
            //     print(tOrOaccum); //! ((_1,_4),_8,_8):((_0,_1),_4,_32)  si ze is 256
            //     printf("tOrOaccum size %d\n",decltype(size(tOrOaccum))::value);  // !
            //     print(tOgOaccum);  //!  ((_1,_4),_8,_8):((_0,_1),4096,_64)
            //     printf("tOgOaccum size %d\n",decltype(size(tOgOaccum))::value); 

            // }
            // // Clear_OOB_K must be false since we don't want to write zeros to gmem
            // flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            //         gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, params.seqlen_q - m_block * kBlockM
            // );  //! copy register to gmem

        }

        {
            // __syncthreads();

            //     if (tidx >= kNThreadsS) { return; } //! only use 128 threads to deal with epilogue

            //     Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
            //     cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

            //     Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
            //     Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // ((MMA=4, X), MMA_M, MMA_K=1)
            //     Tensor taccOcO_row = taccOcO(make_coord(0, _, 0), _, 0);
            //     CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M

            //     if (get<1>(taccOcO_row(0)) == 0) {
            // #pragma unroll
            //         for (int mi = 0; mi < size(lse); ++mi) {
            //             const int row = get<0>(taccOcO_row(mi));
            //             if (row < params.seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }  //! todo 
            //         }
            //     }

            //     // Construct identity layout for sO
            //     Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
            //     // Repeat the partitioning with identity layouts
            //     Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
            //     Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
            //     // Clear_OOB_K must be false since we don't want to write zeros to gmem
            //     flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            //             gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, params.seqlen_q - m_block * kBlockM
            //     ); //! copy register to gmem
        }


    }
}

template<typename Kernel_traits, bool Is_causal, typename SharedStorage, int CLUSTER_DIM>
__global__ void __cluster_dims__(1, 1, CLUSTER_DIM) __launch_bounds__(Kernel_traits::kNThreads, 1)  // ! todo __launch_bounds__(Kernel_traits::kNThreads, 1, 1)
flash_fwd_splitkv_mla_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int begin_seqlen = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int end_seqlen = tile_scheduler_metadata.w;
    if (begin_idx >= params.b) return;
    int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);
    
    // T0_DEBUG_PRINT("bidh: %d, m_block: %d, partition_idx: %d, begin_idx: %d, end_idx: %d, begin_seqlen: %d, end_seqlen: %d, begin_n_split_idx: %d\n", bidh, m_block, partition_idx, begin_idx, end_idx, begin_seqlen, end_seqlen, begin_n_split_idx);

#pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
        const int seqlen_k = __ldg(params.cu_seqlens_k + batch_id);
        const int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
        const int n_block_max = batch_id == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(seqlen_k, kBlockN);
        const bool NoSplit = n_block_min == 0 && n_block_max == cute::ceil_div(seqlen_k, kBlockN);
        if (batch_id > begin_idx) {
            __syncthreads();  // Barrier between two tiles.
        }
        flash::compute_attn_1rowblock_splitkv_mla<Kernel_traits, Is_causal,CLUSTER_DIM>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, NoSplit, shared_storage);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Element, typename ElementAccum, typename index_t, int kHeadDimV, int kMaxSplits>
__global__ void __launch_bounds__(256, 1, 1)
flash_fwd_splitkv_mla_combine_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    constexpr int kNThreads = 128;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int hs = params.h * params.seqlen_q;
    const int batch_idx = bidx / hs;
    const int hs_idx = bidx % hs;

    const int split_offset = __ldg(params.num_splits_ptr + batch_idx);
    const int actual_num_splits = __ldg(params.num_splits_ptr + batch_idx + 1) - split_offset;
    
    G0_DEBUG_PRINT("actual_num_splits: %d\n", actual_num_splits);  //! 

    FLASH_DEVICE_ASSERT(actual_num_splits <= kMaxSplits);
    if (actual_num_splits <= 1) return;  //! mod == -> <= 

    __shared__ ElementAccum sLseScale[kMaxSplits];

    const index_t row_offset_lseaccum = split_offset * hs + hs_idx;
    const index_t row_offset_lse = bidx;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                   Shape<Int<kMaxSplits>>{}, make_stride(hs));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<_1>{}, Stride<_1>{});

    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0) {
        constexpr int kNLsePerThread = cute::ceil_div(kMaxSplits, 32);

        float local_lse[kNLsePerThread];
        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            local_lse[i] = split < actual_num_splits ? gLSEaccum(split) : -INFINITY;
        }

        float max_lse = -INFINITY;
        for (int i = 0; i < kNLsePerThread; ++i) max_lse = max(max_lse, local_lse[i]);
        for (int offset = 16; offset >= 1; offset /= 2) max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

        float sum_lse = 0;
        for (int i = 0; i < kNLsePerThread; ++i) sum_lse = sum_lse + expf(local_lse[i] - max_lse);
        for (int offset = 16; offset >= 1; offset /= 2) sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

        float global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? INFINITY : logf(sum_lse) + max_lse;
        if (tidx == 0) gLSE(0) = global_lse;

        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            if (split < actual_num_splits) sLseScale[split] = expf(local_lse[i] - global_lse);
        }
    }
    __syncthreads();

    static_assert(kHeadDimV % kNThreads == 0);
    constexpr int Elements = kHeadDimV / kNThreads;
    const index_t row_offset_oaccum = (split_offset * hs + hs_idx) * kHeadDimV;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kHeadDimV>>{}, Stride<_1>{});
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            Layout<Shape<Int<kNThreads>>>{},
            Layout<Shape<Int<Elements>>>{}));
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    for (int split = 0; split < actual_num_splits; ++split) {
        cute::copy(tOgOaccum, tOrOaccum);
        ElementAccum lse_scale = sLseScale[split];
        for (int i = 0; i < size(tOrO); ++i) {
            tOrO(i) += lse_scale * tOrOaccum(i);
        }
        tOgOaccum.data() = tOgOaccum.data() + hs * kHeadDimV;
    }

    Tensor rO = flash::convert_type<Element>(tOrO);
    const int head_idx = (bidx - batch_idx * hs) / params.seqlen_q;
    const int row = bidx - batch_idx * hs - head_idx * params.seqlen_q;
    auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride + head_idx * params.o_head_stride + row * params.o_row_stride;
    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + tidx * Elements), Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
    cute::copy(rO, gO);
}

} // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename SharedStorage>
void run_flash_splitkv_fwd_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.page_block_size == Kernel_traits::kBlockN);
    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {

        int all_batch = params.b *  (132/params.num_sm_parts);  //!
        int cluster_dim  = 1; //!
        if(all_batch < 4){ //!
            cluster_dim = 8; //!
            params.num_sm_parts = 120/ (132/params.num_sm_parts); //!
        }else if(all_batch < 16){ //!
            cluster_dim = 4; //!
            params.num_sm_parts = 120/ (132/params.num_sm_parts); //! 
        }else if(all_batch < 64){ //! 
            cluster_dim = 2; //!
        } //!
        params.num_sm_parts = params.num_sm_parts/cluster_dim*cluster_dim;  //!

        // auto kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, SharedStorage>;
        // cudaKernel_t kernel;  //!
        auto kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, SharedStorage, 1>;
        if(cluster_dim == 8){  //!
            kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, SharedStorage, 8>; //!
        }else if(cluster_dim == 4){  //!
            kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, SharedStorage, 4>;
        }else if(cluster_dim == 2){
            kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, SharedStorage, 2>;
        }
        // else{
        //     kernel = (cudaKernel_t)&flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, SharedStorage, 1>;
        // }

        constexpr size_t smem_size = sizeof(SharedStorage);
        DEBUG_PRINT("smem_size %d\n",smem_size);
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });

    CHECK_CUDA_KERNEL_LAUNCH();

    dim3 grid_combine(params.b * params.h * params.seqlen_q);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_kernel<
                typename Kernel_traits::Element, typename Kernel_traits::ElementAccum, typename Kernel_traits::index_t, Kernel_traits::kHeadDimV, kMaxSplits>;
        combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    static_assert(Headdim == 576);
    FLASH_ASSERT(params.d_v == 512);
    FLASH_ASSERT(params.k_ptr == params.v_ptr);  // Shared_KV
    using Kernel_traits = Flash_fwd_kernel_traits_mla<576, 64, 64, 8, T, 512>;
    run_flash_splitkv_fwd_mla<Kernel_traits, flash::SharedStorageMLA<Kernel_traits>>(params, stream);
}

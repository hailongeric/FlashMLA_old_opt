#include "flash_fwd_mla_kernel.h"

static constexpr int MaxBatchSize = 4096;

__global__ void __launch_bounds__(256, 1, 1)
get_mla_metadata_kernel(__grid_constant__ const Mla_metadata_params params) {
    int *seqlens_k_ptr = params.seqlens_k_ptr;
    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.batch_size;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;

    __shared__ int num_blocks_shared[MaxBatchSize];
    __shared__ int num_splits_shared[MaxBatchSize];

    int total_num_blocks = 0;
    for (int i = threadIdx.x; i < batch_size; i += 32) {
        int num_blocks = cutlass::ceil_div(seqlens_k_ptr[i], block_size_n);
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
        num_blocks_shared[i] = num_blocks;
    }
    for (int offset = 16; offset >= 1; offset /= 2) {
        total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
    }
    __syncwarp();

    if (threadIdx.x == 0) {
        int all_batch = batch_size *  (132/num_sm_parts);  //!
        int cluster_dim = 1; //!

        if(all_batch < 4){ //!
            cluster_dim = 8; //!
            num_sm_parts = 120/ (132/num_sm_parts); //!
        }else if(all_batch < 16){ //!
            cluster_dim = 4; //!
            num_sm_parts = 120/ (132/num_sm_parts); //!
        }else if(all_batch < 64){ //!
            cluster_dim = 2; //!
        } //!
        num_sm_parts = num_sm_parts/cluster_dim*cluster_dim; //!


        // printf("total_num_blocks: %d num_sm_parts %d\n", total_num_blocks, num_sm_parts);
        int payload = max(cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks,fixed_overhead_num_blocks*2);
        // int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;  //!

        //! todo debug info 
        // int avg_splits = (total_num_blocks / batch_size) / (total_num_blocks/ num_sm_parts);
        // printf("get_mla_metadata_kernel: batch_size=%d, avg_splits=%d, cluster_dim=%d, num_sm_parts=%d, total_num_blocks=%d, payload=%d\n",
        //       batch_size, avg_splits, cluster_dim, num_sm_parts, total_num_blocks, payload);

        int st_off = 0;  //! 
        int temp_idx = 0; //!
        int now_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
        num_splits_shared[0] = 0;
        // for (int i = 0; i < num_sm_parts; ++i) {
        //     int tile_scheduler_metadata0[4], tile_scheduler_metadata1;
        //     tile_scheduler_metadata0[0] = now_idx;
        //     tile_scheduler_metadata0[1] = now_block * block_size_n;
        //     tile_scheduler_metadata1 = now_n_split_idx;
        //     int remain_payload = payload;
        //     while (now_idx < batch_size) {
        //         int num_blocks = num_blocks_shared[now_idx];
        //         int now_remain_blocks = num_blocks - now_block;
        //         if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
        //             cum_num_splits += now_n_split_idx + 1;
        //             num_splits_shared[now_idx + 1] = cum_num_splits;
        //             remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
        //             ++now_idx;
        //             now_block = 0;
        //             now_n_split_idx = 0;
        //         } else {
        //             if (remain_payload - fixed_overhead_num_blocks > 0) {
        //                 now_block += remain_payload - fixed_overhead_num_blocks;
        //                 ++now_n_split_idx;
        //                 remain_payload = 0;
        //             }
        //             break;
        //         }
        //     }
        //     tile_scheduler_metadata0[2] = now_block > 0 ? now_idx : now_idx - 1;
        //     tile_scheduler_metadata0[3] = now_block > 0 ? now_block * block_size_n : seqlens_k_ptr[now_idx - 1];
        //     *reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr + i * TileSchedulerMetaDataSize) = *reinterpret_cast<int4 *>(tile_scheduler_metadata0);
        //     tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 4] = tile_scheduler_metadata1;
        // }
        for (int c_id = 0; c_id < num_sm_parts/cluster_dim; ++c_id) {
            int last_batch_idx = now_idx;
            for(int rank_id = 0; rank_id < cluster_dim; rank_id++ ){ // !
                int tile_scheduler_metadata0[4], tile_scheduler_metadata1;
                tile_scheduler_metadata0[0] = now_idx;  // batch idx  //! todo use temp variable to recode those
                temp_idx  = now_idx;
                tile_scheduler_metadata0[1] = now_block * block_size_n;  // in batch n: start seqlen
                tile_scheduler_metadata1 = now_n_split_idx;
                int remain_payload = payload;
                while (now_idx < batch_size) {
                    int num_blocks = num_blocks_shared[now_idx]; //! every batch,lens
                    int now_remain_blocks = num_blocks - now_block;
                    if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                        cum_num_splits += now_n_split_idx + 1;
                        num_splits_shared[now_idx + 1] = cum_num_splits;
                        remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                        ++now_idx;
                        now_block = 0;
                        now_n_split_idx = 0;
                    } else {
                        if (remain_payload - fixed_overhead_num_blocks > 0) {
                            now_block += remain_payload - fixed_overhead_num_blocks;
                            ++now_n_split_idx;
                            remain_payload = 0;
                        }
                        break;
                    }
                }
                tile_scheduler_metadata0[2] = now_block > 0 ? now_idx : now_idx - 1; // endidx
                tile_scheduler_metadata0[3] = now_block > 0 ? now_block * block_size_n : seqlens_k_ptr[now_idx - 1];  // endseqlen
                *reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr + (c_id*cluster_dim +rank_id ) * TileSchedulerMetaDataSize) = *reinterpret_cast<int4 *>(tile_scheduler_metadata0);
                tile_scheduler_metadata_ptr[(c_id*cluster_dim +rank_id ) * TileSchedulerMetaDataSize + 4] = tile_scheduler_metadata1;
                if(rank_id == 0){
                    st_off = tile_scheduler_metadata1;
                }
                // printf("i[%d] bs[%d] %d num_splits_shared[bs] %d range(%d-%d)\n",c_id*cluster_dim +rank_id, now_idx,tile_scheduler_metadata1, num_splits_shared[now_idx],tile_scheduler_metadata0[1] ,tile_scheduler_metadata0[3] );
            }
            if( cluster_dim > 1 && (temp_idx < batch_size) && ( now_idx == last_batch_idx || (now_block == 0 && now_idx-1 == last_batch_idx))){ //!  deal with if reduce or not in cluster
                // cum_num_splits-=cluster_dim-1; //! 
                if(now_idx == last_batch_idx){
                    now_n_split_idx-= cluster_dim-1;
                }
                for(int rank_id = 0; rank_id < cluster_dim; rank_id++ ){
                    tile_scheduler_metadata_ptr[(c_id*cluster_dim +rank_id) * TileSchedulerMetaDataSize + 4] = (1<<16) |st_off;  //! reduce flag
                }
                if( now_idx-1 == last_batch_idx){
                    num_splits_shared[now_idx] -= cluster_dim-1;
                    cum_num_splits-=cluster_dim-1; //! 
                }
            } //!
        }
        FLASH_DEVICE_ASSERT(now_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncwarp();

    for (int i = threadIdx.x; i <= batch_size; i += 32) {
        num_splits_ptr[i] = num_splits_shared[i];
    }
}

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.batch_size < MaxBatchSize);
    get_mla_metadata_kernel<<<1, 32, 0, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
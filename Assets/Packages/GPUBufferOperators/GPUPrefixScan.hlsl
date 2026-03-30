#ifndef CS_GPU_PREFIX_SCAN_HLSL
#define CS_GPU_PREFIX_SCAN_HLSL

#define NUM_GROUP_THREADS 512

RWStructuredBuffer<uint> data_buffer;
RWStructuredBuffer<uint> group_sum_buffer;

uint num_elements;
uint group_offset;
uint group_sum_offset;

static const uint num_group_threads = NUM_GROUP_THREADS;
static const uint num_elements_per_group = 4 * NUM_GROUP_THREADS;

static const uint s_max_len = num_group_threads >> 5;
static const uint s_max_len_1 = s_max_len - 1;

groupshared uint4 s_max[s_max_len];
groupshared uint4 s_sum[s_max_len];

// scan input data locally and output total sums within groups
[numthreads(512, 1, 1)]
void PrefixScan(uint GTid : SV_GroupThreadID, uint group_id : SV_GroupID)
{
    group_id += group_offset;

    const uint global_ai = GTid + num_elements_per_group * group_id;
    const uint global_bi = global_ai + num_group_threads;
    const uint global_ci = global_bi + num_group_threads;
    const uint global_di = global_ci + num_group_threads;

    // -------------------------------------------------------
    const uint4 data = uint4(global_ai < num_elements ? data_buffer[global_ai] : 0u,
                             global_bi < num_elements ? data_buffer[global_bi] : 0u,
                             global_ci < num_elements ? data_buffer[global_ci] : 0u,
                             global_di < num_elements ? data_buffer[global_di] : 0u);
        
    uint4 scan = WavePrefixSum(data);
    const uint wave_id = GTid >> 5u;
    if((GTid & 31u) == 31u)
    {
        s_max[wave_id] = scan + data;
    }
    GroupMemoryBarrierWithGroupSync();
    if(GTid < s_max_len)
    {
        const uint4 sum = s_max[GTid];
        s_sum[GTid] = WavePrefixSum(sum);
    }
        
    GroupMemoryBarrierWithGroupSync();
    
    scan += s_sum[wave_id];
    uint4 groupSum = s_sum[s_max_len_1] + s_max[s_max_len_1];
    groupSum.y += groupSum.x;
    groupSum.z += groupSum.y;
    groupSum.w += groupSum.z;
        
    if (global_ai < num_elements)
        data_buffer[global_ai] = scan.x;
    if (global_bi < num_elements)
        data_buffer[global_bi] = scan.y + groupSum.x;
    if (global_ci < num_elements)
        data_buffer[global_ci] = scan.z + groupSum.y;
    if (global_di < num_elements)
        data_buffer[global_di] = scan.w + groupSum.z;
    
    if (GTid == 0u)
    {
        group_sum_buffer[group_id + group_sum_offset] = groupSum.w;
    }
}

// add each group's total sum to its scan output
[numthreads(NUM_GROUP_THREADS, 1, 1)]
void AddGroupSum(uint GTid : SV_GroupThreadID, uint group_id : SV_GroupID)
{
    group_id += group_offset;

    const uint group_sum = group_sum_buffer[group_id];

    const uint global_ai = GTid + num_elements_per_group * group_id;
    const uint global_bi = global_ai + num_group_threads;
    const uint global_ci = global_bi + num_group_threads;
    const uint global_di = global_ci + num_group_threads;

    if (global_ai < num_elements)
        data_buffer[global_ai] += group_sum;
    if (global_bi < num_elements)
        data_buffer[global_bi] += group_sum;
    if (global_ci < num_elements)
        data_buffer[global_ci] += group_sum;
    if (global_di < num_elements)
        data_buffer[global_di] += group_sum;
}


#endif /* CS_GPU_PREFIX_SCAN_HLSL */
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace AVBD.RigidOnly
{
    public class CompressedSparseRows
    {
        public NativeList<int4> Csr; // x: start, y: end, z: color, w: padding
        public NativeList<int4> Data; // x: nodeA, y: nodeB, z: edgeID, w: padding

        public CompressedSparseRows()
        {
            Csr = new NativeList<int4>(Allocator.Persistent);
            Data = new NativeList<int4>(Allocator.Persistent);
        }
        
        [BurstCompile]
        public unsafe struct CounterEdgesJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Manifold> Forces;
            
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<int> Counter;
            
            public void Execute(int i)
            {
                var f = Forces[i];
                
                 Interlocked.Increment(ref UnsafeUtility.ArrayElementAsRef<int>(Counter.GetUnsafePtr(), f.BodyA));
                 Interlocked.Increment(ref UnsafeUtility.ArrayElementAsRef<int>(Counter.GetUnsafePtr(), f.BodyB));
            }
        }
        
        [BurstCompile]
        public struct BuildCSRJob : IJob
        {
            [ReadOnly] public NativeArray<int> Counter;
            public NativeArray<int4> Csr;
            
            public void Execute()
            {
                int ptr = 0;
                for (int i = 0; i < Counter.Length; i++)
                {
                    int end = ptr + Counter[i];
                    var range = new int4(ptr, end, -1, 0);
                    Csr[i] = range;
                    ptr = end;
                }
            }
        }
        
        [BurstCompile]
        public unsafe struct FillDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Manifold> Forces;
            [ReadOnly] public NativeArray<int4> Csr;
            [NativeDisableContainerSafetyRestriction]
            [WriteOnly] public NativeArray<int4> Data;
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<int> Counter;
            
            public void Execute(int i)
            {
                var f = Forces[i];
                int4 data = new int4(f.BodyA, f.BodyB, i, 0);
                int offsetA = Interlocked.Increment(ref UnsafeUtility.ArrayElementAsRef<int>(Counter.GetUnsafePtr(), f.BodyA));
                Data[Csr[f.BodyA].x + offsetA - 1] = data;
                int offsetB = Interlocked.Increment(ref UnsafeUtility.ArrayElementAsRef<int>(Counter.GetUnsafePtr(), f.BodyB));
                Data[Csr[f.BodyB].x + offsetB - 1] = data.yxzw;
            }
        }
        
        [BurstCompile]
        public struct GreedyColoringJob: IJob
        {
            public NativeArray<int4> Csr;
            [ReadOnly] public NativeArray<int4> Data;
            [WriteOnly] public NativeReference<int> ColorCount;
            
            public void Execute()
            {
                uint colorCount = 0;
                for (int v = 0; v < Csr.Length; v++)
                {
                    int4 vertexInfo = Csr[v];
                    int start = vertexInfo.x;
                    int end = vertexInfo.y;
                
                    // 重置标记数组
                    uint usedColors = 0;
                
                    // 标记邻居使用的颜色
                    for (int idx = start; idx < end; idx++)
                    {
                        int4 edge = Data[idx];
                        int neighbor = edge.y;  // 目标节点
                        int4 neighborInfo = Csr[neighbor];
                        int neighborColor = neighborInfo.z;
                    
                        if (neighborColor != -1)
                            usedColors |= 1u << neighborColor;
                    }
                
                    // 选择最小可用颜色
                    int color = usedColors == 0 ? 0 : math.tzcnt(~usedColors);
                
                    // 赋值颜色
                    vertexInfo.z = color;
                    colorCount |= 1u << color;
                    Csr[v] = vertexInfo;
                }
                ColorCount.Value = math.tzcnt(~colorCount);
            }
        }
        
        public void Dispose()
        {
            Csr.Dispose();
            Data.Dispose();
        }
    }
}

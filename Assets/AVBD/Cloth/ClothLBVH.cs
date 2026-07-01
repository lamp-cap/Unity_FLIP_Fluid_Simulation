using System.Collections.Generic;
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD.Cloth
{
    /// <summary>
    /// LBVH(Linear BVH),用 Morton 码排序 + Karras 2012 风格的二叉基数树构建。
    /// 对应 Prop.md 第二节:用 Morton 码排序并行构建 LBVH,分别为三角形和边各建一棵树:
    ///   - 三角形 BVH 处理 vertex-Face(VF)碰撞
    ///   - 边 BVH       处理 edge-edge(EE)碰撞
    /// 每帧开始完全重建(Build),迭代中只 Refit(更新 bounds,不改结构)。
    ///
    /// 这是一个单一图元类型的通用 BVH:把 N 个图元(各带一个 AABB)建成 2N-1 个节点的树。
    /// 内部节点 [0, N-1),叶子 [N-1, 2N-1)。
    ///
    /// 构建与 Refit 全部走 Burst 多线程:
    ///   - Build:GlobalBounds → Morton(并行)→ Sort → LeafInit(并行)→ Karras(并行)→ Refit
    ///   - Refit:原子自底向上(每个内部节点由「第二个到达的孩子」线程合并 bounds)
    /// 父指针拆到独立的 <see cref="ParentIds"/> 数组,使各线程的写互不重叠,无需锁。
    /// </summary>
    public class ClothLBVH
    {
        public struct Node
        {
            public float3 BoundsMin;
            public float3 BoundsMax;
            public int Left;      // 内部节点:左孩子索引;叶子:-1
            public int Right;     // 内部节点:右孩子索引;叶子:图元索引(原始,未排序)
            public bool IsLeaf => Left < 0;
        }

        public int NumPrimitives;
        public int NumNodes;                       // 2N-1
        public NativeArray<Node> Nodes;            // [0..N-2] 内部, [N-1..2N-2] 叶子
        public NativeArray<int2> PrimIds;          // 排序后 (mortonCode, 原始 id)
        public NativeArray<float3> PrimMin;        // 原始图元 AABB(未排序,index=原始 id)
        public NativeArray<float3> PrimMax;

        public NativeArray<int> ParentIds;         // 每个节点的父节点索引(根 = -1)
        private NativeArray<int> _counters;        // 内部节点 refit 原子计数(size = max(N-1,1))
        private NativeArray<float3> _bounds;        // [0]=lo, [1]=hi 整体 AABB

        private bool _allocated;

        public void Allocate(int numPrimitives)
        {
            Dispose();
            NumPrimitives = max(numPrimitives, 1);
            NumNodes = 2 * NumPrimitives - 1;
            Nodes = new NativeArray<Node>(NumNodes, Allocator.Persistent);
            PrimIds = new NativeArray<int2>(NumPrimitives, Allocator.Persistent);
            PrimMin = new NativeArray<float3>(NumPrimitives, Allocator.Persistent);
            PrimMax = new NativeArray<float3>(NumPrimitives, Allocator.Persistent);
            ParentIds = new NativeArray<int>(NumNodes, Allocator.Persistent);
            _counters = new NativeArray<int>(max(NumPrimitives - 1, 1), Allocator.Persistent);
            _bounds = new NativeArray<float3>(2, Allocator.Persistent);
            _allocated = true;
        }

        private struct Int2Comparer : IComparer<int2>
        {
            public int Compare(int2 lhs, int2 rhs) => lhs.x - rhs.x;
        }

        // ---- Morton 码工具 ----
        static uint ExpandBits(uint v)
        {
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        }

        static uint Morton3D(float3 p)
        {
            // p 已归一化到 [0,1]
            float x = clamp(p.x * 1024f, 0f, 1023f);
            float y = clamp(p.y * 1024f, 0f, 1023f);
            float z = clamp(p.z * 1024f, 0f, 1023f);
            uint xx = ExpandBits((uint)x);
            uint yy = ExpandBits((uint)y);
            uint zz = ExpandBits((uint)z);
            return xx * 4 + yy * 2 + zz;
        }

        // 公共距离函数:最高不同 bit 的位置(用排序后下标 i,j)
        static int Delta(in NativeArray<int2> ids, int n, int i, int j)
        {
            if (j < 0 || j >= n) return -1;
            uint ci = (uint)ids[i].x;
            uint cj = (uint)ids[j].x;
            if (ci == cj)
            {
                // 处理重复码:用下标做 tie-break,保证唯一
                return 32 + Clz((uint)i ^ (uint)j);
            }
            return Clz(ci ^ cj);
        }

        static int Clz(uint x)
        {
            if (x == 0) return 32;
            int n = 0;
            if ((x & 0xFFFF0000u) == 0) { n += 16; x <<= 16; }
            if ((x & 0xFF000000u) == 0) { n += 8; x <<= 8; }
            if ((x & 0xF0000000u) == 0) { n += 4; x <<= 4; }
            if ((x & 0xC0000000u) == 0) { n += 2; x <<= 2; }
            if ((x & 0x80000000u) == 0) { n += 1; }
            return n;
        }

        static int Sign(int x) => x > 0 ? 1 : (x < 0 ? -1 : 0);

        // ====================================================================
        //  Jobs
        // ====================================================================

        [BurstCompile]
        struct GlobalBoundsJob : IJob
        {
            [ReadOnly] public NativeArray<float3> PrimMin;
            [ReadOnly] public NativeArray<float3> PrimMax;
            [WriteOnly] public NativeArray<float3> Bounds; // [0]=lo, [1]=hi

            public void Execute()
            {
                float3 lo = new float3(float.MaxValue);
                float3 hi = new float3(float.MinValue);
                int n = PrimMin.Length;
                for (int i = 0; i < n; i++)
                {
                    lo = min(lo, PrimMin[i]);
                    hi = max(hi, PrimMax[i]);
                }
                Bounds[0] = lo;
                Bounds[1] = hi;
            }
        }

        [BurstCompile]
        struct MortonJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> PrimMin;
            [ReadOnly] public NativeArray<float3> PrimMax;
            [ReadOnly] public NativeArray<float3> Bounds;
            [WriteOnly] public NativeArray<int2> PrimIds;

            public void Execute(int i)
            {
                float3 lo = Bounds[0];
                float3 ext = max(Bounds[1] - lo, new float3(1e-6f));
                float3 c = 0.5f * (PrimMin[i] + PrimMax[i]);
                float3 nc = (c - lo) / ext;
                PrimIds[i] = new int2((int)Morton3D(nc), i);
            }
        }

        [BurstCompile]
        struct LeafInitJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> PrimIds;
            [ReadOnly] public NativeArray<float3> PrimMin;
            [ReadOnly] public NativeArray<float3> PrimMax;
            public int N;
            // 写叶子区间 [N-1, 2N-2],在 parallel-for 安全范围之外
            [NativeDisableParallelForRestriction] public NativeArray<Node> Nodes;

            public void Execute(int i)
            {
                int prim = PrimIds[i].y;
                Nodes[(N - 1) + i] = new Node
                {
                    Left = -1,
                    Right = prim,
                    BoundsMin = PrimMin[prim],
                    BoundsMax = PrimMax[prim],
                };
            }
        }

        // Karras 基数树:每个内部节点 i in [0, n-2] 独立计算自己的范围/split。
        // 写 Nodes[i].Left/Right(本元素)与 ParentIds[左右孩子](每个孩子唯一一个父)。
        [BurstCompile]
        struct KarrasJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> PrimIds;
            public int N;
            public NativeArray<Node> Nodes;                       // 只写 Nodes[i],在安全范围内
            [NativeDisableParallelForRestriction] public NativeArray<int> ParentIds;

            public void Execute(int i)
            {
                int n = N;

                // 决定该内部节点覆盖的范围方向
                int d = Sign(Delta(PrimIds, n, i, i + 1) - Delta(PrimIds, n, i, i - 1));
                int deltaMin = Delta(PrimIds, n, i, i - d);
                int lmax = 2;
                while (Delta(PrimIds, n, i, i + lmax * d) > deltaMin) lmax *= 2;

                int l = 0;
                for (int t = lmax / 2; t >= 1; t /= 2)
                {
                    if (Delta(PrimIds, n, i, i + (l + t) * d) > deltaMin) l += t;
                }
                int j = i + l * d;

                // 找 split
                int deltaNode = Delta(PrimIds, n, i, j);
                int s = 0;
                int div = 2;
                int t2 = (l + div - 1) / div;
                while (t2 >= 1)
                {
                    if (Delta(PrimIds, n, i, i + (s + t2) * d) > deltaNode) s += t2;
                    if (t2 == 1) break;
                    div *= 2;
                    t2 = (l + div - 1) / div;
                }
                int gamma = i + s * d + min(d, 0);

                int left, right;
                int rangeMin = min(i, j);
                int rangeMax = max(i, j);

                // 左孩子
                if (rangeMin == gamma) left = (n - 1) + gamma;        // 叶子
                else left = gamma;                                    // 内部
                // 右孩子
                if (rangeMax == gamma + 1) right = (n - 1) + gamma + 1;
                else right = gamma + 1;

                var node = Nodes[i];
                node.Left = left;
                node.Right = right;
                Nodes[i] = node;

                ParentIds[left] = i;
                ParentIds[right] = i;
            }
        }

        // Refit 前清零:计数器归零 + 根 parent = -1(幂等)。
        [BurstCompile]
        struct ResetJob : IJob
        {
            [WriteOnly] public NativeArray<int> Counters;
            [WriteOnly] public NativeArray<int> ParentIds;

            public void Execute()
            {
                for (int i = 0; i < Counters.Length; i++) Counters[i] = 0;
                ParentIds[0] = -1;
            }
        }

        // 原子自底向上 refit。index i 遍历叶子(0..n-1):刷新叶子 bounds 后沿 ParentIds 向上爬。
        // 每个内部节点的计数器记录到达孩子数:第一个到达者停下,第二个到达者合并两孩子 bounds 并继续上爬。
        // Interlocked.Increment 提供 happens-before,保证第二个孩子看得到第一个孩子写入的 bounds。
        [BurstCompile]
        unsafe struct RefitJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> PrimMin;
            [ReadOnly] public NativeArray<float3> PrimMax;
            [ReadOnly] public NativeArray<int> ParentIds;
            public int N;
            [NativeDisableParallelForRestriction] public NativeArray<Node> Nodes;
            [NativeDisableParallelForRestriction] public NativeArray<int> Counters;

            public void Execute(int i)
            {
                int leaf = (N - 1) + i;

                // 刷新叶子 bounds
                var ln = Nodes[leaf];
                ln.BoundsMin = PrimMin[ln.Right];
                ln.BoundsMax = PrimMax[ln.Right];
                Nodes[leaf] = ln;

                int* counters = (int*)Counters.GetUnsafePtr();
                int parent = ParentIds[leaf];
                while (parent != -1)
                {
                    int prev = Interlocked.Increment(ref *(counters + parent)) - 1;
                    if (prev == 0) break; // 第一个到达的孩子:兄弟子树还没算完,停

                    // 第二个到达:两孩子 bounds 均已就绪
                    var p = Nodes[parent];
                    var cl = Nodes[p.Left];
                    var cr = Nodes[p.Right];
                    p.BoundsMin = min(cl.BoundsMin, cr.BoundsMin);
                    p.BoundsMax = max(cl.BoundsMax, cr.BoundsMax);
                    Nodes[parent] = p;

                    parent = ParentIds[parent];
                }
            }
        }

        // n == 1:根即叶子。
        [BurstCompile]
        struct SingleLeafJob : IJob
        {
            [ReadOnly] public NativeArray<float3> PrimMin;
            [ReadOnly] public NativeArray<float3> PrimMax;
            [WriteOnly] public NativeArray<Node> Nodes;
            [WriteOnly] public NativeArray<int> ParentIds;

            public void Execute()
            {
                Nodes[0] = new Node
                {
                    Left = -1,
                    Right = 0,
                    BoundsMin = PrimMin[0],
                    BoundsMax = PrimMax[0],
                };
                ParentIds[0] = -1;
            }
        }

        // ====================================================================
        //  调度 API
        // ====================================================================

        /// <summary>
        /// 调度完整重建。调用方需已填好 PrimMin/PrimMax(或其写入 job 已被 dep 覆盖)。
        /// 返回最终 handle;不阻塞。
        /// </summary>
        public JobHandle ScheduleBuild(JobHandle dependsOn = default)
        {
            int n = NumPrimitives;

            if (n == 1)
            {
                return new SingleLeafJob
                {
                    PrimMin = PrimMin,
                    PrimMax = PrimMax,
                    Nodes = Nodes,
                    ParentIds = ParentIds,
                }.Schedule(dependsOn);
            }

            var boundsHandle = new GlobalBoundsJob
            {
                PrimMin = PrimMin,
                PrimMax = PrimMax,
                Bounds = _bounds,
            }.Schedule(dependsOn);

            var mortonHandle = new MortonJob
            {
                PrimMin = PrimMin,
                PrimMax = PrimMax,
                Bounds = _bounds,
                PrimIds = PrimIds,
            }.Schedule(n, 64, boundsHandle);

            var sortHandle = PrimIds.SortJob(new Int2Comparer()).Schedule(mortonHandle);

            var leafHandle = new LeafInitJob
            {
                PrimIds = PrimIds,
                PrimMin = PrimMin,
                PrimMax = PrimMax,
                N = n,
                Nodes = Nodes,
            }.Schedule(n, 64, sortHandle);

            // Karras 仅写 Nodes[i](内部区间)与 ParentIds;链在 leaf 之后避免 Nodes 写-写冲突。
            var karrasHandle = new KarrasJob
            {
                PrimIds = PrimIds,
                N = n,
                Nodes = Nodes,
                ParentIds = ParentIds,
            }.Schedule(n - 1, 64, leafHandle);

            return ScheduleRefitCore(karrasHandle);
        }

        /// <summary>调度 refit(只更新 bounds,不改结构)。返回最终 handle;不阻塞。</summary>
        public JobHandle ScheduleRefit(JobHandle dependsOn = default)
        {
            if (NumPrimitives == 1)
            {
                return new SingleLeafJob
                {
                    PrimMin = PrimMin,
                    PrimMax = PrimMax,
                    Nodes = Nodes,
                    ParentIds = ParentIds,
                }.Schedule(dependsOn);
            }
            return ScheduleRefitCore(dependsOn);
        }

        JobHandle ScheduleRefitCore(JobHandle dependsOn)
        {
            int n = NumPrimitives;

            var resetHandle = new ResetJob
            {
                Counters = _counters,
                ParentIds = ParentIds,
            }.Schedule(dependsOn);

            return new RefitJob
            {
                PrimMin = PrimMin,
                PrimMax = PrimMax,
                ParentIds = ParentIds,
                N = n,
                Nodes = Nodes,
                Counters = _counters,
            }.Schedule(n, 64, resetHandle);
        }

        /// <summary>完整重建(阻塞)。等价于 ScheduleBuild(default).Complete()。</summary>
        public void Build() => ScheduleBuild().Complete();

        /// <summary>只更新 bounds(阻塞)。等价于 ScheduleRefit(default).Complete()。</summary>
        public void Refit() => ScheduleRefit().Complete();

        // ====================================================================
        //  查询
        // ====================================================================

        /// <summary>
        /// 用一个查询 AABB 遍历树,把所有重叠的叶子图元 id 写入 results。
        /// 调用方在 results 里自行做精确的 VF/EE 测试。
        /// </summary>
        public void QueryAABB(float3 qMin, float3 qMax, NativeList<int> results)
        {
            results.Clear();
            if (NumPrimitives == 1)
            {
                if (Overlap(Nodes[0].BoundsMin, Nodes[0].BoundsMax, qMin, qMax))
                    results.Add(Nodes[0].Right);
                return;
            }

            var stack = new NativeList<int>(64, Allocator.Temp);
            stack.Add(0);
            while (stack.Length > 0)
            {
                int idx = stack[^1];
                stack.RemoveAt(stack.Length - 1);
                var node = Nodes[idx];
                if (!Overlap(node.BoundsMin, node.BoundsMax, qMin, qMax)) continue;
                if (node.IsLeaf)
                {
                    results.Add(node.Right);
                }
                else
                {
                    stack.Add(node.Left);
                    stack.Add(node.Right);
                }
            }
            stack.Dispose();
        }

        static bool Overlap(float3 aMin, float3 aMax, float3 bMin, float3 bMax)
        {
            return all(aMin <= bMax) && all(bMin <= aMax);
        }

        /// <summary>
        /// Burst 可调用的静态查询:遍历 nodes,把重叠叶子的图元 id 写入 results。
        /// stack 由调用方提供(Allocator.Temp),避免每次查询重新分配。
        /// nodes/numPrimitives 来自某棵已 Build 的 ClothLBVH(通过 Nodes / NumPrimitives 取得)。
        /// </summary>
        public static void QueryAABBStatic(
            in NativeArray<Node> nodes, int numPrimitives,
            float3 qMin, float3 qMax,
            ref NativeList<int> stack, ref NativeList<int> results)
        {
            results.Clear();
            if (numPrimitives == 1)
            {
                if (Overlap(nodes[0].BoundsMin, nodes[0].BoundsMax, qMin, qMax))
                    results.Add(nodes[0].Right);
                return;
            }

            stack.Clear();
            stack.Add(0);
            while (stack.Length > 0)
            {
                int idx = stack[^1];
                stack.RemoveAt(stack.Length - 1);
                var node = nodes[idx];
                if (!Overlap(node.BoundsMin, node.BoundsMax, qMin, qMax)) continue;
                if (node.IsLeaf)
                {
                    results.Add(node.Right);
                }
                else
                {
                    stack.Add(node.Left);
                    stack.Add(node.Right);
                }
            }
        }

        public void Dispose()
        {
            if (!_allocated) return;
            if (Nodes.IsCreated) Nodes.Dispose();
            if (PrimIds.IsCreated) PrimIds.Dispose();
            if (PrimMin.IsCreated) PrimMin.Dispose();
            if (PrimMax.IsCreated) PrimMax.Dispose();
            if (ParentIds.IsCreated) ParentIds.Dispose();
            if (_counters.IsCreated) _counters.Dispose();
            if (_bounds.IsCreated) _bounds.Dispose();
            _allocated = false;
        }
    }
}

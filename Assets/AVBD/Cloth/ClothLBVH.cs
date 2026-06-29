using Unity.Collections;
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
    /// 第一阶段以正确性为主,Build/Refit/Query 用串行实现但保持 Job 友好的数据布局;
    /// 后续可把各阶段封装成 IJob/IJobParallelFor。
    /// </summary>
    public class ClothLBVH
    {
        public struct Node
        {
            public float3 BoundsMin;
            public float3 BoundsMax;
            public int Left;      // 内部节点:左孩子索引;叶子:-1
            public int Right;     // 内部节点:右孩子索引;叶子:图元索引(原始,未排序)
            public int Parent;
            public bool IsLeaf => Left < 0;
        }

        public int NumPrimitives;
        public int NumNodes;                       // 2N-1
        public NativeArray<Node> Nodes;            // [0..N-2] 内部, [N-1..2N-2] 叶子
        public NativeArray<uint> MortonCodes;      // 排序后
        public NativeArray<int> SortedPrimIds;     // 排序后 -> 原始图元 id
        public NativeArray<float3> PrimMin;        // 原始图元 AABB(未排序,index=原始 id)
        public NativeArray<float3> PrimMax;

        private bool _allocated;

        public void Allocate(int numPrimitives)
        {
            Dispose();
            NumPrimitives = max(numPrimitives, 1);
            NumNodes = 2 * NumPrimitives - 1;
            Nodes = new NativeArray<Node>(NumNodes, Allocator.Persistent);
            MortonCodes = new NativeArray<uint>(NumPrimitives, Allocator.Persistent);
            SortedPrimIds = new NativeArray<int>(NumPrimitives, Allocator.Persistent);
            PrimMin = new NativeArray<float3>(NumPrimitives, Allocator.Persistent);
            PrimMax = new NativeArray<float3>(NumPrimitives, Allocator.Persistent);
            _allocated = true;
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

        /// <summary>
        /// 完整重建。primMin/primMax 为每个图元的 AABB(index=原始图元 id)。
        /// 调用方负责先填好 PrimMin/PrimMax(用 SetPrimitiveBounds),或直接传入。
        /// </summary>
        public void Build()
        {
            int n = NumPrimitives;

            // 1. 计算整体 AABB(用于归一化质心)
            float3 lo = new float3(float.MaxValue);
            float3 hi = new float3(float.MinValue);
            for (int i = 0; i < n; i++)
            {
                lo = min(lo, PrimMin[i]);
                hi = max(hi, PrimMax[i]);
            }
            float3 ext = max(hi - lo, new float3(1e-6f));

            // 2. Morton 码 + 初始 id
            for (int i = 0; i < n; i++)
            {
                float3 c = 0.5f * (PrimMin[i] + PrimMax[i]);
                float3 nc = (c - lo) / ext;
                MortonCodes[i] = Morton3D(nc);
                SortedPrimIds[i] = i;
            }

            // 3. 按 Morton 码排序(配合原始 id)
            SortByMorton(0, n - 1);

            if (n == 1)
            {
                // 单图元:根即叶子
                var leaf = new Node
                {
                    Left = -1,
                    Right = SortedPrimIds[0],
                    Parent = -1,
                    BoundsMin = PrimMin[SortedPrimIds[0]],
                    BoundsMax = PrimMax[SortedPrimIds[0]],
                };
                Nodes[0] = leaf;
                return;
            }

            // 4. Karras 基数树:内部节点 i in [0, n-2],叶子 j in [n-1, 2n-2]
            //    叶子节点 k 对应排序后第 k 个图元。
            for (int i = 0; i < n; i++)
            {
                int leafIdx = (n - 1) + i;
                Nodes[leafIdx] = new Node
                {
                    Left = -1,
                    Right = SortedPrimIds[i],
                    Parent = -1,
                    BoundsMin = PrimMin[SortedPrimIds[i]],
                    BoundsMax = PrimMax[SortedPrimIds[i]],
                };
            }

            for (int i = 0; i < n - 1; i++)
            {
                // 决定该内部节点覆盖的范围方向
                int d = Sign(Delta(i, i + 1) - Delta(i, i - 1));
                int deltaMin = Delta(i, i - d);
                int lmax = 2;
                while (Delta(i, i + lmax * d) > deltaMin) lmax *= 2;

                int l = 0;
                for (int t = lmax / 2; t >= 1; t /= 2)
                {
                    if (Delta(i, i + (l + t) * d) > deltaMin) l += t;
                }
                int j = i + l * d;

                // 找 split
                int deltaNode = Delta(i, j);
                int s = 0;
                int div = 2;
                int t2 = (l + div - 1) / div;
                while (t2 >= 1)
                {
                    if (Delta(i, i + (s + t2) * d) > deltaNode) s += t2;
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

                var ln = Nodes[left]; ln.Parent = i; Nodes[left] = ln;
                var rn = Nodes[right]; rn.Parent = i; Nodes[right] = rn;
            }

            // 根节点 parent = -1
            var root = Nodes[0]; root.Parent = -1; Nodes[0] = root;

            // 5. 自底向上 Refit 内部节点 bounds
            Refit();
        }

        // 公共距离函数:最高不同 bit 的位置(用排序后下标 i,j)
        int Delta(int i, int j)
        {
            int n = NumPrimitives;
            if (j < 0 || j >= n) return -1;
            uint ci = MortonCodes[i];
            uint cj = MortonCodes[j];
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

        /// <summary>
        /// 只更新 bounds,不改结构(refit)。迭代中调用。
        /// 叶子 bounds 由调用方先用 SetLeafBounds/直接改 PrimMin/PrimMax 后,这里向上传播。
        /// </summary>
        public void Refit()
        {
            int n = NumPrimitives;
            if (n == 1) return;

            // 刷新叶子 bounds(从 PrimMin/PrimMax 取)
            for (int i = 0; i < n; i++)
            {
                int leafIdx = (n - 1) + i;
                int prim = Nodes[leafIdx].Right;
                var nd = Nodes[leafIdx];
                nd.BoundsMin = PrimMin[prim];
                nd.BoundsMax = PrimMax[prim];
                Nodes[leafIdx] = nd;
            }

            // 内部节点逆序传播(i 从 n-2 到 0,因为孩子索引若是内部节点恒 > 父?
            // Karras 树中内部孩子索引不保证 > 父,稳妥起见用后序遍历)
            RefitRecursive(0);
        }

        // 后序:先孩子后父
        float3x2 RefitRecursive(int nodeIdx)
        {
            var node = Nodes[nodeIdx];
            if (node.IsLeaf)
                return new float3x2(node.BoundsMin, node.BoundsMax);

            float3x2 l = RefitRecursive(node.Left);
            float3x2 r = RefitRecursive(node.Right);
            node.BoundsMin = min(l.c0, r.c0);
            node.BoundsMax = max(l.c1, r.c1);
            Nodes[nodeIdx] = node;
            return new float3x2(node.BoundsMin, node.BoundsMax);
        }

        // ---- 简单的归并排序(稳定),按 MortonCodes 排序并同步 SortedPrimIds ----
        void SortByMorton(int lo, int hi)
        {
            if (lo >= hi) return;
            var tmpCode = new NativeArray<uint>(hi - lo + 1, Allocator.Temp);
            var tmpId = new NativeArray<int>(hi - lo + 1, Allocator.Temp);
            MergeSort(lo, hi, tmpCode, tmpId);
            tmpCode.Dispose();
            tmpId.Dispose();
        }

        void MergeSort(int lo, int hi, NativeArray<uint> tmpCode, NativeArray<int> tmpId)
        {
            if (lo >= hi) return;
            int mid = (lo + hi) / 2;
            MergeSort(lo, mid, tmpCode, tmpId);
            MergeSort(mid + 1, hi, tmpCode, tmpId);

            int i = lo, j = mid + 1, k = 0;
            while (i <= mid && j <= hi)
            {
                if (MortonCodes[i] <= MortonCodes[j])
                { tmpCode[k] = MortonCodes[i]; tmpId[k] = SortedPrimIds[i]; i++; }
                else
                { tmpCode[k] = MortonCodes[j]; tmpId[k] = SortedPrimIds[j]; j++; }
                k++;
            }
            while (i <= mid) { tmpCode[k] = MortonCodes[i]; tmpId[k] = SortedPrimIds[i]; i++; k++; }
            while (j <= hi) { tmpCode[k] = MortonCodes[j]; tmpId[k] = SortedPrimIds[j]; j++; k++; }
            for (int x = 0; x < k; x++)
            {
                MortonCodes[lo + x] = tmpCode[x];
                SortedPrimIds[lo + x] = tmpId[x];
            }
        }

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
                int idx = stack[stack.Length - 1];
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
                int idx = stack[stack.Length - 1];
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
            if (MortonCodes.IsCreated) MortonCodes.Dispose();
            if (SortedPrimIds.IsCreated) SortedPrimIds.Dispose();
            if (PrimMin.IsCreated) PrimMin.Dispose();
            if (PrimMax.IsCreated) PrimMax.Dispose();
            _allocated = false;
        }
    }
}

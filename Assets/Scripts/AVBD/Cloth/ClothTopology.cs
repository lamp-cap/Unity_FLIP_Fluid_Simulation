using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;

namespace AVBD.Cloth
{
    /// <summary>
    /// 布料网格拓扑 + 静止形状预计算 + 图着色。
    /// 对应参考实现中由框架层 TriMeshFEM 提供的部分:
    ///   - 邻接面/边 (numNeiFaces / getVertexIthNeiFace / getVertexIthNeiFaceOrder ...)
    ///   - EdgeInfo (eV1, eV2, eV12Next, eV21Next, fId1, fId2)
    ///   - DmInv (calculateDeformationGradient 用)
    ///   - faceRestposeArea
    ///   - verticesColoringCategories (图着色并行分组)
    ///   - edgeLaplacianQuadraticForms (弯曲能 4x4 二次型 Q)
    ///
    /// 我们只模拟布料,所以所有网格在求解时统一展平为一组全局顶点/面/边数组,
    /// 没有 iMesh 维度。多块布料在导入时各自加偏移合并到同一组数组里。
    ///
    /// 使用 SoA 布局 + Native 容器,便于后续 Job/Burst 化。
    /// 所有 NativeArray 由本类持有,Dispose() 释放。
    /// </summary>
    public class ClothTopology
    {
        // ---------- 顶点 ----------
        public int NumVertices;
        public NativeArray<float3> RestPositions;   // 静止位置(局部/世界,导入时已是世界)
        public NativeArray<float> VertexMass;        // 顶点质量
        public NativeArray<float> VertexInvMass;     // 1/mass(固定点为 0)
        public NativeArray<bool> FixedMask;          // 固定点

        // ---------- 三角面 ----------
        public int NumFaces;
        public NativeArray<int3> FaceVerts;          // 每个面三个顶点全局索引
        public NativeArray<float2x2> DmInv;          // 每个面的逆材料矩阵
        public NativeArray<float> FaceRestArea;      // 每个面静止面积
        // 跨边邻接面:FaceAdjacent[f][k] = 与 f 共享"对着局部顶点 k 的那条边"的邻接面(无则 -1)。
        //   k=0 -> 边(v1,v2)(对着 v0);k=1 -> 边(v2,v0);k=2 -> 边(v0,v1)。
        // 供自碰撞 OGC 特征法向:最近点落在三角形某条边上时,取两相邻面法向的角度加权。
        public NativeArray<int3> FaceAdjacent;

        // ---------- 边 ----------
        public int NumEdges;
        public NativeArray<EdgeInfo> Edges;          // 每条边信息(含弯曲用对角顶点)
        public NativeArray<float4x4> EdgeQ;          // 弯曲能 4x4 二次型 Q(boundary 边为 0)

        // ---------- 顶点 -> 邻接面 (CSR) ----------
        // VertexFaceStart[v]..VertexFaceStart[v+1] 区间内是 (faceId, faceVertexOrder) 对
        public NativeArray<int> VertexFaceStart;
        public NativeArray<int2> VertexFaceList;     // (faceId, orderInFace 0/1/2)

        // ---------- 顶点 -> 邻接边 (CSR) ----------
        // 用于弯曲能:每个顶点参与的"相关弯曲边" + 该顶点在 Q 里的 order(0..3)
        public NativeArray<int> VertexEdgeStart;
        public NativeArray<int2> VertexEdgeList;     // (edgeId, edgeVertexOrder 0..3)

        // ---------- 顶点 -> 一环邻居顶点 (CSR) ----------
        // 通过共享面定义的拓扑一环邻居。用于自碰撞检测排除拓扑近邻,
        // 避免静止平铺态下顶点与"相邻图元"被当成接触(假接触 -> rest 抖动/膨胀)。
        public NativeArray<int> VertexNeighborStart;
        public NativeArray<int> VertexNeighborList;  // 邻居顶点全局索引,按顶点分段

        // ---------- 图着色并行分组 ----------
        // 同一组内顶点互不相邻,可并行。ColorStart[c]..ColorStart[c+1] 为该组顶点索引。
        public int NumColors;
        public NativeArray<int> ColorStart;
        public NativeArray<int> ColorVertices;       // 顶点全局索引,按颜色分段

        /// <summary>
        /// 边信息。对应参考的 EdgeInfo:
        ///   eV1, eV2  : 边两端点
        ///   eV12Next  : fId1 中除 eV1,eV2 外的第三点(对角点 1)
        ///   eV21Next  : fId2 中除 eV1,eV2 外的第三点(对角点 2),boundary 边为 -1
        ///   fId1, fId2: 相邻两面(boundary 边 fId2 = -1)
        /// 弯曲能 Xs 行序: [eV1, eV2, eV12Next, eV21Next]
        /// </summary>
        public struct EdgeInfo
        {
            public int eV1, eV2;
            public int eV12Next, eV21Next;
            public int fId1, fId2;
        }

        private bool _allocated;

        // ===============================================================
        // 从多块 Unity Mesh(世界空间顶点)构建统一拓扑
        // meshes: 每块布料的 (worldVertices, triangles)
        // fixedPredicate: 可选,返回该全局顶点是否固定
        // density: 面密度(kg/m^2),顶点质量按邻接面面积 1/3 累加
        // ===============================================================
        public void Build(
            List<float3[]> meshVertices,
            List<int[]> meshTriangles,
            float density,
            System.Func<int, float3, bool> fixedPredicate = null)
        {
            Dispose();

            // ---- 1. 合并顶点/面到全局数组(加偏移) ----
            var allVerts = new List<float3>();
            var allFaces = new List<int3>();
            foreach (var v in meshVertices)
            {
                foreach (var p in v) allVerts.Add(p);
            }
            int vOffset = 0;
            for (int m = 0; m < meshVertices.Count; m++)
            {
                int[] tris = meshTriangles[m];
                for (int t = 0; t < tris.Length; t += 3)
                {
                    allFaces.Add(new int3(tris[t] + vOffset, tris[t + 1] + vOffset, tris[t + 2] + vOffset));
                }
                vOffset += meshVertices[m].Length;
            }

            NumVertices = allVerts.Count;
            NumFaces = allFaces.Count;

            RestPositions = new NativeArray<float3>(NumVertices, Allocator.Persistent);
            VertexMass = new NativeArray<float>(NumVertices, Allocator.Persistent);
            VertexInvMass = new NativeArray<float>(NumVertices, Allocator.Persistent);
            FixedMask = new NativeArray<bool>(NumVertices, Allocator.Persistent);
            FaceVerts = new NativeArray<int3>(NumFaces, Allocator.Persistent);
            DmInv = new NativeArray<float2x2>(NumFaces, Allocator.Persistent);
            FaceRestArea = new NativeArray<float>(NumFaces, Allocator.Persistent);

            for (int i = 0; i < NumVertices; i++) RestPositions[i] = allVerts[i];
            for (int i = 0; i < NumFaces; i++) FaceVerts[i] = allFaces[i];

            // ---- 2. 每面 DmInv / 面积 / 质量累加 ----
            // Dm = [u1-u0, u2-u0] 用静止位置在面内的 2D 参数化。
            // 这里用 3D 静止三角形构造局部 2D 基底(等距展开)。
            var massAccum = new double[NumVertices];
            for (int f = 0; f < NumFaces; f++)
            {
                int3 fv = FaceVerts[f];
                float3 p0 = RestPositions[fv.x];
                float3 p1 = RestPositions[fv.y];
                float3 p2 = RestPositions[fv.z];

                float3 e1 = p1 - p0;
                float3 e2 = p2 - p0;
                float3 n = cross(e1, e2);
                float area = 0.5f * length(n);
                FaceRestArea[f] = area;

                // 局部 2D 基底:x 轴沿 e1,y 轴在面内垂直
                float3 ax = normalizesafe(e1);
                float3 az = normalizesafe(n);
                float3 ay = cross(az, ax);

                // Dm 列 = e1,e2 在 (ax,ay) 下的 2D 坐标
                float2 u1 = new float2(dot(e1, ax), dot(e1, ay));
                float2 u2 = new float2(dot(e2, ax), dot(e2, ay));
                float2x2 Dm = new float2x2(u1.x, u2.x,
                                           u1.y, u2.y);
                float det = Dm.c0.x * Dm.c1.y - Dm.c1.x * Dm.c0.y;
                if (abs(det) < 1e-12f)
                {
                    DmInv[f] = float2x2(1,0,0,1);
                }
                else
                {
                    float inv = 1f / det;
                    DmInv[f] = new float2x2(
                        Dm.c1.y * inv, -Dm.c1.x * inv,
                        -Dm.c0.y * inv, Dm.c0.x * inv);
                }

                double third = area * density / 3.0;
                massAccum[fv.x] += third;
                massAccum[fv.y] += third;
                massAccum[fv.z] += third;
            }

            for (int v = 0; v < NumVertices; v++)
            {
                float m = (float)massAccum[v];
                if (m < 1e-12f) m = 1e-6f;
                VertexMass[v] = m;
                bool fixedV = fixedPredicate != null && fixedPredicate(v, RestPositions[v]);
                FixedMask[v] = fixedV;
                VertexInvMass[v] = fixedV ? 0f : 1f / m;
            }

            // ---- 3. 顶点->邻接面 CSR ----
            BuildVertexFaceAdjacency();

            // ---- 4. 边 + EdgeInfo + 顶点->边 CSR ----
            BuildEdges();

            // ---- 5. 弯曲能 Q ----
            BuildBendingQ();

            // ---- 6. 图着色 ----
            BuildColoring();

            _allocated = true;
        }

        private void BuildVertexFaceAdjacency()
        {
            var lists = new List<int2>[NumVertices];
            for (int v = 0; v < NumVertices; v++) lists[v] = new List<int2>();
            for (int f = 0; f < NumFaces; f++)
            {
                int3 fv = FaceVerts[f];
                lists[fv.x].Add(new int2(f, 0));
                lists[fv.y].Add(new int2(f, 1));
                lists[fv.z].Add(new int2(f, 2));
            }
            VertexFaceStart = new NativeArray<int>(NumVertices + 1, Allocator.Persistent);
            int total = 0;
            for (int v = 0; v < NumVertices; v++) { VertexFaceStart[v] = total; total += lists[v].Count; }
            VertexFaceStart[NumVertices] = total;
            VertexFaceList = new NativeArray<int2>(total, Allocator.Persistent);
            int idx = 0;
            for (int v = 0; v < NumVertices; v++)
                foreach (var e in lists[v]) VertexFaceList[idx++] = e;
        }

        private void BuildEdges()
        {
            // key: (min,max) -> 临时边记录
            var edgeMap = new Dictionary<long, int>();
            var eV1 = new List<int>();
            var eV2 = new List<int>();
            var eOpp1 = new List<int>(); // fId1 对角点
            var eOpp2 = new List<int>(); // fId2 对角点
            var eF1 = new List<int>();
            var eF2 = new List<int>();

            long Key(int a, int b)
            {
                int lo = min(a, b), hi = max(a, b);
                return ((long)lo << 32) | (uint)hi;
            }

            void AddFaceEdge(int a, int b, int opp, int faceId)
            {
                long k = Key(a, b);
                if (!edgeMap.TryGetValue(k, out int eid))
                {
                    eid = eV1.Count;
                    edgeMap[k] = eid;
                    eV1.Add(min(a, b));
                    eV2.Add(max(a, b));
                    eOpp1.Add(opp);
                    eOpp2.Add(-1);
                    eF1.Add(faceId);
                    eF2.Add(-1);
                }
                else
                {
                    // 第二个相邻面
                    if (eF2[eid] == -1)
                    {
                        eOpp2[eid] = opp;
                        eF2[eid] = faceId;
                    }
                    // 非流形(>2 面)忽略额外面
                }
            }

            for (int f = 0; f < NumFaces; f++)
            {
                int3 fv = FaceVerts[f];
                AddFaceEdge(fv.x, fv.y, fv.z, f);
                AddFaceEdge(fv.y, fv.z, fv.x, f);
                AddFaceEdge(fv.z, fv.x, fv.y, f);
            }

            NumEdges = eV1.Count;
            Edges = new NativeArray<EdgeInfo>(NumEdges, Allocator.Persistent);
            for (int e = 0; e < NumEdges; e++)
            {
                Edges[e] = new EdgeInfo
                {
                    eV1 = eV1[e],
                    eV2 = eV2[e],
                    eV12Next = eOpp1[e],
                    eV21Next = eOpp2[e],
                    fId1 = eF1[e],
                    fId2 = eF2[e],
                };
            }

            // 顶点 -> 相关弯曲边 CSR(只含内部边,即 fId2 != -1;但 order 仍按 4 点)
            var vlists = new List<int2>[NumVertices];
            for (int v = 0; v < NumVertices; v++) vlists[v] = new List<int2>();
            for (int e = 0; e < NumEdges; e++)
            {
                var ei = Edges[e];
                if (ei.fId2 == -1) continue; // boundary 边无弯曲
                vlists[ei.eV1].Add(new int2(e, 0));
                vlists[ei.eV2].Add(new int2(e, 1));
                vlists[ei.eV12Next].Add(new int2(e, 2));
                vlists[ei.eV21Next].Add(new int2(e, 3));
            }
            VertexEdgeStart = new NativeArray<int>(NumVertices + 1, Allocator.Persistent);
            int total = 0;
            for (int v = 0; v < NumVertices; v++) { VertexEdgeStart[v] = total; total += vlists[v].Count; }
            VertexEdgeStart[NumVertices] = total;
            VertexEdgeList = new NativeArray<int2>(total, Allocator.Persistent);
            int idx = 0;
            for (int v = 0; v < NumVertices; v++)
                foreach (var x in vlists[v]) VertexEdgeList[idx++] = x;

            // 面跨边邻接:FaceAdjacent[f] = (跨边(v0,v1)的邻面, 跨(v1,v2), 跨(v2,v0))
            // 与 AddFaceEdge 的加边顺序一致。边界边为 -1。供 VF 特征法向取边特征的两相邻面。
            FaceAdjacent = new NativeArray<int3>(NumFaces, Allocator.Persistent);
            for (int f = 0; f < NumFaces; f++)
            {
                int3 fv = FaceVerts[f];
                FaceAdjacent[f] = new int3(
                    OtherFace(edgeMap, eF1, eF2, fv.x, fv.y, f),
                    OtherFace(edgeMap, eF1, eF2, fv.y, fv.z, f),
                    OtherFace(edgeMap, eF1, eF2, fv.z, fv.x, f));
            }

            int OtherFace(Dictionary<long, int> map, List<int> f1, List<int> f2, int a, int b, int self)
            {
                int eid = map[Key(a, b)];
                int o = f1[eid] == self ? f2[eid] : f1[eid];
                return o;
            }
        }

        // 弯曲能二次型 Q (4x4),基于静止形状的 cotangent 公式。
        // 采用 Bergou et al. / 经典 quadratic bending(线性化薄板弯曲)。
        // Xs 行序: [eV1, eV2, eV12Next(=x3), eV21Next(=x4)]
        private void BuildBendingQ()
        {
            EdgeQ = new NativeArray<float4x4>(NumEdges, Allocator.Persistent);
            for (int e = 0; e < NumEdges; e++)
            {
                var ei = Edges[e];
                if (ei.fId2 == -1) { EdgeQ[e] = float4x4(0); continue; }

                float3 x0 = RestPositions[ei.eV1];
                float3 x1 = RestPositions[ei.eV2];
                float3 x2 = RestPositions[ei.eV12Next];
                float3 x3 = RestPositions[ei.eV21Next];

                // 两个三角形: (x0,x1,x2) 和 (x0,x1,x3) 共享边 x0-x1
                // quadratic bending stencil K = (3/A) * [c0;c1;c2;c3]*[...]^T 的对称构造
                // 这里用 Wardetzky 2007 "Discrete Quadratic Curvature Energies" 的 K 向量。
                float3 e0 = x1 - x0; // 共享边
                float3 e1 = x2 - x0;
                float3 e2 = x3 - x0;
                float3 e3 = x2 - x1;
                float3 e4 = x3 - x1;

                float c01 = Cot(e0, e1);
                float c02 = Cot(e0, e2);
                float c03 = Cot(-e0, e3);
                float c04 = Cot(-e0, e4);

                // K 系数(对应顶点 x0,x1,x2,x3),来自 Wardetzky 公式
                float4 K = new float4(
                    c03 + c04,
                    c01 + c02,
                    -c01 - c03,
                    -c02 - c04);

                float A0 = 0.5f * length(cross(e1, e0));   // 三角(x0,x1,x2) 面积
                float A1 = 0.5f * length(cross(e0, e2));   // 三角(x0,x1,x3) 面积
                float areaSum = A0 + A1;
                float scale = (areaSum > 1e-12f) ? (3.0f / areaSum) : 0f;

                // Q = scale * K * K^T  (4x4 对称半正定)
                float4x4 Q = float4x4(0);
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        Q[j][i] = scale * K[i] * K[j]; // 列主序: Q[col][row]
                EdgeQ[e] = Q;
            }
        }

        private static float Cot(float3 a, float3 b)
        {
            float3 c = cross(a, b);
            float s = length(c);
            if (s < 1e-12f) s = 1e-12f;
            return dot(a, b) / s;
        }

        // 贪心图着色:相邻(共享面)顶点不同色
        private void BuildColoring()
        {
            // 构建顶点邻接(通过面)
            var adj = new HashSet<int>[NumVertices];
            for (int v = 0; v < NumVertices; v++) adj[v] = new HashSet<int>();
            for (int f = 0; f < NumFaces; f++)
            {
                int3 fv = FaceVerts[f];
                adj[fv.x].Add(fv.y); adj[fv.x].Add(fv.z);
                adj[fv.y].Add(fv.x); adj[fv.y].Add(fv.z);
                adj[fv.z].Add(fv.x); adj[fv.z].Add(fv.y);
            }

            // 持久化一环邻居 CSR(供自碰撞检测排除拓扑近邻)。
            BuildVertexNeighborCSR(adj);

            var color = new int[NumVertices];
            for (int v = 0; v < NumVertices; v++) color[v] = -1;
            int maxColor = -1;
            var used = new HashSet<int>();
            for (int v = 0; v < NumVertices; v++)
            {
                used.Clear();
                foreach (int nb in adj[v])
                    if (color[nb] >= 0) used.Add(color[nb]);
                int c = 0;
                while (used.Contains(c)) c++;
                color[v] = c;
                if (c > maxColor) maxColor = c;
            }

            NumColors = maxColor + 1;
            var buckets = new List<int>[NumColors];
            for (int c = 0; c < NumColors; c++) buckets[c] = new List<int>();
            for (int v = 0; v < NumVertices; v++) buckets[color[v]].Add(v);

            ColorStart = new NativeArray<int>(NumColors + 1, Allocator.Persistent);
            int total = 0;
            for (int c = 0; c < NumColors; c++) { ColorStart[c] = total; total += buckets[c].Count; }
            ColorStart[NumColors] = total;
            ColorVertices = new NativeArray<int>(total, Allocator.Persistent);
            int idx = 0;
            for (int c = 0; c < NumColors; c++)
                foreach (int v in buckets[c]) ColorVertices[idx++] = v;
        }

        // 把一环邻接(共享面定义)展平为 CSR。供自碰撞检测排除拓扑近邻。
        // 每个顶点的邻居列表内含自身(查询时 v 也要被排除),且已排序便于二分。
        private void BuildVertexNeighborCSR(HashSet<int>[] adj)
        {
            VertexNeighborStart = new NativeArray<int>(NumVertices + 1, Allocator.Persistent);
            int total = 0;
            for (int v = 0; v < NumVertices; v++)
            {
                VertexNeighborStart[v] = total;
                total += adj[v].Count + 1; // +1 容纳自身
            }
            VertexNeighborStart[NumVertices] = total;

            VertexNeighborList = new NativeArray<int>(max(total, 1), Allocator.Persistent);
            var tmp = new List<int>();
            int idx = 0;
            for (int v = 0; v < NumVertices; v++)
            {
                tmp.Clear();
                tmp.Add(v);
                foreach (int nb in adj[v]) tmp.Add(nb);
                tmp.Sort(); // 升序,供 IsNeighbor 二分
                for (int i = 0; i < tmp.Count; i++) VertexNeighborList[idx++] = tmp[i];
            }
        }

        // v 是否为 q 的一环邻居(或 v == q)。
        // 一环邻居通常只有 6-8 个,线性扫描比二分更快(无分支预测错 + 无除法)。
        public static bool IsNeighbor(
            NativeArray<int> neighborStart, NativeArray<int> neighborList, int q, int v)
        {
            for (int i = neighborStart[q], end = neighborStart[q + 1]; i < end; i++)
            {
                if (neighborList[i] == v) return true;
            }
            return false;
        }

        public void Dispose()
        {
            if (!_allocated && !RestPositions.IsCreated) return;
            if (RestPositions.IsCreated) RestPositions.Dispose();
            if (VertexMass.IsCreated) VertexMass.Dispose();
            if (VertexInvMass.IsCreated) VertexInvMass.Dispose();
            if (FixedMask.IsCreated) FixedMask.Dispose();
            if (FaceVerts.IsCreated) FaceVerts.Dispose();
            if (DmInv.IsCreated) DmInv.Dispose();
            if (FaceRestArea.IsCreated) FaceRestArea.Dispose();
            if (FaceAdjacent.IsCreated) FaceAdjacent.Dispose();
            if (Edges.IsCreated) Edges.Dispose();
            if (EdgeQ.IsCreated) EdgeQ.Dispose();
            if (VertexFaceStart.IsCreated) VertexFaceStart.Dispose();
            if (VertexFaceList.IsCreated) VertexFaceList.Dispose();
            if (VertexEdgeStart.IsCreated) VertexEdgeStart.Dispose();
            if (VertexEdgeList.IsCreated) VertexEdgeList.Dispose();
            if (VertexNeighborStart.IsCreated) VertexNeighborStart.Dispose();
            if (VertexNeighborList.IsCreated) VertexNeighborList.Dispose();
            if (ColorStart.IsCreated) ColorStart.Dispose();
            if (ColorVertices.IsCreated) ColorVertices.Dispose();
            _allocated = false;
        }
    }
}

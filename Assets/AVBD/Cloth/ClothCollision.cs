using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD.Cloth
{
    /// <summary>
    /// 布料自碰撞 / 布料间碰撞检测 + 接触力/Hessian + 保守边界(安全距离)。
    /// 对应参考实现:
    ///   VBDClothPhysics.cpp ::
    ///     applyCollisionDetection           -> DetectAndUpdateBounds
    ///     accumulateVFContactForceAndHessian -> AccumulateVFContact
    ///     accumulateEEContactForceAndHessian -> AccumulateEEContact
    ///     computeContactRepulsiveForce       -> RepulsiveForce
    ///     computeFriction                    -> Friction
    ///     applyConservativeBoundTruncation   -> ApplyConservativeBoundTruncation
    ///
    /// VF:为每个顶点用三角形 BVH 找邻近三角面,记录最近点/法向/重心。
    /// EE:为每条边用边 BVH 找邻近边,记录两线段最近点。
    /// 我们只模拟布料,所有图元同属一组全局数组(无 iMesh)。
    ///
    /// 注意:VBD 是逐顶点局部求解,接触项需要"该顶点对接触的贡献"。
    /// 这里采用与参考一致的做法:每帧检测一次,把接触结果缓存,迭代中复用
    /// (VBDStepWithExistingCollisions),并在顶点位移超出保守边界时触发重检测。
    /// </summary>
    public class ClothCollision
    {
        // ---- VF 接触点:顶点 v 与某三角面 ----
        public struct VFContact
        {
            public int v;            // 顶点(全局)
            public int face;         // 三角面(全局)
            public float3 bary;      // 在三角形上的重心坐标
            public float3 normal;    // 由接触点指向顶点
            public float dist;       // 距离
        }

        // ---- EE 接触点:边 e1 与边 e2 ----
        public struct EEContact
        {
            public int e1, e2;
            public float mu1, mu2;   // 各自线段参数
            public float3 c1, c2;    // 最近点
            public float3 normal;    // c1 - c2 归一
            public float dist;
        }

        // 顶点 -> 接触结构体(直接存值,不存索引)。
        // 这样消除了"先 Add 列表拿 index 再写反查表"的串行依赖,可用 ParallelWriter 并行写。
        // 同一接触会在其相关的各顶点 key 下各存一份拷贝(VF: v 与三角形 3 顶点;EE: 两边 4 端点)。
        // 消费方(VBDStep)按顶点 key 取出接触并自算该顶点的 order。
        public NativeParallelMultiHashMap<int, VFContact> VertexToVF;
        public NativeParallelMultiHashMap<int, EEContact> VertexToEE;

        private ClothLBVH _faceBVH;
        private ClothLBVH _edgeBVH;
        private bool _allocated;

        // 计数 pass 用:每个图元产生的接触数(主线程求和后据此扩容,避免 ParallelWriter 溢出)
        private NativeArray<int> _vfCounts;   // 长度 = numVertices
        private NativeArray<int> _eeCounts;   // 长度 = numEdges

        // 是否用 Burst 并行做检测(可关闭走标量路径调试)
        public bool UseBurst = true;

        public void Allocate(int numVertices, int numFaces, int numEdges)
        {
            Dispose();
            _faceBVH = new ClothLBVH();
            _faceBVH.Allocate(numFaces);
            _edgeBVH = new ClothLBVH();
            _edgeBVH.Allocate(numEdges);

            // 初始容量给个起步值;真正容量由每次检测前的计数 pass 动态决定(EnsureCapacity)。
            VertexToVF = new NativeParallelMultiHashMap<int, VFContact>(max(numVertices * 8, 16), Allocator.Persistent);
            VertexToEE = new NativeParallelMultiHashMap<int, EEContact>(max(numEdges * 8, 16), Allocator.Persistent);

            _vfCounts = new NativeArray<int>(max(numVertices, 1), Allocator.Persistent);
            _eeCounts = new NativeArray<int>(max(numEdges, 1), Allocator.Persistent);
            _allocated = true;
        }

        public ClothLBVH FaceBVH => _faceBVH;
        public ClothLBVH EdgeBVH => _edgeBVH;

        // ---- 用当前顶点位置刷新两棵 BVH 的图元 AABB ----
        // 纯并行循环:每个图元只写自己的 PrimMin/PrimMax 槽位,无竞争。
        // Burst 路径:面 / 边各一个 IJobParallelFor,并发跑后 Complete。
        public void UpdatePrimitiveBounds(ClothTopology topo, NativeArray<float3> pos, float margin)
        {
            if (UseBurst)
            {
                var faceJob = new FaceBoundsJob
                {
                    Pos = pos,
                    FaceVerts = topo.FaceVerts,
                    Margin = margin,
                    PrimMin = _faceBVH.PrimMin,
                    PrimMax = _faceBVH.PrimMax,
                };
                var faceHandle = faceJob.Schedule(topo.NumFaces, 64);

                var edgeJob = new EdgeBoundsJob
                {
                    Pos = pos,
                    Edges = topo.Edges,
                    Margin = margin,
                    PrimMin = _edgeBVH.PrimMin,
                    PrimMax = _edgeBVH.PrimMax,
                };
                var edgeHandle = edgeJob.Schedule(topo.NumEdges, 64);

                JobHandle.CombineDependencies(faceHandle, edgeHandle).Complete();
                return;
            }

            // 标量路径
            float3 m = new float3(margin);
            for (int f = 0; f < topo.NumFaces; f++)
            {
                int3 fv = topo.FaceVerts[f];
                float3 a = pos[fv.x], b = pos[fv.y], c = pos[fv.z];
                _faceBVH.PrimMin[f] = min(min(a, b), c) - m;
                _faceBVH.PrimMax[f] = max(max(a, b), c) + m;
            }
            for (int e = 0; e < topo.NumEdges; e++)
            {
                var ei = topo.Edges[e];
                float3 a = pos[ei.eV1], b = pos[ei.eV2];
                _edgeBVH.PrimMin[e] = min(a, b) - m;
                _edgeBVH.PrimMax[e] = max(a, b) + m;
            }
        }

        /// <summary>每帧开始:完全重建两棵 BVH。</summary>
        public void RebuildBVH(ClothTopology topo, NativeArray<float3> pos, float margin)
        {
            UpdatePrimitiveBounds(topo, pos, margin);
            _faceBVH.Build();
            _edgeBVH.Build();
        }

        /// <summary>迭代中:只 refit bounds。</summary>
        public void RefitBVH(ClothTopology topo, NativeArray<float3> pos, float margin)
        {
            UpdatePrimitiveBounds(topo, pos, margin);
            _faceBVH.Refit();
            _edgeBVH.Refit();
        }

        /// <summary>
        /// 碰撞检测 + 重算保守边界(安全距离)。对应 applyCollisionDetection。
        /// queryRadius: 检测半径(maxQueryDis);thickness: 布料厚度;
        /// relax: conservativeStepRelaxation。
        /// </summary>
        public void DetectAndUpdateBounds(
            ClothTopology topo, ClothState state,
            float queryRadius, float thickness, float relax,
            NativeArray<AnalyticCollider> colliders, int colliderCount)
        {
            var pos = state.Positions;
            // BVH 重建仍在主线程(递归 + 每帧/每次重检测一次,非热点)
            RebuildBVH(topo, pos, queryRadius);

            VertexToVF.Clear();
            VertexToEE.Clear();

            if (UseBurst)
                DetectBurst(topo, state, queryRadius, thickness, relax, colliders, colliderCount);
            else
                DetectScalar(topo, state, queryRadius, thickness, relax, colliders, colliderCount);
        }

        // ---- Burst 路径:先数后填,避免 ParallelWriter 容量溢出("HashMap is full")----
        // 1) 计数 pass:并行统计每个图元命中的接触数(与检测同样的 BVH 遍历,只 ++ 计数)。
        // 2) 主线程求和 -> EnsureCapacity 扩容到足够大。
        // 3) 填充 pass:并行写接触结构体到 ParallelWriter。
        // ParallelWriter 不会自动扩容,所以必须在写之前保证容量足够。
        void DetectBurst(ClothTopology topo, ClothState state,
            float queryRadius, float thickness, float relax,
            NativeArray<AnalyticCollider> colliders, int colliderCount)
        {
            // ---- 1) 计数 pass ----
            var vfCountJob = new VFCountJob
            {
                Pos = state.Positions,
                FaceVerts = topo.FaceVerts,
                FaceNodes = _faceBVH.Nodes,
                FaceNumPrim = _faceBVH.NumPrimitives,
                QueryRadius = queryRadius,
                NeighborStart = topo.VertexNeighborStart,
                NeighborList = topo.VertexNeighborList,
                Counts = _vfCounts,
            };
            var vfCountHandle = vfCountJob.Schedule(topo.NumVertices, 32);

            var eeCountJob = new EECountJob
            {
                Pos = state.Positions,
                Edges = topo.Edges,
                EdgeNodes = _edgeBVH.Nodes,
                EdgeNumPrim = _edgeBVH.NumPrimitives,
                QueryRadius = queryRadius,
                NeighborStart = topo.VertexNeighborStart,
                NeighborList = topo.VertexNeighborList,
                Counts = _eeCounts,
            };
            var eeCountHandle = eeCountJob.Schedule(topo.NumEdges, 32);
            JobHandle.CombineDependencies(vfCountHandle, eeCountHandle).Complete();

            // ---- 2) 求和 + 扩容 ----
            // 每个命中接触写 4 个 key(VF: v + 三角形 3 顶点;EE: 两边 4 端点)。
            int vfTotal = 0;
            for (int i = 0; i < topo.NumVertices; i++) vfTotal += _vfCounts[i];
            int eeTotal = 0;
            for (int i = 0; i < topo.NumEdges; i++) eeTotal += _eeCounts[i];

            int vfNeeded = vfTotal * 4 + 16;
            int eeNeeded = eeTotal * 4 + 16;
            if (VertexToVF.Capacity < vfNeeded) VertexToVF.Capacity = vfNeeded;
            if (VertexToEE.Capacity < eeNeeded) VertexToEE.Capacity = eeNeeded;

            // ---- 3) 填充 pass ----
            var vfJob = new VFDetectJob
            {
                Pos = state.Positions,
                FaceVerts = topo.FaceVerts,
                FaceAdjacent = topo.FaceAdjacent,
                VertexFaceStart = topo.VertexFaceStart,
                VertexFaceList = topo.VertexFaceList,
                FaceNodes = _faceBVH.Nodes,
                FaceNumPrim = _faceBVH.NumPrimitives,
                QueryRadius = queryRadius,
                NeighborStart = topo.VertexNeighborStart,
                NeighborList = topo.VertexNeighborList,
                Out = VertexToVF.AsParallelWriter(),
            };
            var vfHandle = vfJob.Schedule(topo.NumVertices, 32);

            var eeJob = new EEDetectJob
            {
                Pos = state.Positions,
                Edges = topo.Edges,
                EdgeNodes = _edgeBVH.Nodes,
                EdgeNumPrim = _edgeBVH.NumPrimitives,
                QueryRadius = queryRadius,
                NeighborStart = topo.VertexNeighborStart,
                NeighborList = topo.VertexNeighborList,
                Out = VertexToEE.AsParallelWriter(),
            };
            var eeHandle = eeJob.Schedule(topo.NumEdges, 32);

            JobHandle.CombineDependencies(vfHandle, eeHandle).Complete();

            // 保守边界重算:依赖两张反查表(已 Complete),并行读。
            // 安全距离同时纳入"到解析碰撞体的距离",使截断也能保护布料-碰撞体穿透
            // (Offset Geometric Contact:每次检测算安全距离,迭代中位移超出即截断 + 重检测)。
            var boundsJob = new ConservativeBoundsJob
            {
                Pos = state.Positions,
                VertexToVF = VertexToVF,
                VertexToEE = VertexToEE,
                QueryRadius = queryRadius,
                Thickness = thickness,
                Relax = relax,
                Colliders = colliders,
                ColliderCount = colliderCount,
                ConservativeBounds = state.ConservativeBounds,
                PositionsAtPrevCD = state.PositionsAtPrevCD,
            };
            boundsJob.Schedule(topo.NumVertices, 64).Complete();
        }

        // ---- 标量路径(调试用):逐顶点/逐边串行,行为与 Burst 路径一致 ----
        void DetectScalar(ClothTopology topo, ClothState state,
            float queryRadius, float thickness, float relax,
            NativeArray<AnalyticCollider> colliders, int colliderCount)
        {
            var pos = state.Positions;
            var nbStart = topo.VertexNeighborStart;
            var nbList = topo.VertexNeighborList;
            var stack = new NativeList<int>(64, Allocator.Temp);
            var hits = new NativeList<int>(64, Allocator.Temp);

            for (int v = 0; v < topo.NumVertices; v++)
            {
                float3 p = pos[v];
                float3 q = new float3(queryRadius);
                ClothLBVH.QueryAABBStatic(_faceBVH.Nodes, _faceBVH.NumPrimitives, p - q, p + q, ref stack, ref hits);
                for (int h = 0; h < hits.Length; h++)
                {
                    int f = hits[h];
                    int3 fv = topo.FaceVerts[f];
                    // 排除 v 自身及其一环拓扑邻居所在的面(否则静止平铺态产生假接触)
                    if (ClothTopology.IsNeighbor(nbStart, nbList, v, fv.x) ||
                        ClothTopology.IsNeighbor(nbStart, nbList, v, fv.y) ||
                        ClothTopology.IsNeighbor(nbStart, nbList, v, fv.z)) continue;
                    ClosestPointOnTriangle(p, pos[fv.x], pos[fv.y], pos[fv.z], out float3 closest, out float3 bary);
                    float3 diff = p - closest;
                    float d = length(diff);
                    if (d < queryRadius)
                    {
                        // OGC 特征法向(面/边/顶点角度加权),已按 p-closest 定号
                        float3 n = VFFeatureNormal(p, closest, f, bary,
                            topo.FaceVerts, topo.FaceAdjacent,
                            topo.VertexFaceStart, topo.VertexFaceList, pos);
                        var c = new VFContact { v = v, face = f, bary = bary, normal = n, dist = d };
                        VertexToVF.Add(v, c);
                        VertexToVF.Add(fv.x, c);
                        VertexToVF.Add(fv.y, c);
                        VertexToVF.Add(fv.z, c);
                    }
                }
            }

            for (int e1 = 0; e1 < topo.NumEdges; e1++)
            {
                var ei1 = topo.Edges[e1];
                float3 a1 = pos[ei1.eV1], b1 = pos[ei1.eV2];
                float3 lo = min(a1, b1) - new float3(queryRadius);
                float3 hi = max(a1, b1) + new float3(queryRadius);
                ClothLBVH.QueryAABBStatic(_edgeBVH.Nodes, _edgeBVH.NumPrimitives, lo, hi, ref stack, ref hits);
                for (int h = 0; h < hits.Length; h++)
                {
                    int e2 = hits[h];
                    if (e2 <= e1) continue;
                    var ei2 = topo.Edges[e2];
                    // 排除任一端点互为一环邻居(含共享端点)的边对
                    if (ClothTopology.IsNeighbor(nbStart, nbList, ei1.eV1, ei2.eV1) ||
                        ClothTopology.IsNeighbor(nbStart, nbList, ei1.eV1, ei2.eV2) ||
                        ClothTopology.IsNeighbor(nbStart, nbList, ei1.eV2, ei2.eV1) ||
                        ClothTopology.IsNeighbor(nbStart, nbList, ei1.eV2, ei2.eV2)) continue;

                    float3 a2 = pos[ei2.eV1], b2 = pos[ei2.eV2];
                    ClosestPointsBetweenSegments(a1, b1, a2, b2, out float3 cc1, out float3 cc2, out float mu1, out float mu2);
                    float3 diff = cc1 - cc2;
                    float d = length(diff);
                    if (d < queryRadius)
                    {
                        // OGC EE 法向:两边方向叉乘(良态时),按 diff 定号;近平行退化回退 diff/d
                        float3 n = EEFeatureNormal(b1 - a1, b2 - a2, diff, d);
                        var c = new EEContact { e1 = e1, e2 = e2, mu1 = mu1, mu2 = mu2, c1 = cc1, c2 = cc2, normal = n, dist = d };
                        VertexToEE.Add(ei1.eV1, c);
                        VertexToEE.Add(ei1.eV2, c);
                        VertexToEE.Add(ei2.eV1, c);
                        VertexToEE.Add(ei2.eV2, c);
                    }
                }
            }

            for (int v = 0; v < topo.NumVertices; v++)
            {
                float dMin = queryRadius;
                if (VertexToVF.TryGetFirstValue(v, out VFContact cvf, out var it))
                {
                    do { dMin = min(dMin, cvf.dist); }
                    while (VertexToVF.TryGetNextValue(out cvf, ref it));
                }
                if (VertexToEE.TryGetFirstValue(v, out EEContact cee, out var it2))
                {
                    do { dMin = min(dMin, cee.dist); }
                    while (VertexToEE.TryGetNextValue(out cee, ref it2));
                }
                // 解析碰撞体纳入安全距离:带符号最近距离(穿透为负,会使 gap 变负)。
                for (int i = 0; i < colliderCount; i++)
                {
                    colliders[i].Query(pos[v], out _, out _, out float cd);
                    dMin = min(dMin, cd);
                }
                float safe = dMin - thickness;
                // safe<=0:已进入接触偏移层(或已穿透)。写哨兵 -1 表示本次迭代豁免截断,
                // 让排斥力把顶点推出,而不是被 bound=0 冻结。
                state.ConservativeBounds[v] = safe <= 0f ? -1f : safe * relax;
                state.PositionsAtPrevCD[v] = pos[v];
            }

            stack.Dispose();
            hits.Dispose();
        }

        // ===============================================================
        // 接触力/Hessian(VBDStep 内对单个顶点调用)
        // 排斥力:线性 penalty,对应 computeContactRepulsiveForce case 0。
        // ===============================================================

        /// <summary>computeContactRepulsiveForce(case 0: 线性 penalty)。</summary>
        public static void RepulsiveForce(float dis, float thickness, float contactRadius, float k,
            out float dEdD, out float d2EdDdD)
        {
            float disAdj = dis - thickness;
            float penetration = contactRadius - disAdj;
            // case 0: 二次能量 E = 0.5 k pen^2 -> dE/dD = -k pen, d2E = k
            dEdD = -k * penetration;
            d2EdDdD = k;
        }

        /// <summary>computeFriction(IPC 平滑摩擦)。T:3x2 切空间基底。</summary>
        public static void Friction(float mu, float lambda, float3x2 T, float2 u, float epsU,
            out float3 force, out float3x3 hessian)
        {
            float uNorm = length(u);
            if (uNorm > 0f)
            {
                float f1_over_x;
                if (uNorm > epsU) f1_over_x = 1f / uNorm;
                else f1_over_x = (-uNorm / epsU + 2f) / epsU;

                // force = -mu*lambda * T * f1_over_x * u
                float2 fu = f1_over_x * u;
                force = -mu * lambda * (T.c0 * fu.x + T.c1 * fu.y);

                // hessian = mu*lambda * T * (f1_over_x * I) * T^T
                float s = mu * lambda * f1_over_x;
                // T * I2 * T^T = T.c0 ⊗ T.c0 + T.c1 ⊗ T.c1
                hessian = s * (Outer(T.c0, T.c0) + Outer(T.c1, T.c1));
            }
            else
            {
                force = float3(0);
                hessian = float3x3(0);
            }
        }

        static float3x3 Outer(float3 a, float3 b)
        {
            return new float3x3(
                a.x * b.x, a.x * b.y, a.x * b.z,
                a.y * b.x, a.y * b.y, a.y * b.z,
                a.z * b.x, a.z * b.y, a.z * b.z);
        }

        /// <summary>
        /// 累加一个 VF 接触对顶点 v 的力/Hessian。
        /// contactVertexOrder:v 在该接触里的角色 —— 3 表示 v 是 V 侧顶点,
        /// 0/1/2 表示 v 是 F 侧三角形的第几个顶点。
        /// 对应 accumulateVFContactForceAndHessian。
        /// </summary>
        public void AccumulateVFContact(
            in VFContact c, int contactVertexOrder,
            ClothTopology topo, ClothState state,
            float thickness, float contactRadius, float k,
            bool applyFriction, float frictionMu, float frictionEpsV, float dt,
            ref float3 force, ref float3x3 hessian)
        {
            float dis = c.dist;
            if (dis >= contactRadius + thickness) return;

            // b 权重:V 侧为 +1,F 侧三个顶点为 -bary
            float4 bs = new float4(-c.bary.x, -c.bary.y, -c.bary.z, 1f);
            float b = bs[contactVertexOrder];

            RepulsiveForce(dis, thickness, contactRadius, k, out float dEdD, out float d2EdDdD);
            float lambda = -dEdD;
            float3 n = c.normal;

            force += b * lambda * n;
            hessian += d2EdDdD * b * b * Outer(n, n);

            if (applyFriction)
            {
                // 切向相对位移(简化:用 V 侧顶点相对位移在切空间投影)
                int vSide = c.v;
                float3 dx = state.Positions[vSide] - state.PositionsPrev[vSide];
                int3 fv = topo.FaceVerts[c.face];
                // 切空间基底
                float3 e0 = normalizesafe(state.Positions[fv.y] - state.Positions[fv.x]);
                float3 t1 = normalizesafe(cross(e0, n));
                float3x2 T = new float3x2(e0, t1);
                float2 u = new float2(dot(T.c0, dx), dot(T.c1, dx));
                Friction(frictionMu, lambda, T, u, frictionEpsV * dt, out float3 ff, out float3x3 fh);
                force += b * ff;
                hessian += b * b * fh;
            }
        }

        /// <summary>
        /// 累加一个 EE 接触对顶点的力/Hessian。
        /// contactVertexOrder:顶点在 [e1.v1, e1.v2, e2.v1, e2.v2] 中的序号(0..3)。
        /// 对应 accumulateEEContactForceAndHessian。
        /// </summary>
        public void AccumulateEEContact(
            in EEContact c, int contactVertexOrder,
            ClothTopology topo, ClothState state,
            float thickness, float contactRadius, float k,
            bool applyFriction, float frictionMu, float frictionEpsV, float dt,
            ref float3 force, ref float3x3 hessian)
        {
            float dis = c.dist;
            if (dis >= contactRadius + thickness) return;

            float3 n = c.normal;
            // b: [1-mu1, mu1, -1+mu2, -mu2]
            float4 bs = new float4(1f - c.mu1, c.mu1, -1f + c.mu2, -c.mu2);
            float b = bs[contactVertexOrder];

            RepulsiveForce(dis, thickness, contactRadius, k, out float dEdD, out float d2EdDdD);
            float lambda = -dEdD;

            force += b * lambda * n;
            hessian += d2EdDdD * b * b * Outer(n, n);

            if (applyFriction)
            {
                var ei1 = topo.Edges[c.e1];
                float3 v1 = normalizesafe(state.Positions[ei1.eV2] - state.Positions[ei1.eV1]);
                float3 t1 = normalizesafe(cross(v1, n));
                float3x2 T = new float3x2(v1, t1);
                // 相对位移(用接触点近似)
                float3 dx = (c.c1 - c.c2);
                float3 dxPrev;
                {
                    var ei2 = topo.Edges[c.e2];
                    float3 c1p = (1f - c.mu1) * state.PositionsPrev[ei1.eV1] + c.mu1 * state.PositionsPrev[ei1.eV2];
                    float3 c2p = (1f - c.mu2) * state.PositionsPrev[ei2.eV1] + c.mu2 * state.PositionsPrev[ei2.eV2];
                    dxPrev = c1p - c2p;
                }
                float3 rel = dx - dxPrev;
                float2 u = new float2(dot(T.c0, rel), dot(T.c1, rel));
                Friction(frictionMu, lambda, T, u, frictionEpsV * dt, out float3 ff, out float3x3 fh);
                force += b * ff;
                hessian += b * b * fh;
            }
        }

        /// <summary>
        /// applyConservativeBoundTruncation:若累计位移超出保守边界,截断到边界内。
        /// 返回是否需要触发下次迭代重检测。
        /// bound < 0 是哨兵:顶点已进入接触偏移层(或已穿透解析碰撞体),
        /// 本次不截断(让排斥力自由推出),但仍返回 true 强制下次重检测继续跟踪安全距离。
        /// </summary>
        public static bool ApplyConservativeBoundTruncation(
            ClothState state, int v, ref float3 newPos)
        {
            float bound = state.ConservativeBounds[v];
            if (bound < 0f) return true; // 接触/穿透层:豁免截断,但需重检测

            float3 disp = newPos - state.PositionsAtPrevCD[v];
            float dispLen = length(disp);
            if (dispLen > bound && dispLen > 1e-12f)
            {
                disp *= (bound / dispLen);
                newPos = state.PositionsAtPrevCD[v] + disp;
                return true;
            }
            return false;
        }

        // ===============================================================
        // 几何工具
        // ===============================================================
        public static void ClosestPointOnTriangle(float3 p, float3 a, float3 b, float3 c,
            out float3 closest, out float3 bary)
        {
            float3 ab = b - a, ac = c - a, ap = p - a;
            float d1 = dot(ab, ap), d2 = dot(ac, ap);
            if (d1 <= 0 && d2 <= 0) { closest = a; bary = new float3(1, 0, 0); return; }

            float3 bp = p - b;
            float d3 = dot(ab, bp), d4 = dot(ac, bp);
            if (d3 >= 0 && d4 <= d3) { closest = b; bary = new float3(0, 1, 0); return; }

            float vc = d1 * d4 - d3 * d2;
            if (vc <= 0 && d1 >= 0 && d3 <= 0)
            {
                float v0 = d1 / (d1 - d3);
                closest = a + v0 * ab; bary = new float3(1 - v0, v0, 0); return;
            }

            float3 cp = p - c;
            float d5 = dot(ab, cp), d6 = dot(ac, cp);
            if (d6 >= 0 && d5 <= d6) { closest = c; bary = new float3(0, 0, 1); return; }

            float vb = d5 * d2 - d1 * d6;
            if (vb <= 0 && d2 >= 0 && d6 <= 0)
            {
                float w0 = d2 / (d2 - d6);
                closest = a + w0 * ac; bary = new float3(1 - w0, 0, w0); return;
            }

            float va = d3 * d6 - d5 * d4;
            if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
            {
                float w0 = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                closest = b + w0 * (c - b); bary = new float3(0, 1 - w0, w0); return;
            }

            float denom = 1f / (va + vb + vc);
            float vv = vb * denom, ww = vc * denom;
            closest = a + ab * vv + ac * ww;
            bary = new float3(1 - vv - ww, vv, ww);
        }

        public static void ClosestPointsBetweenSegments(
            float3 p1, float3 q1, float3 p2, float3 q2,
            out float3 c1, out float3 c2, out float s, out float t)
        {
            float3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
            float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
            const float eps = 1e-12f;

            if (a <= eps && e <= eps) { s = 0; t = 0; c1 = p1; c2 = p2; return; }
            if (a <= eps) { s = 0; t = saturate(f / e); }
            else
            {
                float c0 = dot(d1, r);
                if (e <= eps) { t = 0; s = saturate(-c0 / a); }
                else
                {
                    float bb = dot(d1, d2);
                    float denom = a * e - bb * bb;
                    s = denom > eps ? saturate((bb * f - c0 * e) / denom) : 0f;
                    t = (bb * s + f) / e;
                    if (t < 0) { t = 0; s = saturate(-c0 / a); }
                    else if (t > 1) { t = 1; s = saturate((bb - c0) / a); }
                }
            }
            c1 = p1 + d1 * s;
            c2 = p2 + d2 * t;
        }

        // ===============================================================
        // OGC 特征法向(angle-weighted pseudonormal)
        // ---------------------------------------------------------------
        // 按最近点落在三角形的"面 / 边 / 顶点"特征上,取对应特征的法向:
        //   - 面内:面法向
        //   - 边上:该边两个相邻面法向的角度加权(= 二面角平分方向)
        //   - 顶点上:该顶点所有邻接面法向按张角加权
        // 这样接触片内的法向场分段光滑、无切向噪声(原来的 (p-closest)/d 在
        // 边/顶点特征上是辐射方向,相邻顶点法向不一致 -> 剪切力 -> 皱)。
        // 自碰撞无全局内外,最后用 sign(dot(N, p-closest)) 翻到指向查询点一侧。
        // ===============================================================

        /// <summary>面 f 的单位法向(当前位形)。</summary>
        static float3 FaceNormal(int f, NativeArray<int3> faceVerts, NativeArray<float3> pos)
        {
            int3 fv = faceVerts[f];
            return normalizesafe(cross(pos[fv.y] - pos[fv.x], pos[fv.z] - pos[fv.x]));
        }

        /// <summary>面 f 在顶点 vert 处的内角(用于角度加权)。</summary>
        static float FaceCornerAngle(int f, int vert, NativeArray<int3> faceVerts, NativeArray<float3> pos)
        {
            int3 fv = faceVerts[f];
            int o0 = vert, o1, o2;
            if (fv.x == vert) { o1 = fv.y; o2 = fv.z; }
            else if (fv.y == vert) { o1 = fv.z; o2 = fv.x; }
            else { o1 = fv.x; o2 = fv.y; }
            float3 e1 = normalizesafe(pos[o1] - pos[o0]);
            float3 e2 = normalizesafe(pos[o2] - pos[o0]);
            return acos(clamp(dot(e1, e2), -1f, 1f));
        }

        /// <summary>
        /// VF 接触的 OGC 特征法向。bary 来自 ClosestPointOnTriangle。
        /// faceAdj: 面跨边邻接(ClothTopology.FaceAdjacent);vfStart/vfList: 顶点->邻接面 CSR。
        /// 返回已按 (p-closest) 定号的单位法向。
        /// </summary>
        public static float3 VFFeatureNormal(
            float3 p, float3 closest, int face, float3 bary,
            NativeArray<int3> faceVerts, NativeArray<int3> faceAdj,
            NativeArray<int> vfStart, NativeArray<int2> vfList,
            NativeArray<float3> pos)
        {
            const float eps = 1e-4f;
            int3 fv = faceVerts[face];

            // 落在某个顶点上:两个 bary 分量 ~0
            int nearZero = (bary.x < eps ? 1 : 0) + (bary.y < eps ? 1 : 0) + (bary.z < eps ? 1 : 0);

            float3 N;
            if (nearZero >= 2)
            {
                // 顶点特征:取该顶点所有邻接面的角度加权法向
                int vert = bary.x >= eps ? fv.x : (bary.y >= eps ? fv.y : fv.z);
                N = float3(0);
                int s = vfStart[vert], e = vfStart[vert + 1];
                for (int i = s; i < e; i++)
                {
                    int nf = vfList[i].x;
                    N += FaceCornerAngle(nf, vert, faceVerts, pos) * FaceNormal(nf, faceVerts, pos);
                }
                N = normalizesafe(N);
            }
            else if (nearZero == 1)
            {
                // 边特征:零分量对应的"对角顶点",其相对的边即最近边。
                // FaceAdjacent 顺序: (x,y)->.x, (y,z)->.y, (z,x)->.z
                int adj;
                if (bary.z < eps) adj = faceAdj[face].x;      // 边 (x,y)
                else if (bary.x < eps) adj = faceAdj[face].y; // 边 (y,z)
                else adj = faceAdj[face].z;                   // 边 (z,x)

                float3 nf = FaceNormal(face, faceVerts, pos);
                if (adj >= 0)
                {
                    // 角度加权:两个面法向各按其在该边的相对张角加权。
                    // 共享边上两面权重相近,近似用面法向等权平均即可得到平分方向;
                    // 这里用面法向之和归一(等价于半角平分,数值稳定)。
                    float3 na = FaceNormal(adj, faceVerts, pos);
                    N = normalizesafe(nf + na);
                    if (lengthsq(N) < 1e-12f) N = nf; // 两面几乎反向(退化)
                }
                else N = nf; // 边界边
            }
            else
            {
                // 面内特征:面法向
                N = FaceNormal(face, faceVerts, pos);
            }

            // 按当前相对位置定号:指向查询点一侧
            float3 dir = p - closest;
            if (dot(N, dir) < 0f) N = -N;
            return N;
        }

        /// <summary>
        /// EE 接触法向。理想方向是两边方向的叉乘(垂直于两条边),
        /// 良态时比 (c1-c2)/d 更稳(端点特征/近接触时辐射方向有噪声);
        /// 两边近平行(叉乘退化)时回退到 diff/d。最后按 diff 定号(指向边 1 一侧)。
        /// d1=b1-a1, d2=b2-a2, diff=c1-c2, dlen=length(diff)。
        /// </summary>
        public static float3 EEFeatureNormal(float3 d1, float3 d2, float3 diff, float dlen)
        {
            float3 cr = cross(d1, d2);
            float crLen2 = lengthsq(cr);
            // 叉乘长度^2 相对两边长度积的比值,判断是否近平行
            float scale = lengthsq(d1) * lengthsq(d2);
            if (crLen2 > 1e-12f && crLen2 > 1e-8f * scale)
            {
                float3 N = cr * rsqrt(crLen2);
                // 按 diff 定号(diff 退化时保持叉乘方向)
                if (dlen > 1e-8f && dot(N, diff) < 0f) N = -N;
                return N;
            }
            // 近平行:回退辐射方向
            return dlen > 1e-8f ? diff / dlen : normalizesafe(cross(d1, float3(0, 1, 0)));
        }

        public void Dispose()
        {
            if (!_allocated) return;
            _faceBVH?.Dispose();
            _edgeBVH?.Dispose();
            if (VertexToVF.IsCreated) VertexToVF.Dispose();
            if (VertexToEE.IsCreated) VertexToEE.Dispose();
            if (_vfCounts.IsCreated) _vfCounts.Dispose();
            if (_eeCounts.IsCreated) _eeCounts.Dispose();
            _allocated = false;
        }
    }

    // ===================================================================
    // Burst 检测 Job:VF / EE / 保守边界
    // 设计:每个图元(顶点或边)各自把命中的接触结构体写进 ParallelWriter,
    // 同一接触在其相关顶点 key 下各写一份拷贝。无串行索引依赖。
    // BVH 遍历用调用方栈(Allocator.Temp,每个 Execute 各分配各释放)。
    // ===================================================================

    /// <summary>三角形叶子 AABB 刷新:每个面独立写自己的 PrimMin/PrimMax。对应 UpdatePrimitiveBounds 的面循环。</summary>
    [BurstCompile]
    public struct FaceBoundsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeArray<int3> FaceVerts;
        public float Margin;
        [WriteOnly] public NativeArray<float3> PrimMin;
        [WriteOnly] public NativeArray<float3> PrimMax;

        public void Execute(int f)
        {
            float3 m = new float3(Margin);
            int3 fv = FaceVerts[f];
            float3 a = Pos[fv.x], b = Pos[fv.y], c = Pos[fv.z];
            PrimMin[f] = min(min(a, b), c) - m;
            PrimMax[f] = max(max(a, b), c) + m;
        }
    }

    /// <summary>边叶子 AABB 刷新:每条边独立写自己的 PrimMin/PrimMax。对应 UpdatePrimitiveBounds 的边循环。</summary>
    [BurstCompile]
    public struct EdgeBoundsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeArray<ClothTopology.EdgeInfo> Edges;
        public float Margin;
        [WriteOnly] public NativeArray<float3> PrimMin;
        [WriteOnly] public NativeArray<float3> PrimMax;

        public void Execute(int e)
        {
            float3 m = new float3(Margin);
            var ei = Edges[e];
            float3 a = Pos[ei.eV1], b = Pos[ei.eV2];
            PrimMin[e] = min(a, b) - m;
            PrimMax[e] = max(a, b) + m;
        }
    }

    /// <summary>VF 计数:与 VFDetectJob 同样的遍历/过滤,只统计每个顶点命中的接触数。</summary>
    [BurstCompile]
    public struct VFCountJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeArray<int3> FaceVerts;
        [ReadOnly] public NativeArray<ClothLBVH.Node> FaceNodes;
        [ReadOnly] public NativeArray<int> NeighborStart;
        [ReadOnly] public NativeArray<int> NeighborList;
        public int FaceNumPrim;
        public float QueryRadius;
        [WriteOnly] public NativeArray<int> Counts;

        public void Execute(int v)
        {
            float3 p = Pos[v];
            float3 q = new float3(QueryRadius);

            var stack = new NativeList<int>(64, Allocator.Temp);
            var hits = new NativeList<int>(64, Allocator.Temp);
            ClothLBVH.QueryAABBStatic(FaceNodes, FaceNumPrim, p - q, p + q, ref stack, ref hits);

            int n = 0;
            for (int h = 0; h < hits.Length; h++)
            {
                int f = hits[h];
                int3 fv = FaceVerts[f];
                // 排除 v 自身及其一环拓扑邻居所在的面(与 VFDetectJob 过滤一致)
                if (ClothTopology.IsNeighbor(NeighborStart, NeighborList, v, fv.x) ||
                    ClothTopology.IsNeighbor(NeighborStart, NeighborList, v, fv.y) ||
                    ClothTopology.IsNeighbor(NeighborStart, NeighborList, v, fv.z)) continue;
                ClothCollision.ClosestPointOnTriangle(p, Pos[fv.x], Pos[fv.y], Pos[fv.z],
                    out float3 closest, out float3 bary);
                float d = length(p - closest);
                if (d < QueryRadius) n++;
            }
            Counts[v] = n;
            stack.Dispose();
            hits.Dispose();
        }
    }

    /// <summary>EE 计数:与 EEDetectJob 同样的遍历/过滤,只统计每条边命中的接触数。</summary>
    [BurstCompile]
    public struct EECountJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeArray<ClothTopology.EdgeInfo> Edges;
        [ReadOnly] public NativeArray<ClothLBVH.Node> EdgeNodes;
        [ReadOnly] public NativeArray<int> NeighborStart;
        [ReadOnly] public NativeArray<int> NeighborList;
        public int EdgeNumPrim;
        public float QueryRadius;
        [WriteOnly] public NativeArray<int> Counts;

        public void Execute(int e1)
        {
            var ei1 = Edges[e1];
            float3 a1 = Pos[ei1.eV1], b1 = Pos[ei1.eV2];
            float3 lo = min(a1, b1) - new float3(QueryRadius);
            float3 hi = max(a1, b1) + new float3(QueryRadius);

            var stack = new NativeList<int>(64, Allocator.Temp);
            var hits = new NativeList<int>(64, Allocator.Temp);
            ClothLBVH.QueryAABBStatic(EdgeNodes, EdgeNumPrim, lo, hi, ref stack, ref hits);

            int n = 0;
            for (int h = 0; h < hits.Length; h++)
            {
                int e2 = hits[h];
                if (e2 <= e1) continue;
                var ei2 = Edges[e2];
                // 排除任一端点互为一环邻居(含共享端点)的边对(与 EEDetectJob 过滤一致)
                if (ClothTopology.IsNeighbor(NeighborStart, NeighborList, ei1.eV1, ei2.eV1) ||
                    ClothTopology.IsNeighbor(NeighborStart, NeighborList, ei1.eV1, ei2.eV2) ||
                    ClothTopology.IsNeighbor(NeighborStart, NeighborList, ei1.eV2, ei2.eV1) ||
                    ClothTopology.IsNeighbor(NeighborStart, NeighborList, ei1.eV2, ei2.eV2)) continue;

                float3 a2 = Pos[ei2.eV1], b2 = Pos[ei2.eV2];
                ClothCollision.ClosestPointsBetweenSegments(a1, b1, a2, b2,
                    out float3 cc1, out float3 cc2, out float mu1, out float mu2);
                float d = length(cc1 - cc2);
                if (d < QueryRadius) n++;
            }
            Counts[e1] = n;
            stack.Dispose();
            hits.Dispose();
        }
    }

    /// <summary>VF 检测:每个顶点查三角面 BVH,命中写入 VertexToVF。对应原 VF 串行循环。</summary>
    [BurstCompile]
    public struct VFDetectJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeArray<int3> FaceVerts;
        [ReadOnly] public NativeArray<int3> FaceAdjacent;
        [ReadOnly] public NativeArray<int> VertexFaceStart;
        [ReadOnly] public NativeArray<int2> VertexFaceList;
        [ReadOnly] public NativeArray<ClothLBVH.Node> FaceNodes;
        [ReadOnly] public NativeArray<int> NeighborStart;
        [ReadOnly] public NativeArray<int> NeighborList;
        public int FaceNumPrim;
        public float QueryRadius;
        public NativeParallelMultiHashMap<int, ClothCollision.VFContact>.ParallelWriter Out;

        public void Execute(int v)
        {
            float3 p = Pos[v];
            float3 q = new float3(QueryRadius);

            var stack = new NativeList<int>(64, Allocator.Temp);
            var hits = new NativeList<int>(64, Allocator.Temp);
            ClothLBVH.QueryAABBStatic(FaceNodes, FaceNumPrim, p - q, p + q, ref stack, ref hits);

            for (int h = 0; h < hits.Length; h++)
            {
                int f = hits[h];
                int3 fv = FaceVerts[f];
                // 排除 v 自身及其一环拓扑邻居所在的面(否则静止平铺态产生假接触)
                if (ClothTopology.IsNeighbor(NeighborStart, NeighborList, v, fv.x) ||
                    ClothTopology.IsNeighbor(NeighborStart, NeighborList, v, fv.y) ||
                    ClothTopology.IsNeighbor(NeighborStart, NeighborList, v, fv.z)) continue;

                ClothCollision.ClosestPointOnTriangle(p, Pos[fv.x], Pos[fv.y], Pos[fv.z],
                    out float3 closest, out float3 bary);
                float3 diff = p - closest;
                float d = length(diff);
                if (d < QueryRadius)
                {
                    // OGC 特征法向(面/边/顶点角度加权),已按 p-closest 定号
                    float3 n = ClothCollision.VFFeatureNormal(
                        p, closest, f, bary, FaceVerts, FaceAdjacent,
                        VertexFaceStart, VertexFaceList, Pos);
                    var c = new ClothCollision.VFContact { v = v, face = f, bary = bary, normal = n, dist = d };
                    Out.Add(v, c);
                    Out.Add(fv.x, c);
                    Out.Add(fv.y, c);
                    Out.Add(fv.z, c);
                }
            }
            stack.Dispose();
            hits.Dispose();
        }
    }

    /// <summary>EE 检测:每条边查边 BVH,命中写入 VertexToEE。对应原 EE 串行循环。</summary>
    [BurstCompile]
    public struct EEDetectJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeArray<ClothTopology.EdgeInfo> Edges;
        [ReadOnly] public NativeArray<ClothLBVH.Node> EdgeNodes;
        public int EdgeNumPrim;
        public float QueryRadius;
        [ReadOnly] public NativeArray<int> NeighborStart;
        [ReadOnly] public NativeArray<int> NeighborList;
        public NativeParallelMultiHashMap<int, ClothCollision.EEContact>.ParallelWriter Out;

        public void Execute(int e1)
        {
            var ei1 = Edges[e1];
            float3 a1 = Pos[ei1.eV1], b1 = Pos[ei1.eV2];
            float3 lo = min(a1, b1) - new float3(QueryRadius);
            float3 hi = max(a1, b1) + new float3(QueryRadius);

            var stack = new NativeList<int>(64, Allocator.Temp);
            var hits = new NativeList<int>(64, Allocator.Temp);
            ClothLBVH.QueryAABBStatic(EdgeNodes, EdgeNumPrim, lo, hi, ref stack, ref hits);

            for (int h = 0; h < hits.Length; h++)
            {
                int e2 = hits[h];
                if (e2 <= e1) continue; // 去重 + 跳过自身
                var ei2 = Edges[e2];
                // 排除任一端点互为一环邻居(含共享端点)的边对
                if (ClothTopology.IsNeighbor(NeighborStart, NeighborList, ei1.eV1, ei2.eV1) ||
                    ClothTopology.IsNeighbor(NeighborStart, NeighborList, ei1.eV1, ei2.eV2) ||
                    ClothTopology.IsNeighbor(NeighborStart, NeighborList, ei1.eV2, ei2.eV1) ||
                    ClothTopology.IsNeighbor(NeighborStart, NeighborList, ei1.eV2, ei2.eV2)) continue;

                float3 a2 = Pos[ei2.eV1], b2 = Pos[ei2.eV2];
                ClothCollision.ClosestPointsBetweenSegments(a1, b1, a2, b2,
                    out float3 cc1, out float3 cc2, out float mu1, out float mu2);
                float3 diff = cc1 - cc2;
                float d = length(diff);
                if (d < QueryRadius)
                {
                    // OGC EE 法向:两边方向叉乘(良态时),按 diff 定号;近平行退化回退 diff/d
                    float3 n = ClothCollision.EEFeatureNormal(b1 - a1, b2 - a2, diff, d);
                    var c = new ClothCollision.EEContact { e1 = e1, e2 = e2, mu1 = mu1, mu2 = mu2, c1 = cc1, c2 = cc2, normal = n, dist = d };
                    Out.Add(ei1.eV1, c);
                    Out.Add(ei1.eV2, c);
                    Out.Add(ei2.eV1, c);
                    Out.Add(ei2.eV2, c);
                }
            }
            stack.Dispose();
            hits.Dispose();
        }
    }

    /// <summary>保守边界(安全距离)重算。对应 applyCollisionDetection 末尾。</summary>
    [BurstCompile]
    public struct ConservativeBoundsJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeParallelMultiHashMap<int, ClothCollision.VFContact> VertexToVF;
        [ReadOnly] public NativeParallelMultiHashMap<int, ClothCollision.EEContact> VertexToEE;
        [ReadOnly] public NativeArray<AnalyticCollider> Colliders;
        public int ColliderCount;
        public float QueryRadius;
        public float Thickness;
        public float Relax;
        [WriteOnly] public NativeArray<float> ConservativeBounds;
        [WriteOnly] public NativeArray<float3> PositionsAtPrevCD;

        public void Execute(int v)
        {
            // gap = 到最近图元的距离。自碰撞为非负距离,解析碰撞体为带符号距离(穿透为负)。
            float gap = QueryRadius;
            if (VertexToVF.TryGetFirstValue(v, out ClothCollision.VFContact cvf, out var it))
            {
                do { gap = min(gap, cvf.dist); }
                while (VertexToVF.TryGetNextValue(out cvf, ref it));
            }
            if (VertexToEE.TryGetFirstValue(v, out ClothCollision.EEContact cee, out var it2))
            {
                do { gap = min(gap, cee.dist); }
                while (VertexToEE.TryGetNextValue(out cee, ref it2));
            }
            // 解析碰撞体纳入安全距离:带符号最近距离(穿透为负,会使 gap 变负)。
            for (int i = 0; i < ColliderCount; i++)
            {
                Colliders[i].Query(Pos[v], out _, out _, out float cd);
                gap = min(gap, cd);
            }

            float safe = gap - Thickness;
            // safe<=0:已进入接触偏移层(或已穿透)。不能用 0 冻结顶点,否则排斥力推不出去。
            // 写哨兵 -1 表示"本次迭代豁免截断",由消费端跳过截断并强制下次重检测继续跟踪。
            ConservativeBounds[v] = safe <= 0f ? -1f : safe * Relax;
            PositionsAtPrevCD[v] = Pos[v];
        }
    }
}

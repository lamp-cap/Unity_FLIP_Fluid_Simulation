using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Profiling;
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

        // CSR 平坦数组(替代 HashMap):顶点 v 的接触数据位于 [v*MaxPerVertex, v*MaxPerVertex+Counts[v])
        // 对 GPU 迁移友好(单指针 + 偏移即可访问所有接触)。写侧用原子计数,无计数 pass。
        public const int MAX_VF_PER_VERTEX = 64;
        public const int MAX_EE_PER_VERTEX = 128;

        public NativeArray<VFContact> VFContacts;   // N * MAX_VF_PER_VERTEX
        public NativeArray<EEContact> EEContacts;    // N * MAX_EE_PER_VERTEX
        public NativeArray<int> VFCounts;            // N, 填充后的接触数(写后读,无需原子读)
        public NativeArray<int> EECounts;            // N
        // DEBUG: VF 各阶段计数 [N*4]: [BVHhits, vertexFilterPass, AABBpass, geometryPass]
        public NativeArray<int> VFDbg;

        // 暴露给读侧(VBDStepJob)的跨度常量
        public int MaxVFPerVertex => MAX_VF_PER_VERTEX;
        public int MaxEEPerVertex => MAX_EE_PER_VERTEX;

        private ClothLBVH _faceBVH;
        private ClothLBVH _edgeBVH;
        private bool _allocated;

        public void Allocate(int numVertices, int numFaces, int numEdges)
        {
            Dispose();
            _faceBVH = new ClothLBVH();
            _faceBVH.Allocate(numFaces);
            _edgeBVH = new ClothLBVH();
            _edgeBVH.Allocate(numEdges);

            int nv = max(numVertices, 1);

            VFContacts = new NativeArray<VFContact>(nv * MAX_VF_PER_VERTEX, Allocator.Persistent);
            EEContacts = new NativeArray<EEContact>(nv * MAX_EE_PER_VERTEX, Allocator.Persistent);
            VFCounts   = new NativeArray<int>(nv, Allocator.Persistent);
            EECounts   = new NativeArray<int>(nv, Allocator.Persistent);
            VFDbg      = new NativeArray<int>(nv * 4, Allocator.Persistent);
            _allocated = true;
        }

        public ClothLBVH FaceBVH => _faceBVH;
        public ClothLBVH EdgeBVH => _edgeBVH;

        // ---- 用当前顶点位置刷新两棵 BVH 的图元 AABB ----
        // 纯并行循环:每个图元只写自己的 PrimMin/PrimMax 槽位,无竞争。
        // Burst 路径:面 / 边各一个 IJobParallelFor,并发跑后 Complete。
        public void UpdatePrimitiveBounds(ClothTopology topo, NativeArray<float3> pos, float margin)
        {
            var faceHandle =  new FaceBoundsJob
            {
                Pos = pos,
                FaceVerts = topo.FaceVerts,
                Margin = margin,
                PrimMin = _faceBVH.PrimMin,
                PrimMax = _faceBVH.PrimMax,
            }.Schedule(topo.NumFaces, 64);

            var edgeHandle = new EdgeBoundsJob
            {
                Pos = pos,
                Edges = topo.Edges,
                Margin = margin,
                PrimMin = _edgeBVH.PrimMin,
                PrimMax = _edgeBVH.PrimMax,
            }.Schedule(topo.NumEdges, 64);

            JobHandle.CombineDependencies(faceHandle, edgeHandle).Complete();
        }

        /// <summary>每帧开始:完全重建两棵 BVH。两棵树并发构建。</summary>
        public void RebuildBVH(ClothTopology topo, NativeArray<float3> pos, float margin)
        {
            UpdatePrimitiveBounds(topo, pos, margin);
            var faceHandle = _faceBVH.ScheduleBuild();
            var edgeHandle = _edgeBVH.ScheduleBuild();
            JobHandle.CombineDependencies(faceHandle, edgeHandle).Complete();
        }

        /// <summary>迭代中:只 refit bounds。两棵树并发 refit。</summary>
        public void RefitBVH(ClothTopology topo, NativeArray<float3> pos, float margin)
        {
            UpdatePrimitiveBounds(topo, pos, margin);
            var faceHandle = _faceBVH.ScheduleRefit();
            var edgeHandle = _edgeBVH.ScheduleRefit();
            JobHandle.CombineDependencies(faceHandle, edgeHandle).Complete();
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
            Profiler.BeginSample("Build BVH");
            RebuildBVH(topo, pos, queryRadius);
            Profiler.EndSample();

            Profiler.BeginSample("Detect Collisions");
            DetectBurst(topo, state, queryRadius, thickness, relax, colliders, colliderCount);
            Profiler.EndSample();
        }

        // 单 pass:清除计数器 → VF 检测 → EE 检测 → 保守边界。无计数 pass,BVH 遍历量减半。
        void DetectBurst(ClothTopology topo, ClothState state,
            float queryRadius, float thickness, float relax,
            NativeArray<AnalyticCollider> colliders, int colliderCount)
        {
            // 清零计数器(原子写目标,每次检测前必须清零)
            Profiler.BeginSample("Query Collisions");
            new ClearCountsJob { VFCounts = VFCounts, EECounts = EECounts }
                .Schedule(topo.NumVertices, 64).Complete();

            // VF 检测
            new VFDetectJob
            {
                Pos = state.Positions,
                FaceVerts = topo.FaceVerts,
                FaceNodes = _faceBVH.Nodes,
                FaceNumPrim = _faceBVH.NumPrimitives,
                QueryRadius = queryRadius,
                NbrStart = topo.VertexNeighborStart,
                NbrList = topo.VertexNeighborList,
                FaceAdjacent = topo.FaceAdjacent,
                VFContacts = VFContacts,
                VFCounts = VFCounts,
                VFDbg = VFDbg,
                MaxPerVertex = MAX_VF_PER_VERTEX,
            }.Schedule(topo.NumVertices, 8).Complete();

            // EE 检测
            new EEDetectJob
            {
                Pos = state.Positions,
                Edges = topo.Edges,
                EdgeNodes = _edgeBVH.Nodes,
                EdgeNumPrim = _edgeBVH.NumPrimitives,
                QueryRadius = queryRadius,
                EEContacts = EEContacts,
                EECounts = EECounts,
                MaxPerVertex = MAX_EE_PER_VERTEX,
            }.Schedule(topo.NumEdges, 8).Complete();

            Profiler.EndSample();

            // 保守边界重算
            var boundsJob = new ConservativeBoundsJob
            {
                Pos = state.Positions,
                VFContacts = VFContacts,
                VFCounts = VFCounts,
                EEContacts = EEContacts,
                EECounts = EECounts,
                MaxVFPerVertex = MAX_VF_PER_VERTEX,
                MaxEEPerVertex = MAX_EE_PER_VERTEX,
                QueryRadius = queryRadius,
                Thickness = thickness,
                Relax = relax,
                ConservativeBounds = state.ConservativeBounds,
                PositionsAtPrevCD = state.PositionsAtPrevCD,
            };
            boundsJob.Schedule(topo.NumVertices, 64).Complete();

            // DEBUG
            // int dbgVF = 0, dbgEE = 0;
            // int sBVH = 0, sVtx = 0, sAABB = 0, sGeo = 0;
            // for (int i = 0; i < topo.NumVertices; i++)
            // {
            //     dbgVF += VFCounts[i];
            //     dbgEE += EECounts[i];
            //     sBVH += VFDbg[i * 4 + 0];
            //     sVtx += VFDbg[i * 4 + 1];
            //     sAABB += VFDbg[i * 4 + 2];
            //     sGeo += VFDbg[i * 4 + 3];
            // }
            // UnityEngine.Debug.Log($"[VFDetect] contacts={dbgVF} stages: BVH={sBVH} vtxFilter={sVtx} aabb={sAABB} geo={sGeo} | [EE] contacts={dbgEE}");
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

        // ===============================================================
        // 接触特征分类 / Feasible Region (参照参考实现 TriMeshCollisionGeometry / ClothContactDetector)
        // ===============================================================

        /// <summary>VF 最近点特征类型,从 bary 分类。</summary>
        public enum VFPointType : byte
        {
            Interior = 0, AtA = 1, AtB = 2, AtC = 3, AtAB = 4, AtBC = 5, AtAC = 6
        }

        /// <summary>通过重心坐标判断 VF 最近点落在三角形的哪个特征上。</summary>
        public static VFPointType ClassifyVFPoint(float3 bary, float eps = 1e-4f)
        {
            bool bx = bary.x < eps;
            bool by = bary.y < eps;
            bool bz = bary.z < eps;
            if (bx && by) return VFPointType.AtC;
            if (by && bz) return VFPointType.AtA;
            if (bx && bz) return VFPointType.AtB;
            if (bx) return VFPointType.AtBC;
            if (by) return VFPointType.AtAC;
            if (bz) return VFPointType.AtAB;
            return VFPointType.Interior;
        }

        /// <summary>
        /// EE feasible region(参照 triMeshEdgeContactQueryFunc 行 784-792):
        /// 至少一个最近点必须投影在对应线段范围内。
        /// </summary>
        public static bool EEFeasibleRegion(
            float3 a1, float3 b1, float3 e1Dir,
            float3 a2, float3 b2, float3 e2Dir,
            float3 c1, float3 c2)
        {
            return (dot(c2 - a1, e1Dir) >= 0f && dot(b1 - c2, e1Dir) >= 0f)
                || (dot(c1 - a2, e2Dir) >= 0f && dot(b2 - c1, e2Dir) >= 0f);
        }

        /// <summary>
        /// VF 边 feasible region(参照 TriMeshCollisionGeometry::checkEdgeFeasibleRegion 行 91-153):
        /// 1) 查询点投影在该边范围内
        /// 2) 在相邻面的分割面正确侧(避免一个接触被两个共享边的面重复报告)
        /// v1, v2: 边两端点; oppCurr: 当前面的对角顶点
        /// adjFaceId: 邻接面(-1 表示边界边,跳过第二个分割面检查)
        /// </summary>
        public static bool VFEdgeFeasible(
            float3 p, float3 v1, float3 v2, float3 oppCurr,
            int adjFaceId, NativeArray<int3> faceVerts, NativeArray<float3> pos, float relax)
        {
            float3 AP = p - v1;
            float3 BP = p - v2;
            float3 AB = v2 - v1;

            // 查询点必须投影在边范围内
            if (dot(AP, AB) < -relax) return false;
            if (dot(BP, AB) > relax) return false;

            float ABsq = lengthsq(AB);
            if (ABsq < 1e-12f) return false;

            // 分割面 1 (当前面对角点): 查询点必须在对角点的另一侧
            {
                float t = dot(AB, oppCurr - v1) / ABsq;
                float3 foot = v1 + t * AB;
                if (dot(oppCurr - foot, p - foot) > relax) return false;
            }

            // 分割面 2 (邻接面对角点,如有)
            if (adjFaceId >= 0)
            {
                int3 adjFv = faceVerts[adjFaceId];
                // 找邻接面中不在共享边上的对角顶点(通过位置比较排除两端点)
                float3 ap = pos[adjFv.x], bp = pos[adjFv.y], cp = pos[adjFv.z];
                float3 oppAdj;
                bool aIsEp = lengthsq(ap - v1) < 1e-8f || lengthsq(ap - v2) < 1e-8f;
                bool bIsEp = lengthsq(bp - v1) < 1e-8f || lengthsq(bp - v2) < 1e-8f;
                if (!aIsEp) oppAdj = ap;
                else if (!bIsEp) oppAdj = bp;
                else oppAdj = cp;
                float t = dot(AB, oppAdj - v1) / ABsq;
                float3 foot = v1 + t * AB;
                if (dot(oppAdj - foot, p - foot) > relax) return false;
            }

            return true;
        }

        /// <summary>
        /// VF 顶点 feasible region(参照 TriMeshCollisionGeometry::checkVertexFeasibleRegion 行 155-186):
        /// 对顶点的每个邻居,检查查询点在"以邻居为参考方向的正确半空间"内。
        /// p: 查询点位置; vId: 最近顶点; 邻居数据来自拓扑 CSR。
        /// </summary>
        public static bool VFVertexFeasible(
            float3 p, int vId,
            NativeArray<int> nbrStart, NativeArray<int> nbrList,
            NativeArray<float3> pos, float relax)
        {
            float3 A = pos[vId];
            float3 AP = p - A;
            for (int i = nbrStart[vId], end = nbrStart[vId + 1]; i < end; i++)
            {
                int nbr = nbrList[i];
                if (nbr == vId) continue; // 跳过自身
                float3 BA = A - pos[nbr];
                if (dot(AP, BA) < -relax) return false;
            }
            return true;
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
        // VF 接触法向(OGC 论文规则)
        // ---------------------------------------------------------------
        // 按最近点落在三角形的"面 / 边 / 顶点"特征上取法向:
        //   - 面内:面法向(良态、唯一,不受查询点辐射噪声影响)
        //   - 边 / 顶点:直接用最近点连线方向 (p-closest)/d
        // 边/顶点特征上没有唯一的"表面法向",论文直接取连线方向即排斥方向,
        // 省掉了角度加权伪法向那套昂贵计算。
        // ===============================================================

        /// <summary>面 f 的单位法向(当前位形)。</summary>
        static float3 FaceNormal(int f, NativeArray<int3> faceVerts, NativeArray<float3> pos)
        {
            int3 fv = faceVerts[f];
            return normalizesafe(cross(pos[fv.y] - pos[fv.x], pos[fv.z] - pos[fv.x]));
        }

        /// <summary>
        /// VF 接触法向。bary 来自 ClosestPointOnTriangle。
        /// 最近点在面内 -> 面法向(按 p-closest 定号);在边/顶点 -> 最近点连线方向。
        /// </summary>
        public static float3 VFFeatureNormal(
            float3 p, float3 closest, int face, float3 bary,
            NativeArray<int3> faceVerts, NativeArray<float3> pos)
        {
            const float eps = 1e-4f;

            // 任一 bary 分量 ~0 即落在边或顶点特征上
            bool onBoundary = bary.x < eps || bary.y < eps || bary.z < eps;

            float3 dir = p - closest;
            if (onBoundary)
            {
                // 边/顶点特征:用最近点连线方向(退化时回退面法向定号)
                float dlen = length(dir);
                if (dlen > 1e-8f) return dir / dlen;
                float3 nf = FaceNormal(face, faceVerts, pos);
                return dot(nf, dir) < 0f ? -nf : nf;
            }

            // 面内特征:面法向,按 p-closest 定号指向查询点一侧
            float3 N = FaceNormal(face, faceVerts, pos);
            if (dot(N, dir) < 0f) N = -N;
            return N;
        }

        /// <summary>
        /// EE 接触法向(OGC 论文规则):两条边上最近点连线方向 (c1-c2)/d。
        /// 与 EEContact.normal / 受力代码的约定一致(diff=c1-c2,指向边 1 最近点一侧)。
        /// diff=c1-c2, dlen=length(diff);两边相交(dlen≈0)退化时回退两边方向叉乘。
        /// d1=b1-a1, d2=b2-a2 仅用于退化回退。
        /// </summary>
        public static float3 EEFeatureNormal(float3 d1, float3 d2, float3 diff, float dlen)
        {
            // 最近点连线方向即排斥方向
            if (dlen > 1e-8f) return diff / dlen;
            // 退化(两边相交,最近点重合):回退到垂直两边的方向
            float3 cr = cross(d1, d2);
            float crLen2 = lengthsq(cr);
            if (crLen2 > 1e-12f) return cr * rsqrt(crLen2);
            return normalizesafe(cross(d1, float3(0, 1, 0)));
        }

        public void Dispose()
        {
            if (!_allocated) return;
            _faceBVH?.Dispose();
            _edgeBVH?.Dispose();
            if (VFContacts.IsCreated) VFContacts.Dispose();
            if (EEContacts.IsCreated) EEContacts.Dispose();
            if (VFCounts.IsCreated) VFCounts.Dispose();
            if (EECounts.IsCreated) EECounts.Dispose();
            if (VFDbg.IsCreated) VFDbg.Dispose();
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

    /// <summary>清零 VF/EE 计数器(检测前调用,两个数组等长 = numVertices)</summary>
    [BurstCompile]
    struct ClearCountsJob : IJobParallelFor
    {
        [NativeDisableContainerSafetyRestriction, WriteOnly] public NativeArray<int> VFCounts;
        [NativeDisableContainerSafetyRestriction, WriteOnly] public NativeArray<int> EECounts;
        public void Execute(int i)
        {
            VFCounts[i] = 0;
            EECounts[i] = 0;
        }
    }

    /// <summary>VF 检测:每个顶点查三角面 BVH,命中间接原子写入 CSR 平坦数组。</summary>
    [BurstCompile]
    public unsafe struct VFDetectJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeArray<int3> FaceVerts;
        [ReadOnly] public NativeArray<ClothLBVH.Node> FaceNodes;
        [ReadOnly] public NativeArray<int> NbrStart;
        [ReadOnly] public NativeArray<int> NbrList;
        [ReadOnly] public NativeArray<int3> FaceAdjacent;
        public int FaceNumPrim;
        public float QueryRadius;
        public int MaxPerVertex;
        [NativeDisableParallelForRestriction] public NativeArray<ClothCollision.VFContact> VFContacts;
        [NativeDisableParallelForRestriction] public NativeArray<int> VFCounts;
        [NativeDisableParallelForRestriction] public NativeArray<int> VFDbg;

        public void Execute(int v)
        {
            float3 p = Pos[v];
            float qrSq = QueryRadius * QueryRadius;
            float3 q = new float3(QueryRadius);

            var stack = new NativeList<int>(64, Allocator.Temp);
            var hits = new NativeList<int>(64, Allocator.Temp);
            ClothLBVH.QueryAABBStatic(FaceNodes, FaceNumPrim, p - q, p + q, stack, hits);

            int baseV = v * MaxPerVertex;
            int* cntPtr = (int*)VFCounts.GetUnsafePtr();
            int nBVH = hits.Length, nVtx = 0, nAABB = 0, nGeo = 0;
            for (int h = 0; h < hits.Length; h++)
            {
                int f = hits[h];
                int3 fv = FaceVerts[f];
                if (v == fv.x || v == fv.y || v == fv.z) continue;
                nVtx++;

                float3 v0 = Pos[fv.x], v1 = Pos[fv.y], v2 = Pos[fv.z];
                float3 fMin = min(min(v0, v1), v2), fMax = max(max(v0, v1), v2);
                if (lengthsq(p - clamp(p, fMin, fMax)) > qrSq) continue;
                nAABB++;

                ClothCollision.ClosestPointOnTriangle(p, v0, v1, v2,
                    out float3 closest, out float3 bary);
                float d = length(p - closest);
                if (d > QueryRadius) continue;
                nGeo++;

                var pt = ClothCollision.ClassifyVFPoint(bary);
                bool inFeasible = true;
                const float eps = 1e-6f;
                switch (pt)
                {
                    case ClothCollision.VFPointType.Interior: break;
                    case ClothCollision.VFPointType.AtAB: inFeasible = ClothCollision.VFEdgeFeasible(p, Pos[fv.x], Pos[fv.y], Pos[fv.z], FaceAdjacent[f].x, FaceVerts, Pos, eps); break;
                    case ClothCollision.VFPointType.AtBC: inFeasible = ClothCollision.VFEdgeFeasible(p, Pos[fv.y], Pos[fv.z], Pos[fv.x], FaceAdjacent[f].y, FaceVerts, Pos, eps); break;
                    case ClothCollision.VFPointType.AtAC: inFeasible = ClothCollision.VFEdgeFeasible(p, Pos[fv.z], Pos[fv.x], Pos[fv.y], FaceAdjacent[f].z, FaceVerts, Pos, eps); break;
                    case ClothCollision.VFPointType.AtA:  inFeasible = ClothCollision.VFVertexFeasible(p, fv.x, NbrStart, NbrList, Pos, eps); break;
                    case ClothCollision.VFPointType.AtB:  inFeasible = ClothCollision.VFVertexFeasible(p, fv.y, NbrStart, NbrList, Pos, eps); break;
                    case ClothCollision.VFPointType.AtC:  inFeasible = ClothCollision.VFVertexFeasible(p, fv.z, NbrStart, NbrList, Pos, eps); break;
                }
                if (!inFeasible) continue;

                float3 n = ClothCollision.VFFeatureNormal(p, closest, f, bary, FaceVerts, Pos);
                var c = new ClothCollision.VFContact { v = v, face = f, bary = bary, normal = n, dist = d };

                // 原子写入(4 个 key: v + 三角形三顶点)
                int slot = System.Threading.Interlocked.Increment(ref cntPtr[v]) - 1;
                if (slot < MaxPerVertex) VFContacts[baseV + slot] = c;
                int baseFx = fv.x * MaxPerVertex; slot = System.Threading.Interlocked.Increment(ref cntPtr[fv.x]) - 1; if (slot < MaxPerVertex) VFContacts[baseFx + slot] = c;
                int baseFy = fv.y * MaxPerVertex; slot = System.Threading.Interlocked.Increment(ref cntPtr[fv.y]) - 1; if (slot < MaxPerVertex) VFContacts[baseFy + slot] = c;
                int baseFz = fv.z * MaxPerVertex; slot = System.Threading.Interlocked.Increment(ref cntPtr[fv.z]) - 1; if (slot < MaxPerVertex) VFContacts[baseFz + slot] = c;
            }
            // DEBUG: 每顶点独享 slot,无需原子
            int dbgOff = v * 4;
            VFDbg[dbgOff + 0] = nBVH;
            VFDbg[dbgOff + 1] = nVtx;
            VFDbg[dbgOff + 2] = nAABB;
            VFDbg[dbgOff + 3] = nGeo;
            stack.Dispose();
            hits.Dispose();
        }
    }

    /// <summary>EE 检测:每条边查边 BVH,命中原子写入 CSR 平坦数组。</summary>
    [BurstCompile]
    public unsafe struct EEDetectJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeArray<ClothTopology.EdgeInfo> Edges;
        [ReadOnly] public NativeArray<ClothLBVH.Node> EdgeNodes;
        public int EdgeNumPrim;
        public float QueryRadius;
        public int MaxPerVertex;
        [NativeDisableParallelForRestriction] public NativeArray<ClothCollision.EEContact> EEContacts;
        [NativeDisableParallelForRestriction] public NativeArray<int> EECounts;

        public void Execute(int e1)
        {
            var ei1 = Edges[e1];
            float3 a1 = Pos[ei1.eV1], b1 = Pos[ei1.eV2];
            float qrSq = QueryRadius * QueryRadius;
            float3 e1Min = min(a1, b1), e1Max = max(a1, b1);
            float3 lo = e1Min - new float3(QueryRadius);
            float3 hi = e1Max + new float3(QueryRadius);

            var stack = new NativeList<int>(64, Allocator.Temp);
            var hits = new NativeList<int>(64, Allocator.Temp);
            ClothLBVH.QueryAABBStatic(EdgeNodes, EdgeNumPrim, lo, hi, stack, hits);

            int* cntPtr = (int*)EECounts.GetUnsafePtr();
            for (int h = 0; h < hits.Length; h++)
            {
                int e2 = hits[h];
                if (e2 <= e1) continue;
                var ei2 = Edges[e2];
                if (ei1.eV1 == ei2.eV1 || ei1.eV1 == ei2.eV2 ||
                    ei1.eV2 == ei2.eV1 || ei1.eV2 == ei2.eV2) continue;

                float3 a2 = Pos[ei2.eV1], b2 = Pos[ei2.eV2];
                float3 e2Min = min(a2, b2), e2Max = max(a2, b2);
                float3 sep = max(0f, max(e1Min - e2Max, e2Min - e1Max));
                if (lengthsq(sep) > qrSq) continue;

                ClothCollision.ClosestPointsBetweenSegments(a1, b1, a2, b2,
                    out float3 cc1, out float3 cc2, out float mu1, out float mu2);
                float d = length(cc1 - cc2);
                if (d > QueryRadius) continue;

                float3 e1Dir = b1 - a1, e2Dir = b2 - a2;
                if (!ClothCollision.EEFeasibleRegion(a1, b1, e1Dir, a2, b2, e2Dir, cc1, cc2))
                    continue;

                float3 n = ClothCollision.EEFeatureNormal(e1Dir, e2Dir, cc1 - cc2, d);
                var c = new ClothCollision.EEContact { e1 = e1, e2 = e2, mu1 = mu1, mu2 = mu2, c1 = cc1, c2 = cc2, normal = n, dist = d };

                // 原子写入(4 个 key)
                int slot = System.Threading.Interlocked.Increment(ref cntPtr[ei1.eV1]) - 1; if (slot < MaxPerVertex) EEContacts[ei1.eV1 * MaxPerVertex + slot] = c;
                slot = System.Threading.Interlocked.Increment(ref cntPtr[ei1.eV2]) - 1; if (slot < MaxPerVertex) EEContacts[ei1.eV2 * MaxPerVertex + slot] = c;
                slot = System.Threading.Interlocked.Increment(ref cntPtr[ei2.eV1]) - 1; if (slot < MaxPerVertex) EEContacts[ei2.eV1 * MaxPerVertex + slot] = c;
                slot = System.Threading.Interlocked.Increment(ref cntPtr[ei2.eV2]) - 1; if (slot < MaxPerVertex) EEContacts[ei2.eV2 * MaxPerVertex + slot] = c;
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
        [ReadOnly] public NativeArray<ClothCollision.VFContact> VFContacts;
        [ReadOnly] public NativeArray<int> VFCounts;
        [ReadOnly] public NativeArray<ClothCollision.EEContact> EEContacts;
        [ReadOnly] public NativeArray<int> EECounts;
        public int MaxVFPerVertex;
        public int MaxEEPerVertex;
        public float QueryRadius;
        public float Thickness;
        public float Relax;
        [WriteOnly] public NativeArray<float> ConservativeBounds;
        [WriteOnly] public NativeArray<float3> PositionsAtPrevCD;

        public void Execute(int v)
        {
            float gap = QueryRadius;
            // VF
            int baseV = v * MaxVFPerVertex;
            int cnt = VFCounts[v];
            if (cnt > MaxVFPerVertex) cnt = MaxVFPerVertex;
            for (int i = 0; i < cnt; i++)
                gap = min(gap, VFContacts[baseV + i].dist);
            // EE
            baseV = v * MaxEEPerVertex;
            cnt = EECounts[v];
            if (cnt > MaxEEPerVertex) cnt = MaxEEPerVertex;
            for (int i = 0; i < cnt; i++)
                gap = min(gap, EEContacts[baseV + i].dist);

            float safe = max(0f, gap - Thickness);
            ConservativeBounds[v] = safe * Relax;
            PositionsAtPrevCD[v] = Pos[v];
        }
    }
}

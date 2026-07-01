using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD.Cloth
{
    /// <summary>
    /// VBD 求解热路径的 Burst 化。
    ///
    /// 热路径 = 逐顶点 VBD 步 VBDStepVertex(顶点数 × 颜色组 × 迭代 × 子步 × 帧),
    /// 是整个求解里调用频次最高、且为纯 NativeArray 数学的部分,Burst 收益最大、风险最低。
    /// 这里把它完整搬进 [BurstCompile] IJobParallelFor,按颜色组 dispatch(组内并行,组间串行),
    /// 与 ClothSolver.VBDStepVertex 的标量实现逐行对应,作为可切换的等价实现。
    ///
    /// LBVH 建树 / 广相查询用了递归与 Temp 分配,风险高,本阶段不 Burst 化。
    /// </summary>
    public struct VBDSolverParams
    {
        public float dt;
        public float stepSize;
        public float miu;
        public float lambda;
        public float bendingStiffness;
        public float dampingStVK;
        public float dampingBending;

        public byte handleCollision;
        public float thickness;
        public float contactRadius;
        public float contactStiffness;
        public float queryRadius;
        public byte applyFriction;
        public float frictionMu;
        public float frictionEpsV;

        public byte degenerateTriangleThresEnabled;
        public float degenerateTriangleThres;

        public byte hasColliders;
    }

    /// <summary>
    /// 处理单个颜色组的逐顶点 VBD 步。Execute(i) 对应组内第 i 个顶点。
    /// 同组顶点互不相邻(图着色),写各自的 PositionsNext[v],无竞争。
    /// </summary>
    [BurstCompile]
    public struct VBDStepJob : IJobParallelFor
    {
        public VBDSolverParams P;

        // 当前颜色组:组内第 i 个顶点 = ColorVertices[ColorStart + i]
        public int ColorStart;
        [ReadOnly] public NativeArray<int> ColorVertices;

        // ---- 拓扑(只读) ----
        [ReadOnly] public NativeArray<int3> FaceVerts;
        [ReadOnly] public NativeArray<float2x2> DmInv;
        [ReadOnly] public NativeArray<float> FaceRestArea;
        [ReadOnly] public NativeArray<float> VertexMass;
        [ReadOnly] public NativeArray<bool> FixedMask;
        [ReadOnly] public NativeArray<int> VertexFaceStart;
        [ReadOnly] public NativeArray<int2> VertexFaceList;
        [ReadOnly] public NativeArray<int> VertexEdgeStart;
        [ReadOnly] public NativeArray<int2> VertexEdgeList;
        [ReadOnly] public NativeArray<ClothTopology.EdgeInfo> Edges;
        [ReadOnly] public NativeArray<float4x4> EdgeQ;

        // ---- 状态 ----
        [ReadOnly] public NativeArray<float3> Positions;     // 读邻居当前位置
        [ReadOnly] public NativeArray<float3> PositionsPrev;
        [ReadOnly] public NativeArray<float3> Inertia;
        [ReadOnly] public NativeArray<float> ConservativeBounds;
        [ReadOnly] public NativeArray<float3> PositionsAtPrevCD;

        // 仅写各自顶点槽位;组内顶点全不同,关闭并行写检查
        [NativeDisableParallelForRestriction] public NativeArray<float3> PositionsNext;
        // 截断标志:写组内 index i;主线程归约
        [WriteOnly] public NativeArray<byte> TruncFlags;

        // ---- 碰撞(已检测结果,只读;无碰撞时为空容器) ----
        // 顶点 -> 接触结构体(值直接存),与 ClothCollision 的新数据结构一致。
        [ReadOnly] public NativeParallelMultiHashMap<int, ClothCollision.VFContact> VertexToVF;
        [ReadOnly] public NativeParallelMultiHashMap<int, ClothCollision.EEContact> VertexToEE;

        // ---- 解析碰撞体 ----
        [ReadOnly] public NativeArray<AnalyticCollider> Colliders;

        public void Execute(int i)
        {
            int v = ColorVertices[ColorStart + i];
            TruncFlags[i] = 0;
            if (FixedMask[v]) return;

            float dt = P.dt;
            float3 force = float3(0);
            float3x3 hessian = float3x3(0);

            // ---- 接触力(复用已检测结果) ----
            if (P.handleCollision != 0)
                AccumulateContacts(v, dt, ref force, ref hessian);

            // ---- 惯性项: m/dt^2 * (y - x) ----
            float m = VertexMass[v];
            float invDt2 = 1f / (dt * dt);
            float3 xi = Positions[v];
            force += (m * invDt2) * (Inertia[v] - xi);
            hessian.c0.x += m * invDt2;
            hessian.c1.y += m * invDt2;
            hessian.c2.z += m * invDt2;

            // ---- 材料力(StVK 膜 + 弯曲) ----
            AccumulateMaterial(v, dt, ref force, ref hessian);

            // ---- 解析碰撞体 ----
            if (P.hasColliders != 0)
                AccumulateAnalyticColliders(v, dt, ref force, ref hessian);

            // ---- 解 H dx = f ----
            if (!ClothMath.Solve3x3PSD(hessian, force, out float3 dx))
            {
                PositionsNext[v] = Positions[v];
                return;
            }

            float3 newPos = Positions[v] + P.stepSize * dx;

            // 保守边界截断(对应 applyConservativeBoundTruncation)。
            // bound 恒 >= 0:位移超出即截断回边界内,并标记下次迭代重检测。
            if (P.handleCollision != 0)
            {
                float bound = ConservativeBounds[v];
                float3 disp = newPos - PositionsAtPrevCD[v];
                float dispLen = length(disp);
                if (dispLen > bound && dispLen > 1e-12f)
                {
                    disp *= (bound / dispLen);
                    newPos = PositionsAtPrevCD[v] + disp;
                    TruncFlags[i] = 1;
                }
            }

            PositionsNext[v] = newPos;
        }

        // ---- 材料:StVK 膜 + 阻尼 + 弯曲 ----
        void AccumulateMaterial(int v, float dt, ref float3 force, ref float3x3 hessian)
        {
            float3x3 hBefore = hessian;

            int fStart = VertexFaceStart[v];
            int fEnd = VertexFaceStart[v + 1];
            for (int i = fStart; i < fEnd; i++)
            {
                int2 fo = VertexFaceList[i];
                int f = fo.x;
                int order = fo.y;
                int3 fv = FaceVerts[f];
                ClothMath.AccumulateStVKFace(
                    Positions[fv.x], Positions[fv.y], Positions[fv.z],
                    DmInv[f], FaceRestArea[f],
                    P.lambda, P.miu, order,
                    ref force, ref hessian);
            }

            if (P.dampingStVK > 0f)
            {
                float3x3 dampingH = (hessian - hBefore) * (P.dampingStVK / dt);
                float3 disp = Positions[v] - PositionsPrev[v];
                force -= mul(dampingH, disp);
                hessian += dampingH;
            }

            AccumulateBending(v, dt, ref force, ref hessian);
        }

        void AccumulateBending(int v, float dt, ref float3 force, ref float3x3 hessian)
        {
            float ks = P.bendingStiffness;
            if (ks <= 0f) return;
            float damping = P.dampingBending;

            int eStart = VertexEdgeStart[v];
            int eEnd = VertexEdgeStart[v + 1];
            for (int i = eStart; i < eEnd; i++)
            {
                int2 eo = VertexEdgeList[i];
                int e = eo.x;
                int order = eo.y; // 0..3 对应 [eV1, eV2, eV12Next, eV21Next]
                var ei = Edges[e];
                if (ei.fId2 == -1) continue;

                float4x4 Q = EdgeQ[e];
                float3 x0 = Positions[ei.eV1];
                float3 x1 = Positions[ei.eV2];
                float3 x2 = Positions[ei.eV12Next];
                float3 x3 = Positions[ei.eV21Next];

                if (P.degenerateTriangleThresEnabled != 0)
                {
                    float3 n1 = cross(x1 - x0, x2 - x0);
                    float3 n2 = cross(x2 - x3, x1 - x3);
                    float thr = P.degenerateTriangleThres * P.degenerateTriangleThres;
                    if (lengthsq(n1) < thr || lengthsq(n2) < thr) continue;
                }

                // (Q*Xs) 行 order: Q(order,j) = Q[j][order]
                float qr0 = Q[0][order];
                float qr1 = Q[1][order];
                float qr2 = Q[2][order];
                float qr3 = Q[3][order];
                float3 dE_row = ks * (qr0 * x0 + qr1 * x1 + qr2 * x2 + qr3 * x3);

                float qDiag = Q[order][order];
                float3x3 hTemp = (ks * qDiag) * float3x3(1,0,0,0,1,0,0,0,1);

                float3 dispV = Positions[v] - PositionsPrev[v];
                if (damping > 0f)
                {
                    force -= dE_row + mul(hTemp, dispV) * (damping / dt);
                    hessian += hTemp * (1f + damping / dt);
                }
                else
                {
                    force -= dE_row;
                    hessian += hTemp;
                }
            }
        }

        // ---- 复用已检测接触(VF/EE) ----
        void AccumulateContacts(int v, float dt, ref float3 force, ref float3x3 hessian)
        {
            if (VertexToVF.TryGetFirstValue(v, out ClothCollision.VFContact c, out var it))
            {
                do
                {
                    int order = VFOrder(c, v);
                    AccumulateVFContact(c, order, dt, ref force, ref hessian);
                }
                while (VertexToVF.TryGetNextValue(out c, ref it));
            }

            if (VertexToEE.TryGetFirstValue(v, out ClothCollision.EEContact ce, out var it2))
            {
                do
                {
                    int order = EEOrder(ce, v);
                    if (order >= 0)
                        AccumulateEEContact(ce, order, dt, ref force, ref hessian);
                }
                while (VertexToEE.TryGetNextValue(out ce, ref it2));
            }
        }

        void AccumulateVFContact(in ClothCollision.VFContact c, int order, float dt,
            ref float3 force, ref float3x3 hessian)
        {
            // 用当前位置 + 冻结重心实时重算法向/距离(对应参考 accumulateVFContactForceAndHessian
            // 的 n=(x-contactPoint).normalized()),避免冻结伪法向锁死在穿透侧导致稳定穿模。
            int3 fvN = FaceVerts[c.face];
            float3 contactPoint = c.bary.x * Positions[fvN.x]
                                + c.bary.y * Positions[fvN.y]
                                + c.bary.z * Positions[fvN.z];
            float3 diff = Positions[c.v] - contactPoint; // 接触点 -> 顶点
            float dis = length(diff);
            if (dis >= P.contactRadius + P.thickness) return;
            float3 n = dis > 1e-8f ? diff / dis : c.normal;

            float4 bs = new float4(-c.bary.x, -c.bary.y, -c.bary.z, 1f);
            float b = bs[order];

            ClothCollision.RepulsiveForce(dis, P.thickness, P.contactRadius, P.contactStiffness,
                out float dEdD, out float d2EdDdD);
            float lambda = -dEdD;

            force += b * lambda * n;
            hessian += d2EdDdD * b * b * Outer(n, n);

            if (P.applyFriction != 0)
            {
                int vSide = c.v;
                float3 dx = Positions[vSide] - PositionsPrev[vSide];
                int3 fv = FaceVerts[c.face];
                float3 e0 = normalizesafe(Positions[fv.y] - Positions[fv.x]);
                float3 t1 = normalizesafe(cross(e0, n));
                float3x2 T = new float3x2(e0, t1);
                float2 u = new float2(dot(T.c0, dx), dot(T.c1, dx));
                ClothCollision.Friction(P.frictionMu, lambda, T, u, P.frictionEpsV * dt,
                    out float3 ff, out float3x3 fh);
                force += b * ff;
                hessian += b * b * fh;
            }
        }

        void AccumulateEEContact(in ClothCollision.EEContact c, int order, float dt,
            ref float3 force, ref float3x3 hessian)
        {
            // 用当前位置 + 冻结线段参数实时重算最近点/法向/距离
            // (对应参考 accumulateEEContactForceAndHessian: n=diff/dis, diff=c1-c2)。
            var ei1N = Edges[c.e1];
            var ei2N = Edges[c.e2];
            float3 c1N = (1f - c.mu1) * Positions[ei1N.eV1] + c.mu1 * Positions[ei1N.eV2];
            float3 c2N = (1f - c.mu2) * Positions[ei2N.eV1] + c.mu2 * Positions[ei2N.eV2];
            float3 diff = c1N - c2N; // 边2接触点 -> 边1接触点
            float dis = length(diff);
            if (dis >= P.contactRadius + P.thickness) return;
            float3 n = dis > 1e-8f ? diff / dis : c.normal;

            float4 bs = new float4(1f - c.mu1, c.mu1, -1f + c.mu2, -c.mu2);
            float b = bs[order];

            ClothCollision.RepulsiveForce(dis, P.thickness, P.contactRadius, P.contactStiffness,
                out float dEdD, out float d2EdDdD);
            float lambda = -dEdD;

            force += b * lambda * n;
            hessian += d2EdDdD * b * b * Outer(n, n);

            if (P.applyFriction != 0)
            {
                var ei1 = Edges[c.e1];
                var ei2 = Edges[c.e2];
                float3 v1 = normalizesafe(Positions[ei1.eV2] - Positions[ei1.eV1]);
                float3 t1 = normalizesafe(cross(v1, n));
                float3x2 T = new float3x2(v1, t1);
                float3 c1p = (1f - c.mu1) * PositionsPrev[ei1.eV1] + c.mu1 * PositionsPrev[ei1.eV2];
                float3 c2p = (1f - c.mu2) * PositionsPrev[ei2.eV1] + c.mu2 * PositionsPrev[ei2.eV2];
                float3 rel = (c1N - c2N) - (c1p - c2p);
                float2 u = new float2(dot(T.c0, rel), dot(T.c1, rel));
                ClothCollision.Friction(P.frictionMu, lambda, T, u, P.frictionEpsV * dt,
                    out float3 ff, out float3x3 fh);
                force += b * ff;
                hessian += b * b * fh;
            }
        }

        int VFOrder(in ClothCollision.VFContact c, int v)
        {
            if (c.v == v) return 3;
            int3 fv = FaceVerts[c.face];
            if (fv.x == v) return 0;
            if (fv.y == v) return 1;
            return 2;
        }

        int EEOrder(in ClothCollision.EEContact c, int v)
        {
            var e1 = Edges[c.e1];
            var e2 = Edges[c.e2];
            if (e1.eV1 == v) return 0;
            if (e1.eV2 == v) return 1;
            if (e2.eV1 == v) return 2;
            if (e2.eV2 == v) return 3;
            return -1;
        }

        void AccumulateAnalyticColliders(int v, float dt, ref float3 force, ref float3x3 hessian)
        {
            float3 p = Positions[v];

            for (int i = 0; i < Colliders.Length; i++)
            {
                var col = Colliders[i];
                col.Query(p, out float3 closest, out float3 n, out float dist);

                // 与自碰撞一致:在 dist < contactRadius + thickness 的接触带内即施排斥力,
                // 用同一个 RepulsiveForce(线性 penalty)。dist 为带符号距离(穿透为负),
                // n 指向碰撞体外侧。这样布料在接触带外缘就被平滑推开,不必先陷进去。
                if (dist >= P.contactRadius + P.thickness) continue;

                ClothCollision.RepulsiveForce(dist, P.thickness, P.contactRadius, P.contactStiffness,
                    out float dEdD, out float d2EdDdD);
                float lambda = -dEdD;
                force += lambda * n;
                hessian += d2EdDdD * Outer(n, n);

                if (P.applyFriction != 0)
                {
                    float3 dx = p - PositionsPrev[v];
                    float3 axis = abs(n.x) < 0.9f ? new float3(1, 0, 0) : new float3(0, 1, 0);
                    float3 t0 = normalizesafe(cross(n, axis));
                    float3 t1 = normalizesafe(cross(n, t0));
                    float3x2 T = new float3x2(t0, t1);
                    float2 u = new float2(dot(t0, dx), dot(t1, dx));
                    ClothCollision.Friction(col.FrictionDynamic, lambda, T, u, col.FrictionEpsV * dt,
                        out float3 ff, out float3x3 fh);
                    force += ff;
                    hessian += fh;
                }
            }
        }

        static float3x3 Outer(float3 a, float3 b)
        {
            return new float3x3(
                a.x * b.x, a.x * b.y, a.x * b.z,
                a.y * b.x, a.y * b.y, a.y * b.z,
                a.z * b.x, a.z * b.y, a.z * b.z);
        }
    }

    /// <summary>
    /// 颜色组算完后把 PositionsNext 拷回 Positions(保证 GS 语义)。
    /// 与 VBDStepJob 同样按组 dispatch。
    /// </summary>
    [BurstCompile]
    public struct CopyBackJob : IJobParallelFor
    {
        public int ColorStart;
        [ReadOnly] public NativeArray<int> ColorVertices;
        [ReadOnly] public NativeArray<bool> FixedMask;
        [ReadOnly] public NativeArray<float3> PositionsNext;
        [NativeDisableParallelForRestriction] public NativeArray<float3> Positions;

        public void Execute(int i)
        {
            int v = ColorVertices[ColorStart + i];
            if (FixedMask[v]) return;
            Positions[v] = PositionsNext[v];
        }
    }
}

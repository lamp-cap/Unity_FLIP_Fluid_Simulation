using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD.Cloth
{
    /// <summary>
    /// VBD 布料求解器核心。对应参考实现 VBDClothPhysics.cpp:
    ///   runStep / runStep_CDEveryIter   -> Step
    ///   applyInitialGuess               -> ApplyInitialGuess
    ///   VBDStepWithExistingCollisions   -> VBDStepVertex
    ///   accumlateInertiaForceAndHessian -> (内联于 VBDStepVertex)
    ///   accumlateMaterialForceAndHessian-> AccumulateMaterial (StVK + Bending)
    ///   accumlateBoundaryForceAndHessian-> AccumulateBoundary (世界包围盒, 可选)
    ///   updateVelocity                  -> UpdateVelocity
    ///
    /// 我们只模拟布料:所有网格展平为一组全局顶点/面/边(无 iMesh)。
    /// 按图着色分组做 Gauss-Seidel:同组顶点并行写 PositionsNext,整组算完统一拷回 Positions。
    ///
    /// 第一阶段:求解循环在主线程串行驱动(组间串行,组内可并行)。
    /// 数据布局已 Job 友好,后续可把 VBDStepVertex 封装成 IJobParallelFor。
    /// </summary>
    public class ClothSolverParameters
    {
        public float dt = 1f / 60f;
        public int numSubsteps = 1;
        public int iterations = 20;
        public float3 gravity = new float3(0, -9.8f, 0);
        public float stepSize = 1f;

        // 材料(StVK)
        public float miu = 1e4f;
        public float lambda = 1e4f;
        public float bendingStiffness = 1e-3f;
        public float dampingStVK = 0f;
        public float dampingBending = 0f;

        // 速度阻尼
        public float exponentialVelDamping = 0.999f;
        public float constantVelDamping = 0f;

        // 碰撞
        public bool handleCollision = true;
        public float thickness = 0.01f;          // 布料厚度
        public float contactRadius = 0.02f;       // 接触作用半径
        public float contactStiffness = 1e5f;
        public float queryRadius = 0.05f;         // 检测半径(maxQueryDis)
        public float conservativeStepRelaxation = 0.8f;
        public bool applyFriction = true;
        public float frictionMu = 0.3f;
        public float frictionEpsV = 1e-3f;

        public bool degenerateTriangleThresEnabled = true;
        public float degenerateTriangleThres = 1e-7f;

        // 热路径 Burst 化开关。true: 逐顶点 VBD 步用 [BurstCompile] IJobParallelFor;
        // false: 走主线程标量路径(便于断点调试)。Burst 也可在 Unity 菜单全局关闭。
        public bool useBurst = true;
    }

    public class ClothSolver
    {
        public ClothTopology Topo;
        public ClothState State;
        public ClothCollision Collision;
        public ColliderSet Colliders;
        public ClothSolverParameters P;

        private bool _collisionDetectionRequired = true;

        // Burst 路径用:Job 字段的 Native 容器必须 IsCreated。
        // 无碰撞/无碰撞体时用这些空容器占位。截断标志按"最大颜色组大小"分配。
        private NativeParallelMultiHashMap<int, ClothCollision.VFContact> _emptyVFMap;
        private NativeParallelMultiHashMap<int, ClothCollision.EEContact> _emptyEEMap;
        private NativeArray<AnalyticCollider> _emptyColliders;
        private NativeArray<byte> _truncFlags;
        private bool _burstAllocated;

        public void Initialize(ClothTopology topo, ClothState state, ClothSolverParameters p,
            ColliderSet colliders = null, ClothCollision collision = null)
        {
            Topo = topo;
            State = state;
            P = p;
            Colliders = colliders;
            Collision = collision;
            _collisionDetectionRequired = true;
            AllocateBurstScratch();
        }

        void AllocateBurstScratch()
        {
            DisposeBurstScratch();
            _emptyVFMap = new NativeParallelMultiHashMap<int, ClothCollision.VFContact>(1, Allocator.Persistent);
            _emptyEEMap = new NativeParallelMultiHashMap<int, ClothCollision.EEContact>(1, Allocator.Persistent);
            _emptyColliders = new NativeArray<AnalyticCollider>(0, Allocator.Persistent);

            // 最大颜色组大小
            int maxGroup = 1;
            for (int c = 0; c < Topo.NumColors; c++)
                maxGroup = max(maxGroup, Topo.ColorStart[c + 1] - Topo.ColorStart[c]);
            _truncFlags = new NativeArray<byte>(maxGroup, Allocator.Persistent);
            _burstAllocated = true;
        }

        void DisposeBurstScratch()
        {
            if (!_burstAllocated) return;
            if (_emptyVFMap.IsCreated) _emptyVFMap.Dispose();
            if (_emptyEEMap.IsCreated) _emptyEEMap.Dispose();
            if (_emptyColliders.IsCreated) _emptyColliders.Dispose();
            if (_truncFlags.IsCreated) _truncFlags.Dispose();
            _burstAllocated = false;
        }

        public void Dispose()
        {
            DisposeBurstScratch();
        }

        // ===============================================================
        // runStep -> runStep_CDEveryIter
        // ===============================================================
        public void Step()
        {
            for (int substep = 0; substep < P.numSubsteps; substep++)
            {
                StepOnce(P.dt / P.numSubsteps);
            }
        }

        void StepOnce(float dt)
        {
            float dtSub = dt;

            for (int iter = 0; iter < P.iterations; iter++)
            {
                if (iter == 0)
                {
                    ApplyInitialGuess(dtSub);
                    // 帧首激活掩码清零(initializeActiveCollisionMask)
                    for (int v = 0; v < State.NumVertices; v++) State.ActiveCollisionMask[v] = false;
                }

                // 碰撞检测:首次迭代或被标记需要时(对应 collisionDetectionRequired || iIter==0)
                if (P.handleCollision && Collision != null && (_collisionDetectionRequired || iter == 0))
                {
                    bool hasCols = Colliders != null && Colliders.Count > 0;
                    var cols = hasCols ? Colliders.Colliders : _emptyColliders;
                    int colCount = hasCols ? Colliders.Count : 0;
                    Collision.DetectAndUpdateBounds(Topo, State, P.queryRadius, P.thickness, P.conservativeStepRelaxation,
                        cols, colCount);
                    _collisionDetectionRequired = false;
                }
                else if (!P.handleCollision)
                {
                    for (int v = 0; v < State.NumVertices; v++)
                        State.ConservativeBounds[v] = float.MaxValue;
                }

                // 按颜色分组做 Gauss-Seidel
                if (P.useBurst)
                    SolveGroupsBurst(dtSub);
                else
                    SolveGroupsScalar(dtSub);
            }

            UpdateVelocity(dtSub);
        }

        // ---- 标量路径:主线程逐组逐顶点(便于断点调试) ----
        void SolveGroupsScalar(float dtSub)
        {
            for (int c = 0; c < Topo.NumColors; c++)
            {
                int start = Topo.ColorStart[c];
                int end = Topo.ColorStart[c + 1];

                for (int idx = start; idx < end; idx++)
                {
                    int v = Topo.ColorVertices[idx];
                    if (Topo.FixedMask[v]) continue;
                    VBDStepVertex(v, dtSub);
                }
                for (int idx = start; idx < end; idx++)
                {
                    int v = Topo.ColorVertices[idx];
                    if (Topo.FixedMask[v]) continue;
                    State.Positions[v] = State.PositionsNext[v];
                }
            }
        }

        // ---- Burst 路径:每个颜色组一个 IJobParallelFor(组内并行,组间串行,保持 GS 语义) ----
        void SolveGroupsBurst(float dtSub)
        {
            bool hasCollision = P.handleCollision && Collision != null;
            bool hasColliders = Colliders != null && Colliders.Count > 0;

            var sp = new VBDSolverParams
            {
                dt = dtSub,
                stepSize = P.stepSize,
                miu = P.miu,
                lambda = P.lambda,
                bendingStiffness = P.bendingStiffness,
                dampingStVK = P.dampingStVK,
                dampingBending = P.dampingBending,
                handleCollision = (byte)(hasCollision ? 1 : 0),
                thickness = P.thickness,
                contactRadius = P.contactRadius,
                contactStiffness = P.contactStiffness,
                queryRadius = P.queryRadius,
                applyFriction = (byte)(P.applyFriction ? 1 : 0),
                frictionMu = P.frictionMu,
                frictionEpsV = P.frictionEpsV,
                degenerateTriangleThresEnabled = (byte)(P.degenerateTriangleThresEnabled ? 1 : 0),
                degenerateTriangleThres = P.degenerateTriangleThres,
                hasColliders = (byte)(hasColliders ? 1 : 0),
            };

            var v2vf = hasCollision ? Collision.VertexToVF : _emptyVFMap;
            var v2ee = hasCollision ? Collision.VertexToEE : _emptyEEMap;
            var cols = hasColliders ? Colliders.Colliders : _emptyColliders;

            for (int c = 0; c < Topo.NumColors; c++)
            {
                int start = Topo.ColorStart[c];
                int count = Topo.ColorStart[c + 1] - start;
                if (count <= 0) continue;

                var stepJob = new VBDStepJob
                {
                    P = sp,
                    ColorStart = start,
                    ColorVertices = Topo.ColorVertices,
                    FaceVerts = Topo.FaceVerts,
                    DmInv = Topo.DmInv,
                    FaceRestArea = Topo.FaceRestArea,
                    VertexMass = Topo.VertexMass,
                    FixedMask = Topo.FixedMask,
                    VertexFaceStart = Topo.VertexFaceStart,
                    VertexFaceList = Topo.VertexFaceList,
                    VertexEdgeStart = Topo.VertexEdgeStart,
                    VertexEdgeList = Topo.VertexEdgeList,
                    Edges = Topo.Edges,
                    EdgeQ = Topo.EdgeQ,
                    Positions = State.Positions,
                    PositionsPrev = State.PositionsPrev,
                    Inertia = State.Inertia,
                    ConservativeBounds = State.ConservativeBounds,
                    PositionsAtPrevCD = State.PositionsAtPrevCD,
                    PositionsNext = State.PositionsNext,
                    TruncFlags = _truncFlags,
                    VertexToVF = v2vf,
                    VertexToEE = v2ee,
                    Colliders = cols,
                };
                stepJob.Schedule(count, 64).Complete();

                var copyJob = new CopyBackJob
                {
                    ColorStart = start,
                    ColorVertices = Topo.ColorVertices,
                    FixedMask = Topo.FixedMask,
                    PositionsNext = State.PositionsNext,
                    Positions = State.Positions,
                };
                copyJob.Schedule(count, 64).Complete();

                // 归约截断标志:任一截断 -> 下次迭代重检测
                if (hasCollision)
                {
                    for (int i = 0; i < count; i++)
                        if (_truncFlags[i] != 0) { _collisionDetectionRequired = true; break; }
                }
            }
        }

        // ===============================================================
        // applyInitialGuess + evaluateExternalForce
        // 惯性初值: x = x_prev + dt*v + dt^2 * g  (惯性目标 y 同值)
        // ===============================================================
        void ApplyInitialGuess(float dt)
        {
            float3 g = P.gravity;
            for (int v = 0; v < State.NumVertices; v++)
            {
                State.PositionsPrev[v] = State.Positions[v];

                if (Topo.FixedMask[v])
                {
                    State.Inertia[v] = State.Positions[v];
                    State.ExternalForces[v] = float3(0,0,0);
                    State.PositionsNext[v] = State.Positions[v];
                    continue;
                }

                float3 xPrev = State.Positions[v];
                float3 vel = State.Velocities[v];
                float3 y = xPrev + dt * vel + dt * dt * g;  // 惯性目标
                State.Inertia[v] = y;
                State.Positions[v] = y;                      // 初值 = 惯性目标
                State.PositionsNext[v] = y;
                // 外力此处只有重力(已并入惯性项),保留接口
                State.ExternalForces[v] = float3(0,0,0);
            }
        }

        // ===============================================================
        // VBDStepWithExistingCollisions(vId)
        // 累加 力/Hessian: 接触(已检测) + 惯性 + StVK + 弯曲 + 边界
        // 解 H dx = f,截断到保守边界,写 PositionsNext
        // ===============================================================
        void VBDStepVertex(int v, float dt)
        {
            float3 force = float3(0,0,0);
            float3x3 hessian = new float3x3();

            // ---- 接触力(复用已检测结果) ----
            if (P.handleCollision && Collision != null)
            {
                AccumulateContacts(v, dt, ref force, ref hessian);
            }

            // ---- 惯性项: m/dt^2 * (y - x) ----
            float m = Topo.VertexMass[v];
            float invDt2 = 1f / (dt * dt);
            float3 xi = State.Positions[v];
            force += (m * invDt2) * (State.Inertia[v] - xi);
            hessian.c0.x += m * invDt2;
            hessian.c1.y += m * invDt2;
            hessian.c2.z += m * invDt2;

            // ---- 材料力(StVK 膜 + 弯曲) ----
            AccumulateMaterial(v, dt, ref force, ref hessian);

            // ---- 解析碰撞体(胶囊/球/平面) ----
            if (Colliders != null && Colliders.Count > 0)
            {
                AccumulateAnalyticColliders(v, dt, ref force, ref hessian);
            }

            // ---- 解 H dx = f ----
            if (!ClothMath.Solve3x3PSD(hessian, force, out float3 dx))
            {
                State.PositionsNext[v] = State.Positions[v];
                return;
            }

            float3 newPos = State.Positions[v] + P.stepSize * dx;

            // 保守边界截断 -> 触发下次迭代重检测
            if (P.handleCollision && Collision != null)
            {
                bool truncated = ClothCollision.ApplyConservativeBoundTruncation(State, v, ref newPos);
                if (truncated) _collisionDetectionRequired = true;
            }

            State.PositionsNext[v] = newPos;
        }

        // accumlateMaterialForceAndHessian = StVK + Bending
        void AccumulateMaterial(int v, float dt, ref float3 force, ref float3x3 hessian)
        {
            float3x3 hBefore = hessian;

            // ---- StVK 膜能 (accumlateStVKForceAndHessian) ----
            int fStart = Topo.VertexFaceStart[v];
            int fEnd = Topo.VertexFaceStart[v + 1];
            for (int i = fStart; i < fEnd; i++)
            {
                int2 fo = Topo.VertexFaceList[i];
                int f = fo.x;
                int order = fo.y;
                int3 fv = Topo.FaceVerts[f];
                ClothMath.AccumulateStVKFace(
                    State.Positions[fv.x], State.Positions[fv.y], State.Positions[fv.z],
                    Topo.DmInv[f], Topo.FaceRestArea[f],
                    P.lambda, P.miu, order,
                    ref force, ref hessian);
            }

            // StVK 阻尼: dampingH = (H - H0) * damping/dt; force -= dampingH*(x - xPrev); H += dampingH
            if (P.dampingStVK > 0f)
            {
                float3x3 dampingH = (hessian - hBefore) * (P.dampingStVK / dt);
                float3 disp = State.Positions[v] - State.PositionsPrev[v];
                force -= mul(dampingH, disp);
                hessian += dampingH;
            }

            // ---- 弯曲能 (accumlateBendingForceAndHessian) ----
            AccumulateBending(v, dt, ref force, ref hessian);
        }

        // accumlateBendingForceAndHessian
        void AccumulateBending(int v, float dt, ref float3 force, ref float3x3 hessian)
        {
            float ks = P.bendingStiffness;
            if (ks <= 0f) return;
            float damping = P.dampingBending;

            int eStart = Topo.VertexEdgeStart[v];
            int eEnd = Topo.VertexEdgeStart[v + 1];
            for (int i = eStart; i < eEnd; i++)
            {
                int2 eo = Topo.VertexEdgeList[i];
                int e = eo.x;
                int order = eo.y; // 0..3 对应 [eV1, eV2, eV12Next, eV21Next]
                var ei = Topo.Edges[e];
                if (ei.fId2 == -1) continue; // boundary

                float4x4 Q = Topo.EdgeQ[e];

                // Xs 行: x0..x3
                float3 x0 = State.Positions[ei.eV1];
                float3 x1 = State.Positions[ei.eV2];
                float3 x2 = State.Positions[ei.eV12Next];
                float3 x3 = State.Positions[ei.eV21Next];

                // 退化三角形跳过(对应 degenerateTriangleThres)
                if (P.degenerateTriangleThresEnabled)
                {
                    float3 n1 = cross(x1 - x0, x2 - x0);
                    float3 n2 = cross(x2 - x3, x1 - x3);
                    float thr = P.degenerateTriangleThres * P.degenerateTriangleThres;
                    if (lengthsq(n1) < thr || lengthsq(n2) < thr) continue;
                }

                // dE_dXs.row(order) = ks * (Q * Xs).row(order)
                //   (Q*Xs) row(order) = sum_j Q(order,j) * Xs.row(j)
                // Q[col][row]: 行 order 的元素 Q(order, 0..3) = Q[0][order], Q[1][order], Q[2][order], Q[3][order]
                float qr0 = Q[0][order];
                float qr1 = Q[1][order];
                float qr2 = Q[2][order];
                float qr3 = Q[3][order];
                float3 dE_row = ks * (qr0 * x0 + qr1 * x1 + qr2 * x2 + qr3 * x3);

                float qDiag = Q[order][order];
                float3x3 hTemp = (ks * qDiag) * new float3x3(1,0,0,0,1,0,0,0,1);

                // force -= dE_row + hTemp*(x_v - xPrev_v)*(damping/dt)
                float3 dispV = State.Positions[v] - State.PositionsPrev[v];
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

        // 复用已检测的接触结果累加到顶点 v
        void AccumulateContacts(int v, float dt, ref float3 force, ref float3x3 hessian)
        {
            // VF:从顶点 key 直接取接触结构体
            if (Collision.VertexToVF.TryGetFirstValue(v, out var c, out var it))
            {
                do
                {
                    int order = VFOrder(c, v);
                    Collision.AccumulateVFContact(c, order, Topo, State,
                        P.thickness, P.contactRadius, P.contactStiffness,
                        P.applyFriction, P.frictionMu, P.frictionEpsV, dt,
                        ref force, ref hessian);
                }
                while (Collision.VertexToVF.TryGetNextValue(out c, ref it));
            }

            // EE
            if (Collision.VertexToEE.TryGetFirstValue(v, out var ce, out var it2))
            {
                do
                {
                    int order = EEOrder(ce, v);
                    if (order < 0) continue;
                    Collision.AccumulateEEContact(ce, order, Topo, State,
                        P.thickness, P.contactRadius, P.contactStiffness,
                        P.applyFriction, P.frictionMu, P.frictionEpsV, dt,
                        ref force, ref hessian);
                }
                while (Collision.VertexToEE.TryGetNextValue(out ce, ref it2));
            }
        }

        int VFOrder(in ClothCollision.VFContact c, int v)
        {
            if (c.v == v) return 3; // V 侧
            int3 fv = Topo.FaceVerts[c.face];
            if (fv.x == v) return 0;
            if (fv.y == v) return 1;
            return 2;
        }

        int EEOrder(in ClothCollision.EEContact c, int v)
        {
            var e1 = Topo.Edges[c.e1];
            var e2 = Topo.Edges[c.e2];
            if (e1.eV1 == v) return 0;
            if (e1.eV2 == v) return 1;
            if (e2.eV1 == v) return 2;
            if (e2.eV2 == v) return 3;
            return -1;
        }

        // 解析碰撞体:排斥 penalty + 摩擦
        void AccumulateAnalyticColliders(int v, float dt, ref float3 force, ref float3x3 hessian)
        {
            float3 p = State.Positions[v];
            float k = P.contactStiffness;
            float r = P.thickness; // 布料半厚作为接触偏移

            for (int i = 0; i < Colliders.Count; i++)
            {
                var col = Colliders.Colliders[i];
                col.Query(p, out float3 closest, out float3 n, out float dist);

                float penetration = r - dist; // dist<r 时为正(需要排斥)
                if (penetration > 0f)
                {
                    float lambda = k * penetration;
                    force += lambda * n;
                    hessian += k * ClothMathOuter(n, n);

                    if (P.applyFriction)
                    {
                        float3 dx = p - State.PositionsPrev[v];
                        // 切空间基底
                        float3 t0 = normalizesafe(OrthoVector(n));
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
        }

        static float3 OrthoVector(float3 n)
        {
            // 取与 n 不平行的轴叉乘
            float3 a = abs(n.x) < 0.9f ? new float3(1, 0, 0) : new float3(0, 1, 0);
            return cross(n, a);
        }

        static float3x3 ClothMathOuter(float3 a, float3 b)
        {
            return new float3x3(
                a.x * b.x, a.x * b.y, a.x * b.z,
                a.y * b.x, a.y * b.y, a.y * b.z,
                a.z * b.x, a.z * b.y, a.z * b.z);
        }

        // ===============================================================
        // updateVelocity: v = (x - xPrev)/dt + 阻尼
        // ===============================================================
        void UpdateVelocity(float dt)
        {
            float invDt = 1f / dt;
            for (int v = 0; v < State.NumVertices; v++)
            {
                if (Topo.FixedMask[v]) { State.Velocities[v] = float3(0,0,0); continue; }

                float3 vel = (State.Positions[v] - State.PositionsPrev[v]) * invDt;
                float vMag = length(vel);
                if (vMag > 1e-6f)
                {
                    float vNew = vMag * P.exponentialVelDamping - P.constantVelDamping;
                    vNew = vNew > 1e-6f ? vNew : 0f;
                    vel *= vNew / vMag;
                }
                else vel = float3(0,0,0);
                State.Velocities[v] = vel;
            }
        }
    }
}

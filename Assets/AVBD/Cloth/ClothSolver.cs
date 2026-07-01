using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Profiling;
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
                Profiler.BeginSample("Iteration");
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

                SolveGroupsBurst(dtSub);
                Profiler.EndSample();
            }

            UpdateVelocity(dtSub);
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

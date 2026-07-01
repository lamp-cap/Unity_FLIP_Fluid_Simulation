using Unity.Collections;
using Unity.Mathematics;

namespace AVBD.Cloth
{
    /// <summary>
    /// VBD 布料求解的运行时状态(SoA, Native 容器)。
    /// 对应参考实现 VBDBaseTriMesh 的运行时成员:
    ///   positions / positionsPrev / velocities / inertia / vertexExternalForces
    ///   positionsNext (vertexPositionNext) / positionPrevIter / positionPrevPrevIter
    ///   vertexConvervativeBounds / positionsAtPrevCollisionDetection / activeCollisionMask
    /// 这里没有 iMesh 维度,所有布料展平为一组全局顶点。
    /// </summary>
    public class ClothState
    {
        public int NumVertices;

        public NativeArray<float3> Positions;       // 当前 x
        public NativeArray<float3> PositionsPrev;    // 上一帧 x (用于 v = (x - xPrev)/dt)
        public NativeArray<float3> Velocities;       // v
        public NativeArray<float3> Inertia;          // 惯性目标 y = x + dt v + dt^2 g
        public NativeArray<float3> ExternalForces;   // 外力(重力等)
        public NativeArray<float3> PositionsNext;    // GS 写缓冲(vertexPositionNext)

        // Chebyshev 加速用
        public NativeArray<float3> PositionPrevIter;
        public NativeArray<float3> PositionPrevPrevIter;

        // 碰撞保守边界 / 安全距离
        public NativeArray<float> ConservativeBounds;            // vertexConvervativeBounds
        public NativeArray<float3> PositionsAtPrevCD;            // positionsAtPrevCollisionDetection
        public NativeArray<bool> ActiveCollisionMask;            // activeCollisionMask

        private bool _allocated;

        public void Allocate(int numVertices, NativeArray<float3> restPositions)
        {
            Dispose();
            NumVertices = numVertices;

            Positions = new NativeArray<float3>(numVertices, Allocator.Persistent);
            PositionsPrev = new NativeArray<float3>(numVertices, Allocator.Persistent);
            Velocities = new NativeArray<float3>(numVertices, Allocator.Persistent);
            Inertia = new NativeArray<float3>(numVertices, Allocator.Persistent);
            ExternalForces = new NativeArray<float3>(numVertices, Allocator.Persistent);
            PositionsNext = new NativeArray<float3>(numVertices, Allocator.Persistent);
            PositionPrevIter = new NativeArray<float3>(numVertices, Allocator.Persistent);
            PositionPrevPrevIter = new NativeArray<float3>(numVertices, Allocator.Persistent);
            ConservativeBounds = new NativeArray<float>(numVertices, Allocator.Persistent);
            PositionsAtPrevCD = new NativeArray<float3>(numVertices, Allocator.Persistent);
            ActiveCollisionMask = new NativeArray<bool>(numVertices, Allocator.Persistent);

            for (int i = 0; i < numVertices; i++)
            {
                float3 p = restPositions[i];
                Positions[i] = p;
                PositionsPrev[i] = p;
                PositionsNext[i] = p;
                PositionsAtPrevCD[i] = p;
            }
            _allocated = true;
        }

        public void Dispose()
        {
            if (!_allocated) return;
            if (Positions.IsCreated) Positions.Dispose();
            if (PositionsPrev.IsCreated) PositionsPrev.Dispose();
            if (Velocities.IsCreated) Velocities.Dispose();
            if (Inertia.IsCreated) Inertia.Dispose();
            if (ExternalForces.IsCreated) ExternalForces.Dispose();
            if (PositionsNext.IsCreated) PositionsNext.Dispose();
            if (PositionPrevIter.IsCreated) PositionPrevIter.Dispose();
            if (PositionPrevPrevIter.IsCreated) PositionPrevPrevIter.Dispose();
            if (ConservativeBounds.IsCreated) ConservativeBounds.Dispose();
            if (PositionsAtPrevCD.IsCreated) PositionsAtPrevCD.Dispose();
            if (ActiveCollisionMask.IsCreated) ActiveCollisionMask.Dispose();
            _allocated = false;
        }
    }
}

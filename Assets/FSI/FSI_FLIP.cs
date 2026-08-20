using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;

namespace FSI
{
    public class FSI_FLIP : MonoBehaviour
    {
        private const uint SOLID = 2;
        private const uint AIR = 1;
        private const uint FLUID = 0;
        private const uint INTERACTIVE_SOLID = 3;

        private const int NumBlocks = 3;

        private const float InvDeltaTime = 90f;
        private const float DeltaTime = 1.0f / InvDeltaTime;
        private const float CellSize = 0.5f;
        private const float InvCellSize = 1.0f / CellSize;

        private const float TargetDensity = 5;

        [Range(-10, 10)] public float gravity = -9;
        [Range(0, 1)] public float flipness = 0.95f;
        public Mesh mesh;
        public Material mat;
        private float _rs;
        

        public const int NumParticles = 256 * 128;
        public const int GridRes = 256;
        public const int NumGrid = GridRes * GridRes;
        private const int Band1 = 2;
        private const int Band2 = 3;

        private NativeArray<float2> _gridVelocity;
        private NativeArray<float2> _gridVelocityCopy;
        private NativeArray<float> _gridDensity;
        private NativeArray<float> _gridDivergence;
        private NativeArray<float> _gridPressure;
        private NativeArray<uint> _gridType;
        private NativeArray<float3> _gridLaplacian;
        private NativeArray<float> _gridSDF;
        private NativeArray<int> _start;
        private NativeArray<int> _end;
        
        private NativeArray<float4> _particlePos;
        private NativeArray<float2> _particleVelocity;
        private NativeArray<float4> _particlePosCopy;
        private NativeArray<float2> _particleVelocityCopy;
        private NativeArray<int> _particleID;
        private NativeArray<int2> _hashes;
        private NativeArray<int2> _range;

        private NativeReference<int> _particleCount;

        private ComputeBuffer _posBuffer;
        private Bounds _bounds;
        private FSI_UAAMGSolver _mgPressureSolver;

        private float2 _oldMousePos;
        private float2 _oldMouseVec;
        private Camera _camera;

        // Rigid-body interaction targets (solids coupled to the fluid).
        private NativeArray<SolidBlock> _blocks;
        private NativeArray<int> _gridBlockID;          // per cell: block index or -1
        private NativeArray<float> _blockJx;            // NumBlocks * NumGrid: coupling Jacobian (orientation, ±1)
        private NativeArray<float> _blockJy;
        private NativeArray<float2> _blockForces;       // NumBlocks: (Sx, Sy) = sum(jx*p, jy*p)
        private NativeArray<float> _blockTorques;       // NumBlocks: torque = sum(r × F)
        private NativeArray<float> _blockCoeff;         // NumBlocks: DeltaTime / mass
        private NativeArray<float> _blockAngularCoeff;  // NumBlocks: DeltaTime / inertia

        private struct SolidBlock
        {
            public float2 Center;       // world units
            public float2 Velocity;     // world units / second
            public float Angle;         // radians
            public float AngularVelocity; // radians / second
            public float Mass;
            public float HalfExtent;    // half side length, world units
            public float Inertia;       // moment of inertia = mass * (2 * halfExtent)^2 / 12 for square
        }

        void Start()
        {
            _camera = Camera.main;
            _gridVelocity = new NativeArray<float2>(NumGrid, Allocator.Persistent);
            _gridVelocityCopy = new NativeArray<float2>(NumGrid, Allocator.Persistent);
            _gridDivergence = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridPressure = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridType = new NativeArray<uint>(NumGrid, Allocator.Persistent);
            _start = new NativeArray<int>(NumGrid, Allocator.Persistent);
            _end = new NativeArray<int>(NumGrid, Allocator.Persistent);
            _range = new NativeArray<int2>(NumGrid, Allocator.Persistent);
            _gridDensity = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridLaplacian = new NativeArray<float3>(NumGrid, Allocator.Persistent);
            _gridSDF = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _particleID = new NativeArray<int>(NumParticles, Allocator.Persistent);
            _particlePos = new NativeArray<float4>(NumParticles, Allocator.Persistent);
            _particleVelocity = new NativeArray<float2>(NumParticles, Allocator.Persistent);
            _particlePosCopy = new NativeArray<float4>(NumParticles, Allocator.Persistent);
            _particleVelocityCopy = new NativeArray<float2>(NumParticles, Allocator.Persistent);
            _hashes = new NativeArray<int2>(NumParticles, Allocator.Persistent);
            _particleCount = new NativeReference<int>(Allocator.Persistent);

            _blocks = new NativeArray<SolidBlock>(NumBlocks, Allocator.Persistent);
            _gridBlockID = new NativeArray<int>(NumGrid, Allocator.Persistent);
            _blockJx = new NativeArray<float>(NumBlocks * NumGrid, Allocator.Persistent);
            _blockJy = new NativeArray<float>(NumBlocks * NumGrid, Allocator.Persistent);
            _blockForces = new NativeArray<float2>(NumBlocks, Allocator.Persistent);
            _blockTorques = new NativeArray<float>(NumBlocks, Allocator.Persistent);
            _blockCoeff = new NativeArray<float>(NumBlocks, Allocator.Persistent);
            _blockAngularCoeff = new NativeArray<float>(NumBlocks, Allocator.Persistent);

            // Three rigid squares above the fluid pool. The pressure solve works in
            // projection units, so the neutral-buoyancy mass is
            //   m_neutral = (width*height in cells) * DeltaTime * CellSize^2
            // (e.g. 12x12 -> 144 * 1/120 * 0.25 ~= 0.3). We use ~2x neutral so the
            // blocks sink gently; raise mass to sink faster, lower it to float.
            _blocks[0] = new SolidBlock { Center = new float2(50, 200) * CellSize, HalfExtent = 6 * CellSize, Mass = 0.1f };   // 12x12
            _blocks[1] = new SolidBlock { Center = new float2(100, 215) * CellSize, HalfExtent = 5 * CellSize, Mass = 0.2f };  // 10x10
            _blocks[2] = new SolidBlock { Center = new float2(150, 230) * CellSize, HalfExtent = 7 * CellSize, Mass = 0.15f };  // 14x14
            for (int b = 0; b < NumBlocks; b++)
            {
                var block = _blocks[b];
                float sideLength = 2 * block.HalfExtent;
                // Moment of inertia for square: I = m * L^2 / 12
                block.Inertia = block.Mass * sideLength * sideLength / 12f;
                block.Angle = 0;
                block.AngularVelocity = 0;
                _blocks[b] = block;

                _blockCoeff[b] = DeltaTime / block.Mass;
                _blockAngularCoeff[b] = DeltaTime / block.Inertia;
            }

            _mgPressureSolver = new FSI_UAAMGSolver(_gridLaplacian, _gridPressure, _gridDivergence, GridRes, CellSize,
                _blockJx, _blockJy, _blockCoeff, NumBlocks);

            // for (int y = 0; y < GridRes; y++)
            // for (int x = 0; x < GridRes; x++)
            // {
            //     int id = y * GridRes + x;
            //     float2 pos = new float2(x, y) * 0.5f + new float2(120.25f, 1.25f);
            //     _particlePos[id] = new float4(pos * CellSize, 0, 1);
            // }
            int2 start = new int2(0, 0);
            int2 end = new int2(200, 150);
            for (int y = 0; y < GridRes; y++)
            for (int x = 0; x < GridRes; x++)
            {
                int2 coord = new int2(x, y);
                int level = -1;
                if (math.all(coord >= start) && math.all(coord <= end))
                {
                    level = GridRes;
                    if (math.any(coord == start) || math.any(coord == end)) 
                        level = 0;
                }
                _gridSDF[y * GridRes + x] = level;
            }
            
            new ComputeDistanceFieldJob(_gridSDF).Run();
            new ParticlesCounterInitJob(_gridSDF, _range, _particleID, _particleCount).Run();
            new ResampleParticlesJob()
            {
                PosRaw = _particlePosCopy,
                VelRaw = _particleVelocityCopy,
                PosNew = _particlePos,
                VelNew = _particleVelocity,
                Ids = _particleID,
                GridVelocity = _gridVelocityCopy,
                Ranges = _range,
            }.Schedule(NumGrid, 32).Complete();

            new MarkSolidBlocksJob
            {
                Blocks = _blocks,
                GridBlockID = _gridBlockID,
            }.Schedule(NumGrid, 32).Complete();

            new SetGridTypeJob
            {
                Range = _range,
                GridType = _gridType,
                GridLevel = _gridSDF,
                GridBlockID = _gridBlockID,
            }.Schedule(NumGrid, 32).Complete();

            _posBuffer = new ComputeBuffer(NumParticles, sizeof(float) * 4);
            mat.SetBuffer("_ParticleBuffer", _posBuffer);
            float cellSize = 0.1f * CellSize;
            _bounds = new Bounds()
            {
                min = new Vector3(0, 0, 0),
                max = new Vector3(GridRes * cellSize, GridRes * cellSize, 0.01f)
            };

            _labelStyle = new GUIStyle()
            {
                alignment = TextAnchor.UpperLeft,
                fontSize = 32,
                normal = { textColor = Color.white }
            };
            
            Debug.Log($"MGPCG_FLIP initialized, particle num: {NumParticles}, grid res: {GridRes}x{GridRes}.");
            
            Test();
        }

        void Update()
        {
            Simulate();
            _posBuffer.SetData(_particlePos);
            Graphics.DrawMeshInstancedProcedural(mesh, 0, mat, _bounds, _particleCount.Value);
        }

        private void OnDestroy()
        {
            _gridVelocity.Dispose();
            _gridVelocityCopy.Dispose();
            _gridDivergence.Dispose();
            _gridPressure.Dispose();
            _gridType.Dispose();
            _start.Dispose();
            _end.Dispose();
            _range.Dispose();
            _gridDensity.Dispose();
            _gridLaplacian.Dispose();
            _gridSDF.Dispose();
            
            _particleID.Dispose();
            _particlePos.Dispose();
            _particleVelocity.Dispose();
            _particlePosCopy.Dispose();
            _particleVelocityCopy.Dispose();
            _hashes.Dispose();
            _posBuffer.Dispose();
            _mgPressureSolver.Dispose();

            _particleCount.Dispose();

            _blocks.Dispose();
            _gridBlockID.Dispose();
            _blockJx.Dispose();
            _blockJy.Dispose();
            _blockForces.Dispose();
            _blockTorques.Dispose();
            _blockCoeff.Dispose();
            _blockAngularCoeff.Dispose();
        }

        private GUIStyle _labelStyle;

        private void OnGUI()
        {
            GUI.Label(new Rect(0, 0, 100, 36), 
                $"mouse pos: {_oldMousePos.x:F1}, {_oldMousePos.y:F1}", _labelStyle);
            GUI.Label(new Rect(0, 36, 100, 36), 
                $"mouse vec: {_oldMouseVec.x:F2}, {_oldMouseVec.y:F2}", _labelStyle);
            GUI.Label(new Rect(0, 108, 100, 36), 
                $"residual: {_rs:F3}", _labelStyle);
            GUI.Label(new Rect(0, 144, 100, 36), 
                $"particles: {_particleCount.Value} / {NumParticles}", _labelStyle);
        }

        private void Simulate()
        {
            int batchCount = 32;

            Profiler.BeginSample("Clear Grid");
            new ClearGridJob()
            {
                Start = _start,
                End = _end,
                Range = _range,
                Pressure = _gridPressure,
                Density = _gridDensity,
            }.Schedule(NumGrid, batchCount).Complete();
            Profiler.EndSample();
            
            Profiler.BeginSample("Build Lut");
            new HashJob
            {
                Ps = _particlePos,
                Hashes = _hashes,
            }.Schedule(_particleCount.Value, batchCount).Complete();

            Profiler.BeginSample("Sort");
            _hashes.Slice(0, _particleCount.Value).SortJob(new Int2Comparer()).Schedule().Complete();
            Profiler.EndSample();
        
            new BuildLutJob
            {
                Hashes = _hashes,
                StartIndices = _start,
                EndIndices = _end
            }.Schedule(_particleCount.Value, batchCount).Complete();
        
            new CombineLutJob
            {
                StartIndices = _start,
                EndIndices = _end,
                Range = _range,
            }.Schedule(NumGrid, batchCount).Complete();

            new ShuffleJob
            {
                Hashes = _hashes,
                PosRaw = _particlePos,
                VelRaw = _particleVelocity,
                PosNew = _particlePosCopy,
                VelNew = _particleVelocityCopy
            }.Schedule(_particleCount.Value, batchCount).Complete();

            (_particlePos, _particlePosCopy) = (_particlePosCopy, _particlePos);
            (_particleVelocity, _particleVelocityCopy) = (_particleVelocityCopy, _particleVelocity);

            Profiler.EndSample();

            Profiler.BeginSample("Resample");

            // NOTE: MarkSolidBlocksJob is NOT called here because solid blocks haven't moved yet.
            // It will be called after UpdateBlocksJob updates their positions.

            new SetGridTypeJob
            {
                Range = _range,
                GridType = _gridType,
                GridLevel = _gridSDF,
                GridBlockID = _gridBlockID,
            }.Schedule(NumGrid, batchCount).Complete();
            
            new SetGridLevelJob(_gridSDF, _gridType).Schedule(NumGrid, batchCount).Complete();
            
            new ComputeDistanceFieldJob(_gridSDF).Run();
            
            new ParticlesCounterJob(_gridSDF, _range, _particleID, _particleCount).Run();
            
            new ResampleParticlesJob()
            {
                PosRaw = _particlePos,
                VelRaw = _particleVelocity,
                PosNew = _particlePosCopy,
                VelNew = _particleVelocityCopy,
                Ids = _particleID,
                GridVelocity = _gridVelocityCopy,
                Ranges = _range,
            }.Schedule(NumGrid, batchCount).Complete();

            (_particlePos, _particlePosCopy) = (_particlePosCopy, _particlePos);
            (_particleVelocity, _particleVelocityCopy) = (_particleVelocityCopy, _particleVelocity);
            
            Profiler.EndSample();
        
            Profiler.BeginSample("P2G");

            new ComputeLaplacianJob
            {
                GridTypes = _gridType,
                GridLaplacian = _gridLaplacian,
            }.Schedule(NumGrid, batchCount).Complete();
        
            new ParticleToGridJob
            {
                ParticlePos = _particlePos,
                ParticleVel = _particleVelocity,
                Range = _range,
                GridVelocity = _gridVelocityCopy,
                GridDensity = _gridDensity,
                GridLevel = _gridSDF
            }.Schedule(NumGrid, batchCount).Complete();
            
            _gridVelocity.CopyFrom(_gridVelocityCopy);

            var mouseVec = float2.zero;
            if (_camera)
            {
                // var mouseRay = _camera.ScreenPointToRay(Input.mousePosition);
                // if (_bounds.IntersectRay(mouseRay, out var dst))
                // {
                //     var hitPos = mouseRay.GetPoint(dst);
                //     float2 hitCoord = new float2((hitPos.x - _bounds.min.x) / _bounds.size.x,
                //         (hitPos.y - _bounds.min.y) / _bounds.size.y) * GridRes;
                //     
                //     if (math.any(_oldMousePos > 0))
                //         mouseVec = math.normalizesafe(hitCoord - _oldMousePos);
                //     
                //     _oldMousePos = hitCoord;
                // }
                // else
                {
                    _oldMousePos = new float2(-100, -100);
                }
            }
            _oldMouseVec = mouseVec;

            new AddForceJob
            {
                GridVelocity = _gridVelocity,
                GridTypes = _gridType,
                Gravity = new float2(0, gravity),
                MousePos = _oldMousePos,
                MouseVec = mouseVec,
            }.Schedule(_gridVelocity.Length, batchCount).Complete();
            Profiler.EndSample();

            Profiler.BeginSample("Solve Pressure");
            // Transfer rigid-body velocity onto faces bordering interactive solids.
            new TransferSolidVelocityJob
            {
                Blocks = _blocks,
                GridTypes = _gridType,
                GridBlockID = _gridBlockID,
                GridVelocity = _gridVelocity,
            }.Schedule(NumGrid, batchCount).Complete();

            new CalcDivergenceJob
            {
                GridVelocity = _gridVelocity,
                GridTypes = _gridType,
                GridDensity = _gridDensity,
                GridDivergence = _gridDivergence,
            }.Schedule(_gridVelocity.Length, batchCount).Complete();

            // Coupling Jacobian for the implicit FSI term in the CG iteration.
            new ComputeCouplingJacobianJob
            {
                GridTypes = _gridType,
                GridBlockID = _gridBlockID,
                BlockJx = _blockJx,
                BlockJy = _blockJy,
            }.Schedule(NumGrid, batchCount).Complete();

            _mgPressureSolver.Solve_MGPCG(4, out _rs);

            new UpdateVelocity
            {
                GridTypes = _gridType,
                GridPressure = _gridPressure,
                GridVelocity = _gridVelocity,
            }.Schedule(NumGrid, batchCount).Complete();

            // ❌ REMOVED: Re-applying solid velocity here would overwrite the pressure projection
            // result and destroy the implicit FSI coupling computed by MGPCG. The implicit
            // solver already ensures fluid velocity at solid boundaries satisfies the coupling
            // constraints, so this second transfer is both unnecessary and harmful.
            //
            // new TransferSolidVelocityJob
            // {
            //     Blocks = _blocks,
            //     GridTypes = _gridType,
            //     GridBlockID = _gridBlockID,
            //     GridVelocity = _gridVelocity,
            // }.Schedule(NumGrid, batchCount).Complete();

            // Accumulate pressure force on each block, then integrate the rigid bodies.
            new ComputeBlockForcesJob
            {
                Blocks = _blocks,
                Pressure = _gridPressure,
                BlockJx = _blockJx,
                BlockJy = _blockJy,
                BlockForces = _blockForces,
                BlockTorques = _blockTorques,
            }.Run();

            new UpdateBlocksJob
            {
                Blocks = _blocks,
                BlockForces = _blockForces,
                BlockTorques = _blockTorques,
                BlockCoeff = _blockCoeff,
                BlockAngularCoeff = _blockAngularCoeff,
                Gravity = gravity,
            }.Run();

            // Update grid cells occupied by solid blocks after their positions changed
            new MarkSolidBlocksJob
            {
                Blocks = _blocks,
                GridBlockID = _gridBlockID,
            }.Schedule(NumGrid, batchCount).Complete();
            Profiler.EndSample();
        
            Profiler.BeginSample("G2P");
            new GridToParticleJob
            {
                GridVelocityOld = _gridVelocityCopy,
                GridVelocityNew = _gridVelocity,
                ParticleVel = _particleVelocity,
                ParticlePos = _particlePos,
                Flipness = flipness,
            }.Schedule(_particleCount.Value, batchCount).Complete();
        
            Profiler.EndSample();
        
            Profiler.BeginSample("Advection");
            new ParticlesAdvectionJob
            {
                GridVelocity = _gridVelocity,
                ParticlePos = _particlePos,
                ParticleVel = _particleVelocity
            }.Schedule(_particleCount.Value, batchCount).Complete();
            
            new GridsAdvectionJob()
            {
                GridVelocity = _gridVelocity,
                GridVelocityAlt = _gridVelocityCopy,
                GridTypes = _gridType
            }.Schedule(NumGrid, batchCount).Complete();
            Profiler.EndSample();
            // (_gridVelocity, _gridVelocityCopy) = (_gridVelocityCopy, _gridVelocity);
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.white;
            Gizmos.DrawWireCube(_bounds.center, _bounds.size);
            // Gizmos.DrawLine(new Vector3(-3.2f, -3.1f, 0), new Vector3(3.1f, -3.1f, 0));
            // Gizmos.DrawLine(new Vector3(3.0f, -3.2f, 0), new Vector3(3.0f, 3.1f, 0));
            if (!Application.isPlaying) return;
            for (int b = 0; b < NumBlocks; b++)
            {
                var block = _blocks[b];
                Gizmos.color = new Color(1, 0.4f, 0.4f, 0.8f);

                // Draw rotated square by connecting 4 corners
                float he = block.HalfExtent;
                float c = math.cos(block.Angle);
                float s = math.sin(block.Angle);

                // 4 corners in local space: (-he, -he), (he, -he), (he, he), (-he, he)
                Vector3[] corners = new Vector3[4];
                float2[] localCorners = new float2[] {
                    new float2(-he, -he),
                    new float2(he, -he),
                    new float2(he, he),
                    new float2(-he, he)
                };

                for (int i = 0; i < 4; i++)
                {
                    float2 local = localCorners[i];
                    // Rotate and translate to world space
                    float2 world = block.Center + new float2(c * local.x - s * local.y, s * local.x + c * local.y);
                    corners[i] = new Vector3(world.x, world.y, 0) * 0.1f;
                }

                // Draw 4 edges
                Gizmos.DrawLine(corners[0], corners[1]);
                Gizmos.DrawLine(corners[1], corners[2]);
                Gizmos.DrawLine(corners[2], corners[3]);
                Gizmos.DrawLine(corners[3], corners[0]);

                // Draw a line indicating the orientation
                Gizmos.color = new Color(1, 1, 0, 0.8f);
                Vector3 center = new Vector3(block.Center.x, block.Center.y, 0) * 0.1f;
                Vector3 front = center + new Vector3(c * he, s * he, 0) * 0.1f;
                Gizmos.DrawLine(center, front);
            }
            // for (int y = 0; y < GridRes; y++)
            // for (int x = 0; x < GridRes; x++)
            // {
            //     float level = _gridSDF[Coord2Idx(x, y)];
            //     if (level < 0) continue;
            //     float v = _gridPressure[Coord2Idx(x, y)] * 0.15f + 0.1f;
            //     // float2 v = _gridVelocityCopy[Coord2Idx(x, y)] * 0.3f + 0.5f;
            //     Gizmos.color = new Color(v, v, v, 0.5f);
            //     Gizmos.DrawCube(new Vector3((x + 0.5f) * CellSize * 0.1f, (y + 0.5f) * CellSize*0.1f, -0.1f), new Vector3(CellSize*0.1f, CellSize*0.1f, 0));
            // }
        }

        #region Utils

        private static float ReadGrid(int2 coord, NativeArray<float> block)
        {
            return block[Coord2Idx(math.clamp(coord, 0, GridRes - 1))];
        }

        private static float2 ReadGrid(int2 coord, NativeArray<float2> block)
        {
            return block[Coord2Idx(math.clamp(coord, 0, GridRes - 1))];
        }
        
        private struct Int2Comparer : IComparer<int2>
        {
            public int Compare(int2 lhs, int2 rhs) => lhs.x - rhs.x;
        }
    
        private static int2 Idx2Coord(int i)
        {
            return new int2(i % GridRes, i / GridRes);
        }
    
        private static int2 GetCoord(float2 pos)
        {
            return (int2)math.floor(pos * InvCellSize);
        }
        private static float2 GetQuadraticWeight(float2 abs_x)
        {
            float2 dst = math.saturate(1.5f - abs_x);
            return math.saturate(math.select(0.5f * dst * dst, 0.75f - abs_x * abs_x, abs_x < 0.5f));
        }

        private static int Coord2Idx(int x, int y)
        {
            return x + y * GridRes;
        }
        private static int Coord2Idx(int2 coord)
        {
            return Coord2Idx(coord.x, coord.y);
        }
        private static bool IsSolidCell(uint gridTypes)
        {
            return (gridTypes & 3u) == SOLID;
        }
        private static bool2 IsSolidCell(uint2 gridTypes)
        {
            return (gridTypes & 3u) == SOLID;
        }
        private static bool IsFluidCell(uint gridTypes)
        {
            return (gridTypes & 3u) == FLUID;
        }
        private static bool IsAirCell(uint gridTypes)
        {
            return (gridTypes & 3u) == AIR;
        }
        private static bool IsInteractiveSolidCell(uint gridTypes)
        {
            return (gridTypes & 3u) == INTERACTIVE_SOLID;
        }
        private static bool IsBlockingCell(uint gridTypes)
        {
            return IsSolidCell(gridTypes) || IsInteractiveSolidCell(gridTypes);
        }
        private static uint2 NeighborGridTypeLB(uint gridTypes)
        {
            return new uint2((gridTypes >> 2) & 3u, (gridTypes >> 6) & 3u);
        }
        private static uint4 NeighborGridTypes( uint gridTypes)
        {
            return new uint4((gridTypes >> 2) & 3u, (gridTypes >> 4) & 3u, (gridTypes >> 6) & 3u, (gridTypes >> 8) & 3u);
        }
        private static uint2 NeighborGridTypeAxis(int axis, uint gridTypes)
        {
            return new uint2((gridTypes >> (axis * 4 + 2)) & 3u,
                (gridTypes >> (axis * 4 + 4)) & 3u);
        }
        private static float2 EnforceBoundaryCondition(float2 velocity, uint gridTypes)
        {
            if (IsSolidCell(gridTypes))
                return float2.zero;
            return math.select(velocity, 0, IsSolidCell(NeighborGridTypeLB(gridTypes)));
        }
        
        private static float2 ClampPosition(float2 pos)
        {
            return math.clamp(pos, 0.001f*CellSize, (GridRes - 0.001f) * CellSize);
        }
        
        #endregion
    
        [BurstCompile]
        private struct ClearGridJob : IJobParallelFor
        {
            [WriteOnly] public NativeArray<int2> Range;
            [WriteOnly] public NativeArray<int> Start;
            [WriteOnly] public NativeArray<int> End;
            [WriteOnly] public NativeArray<float> Pressure;
            [WriteOnly] public NativeArray<float> Density;

            public void Execute(int i)
            {
                Start[i] = 0;
                End[i] = 0;
                Range[i] = int2.zero;
                Pressure[i] = 0;
                Density[i] = 0;
            }
        }

        [BurstCompile]
        private struct HashJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float4> Ps;
            [WriteOnly] public NativeArray<int2> Hashes;

            public void Execute(int i)
            {
                int hash = Coord2Idx(GetCoord(Ps[i].xy));
                Hashes[i] = math.int2(hash, i);
            }
        }
        
        [BurstCompile]
        private struct BuildLutJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> Hashes;
            [NativeDisableParallelForRestriction] public NativeArray<int> StartIndices;
            [NativeDisableParallelForRestriction] public NativeArray<int> EndIndices;

            public void Execute(int i)
            {
                int prev = i == 0 ? NumParticles - 1 : i - 1;
                int next = i == NumParticles - 1 ? 0 : i + 1;
                int currID = Hashes[i].x;
                int prevID = Hashes[prev].x;
                int nextID = Hashes[next].x;
                if (currID != prevID || i == 0) StartIndices[currID] = i;
                if (currID != nextID) EndIndices[currID] = i + 1;
            }
        }
    
        [BurstCompile]
        private struct CombineLutJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int> StartIndices;
            [ReadOnly] public NativeArray<int> EndIndices;
            [WriteOnly] public NativeArray<int2> Range;

            public void Execute(int i)
            {
                Range[i] = new int2(StartIndices[i], EndIndices[i]);
            }
        }
        
        [BurstCompile]
        private struct ShuffleJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> Hashes;
            [ReadOnly] public NativeArray<float4> PosRaw;
            [ReadOnly] public NativeArray<float2> VelRaw;
            [WriteOnly] public NativeArray<float4> PosNew;
            [WriteOnly] public NativeArray<float2> VelNew;

            public void Execute(int i)
            {
                int id = Hashes[i].y;
                PosNew[i] = PosRaw[id];
                VelNew[i] = VelRaw[id];
            }
        }
        
        [BurstCompile]
        private struct SetGridLevelJob : IJobParallelFor
        {
            [ReadOnly] private NativeArray<uint> _gridTypes;
            [WriteOnly] private NativeArray<float> _gridLevel;
            
            public SetGridLevelJob(NativeArray<float> level, NativeArray<uint> gridTypes)
            {
                _gridLevel = level;
                _gridTypes = gridTypes;
            }

            public void Execute(int i)
            {
                uint gridType = _gridTypes[i];
                if (!IsFluidCell(gridType)) _gridLevel[i] = -1;
                else
                {
                    var neighborTypes = NeighborGridTypes(gridType);
                    _gridLevel[i] = math.any(neighborTypes != FLUID) ? 0 : GridRes;
                }
            }
        }
        
        [BurstCompile]
        private struct ComputeDistanceFieldJob : IJob
        {
            private NativeArray<float> _gridLevel;
            
            public ComputeDistanceFieldJob(NativeArray<float> level)
            {
                _gridLevel = level;
            }

            public void Execute()
            {
                int2 offset = new int2(1, 0);
                int rightBound = GridRes - 1;
                for (int i = 0; i < NumGrid; i++)
                {
                    float level = _gridLevel[i];
                    if (level <= 0) continue;
                    int2 coord = Idx2Coord(i);
                    if (coord.x > 0)
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord - offset.xy)]);
                    if (coord.y > 0)
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord - offset.yx)]);
                    // if (math.all(coord > 0))
                    //     level = math.min(level, 1 + _gridLevel[Coord2Idx(coord - offset.xx)]);
                    // if (coord.x < rightBound && coord.y > 0)
                    //     level = math.min(level, 1 + _gridLevel[Coord2Idx(coord - new int2(-1, 1))]);

                    _gridLevel[i] = level;
                }
                
                for (int i = NumGrid - 1; i >= 0; i--)
                {
                    float level = _gridLevel[i];
                    if (level <= 0) continue;
                    int2 coord = Idx2Coord(i);
                    if (coord.x < rightBound)
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord + offset.xy)]);
                    if (coord.y < rightBound)
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord + offset.yx)]);
                    // if (math.all(coord < rightBound))
                    //     level = math.min(level, 1 + _gridLevel[Coord2Idx(coord + offset.xx)]);
                    // if (coord.x > 0 && coord.y < rightBound)
                    //     level = math.min(level, 1 + _gridLevel[Coord2Idx(coord + new int2(-1, 1))]);

                    _gridLevel[i] = level;
                }
            }
        }
        
        [BurstCompile]
        private struct ParticlesCounterInitJob : IJob
        {
            [ReadOnly] private NativeArray<float> _gridLevel;
            private NativeArray<int2> _range;
            [WriteOnly] private NativeArray<int> _particleIDs;
            [WriteOnly] private NativeReference<int> _pCount;
            
            public ParticlesCounterInitJob(NativeArray<float> level, NativeArray<int2> range, NativeArray<int> particleIDs, NativeReference<int> pCount)
            {
                _gridLevel = level;
                _range = range;
                _particleIDs = particleIDs;
                _pCount = pCount;
            }

            public void Execute()
            {
                int ptr = 0;
                for (int i = 0; i < NumGrid; i++)
                {
                    var level = _gridLevel[i];
                    if (level < 0 || level >= Band2) continue;
                    int count = 4; 
                    for (int j = 0; j < count; j++)
                        _particleIDs[ptr + j] = -1;
                    
                    _range[i] = new int2(ptr, ptr + count);
                    ptr += count;
                }
                _pCount.Value = ptr;
            }
        }
        
        [BurstCompile]
        private struct ParticlesCounterJob : IJob
        {
            [ReadOnly] private NativeArray<float> _gridLevel;
            private NativeArray<int2> _range;
            [WriteOnly] private NativeArray<int> _particleIDs;
            [WriteOnly] private NativeReference<int> _pCount;
            
            public ParticlesCounterJob(NativeArray<float> level, NativeArray<int2> range, NativeArray<int> particleIDs, NativeReference<int> pCount)
            {
                _gridLevel = level;
                _range = range;
                _particleIDs = particleIDs;
                _pCount = pCount;
            }

            public void Execute()
            {
                int ptr = 0;
                for (int i = 0; i < NumGrid; i++)
                {
                    var level = _gridLevel[i];
                    if (level < 0 || level >= Band2)
                    {
                        _range[i] = int2.zero;
                        continue;
                    }
                    var range = _range[i];
                    int count = range.y - range.x;
                    int expect = level < 1 ? count : math.max(4, count);
                    for (int j = 0; j < expect; j++)
                    {
                        _particleIDs[ptr + j] = j < count ? range.x + j : - 1; // if need add particle, set -1
                    }
                    _range[i] = new int2(ptr, ptr + expect);
                    ptr += expect;
                }
                _pCount.Value = ptr;
            }
        }
        
        [BurstCompile]
        private struct ResampleParticlesJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int> Ids;
            [ReadOnly] public NativeArray<int2> Ranges;
            [ReadOnly] public NativeArray<float2> GridVelocity;

            [ReadOnly] public NativeArray<float4> PosRaw;
            [ReadOnly] public NativeArray<float2> VelRaw;

            [NativeDisableContainerSafetyRestriction, WriteOnly] public NativeArray<float4> PosNew;
            [NativeDisableContainerSafetyRestriction, WriteOnly] public NativeArray<float2> VelNew;

            public void Execute(int gid)
            {
                var range = Ranges[gid];
                if (range.x >= range.y) return;
                
                int toAdd = 0;
                for (int i = range.x; i < range.y; i++)
                {
                    int id = Ids[i];
                    if (id < 0)
                        toAdd++;
                    else
                    {
                        PosNew[i] = PosRaw[id];
                        VelNew[i] = VelRaw[id];
                    }
                }
                
                if (toAdd == 0) return;
                
                int existCount = range.y - range.x - toAdd;
                var posArr = new NativeArray<float2>(4, Allocator.Temp);
                var velArr = new NativeArray<float2>(4, Allocator.Temp);
                for (int i = 0; i < existCount; i++)
                {
                    int p = Ids[range.x + i];
                    posArr[i] = PosRaw[p].xy;
                    velArr[i] = VelRaw[p].xy;
                }
                
                int2 coord = Idx2Coord(gid);
                float2 origin = ((float2)coord + 0.25f) * CellSize;
                for (int idx = 0; idx < toAdd; idx++)
                {
                    float maxDst = 0;
                    float2 selectedPos = origin;
                    for (int i = 0; i < 4; i++)
                    { 
                        var tryPos = origin + new float2(i & 1, i >> 1) * (CellSize * 0.5f);
                        float minDst = 1000;
                        for (int j = 0; j < existCount; j++)
                        {
                            var vec = posArr[j] - tryPos;
                            float len = math.lengthsq(vec);
                            minDst = math.min(minDst, len);
                        }

                        if (minDst > maxDst)
                        {
                            maxDst = minDst;
                            selectedPos = tryPos;
                        }
                    }
                    ReadGridFaceBilinear(selectedPos, GridVelocity, out var v);
                    int p = range.x + existCount;
                    PosNew[p] = new float4(selectedPos, 0, math.length(v));
                    VelNew[p] = v;
                    posArr[existCount] = selectedPos;
                    existCount++;
                }
            }
        
            private void ReadGridFaceBilinear(float2 pos, NativeArray<float2> block, out float2 v)
            {
                ReadGridFaceBilinear(pos * InvCellSize + new float2(0, -0.5f), 0, block, out var vx);
                ReadGridFaceBilinear(pos * InvCellSize + new float2(-0.5f, 0), 1, block, out var vy);
                v = new float2(vx, vy);
            }
            
            private void ReadGridFaceBilinear(float2 uv, int axis, NativeArray<float2> block, out float v)
            {
                uv = math.clamp(uv, 1e-3f, GridRes - 1e-3f);
                int2 p00 = (int2)math.floor(uv);
                int2 p11 = p00 + 1;
                float2 f = uv - p00;
                float c00 = ReadGrid(p00, block)[axis];
                float c10 = ReadGrid(new int2(p11.x, p00.y), block)[axis];
                float c01 = ReadGrid(new int2(p00.x, p11.y), block)[axis];
                float c11 = ReadGrid(p11, block)[axis];
                float c0 = math.lerp(c00, c10, f.x);
                float c1 = math.lerp(c01, c11, f.x);
                v = math.lerp(c0, c1, f.y);
            }
        }
    
        [BurstCompile]
        private struct SetGridTypeJob :IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> Range;
            [ReadOnly] public NativeArray<float> GridLevel;
            [ReadOnly] public NativeArray<int> GridBlockID;
            [WriteOnly] public NativeArray<uint> GridType;

            public void Execute(int i)
            {
                int2 coord = Idx2Coord(i);

                uint gridType = GetGridType(coord);
                gridType |= GetGridType(coord - new int2(1, 0)) << 2;
                gridType |= GetGridType(coord + new int2(1, 0)) << 4;
                gridType |= GetGridType(coord - new int2(0, 1)) << 6;
                gridType |= GetGridType(coord + new int2(0, 1)) << 8;
                GridType[i] = gridType;
            }

            private uint GetGridType(int2 coord)
            {
                if (math.any(coord < 0) || math.any(coord > GridRes - 1))
                    return SOLID;
                int i = Coord2Idx(coord);
                if (GridBlockID[i] >= 0) return INTERACTIVE_SOLID;
                if (GridLevel[i] >= Band1) return FLUID;
                int2 range = Range[i];
                return range.y > range.x ? FLUID : AIR;
            }
        }

        [BurstCompile]
        private struct ParticleToGridJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float4> ParticlePos;
            [ReadOnly] public NativeArray<float2> ParticleVel;
            [ReadOnly] public NativeArray<int2> Range;
            [ReadOnly] public NativeArray<float> GridLevel;
            public NativeArray<float2> GridVelocity;
            [WriteOnly] public NativeArray<float> GridDensity;

            public void Execute(int i)
            {
                int2 coord = Idx2Coord(i);
                float2 cellCenter = ((float2)coord + 0.5f) * CellSize;
                float2 sdf = ReadGridFacesBilinear(cellCenter, GridLevel);
                if (math.all(sdf > Band1))
                {
                    GridDensity[i] = TargetDensity;
                    return;
                }

                float2 velocity = float2.zero;
                float2 sum = 0;
                float density = 0;
                float2 position_vx = cellCenter + new float2(-0.5f * CellSize, 0.0f);
                float2 position_vy = cellCenter + new float2(0.0f, -0.5f * CellSize);

                for (int x = math.max(coord.x - 2, 0); x <= math.min(coord.x + 1, GridRes - 1); ++x)
                for (int y = math.max(coord.y - 2, 0); y <= math.min(coord.y + 1, GridRes - 1); ++y)
                {
                    var neighborIdx = Coord2Idx(x, y);
                    int2 range = Range[neighborIdx];
                    for (int j = range.x; j < range.y; j++)
                    {
                        float4 p = ParticlePos[j];
                        float2 n_x = p.xy;
                        var n_v = ParticleVel[j];
                        
                        float2 weight = new float2(
                            GetWeight(position_vx - n_x, InvCellSize),
                            GetWeight(position_vy - n_x, InvCellSize));
                        
                        sum += weight;
                        
                        velocity.x += weight.x * n_v.x;
                        velocity.y += weight.y * n_v.y;
                        
                        float2 dist = n_x - cellCenter;
                        density += GetPoly6Weight(dist * InvCellSize);
                    }
                }

                velocity = math.select(GridVelocity[i], velocity / sum, sum > 1e-4f);
                // float2 oldVel = GridVelocity[i];
                // velocity = math.select(velocity, oldVel, sdf >= Band1);
                GridVelocity[i] = velocity;
                GridDensity[i] = density;
            }
            
            private float GetPoly6Weight(float2 pos)
            {
                float r2 = math.lengthsq(pos);
                if (r2 >= 1) return 0;
                float v = 1 - r2;
                return v * v * v * 315f / (64 * math.PI);
            }

            private float2 ReadGridFacesBilinear(float2 pos, NativeArray<float> block)
            {
                return new float2(ReadGridFaceBilinear(pos * InvCellSize + new float2(-0.5f, -1f), block),
                                  ReadGridFaceBilinear(pos * InvCellSize + new float2(-1f, -0.5f), block));
            }
            
            private float ReadGridFaceBilinear(float2 uv, NativeArray<float> block)
            {
                uv = math.clamp(uv, 1e-3f, GridRes - 1e-3f);
                int2 p00 = (int2)math.floor(uv);
                int2 p11 = p00 + 1;
                float2 f = uv - p00;
                float c00 = ReadGrid(p00, block);
                float c10 = ReadGrid(new int2(p11.x, p00.y), block);
                float c01 = ReadGrid(new int2(p00.x, p11.y), block);
                float c11 = ReadGrid(p11, block);
                float c0 = math.lerp(c00, c10, f.x);
                float c1 = math.lerp(c01, c11, f.x);
                return math.lerp(c0, c1, f.y);
            }
        }
    
        [BurstCompile]
        private struct AddForceJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<uint> GridTypes;
            public NativeArray<float2> GridVelocity;
            public float2 Gravity;
            public float2 MousePos;
            public float2 MouseVec;

            public void Execute(int i)
            {
                float2 velocity = GridVelocity[i];
                uint gridType = GridTypes[i];
                velocity += Gravity * DeltaTime;
                // float2 coord = (float2)Idx2Coord(i) + 0.5f;
                // float dst = math.distance(coord, MousePos);
                // if (dst < 8)
                //     velocity += MouseVec * (8 - dst) * 10 * DeltaTime;

                GridVelocity[i] = EnforceBoundaryCondition(velocity, gridType);
            }
        }
    
        [BurstCompile]
        private struct CalcDivergenceJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> GridVelocity;
            [ReadOnly] public NativeArray<uint> GridTypes;
            [ReadOnly] public NativeArray<float> GridDensity;
            [WriteOnly] public NativeArray<float> GridDivergence;
        
            public void Execute(int i)
            {
                int2 cellIdx = Idx2Coord(i);

                float divergence = 0;

                uint gridTypes = GridTypes[i];

                if (IsFluidCell(gridTypes))
                {
                    float2 vel = GridVelocity[i];
                    float v_xn = cellIdx.x + 1 < GridRes ? GridVelocity[Coord2Idx(cellIdx + new int2(1, 0))].x : 0;
                    float v_yn = cellIdx.y + 1 < GridRes ? GridVelocity[Coord2Idx(cellIdx + new int2(0, 1))].y : 0;

                    divergence += InvCellSize * (v_xn - vel.x);
                    divergence += InvCellSize * (v_yn - vel.y);

                    // float deltaDensity = math.max(0, GridDensity[i] - TargetDensity);
                    // deltaDensity = (math.any(cellIdx <= 2) || math.any(cellIdx >= 61)) 
                    //     ? math.max(0, deltaDensity)
                    //     : math.max(-0.1f, deltaDensity);
                    // divergence -= 0.01f * deltaDensity;
                }

                GridDivergence[i] = -divergence;
            }
        }
        
        [BurstCompile]
        private struct UpdateVelocity : IJobParallelFor
        {
            [ReadOnly] public NativeArray<uint> GridTypes;
            [ReadOnly] public NativeArray<float> GridPressure;
            public NativeArray<float2> GridVelocity;
        
            public void Execute(int i)
            {
                int2 cellIdx = Idx2Coord(i);

                float2 velocity = GridVelocity[i];
                uint grid_types = GridTypes[i];
                float pressure = GridPressure[i];

                uint2 lbType = NeighborGridTypeLB(grid_types);
                int c_id_xp = IsBlockingCell(lbType.x) ? i : Coord2Idx(cellIdx + new int2(-1, 0));
                int c_id_yp = IsBlockingCell(lbType.y) ? i : Coord2Idx(cellIdx + new int2(0, -1));

                // float pressure = GridPressure[i];

                velocity.x -= InvCellSize * (pressure - GridPressure[c_id_xp]);
                velocity.y -= InvCellSize * (pressure - GridPressure[c_id_yp]);
                
                GridVelocity[i] = EnforceBoundaryCondition(velocity, grid_types);
            }
        }
    
        [BurstCompile]
        private struct GridToParticleJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float4> ParticlePos;
            [ReadOnly] public NativeArray<float2> GridVelocityOld;
            [ReadOnly] public NativeArray<float2> GridVelocityNew;
            public NativeArray<float2> ParticleVel;
            public float Flipness;
        
            public void Execute(int i)
            {
                float2 pos = ParticlePos[i].xy;
                float2 vel = ParticleVel[i];
                int2 coord = GetCoord(pos);

                // ReadGridFaceBilinear(pos, GridVelocityOld, GridVelocityNew, 
                //     out float2 gOriginVel, out float2 gVel);
                
                // float2 p_pic_vel = gVel;
                // float2 p_flip_vel = vel + (gVel - gOriginVel);
                // ParticleVel[i] = math.lerp(p_pic_vel, p_flip_vel, Flipness);

                float2 new_v = float2.zero;
                float2 old_v = float2.zero;
                
                for (int x = coord.x - 1; x <= coord.x + 2; ++x)
                for (int y = coord.y - 1; y <= coord.y + 2; ++y)
                {
                    int2 nCoord = new int2(x, y);
                    int idx = Coord2Idx(math.clamp(nCoord, 0, GridRes - 1));
                    
                    float2 pos_u = (nCoord + new float2(0, 0.5f)) * CellSize;
                    float2 pos_v = (nCoord + new float2(0.5f, 0)) * CellSize;
                    
                    float2 weights = new float2(GetWeight(pos_u - pos, InvCellSize),
                        GetWeight(pos_v - pos, InvCellSize)); 

                    float2 weightedNewV = weights * GridVelocityNew[idx];
                    float2 weightedOldV = weights * GridVelocityOld[idx];
                    
                    new_v += weightedNewV;
                    old_v += weightedOldV;
                }
                
                ParticleVel[i] = math.lerp(new_v, vel + (new_v - old_v), Flipness);
            }
        
            private void ReadGridFaceBilinear(float2 pos, NativeArray<float2> block0, NativeArray<float2> block1,
                out float2 v0, out float2 v1)
            {
                ReadGridFaceBilinear(pos * InvCellSize + new float2(0, -0.5f), 0, block0, block1, out var v0x, out var v1x);
                ReadGridFaceBilinear(pos * InvCellSize + new float2(-0.5f, 0), 1, block0, block1, out var v0y, out var v1y);
                v0 = new float2(v0x, v0y);
                v1 = new float2(v1x, v1y);
            }
            
            private void ReadGridFaceBilinear(float2 uv, int axis, NativeArray<float2> block0, 
                NativeArray<float2> block1, out float v0, out float v1)
            {
                uv = math.clamp(uv, 1e-3f, GridRes - 1e-3f);
                int2 p00 = (int2)math.floor(uv);
                int2 p11 = p00 + 1;
                float2 f = uv - p00;
                float c00 = ReadGrid(p00, block0)[axis];
                float c10 = ReadGrid(new int2(p11.x, p00.y), block0)[axis];
                float c01 = ReadGrid(new int2(p00.x, p11.y), block0)[axis];
                float c11 = ReadGrid(p11, block0)[axis];
                float c0 = math.lerp(c00, c10, f.x);
                float c1 = math.lerp(c01, c11, f.x);
                v0 = math.lerp(c0, c1, f.y);
                
                c00 = ReadGrid(p00, block1)[axis];
                c10 = ReadGrid(new int2(p11.x, p00.y), block1)[axis];
                c01 = ReadGrid(new int2(p00.x, p11.y), block1)[axis];
                c11 = ReadGrid(p11, block1)[axis];
                c0 = math.lerp(c00, c10, f.x);
                c1 = math.lerp(c01, c11, f.x);
                v1 = math.lerp(c0, c1, f.y);
            }

            private float2 ReadGridFaceBilinear(float2 pos, NativeArray<float2> block)
            {
                return new float2(ReadGridFaceBilinear(pos * InvCellSize + new float2(0, -0.5f), 0, block),
                    ReadGridFaceBilinear(pos * InvCellSize + new float2(-0.5f, 0), 1, block));
            }
            
            private float ReadGridFaceBilinear(float2 uv, int axis, NativeArray<float2> block)
            {
                uv = math.clamp(uv, 1e-3f, GridRes - 1e-3f);
                int2 p00 = (int2)math.floor(uv);
                int2 p11 = p00 + 1;
                float2 f = uv - p00;
                float c00 = ReadGrid(p00, block)[axis];
                float c10 = ReadGrid(new int2(p11.x, p00.y), block)[axis];
                float c01 = ReadGrid(new int2(p00.x, p11.y), block)[axis];
                float c11 = ReadGrid(p11, block)[axis];
                float c0 = math.lerp(c00, c10, f.x);
                float c1 = math.lerp(c01, c11, f.x);
                return math.lerp(c0, c1, f.y);
            }
        }
    
        [BurstCompile]
        private struct ParticlesAdvectionJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> GridVelocity;
            public NativeArray<float4> ParticlePos;
            public NativeArray<float2> ParticleVel;
        
            public void Execute(int i)
            {
                float4 particle = ParticlePos[i];
                float2 pos = particle.xy;

                float2 vel;
                // https://en.wikipedia.org/wiki/List_of_Runge-Kutta_methods
#if USE_RK1
                // advect using RK1 (Forward Euler)
                float2 k1 = ReadGridFaceBilinear(pos, GridVelocity);
                vel = 1.0f * k1;
#elif USE_RK2
                // advect using RK2 (Explicit midpoint method)
                float2 k1 = ReadGridFaceBilinear(pos,  GridVelocity);
                vel = ReadGridFaceBilinear(pos + 0.5f * DeltaTime * k1,  GridVelocity);
#elif USE_RK3
                // advect using RK3 (Ralston's third-order method)
                float2 k1 = ReadGridFaceBilinear(pos,  GridVelocity);
                float2 k2 = ReadGridFaceBilinear(pos + 0.5f * DeltaTime * k1, GridVelocity);
                float2 k3 = ReadGridFaceBilinear(pos + 0.75f * DeltaTime * k2, GridVelocity);
                vel = 2.0f / 9.0f * k1 + 1.0f / 3.0f * k2 + 4.0f / 9.0f * k3;
#else
                // advect using RK4
                float2 k1 = ReadGridFaceBilinear(pos,  GridVelocity);
                float2 k2 = ReadGridFaceBilinear(pos + 0.5f * DeltaTime * k1, GridVelocity);
                float2 k3 = ReadGridFaceBilinear(pos + 0.5f * DeltaTime * k2, GridVelocity);
                float2 k4 = ReadGridFaceBilinear(pos + DeltaTime * k3, GridVelocity);
                vel = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0f;
#endif

                float2 velocity = ParticleVel[i];
                pos += vel * DeltaTime;
                velocity = math.select(velocity, 0, pos <= 0.1f * CellSize); 
                velocity = math.select(velocity, 0, pos >= (GridRes - 0.1f) * CellSize);
                ParticleVel[i] = velocity;
                pos = ClampPosition(pos);
            
                particle.xy = pos;
                particle.w = math.length(vel);
                ParticlePos[i] = particle;
            }

            private float2 ReadGridFaceBilinear(float2 pos, NativeArray<float2> block)
            {
                return new float2(ReadGridFaceBilinear(pos * InvCellSize + new float2(0, -0.5f), 0, block),
                                  ReadGridFaceBilinear(pos * InvCellSize + new float2(-0.5f, 0), 1, block));
            }
            
            private float ReadGridFaceBilinear(float2 uv, int axis, NativeArray<float2> block)
            {
                uv = math.clamp(uv, 1e-3f, GridRes - 1e-3f);
                int2 p00 = (int2)math.floor(uv);
                int2 p11 = p00 + 1;
                float2 f = uv - p00;
                float c00 = ReadGrid(p00, block)[axis];
                float c10 = ReadGrid(new int2(p11.x, p00.y), block)[axis];
                float c01 = ReadGrid(new int2(p00.x, p11.y), block)[axis];
                float c11 = ReadGrid(p11, block)[axis];
                float c0 = math.lerp(c00, c10, f.x);
                float c1 = math.lerp(c01, c11, f.x);
                return math.lerp(c0, c1, f.y);
            }
        }
    
        [BurstCompile]
        private struct GridsAdvectionJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> GridVelocity;
            [ReadOnly] public NativeArray<uint> GridTypes;
            [WriteOnly] public NativeArray<float2> GridVelocityAlt;
        
            public void Execute(int i)
            {
                int2 coord = Idx2Coord(i);
                float2 cellCenter = ((float2)coord + 0.5f) * CellSize;
                // using RK4
                float2 posFaceX = cellCenter + new float2(-0.5f * CellSize, 0);
                float2 traceDirX = BackwardTrace(posFaceX, GridVelocity);
                float vx = ReadGridFaceBilinear(posFaceX - traceDirX * DeltaTime, 0, GridVelocity);

                float2 posFaceY = cellCenter + new float2(0, -0.5f * CellSize);
                float2 traceDirY = BackwardTrace(posFaceY, GridVelocity);
                float vy = ReadGridFaceBilinear(posFaceY - traceDirY * DeltaTime, 1, GridVelocity);
                
                uint grid_types = GridTypes[i];
                GridVelocityAlt[i] = EnforceBoundaryCondition(new float2(vx, vy), grid_types);
            }

            private float2 BackwardTrace(float2 pos, NativeArray<float2> block)
            {
                float2 k1 = ReadGridFacesBilinear(pos,  block);
                float2 k2 = ReadGridFacesBilinear(pos - 0.5f * DeltaTime * k1, block);
                float2 k3 = ReadGridFacesBilinear(pos - 0.5f * DeltaTime * k2, block);
                float2 k4 = ReadGridFacesBilinear(pos - DeltaTime * k3, block);
                return (k1 + 2 * k2 + 2 * k3 + k4) / 6.0f;
            }

            private float2 ReadGridFacesBilinear(float2 pos, NativeArray<float2> block)
            {
                return new float2(ReadGridFaceBilinear(pos, 0, block),
                                  ReadGridFaceBilinear(pos, 1, block));
            }
            
            private float ReadGridFaceBilinear(float2 pos, int axis, NativeArray<float2> block)
            {
                float2 uv = pos * InvCellSize + new float2(axis == 0 ? 0 : -0.5f, axis == 1 ? 0 : -0.5f);
                uv = math.clamp(uv, 1e-3f, GridRes - 1e-3f);
                int2 p00 = (int2)math.floor(uv);
                int2 p11 = p00 + 1;
                float2 f = uv - p00;
                float c00 = ReadGrid(p00, block)[axis];
                float c10 = ReadGrid(new int2(p11.x, p00.y), block)[axis];
                float c01 = ReadGrid(new int2(p00.x, p11.y), block)[axis];
                float c11 = ReadGrid(p11, block)[axis];
                float c0 = math.lerp(c00, c10, f.x);
                float c1 = math.lerp(c01, c11, f.x);
                return math.lerp(c0, c1, f.y);
            }
        }
        
        [BurstCompile]
        private struct ComputeLaplacianJob: IJobParallelFor
        {
            [ReadOnly] public NativeArray<uint> GridTypes;
            [WriteOnly] public NativeArray<float3> GridLaplacian;

            public void Execute(int index)
            {
                uint gridType = GridTypes[index];
                uint2 xAxisType = NeighborGridTypeAxis(0, gridType);
                uint2 yAxisType = NeighborGridTypeAxis(1, gridType);
                
                float center = 4;
                if (IsBlockingCell(xAxisType.x)) center -= 1;
                if (IsBlockingCell(xAxisType.y)) center -= 1;
                if (IsBlockingCell(yAxisType.x)) center -= 1;
                if (IsBlockingCell(yAxisType.y)) center -= 1;
                
                float3 a = float3.zero;
                if (IsFluidCell(gridType))
                {
                    a = new float3(center, 
                        IsFluidCell(xAxisType.x) ? -1 : 0, 
                        IsFluidCell(yAxisType.x) ? -1 : 0);
                }
                
                GridLaplacian[index] = a;
            }
        }

        [BurstCompile]
        private struct MarkSolidBlocksJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<SolidBlock> Blocks;
            [WriteOnly] public NativeArray<int> GridBlockID;

            public void Execute(int i)
            {
                int2 coord = Idx2Coord(i);
                float2 cellCenter = ((float2)coord + 0.5f) * CellSize;
                int blockID = -1;
                for (int b = 0; b < Blocks.Length; b++)
                {
                    var block = Blocks[b];
                    // Transform cell center to block's local space
                    float2 localPos = RotatePoint(cellCenter - block.Center, -block.Angle);
                    // Check if point is inside the axis-aligned square in local space
                    if (math.abs(localPos.x) <= block.HalfExtent && math.abs(localPos.y) <= block.HalfExtent)
                    {
                        blockID = b;
                        break;
                    }
                }
                GridBlockID[i] = blockID;
            }

            private static float2 RotatePoint(float2 p, float angle)
            {
                float c = math.cos(angle);
                float s = math.sin(angle);
                return new float2(c * p.x - s * p.y, s * p.x + c * p.y);
            }
        }

        [BurstCompile]
        private struct TransferSolidVelocityJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<SolidBlock> Blocks;
            [ReadOnly] public NativeArray<uint> GridTypes;
            [ReadOnly] public NativeArray<int> GridBlockID;
            public NativeArray<float2> GridVelocity;

            public void Execute(int i)
            {
                int2 coord = Idx2Coord(i);
                uint gridType = GridTypes[i];
                uint selfType = gridType & 3u;
                uint2 lb = NeighborGridTypeLB(gridType);
                float2 v = GridVelocity[i];

                // Left face (u velocity component)
                float2 facePos = ((float2)coord + new float2(0, 0.5f)) * CellSize;
                if (IsInteractiveSolidCell(selfType))
                {
                    var block = Blocks[GridBlockID[i]];
                    v.x = GetSolidVelocityAt(block, facePos).x;
                }
                else if (IsInteractiveSolidCell(lb.x) && coord.x > 0)
                {
                    var block = Blocks[GridBlockID[Coord2Idx(coord - new int2(1, 0))]];
                    v.x = GetSolidVelocityAt(block, facePos).x;
                }

                // Bottom face (v velocity component)
                facePos = ((float2)coord + new float2(0.5f, 0)) * CellSize;
                if (IsInteractiveSolidCell(selfType))
                {
                    var block = Blocks[GridBlockID[i]];
                    v.y = GetSolidVelocityAt(block, facePos).y;
                }
                else if (IsInteractiveSolidCell(lb.y) && coord.y > 0)
                {
                    var block = Blocks[GridBlockID[Coord2Idx(coord - new int2(0, 1))]];
                    v.y = GetSolidVelocityAt(block, facePos).y;
                }

                GridVelocity[i] = v;
            }

            // Calculate solid velocity at a point: v = v_center + ω × r
            // In 2D: ω × r = ω * (-r.y, r.x) where ω is angular velocity (scalar in 2D)
            private static float2 GetSolidVelocityAt(SolidBlock block, float2 worldPos)
            {
                float2 r = worldPos - block.Center;
                float2 angularVel = block.AngularVelocity * new float2(-r.y, r.x);
                return block.Velocity + angularVel;
            }
        }

        [BurstCompile]
        private struct ComputeCouplingJacobianJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<uint> GridTypes;
            [ReadOnly] public NativeArray<int> GridBlockID;
            [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float> BlockJx;
            [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float> BlockJy;

            public void Execute(int i)
            {
                int numBlocks = BlockJx.Length / NumGrid;
                for (int b = 0; b < numBlocks; b++)
                {
                    BlockJx[b * NumGrid + i] = 0;
                    BlockJy[b * NumGrid + i] = 0;
                }

                uint gridType = GridTypes[i];
                if (!IsFluidCell(gridType)) return;

                int2 coord = Idx2Coord(i);
                uint4 nb = NeighborGridTypes(gridType); // (left, right, bottom, top)

                if (IsInteractiveSolidCell(nb.x))
                {
                    int b = GridBlockID[Coord2Idx(coord - new int2(1, 0))];
                    BlockJx[b * NumGrid + i] = -1;
                }
                if (IsInteractiveSolidCell(nb.y))
                {
                    int b = GridBlockID[Coord2Idx(coord + new int2(1, 0))];
                    BlockJx[b * NumGrid + i] = 1;
                }
                if (IsInteractiveSolidCell(nb.z))
                {
                    int b = GridBlockID[Coord2Idx(coord - new int2(0, 1))];
                    BlockJy[b * NumGrid + i] = -1;
                }
                if (IsInteractiveSolidCell(nb.w))
                {
                    int b = GridBlockID[Coord2Idx(coord + new int2(0, 1))];
                    BlockJy[b * NumGrid + i] = 1;
                }
            }
        }

        [BurstCompile]
        private struct ComputeBlockForcesJob : IJob
        {
            [ReadOnly] public NativeArray<SolidBlock> Blocks;
            [ReadOnly] public NativeArray<float> Pressure;
            [ReadOnly] public NativeArray<float> BlockJx;
            [ReadOnly] public NativeArray<float> BlockJy;
            [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float2> BlockForces;
            [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float> BlockTorques;

            public void Execute()
            {
                int numBlocks = BlockForces.Length;
                for (int b = 0; b < numBlocks; b++)
                {
                    float sx = 0, sy = 0;
                    float torque = 0;
                    int off = b * NumGrid;
                    var block = Blocks[b];

                    for (int i = 0; i < NumGrid; i++)
                    {
                        float p = Pressure[i];
                        float jx = BlockJx[off + i];
                        float jy = BlockJy[off + i];

                        sx += jx * p;
                        sy += jy * p;

                        // Calculate torque: τ = r × F
                        // In 2D: τ = r.x * F.y - r.y * F.x (scalar torque)
                        // Force at cell i: F = (jx * p * CellSize, jy * p * CellSize) * CellSize
                        int2 coord = Idx2Coord(i);
                        float2 cellCenter = ((float2)coord + 0.5f) * CellSize;
                        float2 r = cellCenter - block.Center;

                        // Force components (pressure × area)
                        float fx = jx * p * CellSize;
                        float fy = jy * p * CellSize;

                        // Torque = r × F (2D cross product gives scalar)
                        torque += r.x * fy - r.y * fx;
                    }
                    BlockForces[b] = new float2(sx, sy);
                    BlockTorques[b] = torque;
                }
            }
        }

        [BurstCompile]
        private struct UpdateBlocksJob : IJob
        {
            public NativeArray<SolidBlock> Blocks;
            [ReadOnly] public NativeArray<float2> BlockForces;
            [ReadOnly] public NativeArray<float> BlockTorques;
            [ReadOnly] public NativeArray<float> BlockCoeff;
            [ReadOnly] public NativeArray<float> BlockAngularCoeff;
            public float Gravity;

            public void Execute()
            {
                for (int b = 0; b < Blocks.Length; b++)
                {
                    var block = Blocks[b];
                    float2 S = BlockForces[b];
                    float torque = BlockTorques[b];

                    // Store old position for PBD velocity update
                    float2 oldCenter = block.Center;
                    float oldAngle = block.Angle;

                    // Linear motion: Pressure force F = CellSize² * S (pressure × area = force).
                    // Velocity change: dv = (F / mass) * dt + gravity * dt
                    // Since BlockCoeff = dt / mass, we get: dv = CellSize² * BlockCoeff * S + gravity * dt
                    block.Velocity += CellSize * CellSize * BlockCoeff[b] * S + new float2(0, Gravity) * DeltaTime;

                    // Angular motion: τ = I * α, where α = dω/dt
                    // Angular velocity change: dω = (τ / I) * dt
                    // Since BlockAngularCoeff = dt / I, we get: dω = BlockAngularCoeff * τ
                    block.AngularVelocity += CellSize * BlockAngularCoeff[b] * torque;

                    // Semi-implicit Euler: integrate position using new velocity
                    float2 newCenter = block.Center + block.Velocity * DeltaTime;
                    float newAngle = block.Angle + block.AngularVelocity * DeltaTime;

                    // PBD: Apply position constraints considering rotation
                    // For a rotated square, the worst case is diagonal = sqrt(2) * halfExtent
                    float worldMin = 0;
                    float worldMax = GridRes * CellSize;

                    // Use conservative boundary based on diagonal
                    float rotatedRadius = block.HalfExtent * 1.42f; // slightly more than sqrt(2)
                    float min = rotatedRadius;
                    float max = GridRes * CellSize - rotatedRadius;

                    bool collisionX = false;
                    bool collisionY = false;

                    // First pass: conservative constraint on center
                    if (newCenter.x < min)
                    {
                        newCenter.x = min;
                        collisionX = true;
                    }
                    else if (newCenter.x > max)
                    {
                        newCenter.x = max;
                        collisionX = true;
                    }

                    if (newCenter.y < min)
                    {
                        newCenter.y = min;
                        collisionY = true;
                    }
                    else if (newCenter.y > max)
                    {
                        newCenter.y = max;
                        collisionY = true;
                    }

                    // Second pass: check actual corner positions and correct if needed
                    float c = math.cos(newAngle);
                    float s = math.sin(newAngle);
                    float he = block.HalfExtent;

                    // 4 corners in local space
                    float2 corner0 = new float2(-he, -he);
                    float2 corner1 = new float2(he, -he);
                    float2 corner2 = new float2(he, he);
                    float2 corner3 = new float2(-he, he);

                    // Transform to world space and find worst penetration
                    float maxPenX = 0;
                    float maxPenY = 0;

                    for (int ci = 0; ci < 4; ci++)
                    {
                        float2 local = ci == 0 ? corner0 : ci == 1 ? corner1 : ci == 2 ? corner2 : corner3;
                        float2 world = newCenter + new float2(c * local.x - s * local.y, s * local.x + c * local.y);

                        // Check X penetration
                        if (world.x < worldMin)
                        {
                            float pen = worldMin - world.x;
                            if (pen > maxPenX) maxPenX = pen;
                            collisionX = true;
                        }
                        else if (world.x > worldMax)
                        {
                            float pen = worldMax - world.x;
                            if (pen < maxPenX) maxPenX = pen;
                            collisionX = true;
                        }

                        // Check Y penetration
                        if (world.y < worldMin)
                        {
                            float pen = worldMin - world.y;
                            if (pen > maxPenY) maxPenY = pen;
                            collisionY = true;
                        }
                        else if (world.y > worldMax)
                        {
                            float pen = worldMax - world.y;
                            if (pen < maxPenY) maxPenY = pen;
                            collisionY = true;
                        }
                    }

                    // Apply corner correction
                    newCenter.x += maxPenX;
                    newCenter.y += maxPenY;

                    // PBD: Update velocity based on corrected position (implicit velocity damping from constraint)
                    // v = (x_new - x_old) / dt
                    float2 correctedVelocity = (newCenter - oldCenter) / DeltaTime;

                    // Apply restitution and friction on collision
                    float restitution = 1f;  // 90% bounce
                    float friction = 0.1f;      // friction coefficient
                    if (collisionX)
                    {
                        // Separate normal and tangential components

                        // Normal velocity (reflected with restitution)
                        correctedVelocity.x *= -restitution;

                        // Tangential velocity (reduced by friction)
                        correctedVelocity.y *= (1.0f - friction);

                        // Angular velocity damping on collision
                        block.AngularVelocity *= (1.0f - friction);
                    }

                    if (collisionY)
                    {
                        correctedVelocity.y *= -restitution;
                        correctedVelocity.x *= (1.0f - friction);

                        block.AngularVelocity *= (1.0f - friction);
                    }

                    block.Velocity = correctedVelocity;
                    block.Center = newCenter;

                    // Angular constraint: keep angle in [-π, π]
                    block.Angle = math.fmod(newAngle + math.PI, 2 * math.PI) - math.PI;

                    Blocks[b] = block;
                }
            }
        }

        public void Test()
        {
            var pos = new float2(5 + UnityEngine.Random.value, 6 + UnityEngine.Random.value);
            int2 coord = GetCoord(pos);
                
            float2 weightSum = float2.zero;
                
            for (int x = coord.x - 1; x <= coord.x + 2; ++x)
            for (int y = coord.y - 1; y <= coord.y + 2; ++y)
            {
                int2 nCoord = new int2(x, y);
                int idx = Coord2Idx(math.clamp(nCoord, 0, GridRes - 1));

                float2 pos_u = (nCoord + new float2(0, 0.5f)) * CellSize;
                float2 pos_v = (nCoord + new float2(0.5f, 0)) * CellSize;

                float2 weights = new float2(GetWeight(pos_u - pos, InvCellSize),
                    GetWeight(pos_v - pos, InvCellSize));
                weightSum += weights;

            }
            Debug.Log($"weight sum: {weightSum.x}, {weightSum.y}");
        }
            
        private static float GetWeight(float2 delta_pos, float grid_inv_spacing)
        {
            float2 dist = math.abs(delta_pos * grid_inv_spacing);

            float2 weight = math.saturate(GetQuadraticWeight(dist));

            return weight.x * weight.y;
        }
    }
}

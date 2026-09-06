using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;
using Random = Unity.Mathematics.Random;

namespace PF_FLIP
{
    public abstract class MSBGConstants
    {
        public const uint AIR = 0;
        public const uint FLUID = 1;
        public const uint SOLID = 2;
        
        public const float InvDeltaTime = 60f;
        public const float DeltaTime = 1.0f / InvDeltaTime;

        public const int WidthLevel2 = 2;
        public const int WidthLevel1 = 4;
        public const int WidthLevel0 = 8;
        public const int BaseBlockWidth = WidthLevel0;

        public const float BlockSize = 8;
        public const float SizeLevel0 = BlockSize / WidthLevel0;
        public const float SizeLevel1 = BlockSize / WidthLevel1;
        public const float SizeLevel2 = BlockSize / WidthLevel2;
        public const float BaseCellSize = SizeLevel0;
        public const float InvBaseCellSize = 1.0f / BaseCellSize;

        public const int PoolSize = 16384;
        public const int GridWidth = 16;
        public const int GridCount = GridWidth * GridWidth;
    }
    
    public enum SolverType
    {
        GS,
        MG,
        CG,
        MGPCG,
        None
    }
    
    public class MultiResSparseBlockGrids : System.IDisposable
    {
        #region Params

        private NativeArray<float3> _gridLaplacian;
        private NativeArray<float2> _gridVel;
        private NativeArray<float2> _gridVelCopy;
        private NativeArray<float2> _gridDelaPos;
        private NativeArray<float> _gridDensity;
        private NativeArray<float> _gridDivergence;
        private NativeArray<float> _gridPressure;
        private NativeArray<float> _gridWeight;
        private NativeArray<float> _gridDPressure;
        private NativeArray<uint> _gridTypesPool;
        
        private NativeArray<float2> _gridVelDS;
        private NativeArray<float2> _gridVelCopyDS;
        
        private NativeArray<int> _start;
        private NativeArray<int> _end;
        private NativeArray<int> _gridLevels;
        private NativeArray<int2> _gridRange;
        private NativeArray<int4> _gridLut;
        private NativeArray<int> _gridCounter;
        
        private NativeReference<int> _blockCount;

        private MSBG_Solver _solver;
        
        private const int BatchCount = 64;
        
        #endregion

        public MultiResSparseBlockGrids()
        {
            const int poolSize = MSBGConstants.PoolSize;
            _gridVel = new NativeArray<float2>(poolSize, Allocator.Persistent);
            _gridVelCopy = new NativeArray<float2>(poolSize, Allocator.Persistent);
            _gridDelaPos = new NativeArray<float2>(poolSize, Allocator.Persistent);
            _gridDensity = new NativeArray<float>(poolSize, Allocator.Persistent);
            _gridDivergence = new NativeArray<float>(poolSize, Allocator.Persistent);
            _gridPressure = new NativeArray<float>(poolSize, Allocator.Persistent);
            _gridWeight = new NativeArray<float>(poolSize, Allocator.Persistent);
            _gridDPressure = new NativeArray<float>(poolSize, Allocator.Persistent);
            _gridTypesPool = new NativeArray<uint>(poolSize, Allocator.Persistent);
            _gridLaplacian = new NativeArray<float3>(poolSize, Allocator.Persistent);
            _gridVelDS = new NativeArray<float2>(poolSize / 2, Allocator.Persistent);
            _gridVelCopyDS = new NativeArray<float2>(poolSize / 2, Allocator.Persistent);
            
            const int size = MSBGConstants.GridCount;
            _start = new NativeArray<int>(size, Allocator.Persistent);
            _end = new NativeArray<int>(size, Allocator.Persistent);
            _gridRange = new NativeArray<int2>(size, Allocator.Persistent);
            _gridLevels = new NativeArray<int>(size, Allocator.Persistent);
            _gridLut = new NativeArray<int4>(size, Allocator.Persistent);
            _gridCounter = new NativeArray<int>(size, Allocator.Persistent);
            
            _blockCount = new NativeReference<int>(Allocator.Persistent);

            _solver = new MSBG_Solver(_gridLut, _gridLaplacian, _gridPressure, _gridDivergence);
        }

        public void RandomInit(int iter, SolverType type, out float rs)
        {
            const int testWidth = MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth;
            const int gridWidth = MSBGConstants.GridWidth;
            ClearGrid();
            rs = 0;
            int ptr = 0;
            for (int y = 0; y < gridWidth; y++)
            for (int x = 0; x < gridWidth; x++)
            {
                float fx = (x + 0.5f)/gridWidth;
                float fy = (y + 0.5f)/gridWidth;
                if (math.abs(math.length(new float2(fx, fy) - 0.5f) - 0.3f + 0.2f / gridWidth) < 0.7f / gridWidth)
                    _gridLevels[x + y * gridWidth] = 0;
                else
                    _gridLevels[x + y * gridWidth] = 2;
            }
            new ComputeDistanceFieldJob(_gridLevels).Schedule().Complete();
            for (int y = 0; y < gridWidth; y++)
            for (int x = 0; x < gridWidth; x++)
            {
                float fx = (x + 0.5f)/gridWidth;
                float fy = (y + 0.5f)/gridWidth;
                
                if (math.length(new float2(fx, fy) - 0.5f) > 0.3f + 0.4f / gridWidth)
                {
                    _gridLut[x + y * gridWidth] = new int4(0, 0, -1, -1);
                    continue;
                }
                
                int level = math.min(_gridLevels[x + y * gridWidth], 2);
                int blockWidth = 1 << (4 - level);
                _gridLut[x + y * gridWidth] = new int4(ptr, ptr, level, level);
                ptr += blockWidth * blockWidth;
            }
            
            for (int y = 0; y < gridWidth; y++)
            for (int x = 0; x < gridWidth; x++)
            {
                var info = _gridLut[x + y * gridWidth];
                
                int level = info.z;
                if (level < 0) continue;
                int levelR = x < gridWidth - 1 ? _gridLevels[x + 1 + y * gridWidth] : -1;
                int levelT = y < gridWidth - 1 ? _gridLevels[x + (y + 1) * gridWidth] : -1;
                int levelRT = x < gridWidth - 1 && y < gridWidth - 1 ? _gridLevels[x + 1 + (y + 1) * gridWidth] : -1;
                int4 levels = new int4(level, levelR, levelT, levelRT);
                levels = math.select(levels, 15, levels < 0);
                info.z = PackNeighborsLevel(levels);
                _gridLut[x + y * gridWidth] = info;
            }
            
            if (ptr > MSBGConstants.PoolSize)
            {
                Debug.Log("Pool size is not enough! " + ptr);
                return;
            }
            _solver.ActiveGridCount = ptr;
            var rnd = Random.CreateFromIndex(1);
            for (int gy = 0; gy < gridWidth; gy++)
            for (int gx = 0; gx < gridWidth; gx++)
            {
                int4 info = _gridLut[gx + gy * gridWidth];
                int level = GetCurLevel(info.z);
                if (level < 0) 
                    continue;

                int blockWidth = GetBlockWidth(level);
                int offset = info.x;
                for (int by = 0; by < blockWidth; by++)
                for (int bx = 0; bx < blockWidth; bx++)
                {
                    int x = gx * blockWidth + bx;
                    int y = gy * blockWidth + by;
                    // if (x < 1 || y < 1 || x > testWidth - 2 || y > testWidth - 2) continue;
                    int idx = offset + by * blockWidth + bx;
                    int baseX = x << level;
                    int baseY = y << level;
                    if (math.length(new float2(baseX, baseY) - testWidth * 0.5f) < testWidth * 0.3f)
                    {
                        _gridLaplacian[idx] = new float3(4, -1, -1);
                        _gridVel[idx] = rnd.NextFloat2Direction() * 0.3f;
                    }
                }
            }
            
            var dotResult = new NativeReference<float>(Allocator.TempJob);
            new CalcDivergenceJob()
            {
                GridLut = _gridLut,
                GridVelocity = _gridVel,
                GridLaplacian = _gridLaplacian,
                GridDensity = _gridDensity,
                GridDivergence = _gridDivergence
            }.Schedule(MSBGConstants.GridCount, BatchCount).Complete();
            
            new Dot(_gridDivergence, _gridDivergence, dotResult).Schedule().Complete();
            Debug.Log($"Used pool: {ptr} / {MSBGConstants.PoolSize}, Init with rs: " + dotResult.Value);

            switch (type)
            {
                case SolverType.CG:
                    _solver.Solve_CG(iter, out _, out rs);
                    break;
                case SolverType.GS:
                    _solver.Solve_GS(iter, out rs);
                    break;
                case SolverType.MG:
                    _solver.Solve_MG(iter, out rs);
                    break;
                case SolverType.MGPCG:
                    _solver.Solve_MGPCG(iter, out rs);
                    break;
                default:
                    _solver.BuildLutPyramid().Complete();
                    break;
            }
            
            new UpdateVelocity()
            {
                GridLut = _gridLut,
                GridLaplacian = _gridLaplacian,
                GridVelocity = _gridVel,
                GridPressure = _gridPressure,
            }.Schedule(MSBGConstants.GridCount, BatchCount).Complete();
            // new PostprocessJob()
            // {
            //     GridLut = _gridLut,
            //     GridVelocity = _gridVel,
            // }.Schedule(MSBGConstants.GridCount, BatchCount).Complete();
            new CalcDivergenceJob()
            {
                GridLut = _gridLut,
                GridVelocity = _gridVel,
                GridLaplacian = _gridLaplacian,
                GridDensity = _gridDensity,
                GridDivergence = _gridDivergence
            }.Schedule(MSBGConstants.GridCount, BatchCount).Complete();
            // _solver.CalcDerive();
            
            new Dot(_gridDivergence, _gridDivergence, dotResult).Schedule().Complete();
            Debug.Log("Solved rs: " + dotResult.Value);
            
            dotResult.Dispose();
        }
        
        public void GetLevels(int[] levels)
        {
            for (int i = 0; i < _gridLut.Length; i++)
            {
                levels[i] = GetCurLevel(_gridLut[i].z);
            }
        }
        
        public void SampleField(float[] field, int width, int level)
        {
            _solver.DownSampleField();
            _solver.SampleFieldBilinear(field, width);
            // const int ww = MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth;
            // float w = width;
            // for (int gy = 0; gy < width; gy++)
            // for (int gx = 0; gx < width; gx++)
            // {
            //     int i = gy * width + gx;
            //     float2 coord = new float2(gx/w * ww, gy/w * ww);
            //     // field[i] = _solver.SampleLevel_Point((int2)coord, level);
            //     field[i] = _solver.SampleLevel_Bilinear(coord + 0.5f/ww, level);
            // }
        }
        
        public void Dispose()
        {
            _gridVel.Dispose();
            _gridVelCopy.Dispose();
            _gridDelaPos.Dispose();
            _gridDensity.Dispose();
            _gridDivergence.Dispose();
            _gridPressure.Dispose();
            _gridWeight.Dispose();
            _gridDPressure.Dispose();
            _gridTypesPool.Dispose();
            _gridVelDS.Dispose();
            _gridVelCopyDS.Dispose();
            
            _start.Dispose();
            _end.Dispose();
            _gridRange.Dispose();
            _gridLevels.Dispose();
            _gridLut.Dispose();
            _gridLaplacian.Dispose();
            _solver.Dispose();
            _gridCounter.Dispose();

            _blockCount.Dispose();
        }

        public void DrawGridType()
        {
            for (int gy = 0; gy < MSBGConstants.GridWidth; gy++)
            for (int gx = 0; gx < MSBGConstants.GridWidth; gx++)
            {
                int idx = gx + gy * MSBGConstants.GridWidth;
                int4 info = _gridLut[idx];
                int level = GetCurLevel(info.z);
                if (level < 0) 
                    continue;

                int blockWidth = GetBlockWidth(level);
                int offset = info.x;
                float2 posBase = new float2(gx, gy) * MSBGConstants.BlockSize;
                for (int by = 0; by < blockWidth; by++)
                for (int bx = 0; bx < blockWidth; bx++)
                {
                    int ix = offset + by * blockWidth + bx;
                    if (ix < 0 || ix >= MSBGConstants.PoolSize || !IsFluidCell(_gridTypesPool[ix]))
                        continue;
                    Gizmos.color = level == 0
                        ? new Color(1, 0, 0, 0.5f)
                        : (level == 1 ? new Color(1, 0.5f, 0, 0.5f) : new Color(1, 1, 1, 0.5f));
                    float cellSize = GetCellSize(level);
                    float2 pos = posBase + new float2(bx + 0.5f , by + 0.5f) * cellSize;
                    Gizmos.DrawWireCube(new Vector3(pos.x * 0.1f, pos.y * 0.1f, 0),
                        new Vector3(cellSize * 0.1f, cellSize * 0.1f, 0));
                }
            }
        }
        
        public void ClearGrid(JobHandle handle = default)
        {
            new ClearGridLutJob(_start, _end, _gridCounter).Schedule(MSBGConstants.GridCount, BatchCount, handle).Complete();
            
            new ClearGridPoolJob{
                Velocity = _gridVel,
                DeltaPos = _gridDelaPos,
                Divergence = _gridDivergence,
                Weight = _gridWeight,
                Pressure = _gridPressure,
                DPressure = _gridDPressure,
                Density = _gridDensity
            }.Schedule(MSBGConstants.PoolSize, BatchCount, handle).Complete();
        }
        
        public void BuildSpatialLookup(NativeArray<int2> particleHash, NativeArray<Particle> particles, 
            NativeArray<float2> particleVel, NativeArray<Particle> particlesCopy, 
            NativeArray<float2> particleVelCopy, int activeParticleCount, JobHandle handle = default)
        {
            Profiler.BeginSample("ParticleHash");
            new HashJob(particles, particleHash).Schedule(activeParticleCount, BatchCount, handle).Complete();
            Profiler.EndSample();

            Profiler.BeginSample("Sort");
            particleHash.Slice(0, activeParticleCount).SortJob(new Int2Comparer()).Schedule(handle).Complete();
            Profiler.EndSample();
        
            Profiler.BeginSample("BuildParticleLut");
            new BuildLutJob(particleHash, _start, _end, activeParticleCount)
                .Schedule(activeParticleCount, BatchCount, handle).Complete();
            
            new CombineLutJob(_start, _end, _gridRange).Schedule(_gridRange.Length, BatchCount, handle).Complete();
            Profiler.EndSample();
            
            Profiler.BeginSample("Shuffle");
            new ShuffleJob{
                Hashes = particleHash,
                PosRaw = particles,
                PosNew = particlesCopy,
                VelRaw = particleVel,
                VelNew = particleVelCopy,
            }.Schedule(activeParticleCount, BatchCount, handle).Complete();
            Profiler.EndSample();
        }
        
        public bool AllocateBlocks(out int blockCount, JobHandle handle = default)
        {
            Profiler.BeginSample("ComputeGridLevel");
            new ComputeGridLevelJob
            {
                GridRange = _gridRange,
                GridLut = _gridLut,
                GridType = _gridTypesPool,
                GridLevel = _gridLevels
            }.Schedule(MSBGConstants.GridCount, BatchCount).Complete();
            Profiler.EndSample();

            Profiler.BeginSample("ComputeDistanceField");
            new ComputeDistanceFieldJob(_gridLevels).Schedule().Complete();
            Profiler.EndSample();
            
            Profiler.BeginSample("AllocateBlock");
            new AllocateBlockJob{
                GridLevel = _gridLevels,
                GridRange = _gridRange,
                GridLut = _gridLut,
                BlockCount = _blockCount
            }.Schedule().Complete();
            Profiler.EndSample();
            
            blockCount = _blockCount.Value;
            _solver.ActiveGridCount = blockCount;
            
            return _blockCount.Value < MSBGConstants.PoolSize;
        }
        
        public bool AllocateBlocksStart(JobHandle handle = default)
        {
            new SetGridLevelJob
            {
                GridRange = _gridRange,
                GridLevel = _gridLevels
            }.Schedule(MSBGConstants.GridCount, BatchCount).Complete();
            
            new AllocateBlockJob{
                GridLevel = _gridLevels,
                GridRange = _gridRange,
                GridLut = _gridLut,
                BlockCount = _blockCount
            }.Schedule().Complete();
            
            return _blockCount.Value < MSBGConstants.PoolSize;
        }
        
        public void ParticleToGrid(NativeArray<Particle> particles,
            NativeArray<float2> particleVel, JobHandle handle = default)
        {
            Profiler.BeginSample("ParticleToGrid");
            new ParticleToGridJob{
                ParticleRange = _gridRange,
                GridLut = _gridLut,
                Particles = particles,
                ParticleVel = particleVel,
                GridDensity = _gridDensity,
                GridVelocity = _gridVel
            }.Schedule(MSBGConstants.GridCount, BatchCount, handle).Complete();
            Profiler.EndSample();
            
            _gridVelCopy.CopyFrom(_gridVel);

            Profiler.BeginSample("SetGridType");
            new SetGridTypeJob
            {
                GridLut = _gridLut,
                Density = _gridDensity,
                GridType = _gridTypesPool,
                GridLaplacian = _gridLaplacian
            }.Schedule(MSBGConstants.GridCount, BatchCount, handle).Complete();
            Profiler.EndSample();
        }
        
        public void SolveMultiGridPressure(float2 gravity, out float rs, JobHandle handle = default)
        {
            Profiler.BeginSample("AddForce");
            handle = new AddForceJob
            {
                GridTypes = _gridTypesPool,
                GridVelocity = _gridVel,
                Gravity = gravity
            }.Schedule(_blockCount.Value, BatchCount, handle);
            Profiler.EndSample();
        
            Profiler.BeginSample("CalcDivergence");
            new CalcDivergenceJob
            {
                GridLut = _gridLut,
                GridVelocity = _gridVel,
                GridLaplacian = _gridLaplacian,
                GridDensity = _gridDensity,
                GridDivergence = _gridDivergence
            }.Schedule(MSBGConstants.GridCount, BatchCount, handle).Complete();
            Profiler.EndSample();
        
            Profiler.BeginSample("Solver");
            rs = 1;
            _solver.Solve_GS(4, out rs);
            // _solver.Solve_CG(10, out _, out rs);
            Profiler.EndSample();
            
            Profiler.BeginSample("UpdateVelocity");
            new UpdateVelocity()
            {
                GridLaplacian = _gridLaplacian,
                GridLut = _gridLut,
                GridPressure = _gridPressure,
                GridVelocity = _gridVel
            }.Schedule(MSBGConstants.GridCount, BatchCount, handle).Complete();
            Profiler.EndSample();
            
            Profiler.BeginSample("Postprocess");
            new PostprocessJob()
            {
                GridLut = _gridLut,
                GridVelocity = _gridVel,
            }.Schedule(MSBGConstants.GridCount, BatchCount, handle).Complete();
            Profiler.EndSample();
            
            Profiler.BeginSample("DownSample");
            new DownSample(_gridVel, _gridVelDS, _gridVelCopy, _gridVelCopyDS, _gridLut)
                .Schedule(MSBGConstants.GridCount, BatchCount).Complete();
            Profiler.EndSample();
        }

        public void GridToParticle(NativeArray<Particle> particles,
            NativeArray<float2> particleVel, int activeParticleCount, float flipness, JobHandle handle = default)
        {
            Profiler.BeginSample("GridToParticle");
            new GridToParticleJob()
            {
                Flipness = flipness,
                GridLut = _gridLut,
                GridVelocityNew = _gridVel,
                GridVelocityNewDS = _gridVelDS,
                GridVelocityOld = _gridVelCopy,
                GridVelocityOldDS = _gridVelCopyDS,
                Particles = particles,
                ParticleVel = particleVel
            }.Schedule(activeParticleCount, BatchCount).Complete();
            Profiler.EndSample();
            
            Profiler.BeginSample("Advection");
            new AdvectionJob
            {
                GridVelocity = _gridVel,
                GridVelocityDS = _gridVelDS,
                GridLut = _gridLut,
                Particles = particles,
            }.Schedule(activeParticleCount, BatchCount).Complete();
            Profiler.EndSample();
        }
        
        public bool ResampleParticles(NativeArray<Particle> particles, NativeArray<float2> particleVel, 
            NativeArray<Particle> particlesCopy, NativeArray<float2> particleVelCopy, NativeReference<int> particleCount, JobHandle handle = default)
        {
            int oldCount = particleCount.Value;
            Profiler.BeginSample("ParticleLevel");
            new ParticleLevelJob()
            {
                GridLut = _gridLut,
                Particles = particles,
            }.Schedule(particleCount.Value, BatchCount).Complete();
            Profiler.EndSample();
            uint seed = (uint)Time.frameCount;
            Profiler.BeginSample("GridParticleCompile");
            new GridParticleCompileJob()
            {
                GridLut = _gridLut,
                Particles = particles,
                ParticleRange = _gridRange,
                GridCounter = _gridCounter,
                Seed = seed
            }.Schedule(MSBGConstants.GridCount, BatchCount).Complete();
            
            new GridParticlePrefixSumJob
            {
                GridCounter = _gridCounter,
                TotalCount = particleCount
            }.Schedule().Complete();
            Profiler.EndSample();
            
            if (particleCount.Value < Adpative_FLIP.ParticlePoolSize)
            {
                Profiler.BeginSample("ParticleResample");
                new ParticleResampleJob
                {
                    GridLut = _gridLut,
                    GridCounter = _gridCounter,
                    ParticleRange = _gridRange,
                    Particles = particles,
                    ParticlesVel = particleVel,
                    NewParticles = particlesCopy,
                    NewParticlesVel = particleVelCopy,
                    Seed = seed
                }.Schedule(MSBGConstants.GridCount, BatchCount).Complete();
                Profiler.EndSample();
            }
            else
            {
                particleCount.Value = oldCount;
            }
            
            return particleCount.Value < Adpative_FLIP.ParticlePoolSize;
        }
        
        #region Jobs

        [BurstCompile]
        private struct ClearGridLutJob : IJobParallelFor
        {
            [WriteOnly] private NativeArray<int> _start;
            [WriteOnly] private NativeArray<int> _end;
            [WriteOnly] private NativeArray<int> _counter;

            public ClearGridLutJob(NativeArray<int> start, NativeArray<int> end, NativeArray<int> counter)
            {
                _start = start;
                _end = end;
                _counter = counter;
            }

            public void Execute(int i)
            {
                _start[i] = 0;
                _end[i] = 0;
                _counter[i] = 0;
            }
        }

        [BurstCompile]
        private struct ClearGridPoolJob : IJobParallelFor
        {
            [WriteOnly] public NativeArray<float2> Velocity;
            [WriteOnly] public NativeArray<float2> DeltaPos;
            [WriteOnly] public NativeArray<float> Divergence;
            [WriteOnly] public NativeArray<float> Weight;
            [WriteOnly] public NativeArray<float> Pressure;
            [WriteOnly] public NativeArray<float> DPressure;
            [WriteOnly] public NativeArray<float> Density;

            public void Execute(int i)
            {
                Velocity[i] = float2.zero;
                DeltaPos[i] = float2.zero;
                Divergence[i] = 0;
                Pressure[i] = 0;
                DPressure[i] = 0;
                Density[i] = 0;
                Weight[i] = 0;
            }
        }
        
        [BurstCompile]
        private struct SetGridLevelJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> GridRange;
            [WriteOnly] public NativeArray<int> GridLevel;

            public void Execute(int i)
            {
                int2 range = GridRange[i];
                GridLevel[i] = range.y > range.x ? 1 : 2;
            }
        }
        
        [BurstCompile]
        private struct ComputeGridLevelJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int4> GridLut;
            [ReadOnly] public NativeArray<int2> GridRange;
            [ReadOnly] public NativeArray<uint> GridType;
            [WriteOnly] public NativeArray<int> GridLevel;

            public void Execute(int i)
            {
                int2 range = GridRange[i];
                int4 info = GridLut[i];
                int2 coord = Idx2Coord(i);

                if (GetCurLevel(info.z) < 0 && range.y > range.x)
                {
                    GridLevel[i] = 0;
                    return;
                }

                int rightBound = MSBGConstants.GridWidth - 1;

                int2 rangeL = coord.x > 0 ? GridRange[Coord2Idx(coord.x - 1, coord.y)] : int2.zero;
                int2 rangeD = coord.y > 0 ? GridRange[Coord2Idx(coord.x, coord.y - 1)] : int2.zero;
                int2 rangeR = coord.x < rightBound ? GridRange[Coord2Idx(coord.x + 1, coord.y)] : int2.zero;
                int2 rangeU = coord.y < rightBound ? GridRange[Coord2Idx(coord.x, coord.y + 1)] : int2.zero;
                if (range.y <= range.x && rangeL.y <= rangeL.x && rangeD.y <= rangeD.x && rangeR.y <= rangeR.x &&
                    rangeU.y <= rangeU.x)
                {
                    GridLevel[i] = 2;
                    return;
                }

                float2 coordCenter = ((float2)coord + 0.5f) * MSBGConstants.BlockSize;
                int2 minCoord = math.max(0, coord - 1);
                int2 maxCoord = math.min(MSBGConstants.GridWidth - 1, coord + 1);
                float nearestDist = 100000;
                for (int y = minCoord.y; y <= maxCoord.y; y++)
                for (int x = minCoord.x; x <= maxCoord.x; x++)
                {
                    int neighborIdx = Coord2Idx(x, y);
                    int4 neighborInfo = GridLut[neighborIdx];
                    int neighborLevel = GetCurLevel(neighborInfo.z);
                    if (neighborLevel < 0) continue;
                    int blockWidth = GetBlockWidth(neighborLevel);
                    int ph = neighborInfo.x;
                    float2 blockBase = new float2(x, y) * MSBGConstants.BlockSize;
                    float cellSize = GetCellSize(neighborLevel);
                    for (int yy = 0; yy < blockWidth; yy++)
                    for (int xx = 0; xx < blockWidth; xx++)
                    {
                        if (!IsEdgeFluidCell(GridType[ph + yy * blockWidth + xx])) continue;
                        float2 pos = blockBase + (new float2(xx, yy) + 0.5f) * cellSize;
                        var vec =math.abs(pos - coordCenter) - blockWidth * 0.5f;
                        nearestDist = math.min(nearestDist,  math.max(0, math.max(vec.x, vec.y) - cellSize * 0.5f));
                    }
                }
                
                if (range.y <= range.x && nearestDist <= MSBGConstants.BaseCellSize)
                    GridLevel[i] = 0;
                else if (range.y > range.x && nearestDist <= MSBGConstants.BlockSize * 0.25f)
                    GridLevel[i] = 0;
                else
                    GridLevel[i] = 2;
            }
        }

        [BurstCompile]
        private struct ComputeDistanceFieldJob : IJob
        {
            private NativeArray<int> _gridLevel;
            
            public ComputeDistanceFieldJob(NativeArray<int> level)
            {
                _gridLevel = level;
            }

            public void Execute()
            {
                int2 offset = new int2(1, 0);
                int rightBound = MSBGConstants.GridWidth - 1;
                for (int i = 0; i < MSBGConstants.GridCount; i++)
                {
                    int level = _gridLevel[i];
                    if (level <= 0) continue;
                    int2 coord = Idx2Coord(i);
                    if (coord.x > 0)
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord - offset.xy)]);
                    if (coord.y > 0)
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord - offset.yx)]);
                    if (math.all(coord > 0))
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord - offset.xx)]);
                    if (coord.x < rightBound && coord.y > 0)
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord - new int2(-1, 1))]);

                    _gridLevel[i] = level;
                }
                
                for (int i = MSBGConstants.GridCount - 1; i >= 0; i--)
                {
                    int level = _gridLevel[i];
                    if (level <= 0) continue;
                    int2 coord = Idx2Coord(i);
                    if (coord.x < rightBound)
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord + offset.xy)]);
                    if (coord.y < rightBound)
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord + offset.yx)]);
                    if (math.all(coord < rightBound))
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord + offset.xx)]);
                    if (coord.x > 0 && coord.y < rightBound)
                        level = math.min(level, 1 + _gridLevel[Coord2Idx(coord + new int2(-1, 1))]);

                    _gridLevel[i] = level;
                }
            }
        }

        [BurstCompile]
        private struct HashJob : IJobParallelFor
        {
            [ReadOnly] private NativeArray<Particle> _ps;
            [WriteOnly] private NativeArray<int2> _hashes;
            
            public HashJob(NativeArray<Particle> ps, NativeArray<int2> hashes)
            {
                _ps = ps;
                _hashes = hashes;
            }

            public void Execute(int i)
            {
                int hash = Coord2Idx(GetCoord(_ps[i].Pos));
                _hashes[i] = math.int2(hash, i);
            }

            private int2 GetCoord(float2 pos)
            {
                return (int2)math.floor(pos / MSBGConstants.BlockSize);
            }
        }

        [BurstCompile]
        private struct BuildLutJob : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int2> _hashes;
            [NativeDisableParallelForRestriction] private NativeArray<int> _startIndices;
            [NativeDisableParallelForRestriction] private NativeArray<int> _endIndices;
            private readonly int _particleCount;
            
            public BuildLutJob(NativeArray<int2> hashes,NativeArray<int> start, NativeArray<int> end, int particleCount)
            {
                _particleCount = particleCount;
                _hashes = hashes;
                _startIndices = start;
                _endIndices = end;
            }

            public void Execute(int i)
            {
                int numParticles = _particleCount;
                int prev = i == 0 ? numParticles - 1 : i - 1;
                int next = i == numParticles - 1 ? 0 : i + 1;
                int currID = _hashes[i].x;
                int prevID = _hashes[prev].x;
                int nextID = _hashes[next].x;
                int idx = currID;
                if (currID != prevID) _startIndices[idx] = i;
                if (currID != nextID) _endIndices[idx] = i + 1;
            }
        }
    
        [BurstCompile]
        private struct CombineLutJob : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int> _startIndices;
            [ReadOnly] private NativeArray<int> _endIndices;
            [WriteOnly] private NativeArray<int2> _range;
            
            public CombineLutJob(NativeArray<int> start, NativeArray<int> end, NativeArray<int2> range)
            {
                _range = range;
                _startIndices = start;
                _endIndices = end;
            }
            
            public void Execute(int i)
            {
                _range[i] = new int2(_startIndices[i], _endIndices[i]);
            }
        }
        
        [BurstCompile]
        private struct ShuffleJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> Hashes;
            [ReadOnly] public NativeArray<Particle> PosRaw;
            [ReadOnly] public NativeArray<float2> VelRaw;
            [WriteOnly] public NativeArray<Particle> PosNew;
            [WriteOnly] public NativeArray<float2> VelNew;

            public void Execute(int i)
            {
                int id = Hashes[i].y;
                PosNew[i] = PosRaw[id];
                VelNew[i] = VelRaw[id];
            }
        }

        [BurstCompile]
        private struct AllocateBlockJob : IJob
        {
            [ReadOnly] public NativeArray<int> GridLevel;
            [ReadOnly] public NativeArray<int2> GridRange;
            [WriteOnly] public NativeArray<int4> GridLut;
            [WriteOnly] public NativeReference<int> BlockCount;

            public void Execute()
            {
                // int blockCount = 0;
                int ptr = 0;
                int ptr2 = 0;
                for (int i = 0; i < GridLevel.Length; i++)
                {
                    int2 range = GridRange[i];
                    if (range.y <= range.x && GridLevel[i] > 0)
                        GridLut[i] = new int4(0, 0, -1, -1);
                    else
                    {
                        int2 coord = Idx2Coord(i);
                        int4 level = new int4(GridLevel[i], 
                            ReadLevel(coord + new int2(1, 0)),
                            ReadLevel(coord + new int2(0, 1)),
                            ReadLevel(coord + new int2(1, 1)));

                        int4 level1 = math.select(math.min(level + 1, 2), 15, level < 0);
                        level = math.select(level, 15, level < 0);
                        GridLut[i] = new int4(ptr, ptr2, PackNeighborsLevel(level), PackNeighborsLevel(level1));
                        ptr += GetBlockSize(level.x);
                        ptr2 += GetBlockSize(level1.x);
                        // blockCount++;
                    }
                }
                BlockCount.Value = ptr;
            }

            private int ReadLevel(int2 coord)
            {
                if (math.any(coord >= MSBGConstants.GridWidth)) return -1;
                int i = Coord2Idx(coord);
                int2 range = GridRange[i];
                int level = GridLevel[i];
                if (range.y <= range.x && level > 0) return -1;
                return level;
            }
        }
        
        [BurstCompile]
        private struct ParticleToGridJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> ParticleRange;
            [ReadOnly] public NativeArray<int4> GridLut;
            [ReadOnly] public NativeArray<Particle> Particles;
            [ReadOnly] public NativeArray<float2> ParticleVel;
            [NativeDisableParallelForRestriction] [WriteOnly] public NativeArray<float> GridDensity;
            [NativeDisableParallelForRestriction] [WriteOnly] public NativeArray<float2> GridVelocity;

            public void Execute(int i)
            {
                int4 info = GridLut[i];
                int level = GetCurLevel(info.z);
                if (level < 0) 
                    return;
                
                int2 coord = Idx2Coord(i);

                int blockWidth = GetBlockWidth(level);
                int blockSize = blockWidth * blockWidth;
                
                var densityArr = new NativeArray<float>(blockSize, Allocator.Temp);
                var massArr = new NativeArray<float2>(blockSize, Allocator.Temp);
                var velocityArr = new NativeArray<float2>(blockSize, Allocator.Temp);
                
                float2 blockOrigin = (float2)coord * MSBGConstants.BlockSize;
                float cellSize = GetCellSize(level);
                
                int2 minCoord = math.max(coord - 1, int2.zero);
                int2 maxCoord = math.min(coord + 1, MSBGConstants.GridWidth - 1);
                for (int y = minCoord.y; y <= maxCoord.y; y++)
                for (int x = minCoord.x; x <= maxCoord.x; x++)
                {
                    int2 curr = new int2(x, y);
                    int2 pRange = ParticleRange[Coord2Idx(curr)];
                    for (int j = pRange.x; j < pRange.y; j++)
                    {
                        Particle p = Particles[j];
                        float radius = GetCellSize(p.Level);
                        float h2 = radius * radius;
                        float2 relativePos = p.Pos - blockOrigin;
                        int2 min = math.max(0, (int2)math.floor((relativePos - radius) / cellSize));
                        int2 max = math.min(blockWidth - 1, (int2)math.ceil((relativePos + radius) / cellSize));
                        for (int yy = min.y; yy <= max.y; yy++)
                        for (int xx = min.x; xx <= max.x; xx++)
                        {
                            int2 localCoord = new int2(xx, yy);
                            int idx = BlockCoord2Idx(localCoord, blockWidth);;
                            
                            float2 cellCenter = (new float2(xx, yy) + 0.5f) * cellSize;
                            densityArr[idx] += KernelFunc(math.lengthsq(cellCenter - relativePos), radius);
                            
                            float2 cellLeft = cellCenter - new float2(cellSize * 0.5f, 0);
                            float2 cellBottom = cellCenter - new float2(0, cellSize * 0.5f);
                            // float2 weights = new float2(KernelFunc(math.lengthsq(cellLeft - relativePos), h2),
                            //     KernelFunc(math.lengthsq(cellBottom - relativePos), h2));
                            float2 weights = new float2(KernelFunc(relativePos, cellLeft, radius),
                                KernelFunc(relativePos, cellBottom, radius));
                            velocityArr[idx] += weights * ParticleVel[j];
                            massArr[idx] += weights;
                        }
                    }
                }

                int offset = info.x;
                for (int j = 0; j < blockSize; j++)
                {
                    GridDensity[j + offset] = densityArr[j];
                    float2 weights = massArr[j];
                    float2 velocity = math.select(float2.zero, velocityArr[j] / weights, weights > 1e-5f);
                    GridVelocity[j + offset] = velocity;
                }
            }
            
            private float KernelFunc(float2 p_pos, float2 c_pos, float radius)
            {
                float2 dist = math.abs((p_pos - c_pos)) / radius;

                float2 weight = GetQuadraticWeight(dist);

                return weight.x * weight.y;
            }

            private static float2 GetQuadraticWeight(float2 abs_x)
            {
                float2 dst = math.saturate(1.5f - abs_x);
                return math.select(0.5f * dst * dst, 0.75f - abs_x * abs_x, abs_x < 0.5f);
            }

            private float KernelFunc(float r2, float h2)
            {
                float k = math.max(0, 1 - r2 / h2);
                return k * k * k;
            }
        }
        
        [BurstCompile]
        private struct SetGridTypeJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int4> GridLut;
            [ReadOnly] public NativeArray<float> Density;
            [NativeDisableParallelForRestriction] [WriteOnly] public NativeArray<uint> GridType;
            [NativeDisableParallelForRestriction] [WriteOnly] public NativeArray<float3> GridLaplacian;

            public void Execute(int i)
            {
                int4 info = GridLut[i];
                int level = GetCurLevel(info.z);
                if (level < 0) 
                    return;
                
                int pn = info.x;
                
                int blockWidth = GetBlockWidth(level);
                int haloBlockWidth = blockWidth + 2;
                int haloBlockSize = haloBlockWidth * haloBlockWidth;
                var densityArr = new NativeArray<float>(haloBlockSize, Allocator.Temp);
                int2 baseCoord = Idx2Coord(i) * blockWidth;
                
                FillHaloBlock(Density, GridLut, densityArr, Idx2Coord(i), info);

                for (int by = 0; by < blockWidth; by++)
                for (int bx = 0; bx < blockWidth; bx++)
                {
                    int idx = pn + BlockCoord2Idx(bx, by, blockWidth);
                    float dc = densityArr[BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth)];
                    float3 param = dc > 1e-4f ? new float3(4, -1, -1) : float3.zero;
                    
                    float dl = densityArr[BlockCoord2Idx(bx + 0, by + 1, haloBlockWidth)];
                    float dr = densityArr[BlockCoord2Idx(bx + 2, by + 1, haloBlockWidth)];
                    float db = densityArr[BlockCoord2Idx(bx + 1, by + 0, haloBlockWidth)];
                    float dt = densityArr[BlockCoord2Idx(bx + 1, by + 2, haloBlockWidth)];

                    uint gridType = GetGridType(baseCoord + new int2(bx, by), dc);
                    gridType |= GetGridType(baseCoord + new int2(bx - 1, by), dl) << 2;
                    gridType |= GetGridType(baseCoord + new int2(bx + 1, by), dr) << 4;
                    gridType |= GetGridType(baseCoord + new int2(bx, by - 1), db) << 6;
                    gridType |= GetGridType(baseCoord + new int2(bx, by + 1), dt) << 8;
                    
                    GridLaplacian[idx] = param;
                    GridType[idx] = gridType;
                }

                densityArr.Dispose();
            }

            private uint GetGridType(int2 coord, float density)
            {
                const int baseGridWidth = MSBGConstants.BaseBlockWidth * MSBGConstants.GridWidth;
                if (math.any(coord < 0) || math.any(coord > baseGridWidth - 1))
                    return MSBGConstants.SOLID;
                return density > 1e-4f ? MSBGConstants.FLUID : MSBGConstants.AIR;
            }
        }
        
        [BurstCompile]
        private struct AddForceJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<uint> GridTypes;
            public NativeArray<float2> GridVelocity;
            public float2 Gravity;

            public void Execute(int i)
            {
                float2 velocity = GridVelocity[i];
                uint gridType = GridTypes[i];
                if (!IsSolidCell(gridType))
                    velocity += Gravity * MSBGConstants.DeltaTime;

                GridVelocity[i] = EnforceBoundaryCondition(velocity, gridType);
            }
        }

        [BurstCompile]
        private struct CalcDivergenceJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int4> GridLut;
            [ReadOnly] public NativeArray<float2> GridVelocity;
            [ReadOnly] public NativeArray<float3> GridLaplacian;
            [ReadOnly] public NativeArray<float> GridDensity;

            [NativeDisableParallelForRestriction] [WriteOnly]
            public NativeArray<float> GridDivergence;

            public void Execute(int i)
            {
                int4 info = GridLut[i];
                int level = GetCurLevel(info.z);
                if (level < 0)
                    return;

                int pn = info.x;

                int blockWidth = GetBlockWidth(level);

                int haloBlockWidth = blockWidth + 2;
                int haloBlockSize = haloBlockWidth * haloBlockWidth;
                var velocityArr = new NativeArray<float2>(haloBlockSize, Allocator.Temp);

                FillHaloBlock(GridVelocity, GridLut, velocityArr, Idx2Coord(i), info);

                float cellSize = GetCellSize(level);
                // float invCellSize = 1.0f / cellSize;
                for (int by = 1; by <= blockWidth; by++)
                for (int bx = 1; bx <= blockWidth; bx++)
                {
                    float divergence = 0;
                    int idx = pn + BlockCoord2Idx(bx - 1, by - 1, blockWidth);
                    float3 param = GridLaplacian[idx];

                    if (!InActive(param.x))
                    {
                        float2 vel = velocityArr[BlockCoord2Idx(bx, by, haloBlockWidth)];
                        float un = velocityArr[BlockCoord2Idx(bx + 1, by, haloBlockWidth)].x;
                        float vn = velocityArr[BlockCoord2Idx(bx, by + 1, haloBlockWidth)].y;

                        divergence += cellSize * (un - vel.x);
                        divergence += cellSize * (vn - vel.y);
                    }

                    float density = math.max(-0.1f, GridDensity[idx] - 3.5f);
                    GridDivergence[idx] = divergence - density;
                }
                velocityArr.Dispose();
            }

            public static void FillHaloBlock(NativeArray<float2> v, NativeArray<int4> lut,
                NativeArray<float2> block, int2 coord, int4 info)
            {
                int level = GetCurLevel(info.z);
                int blockWidth = GetBlockWidth(level);
                int haloBlockWidth = blockWidth + 2;
                for (int by = 0; by < blockWidth; by++)
                for (int bx = 0; bx < blockWidth; bx++)
                {
                    int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
                    int physicsIdx = info.x + BlockCoord2Idx(bx, by, blockWidth);

                    block[localIdx] = v[physicsIdx];
                }

                int2 ox = new int2(1, 0);
                int2 oy = new int2(0, 1);

                for (int n = 0; n < 2; n++)
                {
                    int2 dir = new int2(ox[n], oy[n]);
                    int2 curr = coord + dir;
                    if (curr.x < 0 || curr.y < 0 || curr.x >= MSBGConstants.GridWidth ||
                        curr.y >= MSBGConstants.GridWidth)
                        continue;

                    int4 neighborInfo = lut[Coord2Idx(curr)];
                    int nLevel = GetCurLevel(neighborInfo.z);
                    if (nLevel < 0)
                        continue;

                    int phn = neighborInfo.x;
                    if (nLevel == level)
                    {
                        for (int c = 0; c < blockWidth; c++)
                        {
                            int2 nCoord = math.select(math.select(c, 0, dir > 0), blockWidth - 1, dir < 0);
                            int nLocalIdx = BlockCoord2Idx(nCoord, blockWidth);
                            int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                            int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                            block[paddingIdx] = v[phn + nLocalIdx];
                        }
                    }
                    else if (nLevel > level)
                    {
                        int nBlockWidth = GetBlockWidth(nLevel);
                        for (int c = 0; c < blockWidth; c++)
                        {
                            int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                            int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                            int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                            int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                            block[paddingIdx] = v[phn + nLocalIdx];
                        }
                    }
                    else // n_level < level
                    {
                        int nBlockWidth = GetBlockWidth(nLevel);
                        for (int c = 0; c < blockWidth; c++)
                        {
                            int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                            int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                            int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                            int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                            int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                            int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                            block[paddingIdx] = (v[phn + nLocalIdx0] + v[phn + nLocalIdx1]) * 0.5f;
                        }
                    }
                }
            }
        }

        [BurstCompile]
        private struct UpdateVelocity : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int4> GridLut;
            [ReadOnly] public NativeArray<float3> GridLaplacian;
            [ReadOnly] public NativeArray<float> GridPressure;
            [NativeDisableParallelForRestriction] public NativeArray<float2> GridVelocity;
        
            public void Execute(int i)
            {
                int4 info = GridLut[i];
                int level = GetCurLevel(info.z);
                if (level < 0) 
                    return;
                
                int offset = info.x;
                
                int blockWidth = GetBlockWidth(level);
                
                int haloBlockWidth = blockWidth + 2;
                int haloBlockSize = haloBlockWidth * haloBlockWidth;
                var pressureArr = new NativeArray<float>(haloBlockSize, Allocator.Temp);
                var paramArr = new NativeArray<float>(haloBlockSize, Allocator.Temp);
                
                FillHaloBlock(GridPressure, GridLut, pressureArr, paramArr, Idx2Coord(i), info);

                float cellSize = GetCellSize(level);
                float invCellSize = 1.0f / cellSize;
                for (int by = 1; by <= blockWidth; by++)
                for (int bx = 1; bx <= blockWidth; bx++)
                {
                    int idx = offset + (by - 1) * blockWidth + (bx - 1);
                    float2 velocity = GridVelocity[idx];
                    float3 param = GridLaplacian[idx];

                    if (param.x != 0)
                    {
                        float p = pressureArr[BlockCoord2Idx(bx, by, haloBlockWidth)];
                        float up = pressureArr[BlockCoord2Idx(bx - 1, by, haloBlockWidth)];
                        float ua = paramArr[BlockCoord2Idx(bx - 1, by, haloBlockWidth)];
                        float vp = pressureArr[BlockCoord2Idx(bx, by - 1, haloBlockWidth)];
                        float va = paramArr[BlockCoord2Idx(bx, by - 1, haloBlockWidth)];
                    
                        velocity.x += ua != 0 ? (up - p) / ua : 0;
                        velocity.y += va != 0 ? (vp - p) / va : 0;
                    }

                    GridVelocity[idx] = velocity;
                }
            }
        
            private static void FillHaloBlock(NativeArray<float> p, NativeArray<int4> lut, 
                NativeArray<float> block, NativeArray<float> paramsBlock, int2 coord, int4 info)
            {
                int level = GetCurLevel(info.z);
                int blockWidth = GetBlockWidth(level);
                int haloBlockWidth = blockWidth + 2;
                float cellSize = GetCellSize(level);
                for (int by = 0; by < blockWidth; by++)
                for (int bx = 0; bx < blockWidth; bx++)
                {
                    int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
                    int physicsIdx = info.x + BlockCoord2Idx(bx, by, blockWidth);
                        
                    block[localIdx] = p[physicsIdx];
                    paramsBlock[localIdx] = cellSize;
                }
                
                int4 ox = new int4(-1, 0, 1, 0);
                int4 oy = new int4(0, -1, 0, 1);
                
                for (int n = 0; n < 4; n++)
                {
                    int2 dir = new int2(ox[n], oy[n]);
                    int2 curr = coord + dir;
                    if (curr.x < 0 || curr.y < 0 || curr.x >= MSBGConstants.GridWidth || curr.y >= MSBGConstants.GridWidth)
                        continue;
                    
                    int4 neighborInfo = lut[Coord2Idx(curr)];
                    int nLevel = GetCurLevel(neighborInfo.z);
                    if (nLevel < 0)
                        continue;
                    
                    int phn = neighborInfo.x;
                    if (nLevel == level)
                    {
                        for (int c = 0; c < blockWidth; c++)
                        {
                            int2 nCoord = math.select(math.select(c, 0, dir > 0), blockWidth - 1, dir < 0);
                            int nLocalIdx = BlockCoord2Idx(nCoord, blockWidth);
                            int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                            int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                            block[paddingIdx] = p[phn + nLocalIdx];
                            paramsBlock[paddingIdx] = cellSize;
                        }
                    }
                    else if (nLevel > level)
                    {
                        int nBlockWidth = GetBlockWidth(nLevel);
                        float nCellSize = GetCellSize(nLevel);
                        for (int c = 0; c < blockWidth; c++)
                        {
                            int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                            int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                            int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                            int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                            block[paddingIdx] = p[phn + nLocalIdx];
                            paramsBlock[paddingIdx] = 0.5f * (cellSize + nCellSize);
                        }
                    }
                    else // n_level < level
                    {
                        int nBlockWidth = GetBlockWidth(nLevel);
                        float nCellSize = GetCellSize(nLevel);
                        for (int c = 0; c < blockWidth; c++)
                        {
                            int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                            int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                            int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                            int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                            int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                            int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                            block[paddingIdx] = 0.5f * (p[phn + nLocalIdx0] + p[phn + nLocalIdx1]);
                            paramsBlock[paddingIdx] = 0.5f * (cellSize + nCellSize);
                        }
                    }
                }
            }
        }

        [BurstCompile]
        private struct PostprocessJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int4> GridLut;
            [NativeDisableParallelForRestriction] public NativeArray<float2> GridVelocity;

            public void Execute(int i)
            {
                int4 info = GridLut[i];
                int4 levels = GetNeighborsLevel(info.z);
                if (levels.x < 0 || !(levels.x < levels.y || levels.x < levels.z))
                    return;
                
                int level = levels.x;
                int pn = info.x;
                int blockWidth = GetBlockWidth(level);

                int haloBlockWidth = blockWidth + 2;
                int haloBlockSize = haloBlockWidth * haloBlockWidth;
                var velocityArr = new NativeArray<float2>(haloBlockSize, Allocator.Temp);

                CalcDivergenceJob.FillHaloBlock(GridVelocity, GridLut, velocityArr, Idx2Coord(i), info);

                if (level < levels.y)
                {
                    for (int yy = 1; yy < blockWidth; yy += 2)
                    {
                        int ii = pn + BlockCoord2Idx(blockWidth - 1, yy, blockWidth);
                        var uv = velocityArr[BlockCoord2Idx(blockWidth, yy, haloBlockWidth)];
                        var un = velocityArr[BlockCoord2Idx(blockWidth + 1, yy, haloBlockWidth)].x;
                        var vn = velocityArr[BlockCoord2Idx(blockWidth, yy + 1, haloBlockWidth)].y;
                        float divergence = (un - uv.x + vn - uv.y);
                        GridVelocity[ii] -= new float2(0, divergence);
                    }
                }

                if (level < levels.z)
                {
                    for (int xx = 1; xx < blockWidth; xx += 2)
                    {
                        int ii = pn + BlockCoord2Idx(xx, blockWidth - 1, blockWidth);
                        var uv = velocityArr[BlockCoord2Idx(xx, blockWidth, haloBlockWidth)];
                        var un = velocityArr[BlockCoord2Idx(xx + 1, blockWidth, haloBlockWidth)].x;
                        var vn = velocityArr[BlockCoord2Idx(xx, blockWidth + 1, haloBlockWidth)].y;
                        float divergence = (un - uv.x + vn - uv.y);
                        GridVelocity[ii] -= new float2(divergence, 0);
                    }

                }

                velocityArr.Dispose();
            }
        }
    
        [BurstCompile]
        private struct GridToParticleJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Particle> Particles;
            [ReadOnly] public NativeArray<float2> GridVelocityOld;
            [ReadOnly] public NativeArray<float2> GridVelocityNew;
            [ReadOnly] public NativeArray<float2> GridVelocityOldDS;
            [ReadOnly] public NativeArray<float2> GridVelocityNewDS;
            [ReadOnly] public NativeArray<int4> GridLut;
            public NativeArray<float2> ParticleVel;
            public float Flipness;
        
            public void Execute(int i)
            {
                float2 pos = Particles[i].Pos;
                float2 vel = ParticleVel[i];

                SampleGridFaceBilinear(0, pos, GridVelocityOld, GridVelocityOldDS, GridVelocityNew, GridVelocityNewDS, GridLut, 
                    out var gOriginVelX, out var gVelX);
                SampleGridFaceBilinear(1, pos, GridVelocityOld, GridVelocityOldDS, GridVelocityNew, GridVelocityNewDS, GridLut, 
                    out var gOriginVelY, out var gVelY);
                
                float2 gOriginVel = new float2(gOriginVelX, gOriginVelY);
                float2 gVel = new float2(gVelX, gVelY);
                // SampleGridBilinear( pos, GridVelocityOld, GridVelocityOldDS, GridVelocityNew, GridVelocityNewDS, GridLut, 
                //     out var gOriginVel, out var gVel);

                ParticleVel[i] = math.lerp(gVel, vel + (gVel - gOriginVel), Flipness);
            }
            
            private void SampleGridFaceBilinear(int axis, float2 pos, NativeArray<float2> vf1, NativeArray<float2> vc1, 
                NativeArray<float2> vf2, NativeArray<float2> vc2, NativeArray<int4> lut, out float u1, out float u2)
            {
                u1 = 0;
                u2 = 0;
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                const int gridSize = MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth;
                float2 basePos = pos * MSBGConstants.InvBaseCellSize;
                int2 baseCoord = (int2)math.floor(basePos);
                if (math.any(baseCoord < 0) || math.any(baseCoord >= gridSize))
                    return;
                
                int2 blockCoord = baseCoord / baseBlockWidth;
                int4 info = lut[Coord2Idx(blockCoord)];
                int blockLevel = GetCurLevel(info.z);
                if (blockLevel < 0)
                    return;
                
                float2 localPos = (basePos - blockCoord * baseBlockWidth) / (1 << blockLevel);
                int blockWidth = baseBlockWidth >> blockLevel;
                float2 offset = new float2(0.5f, 0.5f);
                offset[axis] = 0;
                float2 localUV = localPos - offset; 
                float2 weight = localUV - math.floor(localUV);

                // Inside block
                if (math.all(localUV > 0 & localUV < blockWidth - 1))
                {
                    int2 c0 = math.max(0, (int2)math.floor(localUV));
                    int2 c1 = c0 + 1;
                    int ptr = info.x;

                    if (math.all(weight < 1e-5f))
                    {
                        int idx = ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth);
                        u1 = vf1[idx][axis];
                        u2 = vf2[idx][axis];
                    }
                    else if (math.all(weight > 0.9999f))
                    {
                        int idx = ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth);
                        u1 = vf1[idx][axis];
                        u2 = vf2[idx][axis];
                    }
                    else
                    {
                        int idx00 = ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth);
                        int idx10 = ptr + BlockCoord2Idx(c1.x, c0.y, blockWidth);
                        int idx01 = ptr + BlockCoord2Idx(c0.x, c1.y, blockWidth);
                        int idx11 = ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth);
                        u1 = LerpBilinear(weight, vf1[idx00][axis], vf1[idx10][axis], vf1[idx01][axis], vf1[idx11][axis]);
                        u2 = LerpBilinear(weight, vf2[idx00][axis], vf2[idx10][axis], vf2[idx01][axis], vf2[idx11][axis]);
                    }
                    return;
                }

                int fineLevel, coarseLevel;
                // edge between same level blocks
                {
                    int2 lbBlockCoord = blockCoord - math.select(int2.zero, 1, localUV < 0 & blockCoord > 0);
                    int4 infoLB = lut[Coord2Idx(lbBlockCoord)];
                    int4 neighborLevel = GetNeighborsLevel(infoLB.z);
                    blockLevel = neighborLevel.x;
                    bool isRight = localUV.x < 0 || localUV.x > blockWidth - 1, isTop = localUV.y < 0 || localUV.y > blockWidth - 1;
                    int levelR = math.select(blockLevel, neighborLevel.y, isRight);
                    int levelT = math.select(blockLevel, neighborLevel.z, isTop);
                    int levelRT = math.select(math.select(neighborLevel.y, neighborLevel.z, isTop), neighborLevel.w,
                        isRight && isTop);
                    fineLevel = math.min(math.min(blockLevel, levelR), math.min(levelT, levelRT));
                    coarseLevel = math.max(math.max(blockLevel, levelR), math.max(levelT, levelRT));
                    if (coarseLevel == fineLevel)
                    {
                        bool2 selector = weight > 0.5f;
                        selector[axis] = false;
                        int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                        int2 c1 = c0 + (1 << blockLevel);
                        SamplePointFine(vf1, vf2, lut, c0.x, c0.y, out float2 lb1, out float2 lb2);
                        SamplePointFine(vf1, vf2, lut, c1.x, c0.y, out float2 rb1, out float2 rb2);
                        SamplePointFine(vf1, vf2, lut, c0.x, c1.y, out float2 lt1, out float2 lt2);
                        SamplePointFine(vf1, vf2, lut, c1.x, c1.y, out float2 rt1, out float2 rt2);
                        u1 = LerpBilinear(weight, lb1[axis], rb1[axis], lt1[axis], rt1[axis]);
                        u2 = LerpBilinear(weight, lb2[axis], rb2[axis], lt2[axis], rt2[axis]);
                        return;
                    }
                }
                
                int2 faceBlockCoord = blockCoord + math.select(int2.zero, new int2(1,0), localUV.x > blockWidth - 0.5f & blockCoord > 0);
                int4 infoFace = lut[Coord2Idx(faceBlockCoord)];
            
                blockLevel = GetCurLevel(infoFace.z);
                blockCoord = faceBlockCoord;
                
                if (blockLevel == coarseLevel)
                {
                    float2 coarsePos = (basePos - blockCoord * baseBlockWidth) / (1 << coarseLevel);
                    float2 coarseUV = coarsePos - offset;
                    float2 coarseWeight = coarseUV - math.floor(coarseUV);
                    bool2 selector = coarseWeight > 0.5f;
                    selector[axis] = false;
                    int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                    int2 c1 = c0 + (1 << blockLevel);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c0.x, c0.y, coarseLevel, out float2 lb1, out float2 lb2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c1.x, c0.y, coarseLevel, out float2 rb1, out float2 rb2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c0.x, c1.y, coarseLevel, out float2 lt1, out float2 lt2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c1.x, c1.y, coarseLevel, out float2 rt1, out float2 rt2);
                    u1 = LerpBilinear(weight, lb1[axis], rb1[axis], lt1[axis], rt1[axis]);
                    u2 = LerpBilinear(weight, lb2[axis], rb2[axis], lt2[axis], rt2[axis]);
                }
                else
                {
                    float2 coarsePos = (basePos - blockCoord * baseBlockWidth) / (1 << coarseLevel);
                    float2 coarseUV = coarsePos - offset;
                    float2 coarseWeight = coarseUV - math.floor(coarseUV);
                    
                    float2 finePos = (basePos - blockCoord * baseBlockWidth) / (1 << fineLevel);
                    float2 fineUV = finePos - offset;
                    float2 fineWeight = fineUV - math.floor(fineUV);
                    
                    
                    bool2 selector = coarseWeight > 0.5f;
                    selector[axis] = false;
                    int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                    int2 c1 = c0 + (1 << coarseLevel);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c0.x, c0.y, coarseLevel, out float2 lb1, out float2 lb2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c1.x, c0.y, coarseLevel, out float2 rb1, out float2 rb2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c0.x, c1.y, coarseLevel, out float2 lt1, out float2 lt2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c1.x, c1.y, coarseLevel, out float2 rt1, out float2 rt2);
                    
                    int4 neighborLevelCur = GetNeighborsLevel(infoFace.z);
                    int4 neighborLevelLB = GetNeighborsLevel(lut[Coord2Idx(math.max(0, faceBlockCoord - 1))].z);
                    int levelR = neighborLevelCur.y;
                    int levelT = neighborLevelCur.z;
                    int levelRT = neighborLevelCur.w;
                    int levelL = neighborLevelLB.z;
                    int levelB = neighborLevelLB.y;
                    int levelLB = neighborLevelLB.x;
                    int levelLT = GetCurLevel(lut[Coord2Idx(math.clamp(faceBlockCoord + new int2(-1, 1), 0, MSBGConstants.GridWidth - 1))].z);
                    int levelRB = GetCurLevel(lut[Coord2Idx(math.clamp(faceBlockCoord + new int2(1, -1), 0, MSBGConstants.GridWidth - 1))].z);
                    float dstToCoarse = 1;
                    localPos = fineUV + new float2(1, 0.5f);
                    float2 subPos = GetBlockWidth(fineLevel) - fineUV - new float2(0.5f, 0.5f);
                    // return math.cmin(localPos);
                    int levelC = GetCurLevel(infoFace.z);
                    if (levelC < levelL) dstToCoarse = math.min(dstToCoarse, localPos.x);
                    if (levelC < levelR) dstToCoarse = math.min(dstToCoarse, subPos.x);
                    if (levelC < levelT) dstToCoarse = math.min(dstToCoarse, subPos.y);
                    if (levelC < levelB) dstToCoarse = math.min(dstToCoarse, localPos.y);
                    if (levelC < levelLB) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, localPos.y));
                    if (levelC < levelRT) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, subPos.y));
                    if (levelC < levelLT) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, subPos.y));
                    if (levelC < levelRB) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, localPos.y));
                    dstToCoarse = math.min(1, dstToCoarse * 2);

                    var valueCoarse1 = LerpBilinear(coarseWeight, lb1[axis], rb1[axis], lt1[axis], rt1[axis]);
                    var valueCoarse2 = LerpBilinear(coarseWeight, lb2[axis], rb2[axis], lt2[axis], rt2[axis]);
                    
                    
                    selector = fineWeight > 0.5f;
                    selector[axis] = false;
                    c0 = baseCoord - math.select(int2.zero, 1 << fineLevel, selector);
                    c1 = c0 + (1 << fineLevel);

                    SamplePointFine(vf1, vf2, lut, c0.x, c0.y, out lb1, out lb2);
                    SamplePointFine(vf1, vf2, lut, c1.x, c0.y, out rb1, out rb2);
                    SamplePointFine(vf1, vf2, lut, c0.x, c1.y, out lt1, out lt2);
                    SamplePointFine(vf1, vf2, lut, c1.x, c1.y, out rt1, out rt2);
                    
                    var valueFine1 = LerpBilinear(fineWeight, lb1[axis], rb1[axis], lt1[axis], rt1[axis]);
                    var valueFine2 = LerpBilinear(fineWeight, lb2[axis], rb2[axis], lt2[axis], rt2[axis]);
                    
                    u1 = math.lerp(valueCoarse1, valueFine1, dstToCoarse);
                    u2 = math.lerp(valueCoarse2, valueFine2, dstToCoarse);
                }
            }

            private void SampleGridBilinear(float2 pos, NativeArray<float2> vf1, NativeArray<float2> vc1, 
                NativeArray<float2> vf2, NativeArray<float2> vc2, NativeArray<int4> lut, out float2 v1, out float2 v2)
            {
                v1 = float2.zero;
                v2 = float2.zero;
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                const int gridSize = MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth;
                float2 basePos = pos * MSBGConstants.InvBaseCellSize;
                int2 baseCoord = (int2)math.floor(basePos);
                if (math.any(baseCoord < 0) || math.any(baseCoord >= gridSize))
                    return;
                
                int2 blockCoord = baseCoord / baseBlockWidth;
                int4 info = lut[Coord2Idx(blockCoord)];
                int blockLevel = GetCurLevel(info.z);
                if (blockLevel < 0)
                    return;
                
                float2 localPos = (basePos - blockCoord * baseBlockWidth) / (1 << blockLevel);
                int blockWidth = baseBlockWidth >> blockLevel;
                float2 localUV = localPos - 0.5f;
                float2 weight = localUV - math.floor(localUV);

                // Inside block
                if (math.all(localUV > 0 & localUV < blockWidth - 1))
                {
                    int2 c0 = math.max(0, (int2)math.floor(localUV));
                    int2 c1 = c0 + 1;
                    int ptr = info.x;

                    if (math.all(weight < 1e-5f))
                    {
                        int idx = ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth);
                        v1 = vf1[idx];
                        v2 = vf2[idx];
                    }
                    else if (math.all(weight > 0.9999f))
                    {
                        int idx = ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth);
                        v1 = vf1[idx];
                        v2 = vf2[idx];
                    }
                    else
                    {
                        int idx00 = ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth);
                        int idx10 = ptr + BlockCoord2Idx(c1.x, c0.y, blockWidth);
                        int idx01 = ptr + BlockCoord2Idx(c0.x, c1.y, blockWidth);
                        int idx11 = ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth);
                        v1 = LerpBilinear(weight, vf1[idx00], vf1[idx10], vf1[idx01], vf1[idx11]);
                        v2 = LerpBilinear(weight, vf2[idx00], vf2[idx10], vf2[idx01], vf2[idx11]);
                    }
                    return;
                }

                int2 lbBlockCoord = blockCoord - math.select(int2.zero, 1, localUV < 0 & blockCoord > 0);
                int4 infoLB = lut[Coord2Idx(lbBlockCoord)];
                int4 neighborLevel = GetNeighborsLevel(infoLB.z);
                blockLevel = neighborLevel.x;
                bool isRight = localUV.x < 0 || localUV.x > blockWidth - 1;
                bool isTop = localUV.y < 0 || localUV.y > blockWidth - 1;
                int levelR = math.select(blockLevel, neighborLevel.y, isRight);
                int levelT = math.select(blockLevel, neighborLevel.z, isTop);
                int levelRT = math.select(math.select(neighborLevel.y, neighborLevel.z, isTop), neighborLevel.w,
                    isRight && isTop);
                int fineLevel = math.min(math.min(blockLevel, levelR), math.min(levelT, levelRT));
                int coarseLevel = math.max(math.max(blockLevel, levelR), math.max(levelT, levelRT));
                if (coarseLevel == fineLevel)
                {
                    int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, weight > 0.5f);
                    int2 c1 = c0 + (1 << blockLevel);
                    SamplePointFine(vf1, vf2, lut, c0.x, c0.y, out float2 lb1, out float2 lb2);
                    SamplePointFine(vf1, vf2, lut, c1.x, c0.y, out float2 rb1, out float2 rb2);
                    SamplePointFine(vf1, vf2, lut, c0.x, c1.y, out float2 lt1, out float2 lt2);
                    SamplePointFine(vf1, vf2, lut, c1.x, c1.y, out float2 rt1, out float2 rt2);
                    v1 = LerpBilinear(weight, lb1, rb1, lt1, rt1);
                    v2 = LerpBilinear(weight, lb2, rb2, lt2, rt2);
                    return;
                }
                
                // UnityEngine.Debug.Assert(coarseLevel - fineLevel == 1, "MSBG SampleBilinear: level difference greater than 1");
                if (GetCurLevel(info.z) == coarseLevel)
                {
                    int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, weight > 0.5f);
                    int2 c1 = c0 + (1 << blockLevel);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c0.x, c0.y, coarseLevel, out float2 lb1, out float2 lb2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c1.x, c0.y, coarseLevel, out float2 rb1, out float2 rb2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c0.x, c1.y, coarseLevel, out float2 lt1, out float2 lt2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c1.x, c1.y, coarseLevel, out float2 rt1, out float2 rt2);
                    v1 = LerpBilinear(weight, lb1, rb1, lt1, rt1);
                    v2 = LerpBilinear(weight, lb2, rb2, lt2, rt2);
                }
                else
                {
                    float2 coarsePos = (basePos - blockCoord * baseBlockWidth) / (1 << coarseLevel);
                    float2 coarseUV = coarsePos - 0.5f;
                    float2 coarseWeight = coarseUV - math.floor(coarseUV);
                    
                    int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, coarseWeight > 0.5f);
                    int2 c1 = c0 + (1 << coarseLevel);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c0.x, c0.y, coarseLevel, out float2 lb1, out float2 lb2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c1.x, c0.y, coarseLevel, out float2 rb1, out float2 rb2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c0.x, c1.y, coarseLevel, out float2 lt1, out float2 lt2);
                    SamplePointLevel(vf1, vc1, vf2, vc2, lut, c1.x, c1.y, coarseLevel, out float2 rt1, out float2 rt2);
                    
                    int4 neighborLevelCur = GetNeighborsLevel(info.z);
                    int4 neighborLevelLB = GetNeighborsLevel(lut[Coord2Idx(math.max(0, blockCoord - 1))].z);
                    levelR = neighborLevelCur.y;
                    levelT = neighborLevelCur.z;
                    levelRT = neighborLevelCur.w;
                    int levelL = neighborLevelLB.z;
                    int levelB = neighborLevelLB.y;
                    int levelLB = neighborLevelLB.x;
                    int levelLT = GetCurLevel(lut[Coord2Idx(math.max(0, blockCoord + new int2(-1, 1)))].z);
                    int levelRB = GetCurLevel(lut[Coord2Idx(math.max(0, blockCoord + new int2(1, -1)))].z);
                    float dstToCoarse = 1;
                    float2 subPos = blockWidth - localPos;
                    int levelC = GetCurLevel(info.z);
                    if (levelC < levelL) dstToCoarse = math.min(dstToCoarse, localPos.x);
                    if (levelC < levelR) dstToCoarse = math.min(dstToCoarse, subPos.x);
                    if (levelC < levelT) dstToCoarse = math.min(dstToCoarse, subPos.y);
                    if (levelC < levelB) dstToCoarse = math.min(dstToCoarse, localPos.y);
                    if (levelC < levelLB) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, localPos.y));
                    if (levelC < levelRT) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, subPos.y));
                    if (levelC < levelLT) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, subPos.y));
                    if (levelC < levelRB) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, localPos.y));
                    dstToCoarse *= 2;
                    
                    var valueCoarse1 = LerpBilinear(coarseWeight, lb1, rb1, lt1, rt1);
                    var valueCoarse2 = LerpBilinear(coarseWeight, lb2, rb2, lt2, rt2);
                    
                    c0 = baseCoord - math.select(int2.zero, 1 << fineLevel, weight > 0.5f);
                    c1 = c0 + (1 << fineLevel);

                    SamplePointFine(vf1, vf2, lut, c0.x, c0.y, out lb1, out lb2);
                    SamplePointFine(vf1, vf2, lut, c1.x, c0.y, out rb1, out rb2);
                    SamplePointFine(vf1, vf2, lut, c0.x, c1.y, out lt1, out lt2);
                    SamplePointFine(vf1, vf2, lut, c1.x, c1.y, out rt1, out rt2);
                    
                    var valueFine1 = LerpBilinear(weight, lb1, rb1, lt1, rt1);
                    var valueFine2 = LerpBilinear(weight, lb2, rb2, lt2, rt2);
                    
                    // dstToCoarse = dstToCoarse < 0.5f
                    //     ? dstToCoarse * 4f / 3f
                    //     : (dstToCoarse + 0.5f) * 2f / 3f;
                    v1 = math.lerp(valueCoarse1, valueFine1, dstToCoarse);
                    v2 = math.lerp(valueCoarse2, valueFine2, dstToCoarse);
                }
            }

            private static void SamplePointFine(NativeArray<float2> v1, NativeArray<float2> v2, NativeArray<int4> lut,
                int x, int y, out float2 r1, out float2 r2)
            {
                var baseCoord = math.clamp(new int2(x, y), 0, MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth - 1);
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                int2 blockCoord = baseCoord / baseBlockWidth;
                int blockIdx = Coord2Idx(blockCoord);
                int4 info = lut[blockIdx];
                int level = GetCurLevel(info.z);
                r1 = float2.zero;
                r2 = float2.zero;
                if (level < 0)
                    return;
                int blockWidth = baseBlockWidth >> level;
                int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
                r1 = v1[info.x + BlockCoord2Idx(localCoord, blockWidth)];
                r2 = v2[info.x + BlockCoord2Idx(localCoord, blockWidth)];
            }

            private static void SamplePointLevel(NativeArray<float2> vf1, NativeArray<float2> vc1, NativeArray<float2> vf2,
                NativeArray<float2> vc2, NativeArray<int4> lut, int x, int y, int targetLevel, out float2 r1, out float2 r2)
            {
                var baseCoord = math.clamp(new int2(x, y), 0, MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth - 1);
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                int2 blockCoord = baseCoord / baseBlockWidth;
                int blockIdx = Coord2Idx(blockCoord);
                int4 info = lut[blockIdx];
                int level = GetCurLevel(info.z);
                
                r1 = float2.zero;
                r2 = float2.zero;
                if (level < 0) return;
                
                if (level == targetLevel)
                {
                    int blockWidth = baseBlockWidth >> level;
                    int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
                    r1 = vf1[info.x + BlockCoord2Idx(localCoord, blockWidth)];
                    r2 = vf2[info.x + BlockCoord2Idx(localCoord, blockWidth)];
                }
                else
                {
                    // UnityEngine.Debug.Assert(level < targetLevel);
                    int blockWidth = baseBlockWidth >> targetLevel;
                    int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> targetLevel;
                    r1 = vc1[info.y + BlockCoord2Idx(localCoord, blockWidth)];
                    r2 = vc2[info.y + BlockCoord2Idx(localCoord, blockWidth)];
                }
            }

        }
    
        [BurstCompile]
        private struct AdvectionJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int4> GridLut;
            [ReadOnly] public NativeArray<float2> GridVelocity;
            [ReadOnly] public NativeArray<float2> GridVelocityDS;
            public NativeArray<Particle> Particles;
            private const float DeltaTime = MSBGConstants.DeltaTime;
        
            public void Execute(int i)
            {
                Particle particle = Particles[i];
                float2 pos = particle.Pos;
                // advect using RK2 (Explicit midpoint method)
                // float2 k1 = SampleGridBilinear(GridVelocity, GridVelocityDS, GridLut, pos);

                SampleGridFaceBilinear(0, pos, GridVelocity, GridVelocityDS, GridLut, out var gOriginVelX);
                SampleGridFaceBilinear(1, pos, GridVelocity, GridVelocityDS, GridLut, out var gOriginVelY);
                
                float2 k1 = new float2(gOriginVelX, gOriginVelY);
                var newPos = pos + 0.5f * DeltaTime * k1;
                
                SampleGridFaceBilinear(0, newPos, GridVelocity, GridVelocityDS, GridLut, out gOriginVelX);
                SampleGridFaceBilinear(1, newPos, GridVelocity, GridVelocityDS, GridLut, out gOriginVelY);
                // float2 k2 = ReadGridBilinear(pos + 0.5f * DeltaTime * k1, GridLut, GridVelocity);
                var vel = new float2(gOriginVelX, gOriginVelY);

                pos += vel * DeltaTime;
                particle.Pos = ClampPosition(pos);
                Particles[i] = particle;
            }
            
            private void SampleGridFaceBilinear(int axis, float2 pos, NativeArray<float2> vf1, NativeArray<float2> vc1, 
                NativeArray<int4> lut, out float u1)
            {
                u1 = 0;
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                const int gridSize = MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth;
                float2 basePos = pos * MSBGConstants.InvBaseCellSize;
                int2 baseCoord = (int2)math.floor(basePos);
                if (math.any(baseCoord < 0) || math.any(baseCoord >= gridSize))
                    return;
                
                int2 blockCoord = baseCoord / baseBlockWidth;
                int4 info = lut[Coord2Idx(blockCoord)];
                int blockLevel = GetCurLevel(info.z);
                if (blockLevel < 0)
                    return;
                
                float2 localPos = (basePos - blockCoord * baseBlockWidth) / (1 << blockLevel);
                int blockWidth = baseBlockWidth >> blockLevel;
                float2 offset = new float2(0.5f, 0.5f);
                offset[axis] = 0;
                float2 localUV = localPos - offset; 
                float2 weight = localUV - math.floor(localUV);

                // Inside block
                if (math.all(localUV > 0 & localUV < blockWidth - 1))
                {
                    int2 c0 = math.max(0, (int2)math.floor(localUV));
                    int2 c1 = c0 + 1;
                    int ptr = info.x;

                    if (math.all(weight < 1e-5f))
                    {
                        int idx = ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth);
                        u1 = vf1[idx][axis];
                    }
                    else if (math.all(weight > 0.9999f))
                    {
                        int idx = ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth);
                        u1 = vf1[idx][axis];
                    }
                    else
                    {
                        int idx00 = ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth);
                        int idx10 = ptr + BlockCoord2Idx(c1.x, c0.y, blockWidth);
                        int idx01 = ptr + BlockCoord2Idx(c0.x, c1.y, blockWidth);
                        int idx11 = ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth);
                        u1 = LerpBilinear(weight, vf1[idx00][axis], vf1[idx10][axis], vf1[idx01][axis], vf1[idx11][axis]);
                    }
                    return;
                }

                int fineLevel, coarseLevel;
                // edge between same level blocks
                {
                    int2 lbBlockCoord = blockCoord - math.select(int2.zero, 1, localUV < 0 & blockCoord > 0);
                    int4 infoLB = lut[Coord2Idx(lbBlockCoord)];
                    int4 neighborLevel = GetNeighborsLevel(infoLB.z);
                    blockLevel = neighborLevel.x;
                    bool isRight = localUV.x < 0 || localUV.x > blockWidth - 1, isTop = localUV.y < 0 || localUV.y > blockWidth - 1;
                    int levelR = math.select(blockLevel, neighborLevel.y, isRight);
                    int levelT = math.select(blockLevel, neighborLevel.z, isTop);
                    int levelRT = math.select(math.select(neighborLevel.y, neighborLevel.z, isTop), neighborLevel.w,
                        isRight && isTop);
                    fineLevel = math.min(math.min(blockLevel, levelR), math.min(levelT, levelRT));
                    coarseLevel = math.max(math.max(blockLevel, levelR), math.max(levelT, levelRT));
                    if (coarseLevel == fineLevel)
                    {
                        bool2 selector = weight > 0.5f;
                        selector[axis] = false;
                        int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                        int2 c1 = c0 + (1 << blockLevel);
                        SamplePointFine(vf1,  lut, c0.x, c0.y, out float2 lb1);
                        SamplePointFine(vf1,  lut, c1.x, c0.y, out float2 rb1);
                        SamplePointFine(vf1,  lut, c0.x, c1.y, out float2 lt1);
                        SamplePointFine(vf1,  lut, c1.x, c1.y, out float2 rt1);
                        u1 = LerpBilinear(weight, lb1[axis], rb1[axis], lt1[axis], rt1[axis]);
                        return;
                    }
                }
                
                int2 faceBlockCoord = blockCoord + math.select(int2.zero, new int2(1,0), localUV.x > blockWidth - 0.5f & blockCoord > 0);
                int4 infoFace = lut[Coord2Idx(faceBlockCoord)];
            
                blockLevel = GetCurLevel(infoFace.z);
                blockCoord = faceBlockCoord;
                
                if (blockLevel == coarseLevel)
                {
                    float2 coarsePos = (basePos - blockCoord * baseBlockWidth) / (1 << coarseLevel);
                    float2 coarseUV = coarsePos - offset;
                    float2 coarseWeight = coarseUV - math.floor(coarseUV);
                    bool2 selector = coarseWeight > 0.5f;
                    selector[axis] = false;
                    int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                    int2 c1 = c0 + (1 << blockLevel);
                    SamplePointLevel(vf1, vc1, lut, c0.x, c0.y, coarseLevel, out float2 lb1);
                    SamplePointLevel(vf1, vc1, lut, c1.x, c0.y, coarseLevel, out float2 rb1);
                    SamplePointLevel(vf1, vc1, lut, c0.x, c1.y, coarseLevel, out float2 lt1);
                    SamplePointLevel(vf1, vc1, lut, c1.x, c1.y, coarseLevel, out float2 rt1);
                    u1 = LerpBilinear(weight, lb1[axis], rb1[axis], lt1[axis], rt1[axis]);
                }
                else
                {
                    float2 coarsePos = (basePos - blockCoord * baseBlockWidth) / (1 << coarseLevel);
                    float2 coarseUV = coarsePos - offset;
                    float2 coarseWeight = coarseUV - math.floor(coarseUV);
                    
                    float2 finePos = (basePos - blockCoord * baseBlockWidth) / (1 << fineLevel);
                    float2 fineUV = finePos - offset;
                    float2 fineWeight = fineUV - math.floor(fineUV);
                    
                    
                    bool2 selector = coarseWeight > 0.5f;
                    selector[axis] = false;
                    int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                    int2 c1 = c0 + (1 << coarseLevel);
                    SamplePointLevel(vf1, vc1, lut, c0.x, c0.y, coarseLevel, out float2 lb1);
                    SamplePointLevel(vf1, vc1, lut, c1.x, c0.y, coarseLevel, out float2 rb1);
                    SamplePointLevel(vf1, vc1, lut, c0.x, c1.y, coarseLevel, out float2 lt1);
                    SamplePointLevel(vf1, vc1, lut, c1.x, c1.y, coarseLevel, out float2 rt1);
                    
                    int4 neighborLevelCur = GetNeighborsLevel(infoFace.z);
                    int4 neighborLevelLB = GetNeighborsLevel(lut[Coord2Idx(math.max(0, faceBlockCoord - 1))].z);
                    int levelR = neighborLevelCur.y;
                    int levelT = neighborLevelCur.z;
                    int levelRT = neighborLevelCur.w;
                    int levelL = neighborLevelLB.z;
                    int levelB = neighborLevelLB.y;
                    int levelLB = neighborLevelLB.x;
                    int levelLT = GetCurLevel(lut[Coord2Idx(math.clamp(faceBlockCoord + new int2(-1, 1), 0, MSBGConstants.GridWidth - 1))].z);
                    int levelRB = GetCurLevel(lut[Coord2Idx(math.clamp(faceBlockCoord + new int2(1, -1), 0, MSBGConstants.GridWidth - 1))].z);
                    float dstToCoarse = 1;
                    localPos = fineUV + new float2(1, 0.5f);
                    float2 subPos = GetBlockWidth(fineLevel) - fineUV - new float2(0.5f, 0.5f);
                    // return math.cmin(localPos);
                    int levelC = GetCurLevel(infoFace.z);
                    if (levelC < levelL) dstToCoarse = math.min(dstToCoarse, localPos.x);
                    if (levelC < levelR) dstToCoarse = math.min(dstToCoarse, subPos.x);
                    if (levelC < levelT) dstToCoarse = math.min(dstToCoarse, subPos.y);
                    if (levelC < levelB) dstToCoarse = math.min(dstToCoarse, localPos.y);
                    if (levelC < levelLB) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, localPos.y));
                    if (levelC < levelRT) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, subPos.y));
                    if (levelC < levelLT) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, subPos.y));
                    if (levelC < levelRB) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, localPos.y));
                    dstToCoarse = math.min(1, dstToCoarse * 2);

                    var valueCoarse1 = LerpBilinear(coarseWeight, lb1[axis], rb1[axis], lt1[axis], rt1[axis]);
                    
                    
                    selector = fineWeight > 0.5f;
                    selector[axis] = false;
                    c0 = baseCoord - math.select(int2.zero, 1 << fineLevel, selector);
                    c1 = c0 + (1 << fineLevel);

                    SamplePointFine(vf1, lut, c0.x, c0.y, out lb1);
                    SamplePointFine(vf1, lut, c1.x, c0.y, out rb1);
                    SamplePointFine(vf1, lut, c0.x, c1.y, out lt1);
                    SamplePointFine(vf1, lut, c1.x, c1.y, out rt1);
                    
                    var valueFine1 = LerpBilinear(fineWeight, lb1[axis], rb1[axis], lt1[axis], rt1[axis]);
                    
                    u1 = math.lerp(valueCoarse1, valueFine1, dstToCoarse);
                }
            }
            
            private static void SamplePointFine(NativeArray<float2> v1,  NativeArray<int4> lut,
                int x, int y, out float2 r1)
            {
                var baseCoord = math.clamp(new int2(x, y), 0, MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth - 1);
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                int2 blockCoord = baseCoord / baseBlockWidth;
                int blockIdx = Coord2Idx(blockCoord);
                int4 info = lut[blockIdx];
                int level = GetCurLevel(info.z);
                r1 = float2.zero;
                if (level < 0)
                    return;
                int blockWidth = baseBlockWidth >> level;
                int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
                r1 = v1[info.x + BlockCoord2Idx(localCoord, blockWidth)];
            }

            private static void SamplePointLevel(NativeArray<float2> vf1, NativeArray<float2> vc1, 
                NativeArray<int4> lut, int x, int y, int targetLevel, out float2 r1)
            {
                var baseCoord = math.clamp(new int2(x, y), 0, MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth - 1);
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                int2 blockCoord = baseCoord / baseBlockWidth;
                int blockIdx = Coord2Idx(blockCoord);
                int4 info = lut[blockIdx];
                int level = GetCurLevel(info.z);
                
                r1 = float2.zero;
                if (level < 0) return;
                
                if (level == targetLevel)
                {
                    int blockWidth = baseBlockWidth >> level;
                    int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
                    r1 = vf1[info.x + BlockCoord2Idx(localCoord, blockWidth)];
                }
                else
                {
                    // UnityEngine.Debug.Assert(level < targetLevel);
                    int blockWidth = baseBlockWidth >> targetLevel;
                    int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> targetLevel;
                    r1 = vc1[info.y + BlockCoord2Idx(localCoord, blockWidth)];
                }
            }

        }
        
        [BurstCompile]
        private struct ParticleLevelJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int4> GridLut;
            public NativeArray<Particle> Particles;
        
            public void Execute(int i)
            {
                var p = Particles[i];
                float2 pos = p.Pos;

                int level = SampleGridLevel(pos, GridLut);
                if (p.Level != level)
                    p.Counter++;
                else p.Counter = 0;

                Particles[i] = p;
            }
            
            private int SampleGridLevel(float2 pos, NativeArray<int4> lut)
            {
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                const int gridSize = MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth;
                float2 basePos = pos * MSBGConstants.InvBaseCellSize;
                int2 baseCoord = (int2)math.floor(basePos);
                if (math.any(baseCoord < 0) || math.any(baseCoord >= gridSize))
                    return 0;
                
                int2 blockCoord = baseCoord / baseBlockWidth;
                int4 info = lut[Coord2Idx(blockCoord)];
                int blockLevel = GetCurLevel(info.z);
                return math.max(0, blockLevel);
            }
        }
        
        private const float prop = 0.25f;
        
        [BurstCompile]
        private struct GridParticleCompileJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Particle> Particles;
            [ReadOnly] public NativeArray<int4> GridLut;
            [ReadOnly] public NativeArray<int2> ParticleRange;
            [WriteOnly] public NativeArray<int> GridCounter;
            public uint Seed;
            private const int Threshold = 30;
        
            public void Execute(int i)
            {
                int4 info = GridLut[i];
                int level = GetCurLevel(info.z);
                if (level < 0)
                    return;

                // Random rnd = new Random(Seed + (uint)i);
                var rnd = ((int)Seed + i) & 3;
                int counter = 0;
                int2 pRange = ParticleRange[i];
                for (int j = pRange.x; j < pRange.y; j++)
                {
                    Particle p = Particles[j];
                    if (p.Level != level && p.Counter >= Threshold)
                    {
                        if (p.Level > level) counter += 4;
                        else if (p.Level < level && (rnd++ & 3) == 0) // p.Level < level, 75% delete
                            counter++;
                    }
                    else
                        counter++;
                }

                GridCounter[i] = counter;
            }
        }
        
        [BurstCompile]
        private struct GridParticlePrefixSumJob : IJob
        {
            public NativeArray<int> GridCounter;
            public NativeReference<int> TotalCount;
        
            public void Execute()
            {
                int ptr = 0;
                for (int i = 0; i < GridCounter.Length; i++)
                {
                    var temp = GridCounter[i];
                    GridCounter[i] = ptr;
                    ptr += temp;
                }
                TotalCount.Value = ptr;
            }
        }
        
        [BurstCompile]
        private struct ParticleResampleJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int4> GridLut;
            [ReadOnly] public NativeArray<int2> ParticleRange;
            [ReadOnly] public NativeArray<int> GridCounter;
            [ReadOnly] public NativeArray<Particle> Particles;
            [ReadOnly] public NativeArray<float2> ParticlesVel;
            [NativeDisableParallelForRestriction] [WriteOnly] public NativeArray<Particle> NewParticles;
            [NativeDisableParallelForRestriction] [WriteOnly] public NativeArray<float2> NewParticlesVel;
            public uint Seed;
            private const int Threshold = 30;
        
            public void Execute(int i)
            {
                int4 info = GridLut[i];
                int level = GetCurLevel(info.z);
                if (level < 0) 
                    return;

                // Random rnd = new Random(Seed + (uint)i);
                var rnd = ((int)Seed + i) & 3;
                int counter = 0;
                int2 pRange = ParticleRange[i];
                int ptr = GridCounter[i];
                float4 ox = new float4(1, 1, -1, -1);
                float4 oy = new float4(1, -1, 1, -1);
                for (int j = pRange.x; j < pRange.y; j++)
                {
                    Particle p = Particles[j];
                    float2 v = ParticlesVel[j];
                    if (p.Level != level && p.Counter >= Threshold)
                    {
                        if (p.Level > level)
                        {
                            float radius = GetCellSize(level);
                            for (int k = 0; k < 4; k++)
                            {
                                float2 offset = new float2(ox[k], oy[k]) * radius * 0.71f;
                                var newP = p;
                                newP.Level = p.Level - 1;
                                newP.Pos += offset;
                                newP.Counter = 0;
                                NewParticles[ptr + counter] = newP;
                                NewParticlesVel[ptr + counter] = v;
                                counter++;
                            }
                        }
                        else if (p.Level < level && (rnd++ & 3) == 0) // p.Level < level, 75% delete
                        {
                            var newP = p;
                            newP.Counter = 0;
                            newP.Level = p.Level + 1;
                            NewParticles[ptr + counter] = newP;
                            NewParticlesVel[ptr + counter] = v;
                            counter++;
                        }
                    }
                    else
                    {
                        NewParticles[ptr + counter] = p;
                        NewParticlesVel[ptr + counter] = v;
                        counter++;
                    }
                }
            }
        }
        
        [BurstCompile]
        private struct Dot : IJob
        {
            [ReadOnly] private NativeArray<float> _lhs;
            [ReadOnly] private NativeArray<float> _rhs;
            [WriteOnly] private NativeReference<float> _result;
            
            public Dot(NativeArray<float> lhs, NativeArray<float> rhs, NativeReference<float> result)
            {
                _lhs = lhs;
                _rhs = rhs;
                _result = result;
            }
            
            public void Execute()
            {
                float sum = 0;
                for (int i = 0; i < _lhs.Length; i++)
                    sum += _lhs[i] * _rhs[i];
                _result.Value = sum;
            }
        }
        
        [BurstCompile]
        private struct DownSample : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int4> _lut;
            [ReadOnly] private NativeArray<float2> _vFine1;
            [ReadOnly] private NativeArray<float2> _vFine2;
            [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float2> _vCoarse1;
            [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float2> _vCoarse2;
            
            public DownSample(NativeArray<float2> vf1, NativeArray<float2> vc1, NativeArray<float2> vf2, NativeArray<float2> vc2, NativeArray<int4> lut)
            {
                _lut = lut;
                _vFine1 = vf1;
                _vCoarse1 = vc1;
                _vFine2 = vf2;
                _vCoarse2 = vc2;
            }

            public void Execute(int i)
            {
                int4 info = _lut[i];
                int levelFine = GetCurLevel(info.z);
                int levelCoarse = GetCurLevel(info.w);
                if (levelFine < 0 || levelCoarse < 0)
                    return;
                
                int phf = info.x;
                int phc = info.y;
                if (levelFine < levelCoarse)
                {
                    int blockWidthCoarse = GetBlockWidth(levelCoarse);
                    int blockWidthFine = GetBlockWidth(levelFine);
                    for (int y = 0; y < blockWidthCoarse; y++)
                    for (int x = 0; x < blockWidthCoarse; x++)
                    {
                        _vCoarse1[phc + BlockCoord2Idx(x, y, blockWidthCoarse)] = 0.25f * (
                            _vFine1[phf + BlockCoord2Idx(x * 2 + 0, y * 2 + 0, blockWidthFine)] +
                            _vFine1[phf + BlockCoord2Idx(x * 2 + 1, y * 2 + 0, blockWidthFine)] +
                            _vFine1[phf + BlockCoord2Idx(x * 2 + 0, y * 2 + 1, blockWidthFine)] +
                            _vFine1[phf + BlockCoord2Idx(x * 2 + 1, y * 2 + 1, blockWidthFine)]);
                        _vCoarse2[phc + BlockCoord2Idx(x, y, blockWidthCoarse)] = 0.25f * (
                            _vFine2[phf + BlockCoord2Idx(x * 2 + 0, y * 2 + 0, blockWidthFine)] +
                            _vFine2[phf + BlockCoord2Idx(x * 2 + 1, y * 2 + 0, blockWidthFine)] +
                            _vFine2[phf + BlockCoord2Idx(x * 2 + 0, y * 2 + 1, blockWidthFine)] +
                            _vFine2[phf + BlockCoord2Idx(x * 2 + 1, y * 2 + 1, blockWidthFine)]);
                    }
                }
                else // levelC == _levelF
                {
                    int blockWidth = GetBlockWidth(levelFine);
                    for (int y = 0; y < blockWidth; y++)
                    for (int x = 0; x < blockWidth; x++)
                    {
                        int ii = BlockCoord2Idx(x, y, blockWidth);
                        _vCoarse1[phc + ii] = _vFine1[phf + ii];
                        _vCoarse2[phc + ii] = _vFine2[phf + ii];
                    }
                }
            }
        }
        
        #endregion

        #region Utils
        
        private struct Int2Comparer : IComparer<int2>
        {
            public int Compare(int2 lhs, int2 rhs) => lhs.x - rhs.x;
        }
        
        private static int2 Idx2Coord(int i) => new int2(i % MSBGConstants.GridWidth, i / MSBGConstants.GridWidth);

        private static int Coord2Idx(int2 i) => Coord2Idx(i.x, i.y);
        private static int Coord2Idx(int x, int y) => x + y * MSBGConstants.GridWidth;
            
        private static int BlockCoord2Idx(int x, int y, int res) => x + y * res;
        private static int BlockCoord2Idx(int2 coord, int res) => coord.x + coord.y * res;
        
        private static bool IsSolidCell(uint gridTypes) => (gridTypes & 3u) == MSBGConstants.SOLID;
        private static bool2 IsSolidCell(uint2 gridTypes) => (gridTypes & 3u) == MSBGConstants.SOLID;
        private static bool IsFluidCell(uint gridTypes) => (gridTypes & 3u) == MSBGConstants.FLUID;
        private static bool IsEdgeFluidCell(uint gridTypes)
        {
            if ((gridTypes & 3u) != MSBGConstants.FLUID) 
                return false;
            
            uint4 types = new uint4((gridTypes >> 2) & 3u, (gridTypes >> 4) & 3u,
                (gridTypes >> 6) & 3u, (gridTypes >> 8) & 3u);
            
            return math.any(types != MSBGConstants.FLUID);
        }

        private static bool IsAirCell(uint gridTypes) => (gridTypes & 3u) == MSBGConstants.AIR;

        private static uint2 NeighborGridTypeLB(uint gridTypes) => 
            new uint2((gridTypes >> 2) & 3u, (gridTypes >> 6) & 3u);

        private static uint2 NeighborGridTypeAxis(int axis, uint gridTypes) =>
            new uint2((gridTypes >> (axis * 4 + 2)) & 3u, (gridTypes >> (axis * 4 + 4)) & 3u);

        private static float2 EnforceBoundaryCondition(float2 velocity, uint gridTypes) => 
            IsSolidCell(gridTypes) ? float2.zero
            : math.select(velocity, 0, IsSolidCell(NeighborGridTypeLB(gridTypes)));

        private static float2 ClampPosition(float2 pos) =>
            math.clamp(pos, 1.5f * MSBGConstants.BaseCellSize, 
                (MSBGConstants.GridWidth * MSBGConstants.WidthLevel0 - 1.5f) * MSBGConstants.BaseCellSize);

        private static void FillHaloBlock(NativeArray<float> v, NativeArray<int4> lut, 
            NativeArray<float> block, int2 coord, int4 info)
        {
            int level = GetCurLevel(info.z);
            int blockWidth = GetBlockWidth(level);
            int haloBlockWidth = blockWidth + 2;
            for (int by = 0; by < blockWidth; by++)
            for (int bx = 0; bx < blockWidth; bx++)
            {
                int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
                int physicsIdx = info.x + BlockCoord2Idx(bx, by, blockWidth);
                    
                block[localIdx] = v[physicsIdx];
            }
            int4 ox = new int4(-1, 0, 1, 0);
            int4 oy = new int4(0, -1, 0, 1);
            
            for (int n = 0; n < 4; n++)
            {
                int2 dir = new int2(ox[n], oy[n]);
                int2 curr = coord + dir;
                if (curr.x < 0 || curr.y < 0 || curr.x >= MSBGConstants.GridWidth || curr.y >= MSBGConstants.GridWidth)
                    continue;
                
                int4 neighborInfo = lut[Coord2Idx(curr)];
                int nLevel = GetCurLevel(neighborInfo.z);
                if (nLevel < 0)
                    continue;
                
                int phn = neighborInfo.x;
                if (nLevel == level)
                {
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 nCoord = math.select(math.select(c, 0, dir > 0), blockWidth - 1, dir < 0);
                        int nLocalIdx = BlockCoord2Idx(nCoord, blockWidth);
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = v[phn + nLocalIdx];
                    }
                }
                else if (nLevel > level)
                {
                    int nBlockWidth = GetBlockWidth(nLevel);
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                        int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = v[phn + nLocalIdx] * 0.5f;
                    }
                }
                else // n_level < level
                {
                    int nBlockWidth = GetBlockWidth(nLevel);
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                        int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                        int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                        int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = (v[phn + nLocalIdx0] + v[phn + nLocalIdx1]);
                    }
                }
            }
        }
        
        private static bool InActive(float x) => math.abs(x) < 1e-6f;
        
        private static int GetBlockWidth(int level) => 1 << (3 - level);
        
        private static int GetCellSize(int level) => 1 << level;

        private static int GetCurLevel(int code) => code < 0 ? -1 : (code & 15);
        private static int GetBlockSize(int level) => 1 << (3 - level) * 2;

        private static int4 GetNeighborsLevel(int code)
        {
            if (code < 0) return -1;
            var levels = new int4((code & 15), (code >> 4) & 15, (code >> 8) & 15, (code >> 12) & 15);
            return math.select(levels, -1, levels == 15);
        }

        private static int PackNeighborsLevel(int4 levels) => 
            (levels.x & 15) | ((levels.y & 15) << 4) | ((levels.z & 15) << 8) | ((levels.w & 15) << 12);

        private static float LerpBilinear(float2 weight, float lb, float rb, float lt, float rt)
        {
            var b = math.lerp(lb, rb, weight.x);
            var t = math.lerp(lt, rt, weight.x);
            return math.lerp(b, t, weight.y);
        }
        
        private static float2 LerpBilinear(float2 weight, float2 lb, float2 rb, float2 lt, float2 rt)
        {
            var b = math.lerp(lb, rb, weight.x);
            var t = math.lerp(lt, rt, weight.x);
            return math.lerp(b, t, weight.y);
        }

        #endregion

    }
}

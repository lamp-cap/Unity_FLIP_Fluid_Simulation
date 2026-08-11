using System;
using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine.Profiling;
using Random = Unity.Mathematics.Random;

[CustomEditor(typeof(AdaptiveNarrowBandFLIP))]
public class AdaptiveNarrowBandFLIPEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        var modular = target as AdaptiveNarrowBandFLIP;
        if (GUILayout.Button("Solve"))
        {
            modular?.Test();
        }
        if (GUILayout.Button("Test V-Cycle Symmetry"))
        {
            modular?.TestVCycleSymmetry();
        }
        if (GUILayout.Button("Test Init"))
        {
            modular?.Init();
        }
    }
}

[ExecuteAlways]
public class AdaptiveNarrowBandFLIP : MonoBehaviour
{
    public enum SolverType
    {
        CG,
        MG,
        MGPCG
    }
    public SolverType solverType;

    [Range(-10, 10)] public float gravity = -9;
    [Range(0, 1)] public float flipness = 0.95f;
    public Mesh mesh;
    public Material mat;
    private float _rs;
        

    private const float InvDeltaTime = 120f;
    private const float DeltaTime = 1.0f / InvDeltaTime;
    public const int NumParticles = 128 * 128;
    private const int Band1 = 2;
    private const int Band2 = 3;

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

    private const int gWidth = MultiresBlockGridSolver.GridWidth;
    private const int bWidth = MultiresBlockGridSolver.BaseBlockSize;
    private const int bCount = gWidth * gWidth;
    private const int cWidth = MultiresBlockGridSolver.BaseCellSize;

    private MultiresBlockGridSolver _mbg;
    
    // Start is called before the first frame update
    void OnEnable()
    {
        _posBuffer = new ComputeBuffer(16384, sizeof(float) * 4);
        _bounds = new Bounds(Vector3.zero, Vector3.one * 10f);
        _particleCount = new NativeReference<int>(Allocator.Persistent);
        const int poolSize = 16384;
        _start = new NativeArray<int>(poolSize, Allocator.Persistent);
        _end = new NativeArray<int>(poolSize, Allocator.Persistent);
        _range = new NativeArray<int2>(poolSize, Allocator.Persistent);
        
        _particleID = new NativeArray<int>(NumParticles, Allocator.Persistent);
        _particlePos = new NativeArray<float4>(NumParticles, Allocator.Persistent);
        _particleVelocity = new NativeArray<float2>(NumParticles, Allocator.Persistent);
        _particlePosCopy = new NativeArray<float4>(NumParticles, Allocator.Persistent);
        _particleVelocityCopy = new NativeArray<float2>(NumParticles, Allocator.Persistent);
        _hashes = new NativeArray<int2>(NumParticles, Allocator.Persistent);
        if (mat != null)
            mat.SetBuffer("_ParticleBuffer", _posBuffer);
    }

    public void Init()
    {
        if (_mbg != null) _mbg.Dispose();
        _mbg = new MultiresBlockGridSolver();

        int2 start = new int2(1, 4);
        int2 end = new int2(92, 92);
        int2 startBlock = start / 8;
        int2 endBlock = end / 8;
        for (int y = 0; y < gWidth; y++)
        for (int x = 0; x < gWidth; x++)
        {
            int2 coord = new int2(x, y);
            int level = -1;
            if (math.all(coord >= startBlock) && math.all(coord <= endBlock))
            {
                if (math.any(coord == startBlock) || math.any(coord == endBlock))
                    level = 0;
                else
                    level = 2;
            }
            _mbg.BlockLevel[y * gWidth + x] = level;
        }
        int count = _mbg.InitBaseCells(new int4(start, end));
        
        for (int y = 0; y < gWidth; y++)
        for (int x = 0; x < gWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int2 info = _mbg.GridInfos[i];
            int level = info.x;
            if (level < 0) continue;
            int ptr = info.y;
            int width = BlockWidth(level);
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int idx = ptr + BlockCoord2Idx(xx, yy, width);
                uint type = _mbg.GridTypes[idx];
                int dist = level > 0 ? 8 : -1;
                if (level == 0 && IsFluidCell(type))
                {
                    uint4 neighborTypes = NeighborGridTypes(type);
                    dist = math.any(neighborTypes == AIR) ? 0 : 8;
                }
                _mbg.SDF[idx] = dist;
            }
        }

        _mbg.IterateCellSDF();

        int pCounter = 0;
        for (int y = 0; y < gWidth; y++)
        for (int x = 0; x < gWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int2 info = _mbg.GridInfos[i];
            int level = info.x;
            if (level != 0) continue;
            int ptr = info.y;
            int width = BlockWidth(level);
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int idx = ptr + BlockCoord2Idx(xx, yy, width);
                float dist = _mbg.SDF[idx];
                if (dist >= 0 && dist < Band2)
                {
                    float2 coord = new float2(x * bWidth + xx, y * bWidth + yy);
                    _particlePos[pCounter] = new float4(coord.x + 0.25f, coord.y + 0.25f, 0, 0);
                    _particlePos[pCounter+1] = new float4(coord.x + 0.75f, coord.y + 0.25f, 0, 0);
                    _particlePos[pCounter+2] = new float4(coord.x + 0.25f, coord.y + 0.75f, 0, 0);
                    _particlePos[pCounter+3] = new float4(coord.x + 0.75f, coord.y + 0.75f, 0, 0);
                    pCounter += 4;
                }
                _mbg.SDF[idx] = dist;
            }
        }

        _particleCount.Value = pCounter;
        
        _posBuffer.SetData(_particlePos);

        var rnd = new Random(123456);
        for (int i = 0; i < count; i++)
        {
            var vel = rnd.NextFloat2();
            var type = _mbg.GridTypes[i];
            if ((type & 3u) == 2) vel = 0;
            if (((type >> 2) & 3u) == 2) vel.x = 0;
            if (((type >> 6) & 3u) == 2) vel.x = 0;
            _mbg.GridVelocity[i] = vel;
        }
        
        _mbg.Solve();
        
        Debug.Log($"MGPCG_FLIP initialized, particle num: {pCounter}, allocate cells: {count}.");

        TestStep();
    }

    // Update is called once per frame
    private void Update()
    {
        if (_particleCount.Value > 0 && mat != null && mesh != null)
        {
            Graphics.DrawMeshInstancedProcedural(mesh, 0, mat, _bounds, _particleCount.Value);
        }
    }

    private void TestStep()
    {
        int batchCount = 32;
        int cellCount = _mbg.CellCount;

        new ClearGridJob()
        {
            Start = _start,
            End = _end,
            Range = _range,
        }.Schedule(_range.Length, batchCount).Complete();
        
        new HashJob
        {
            Ps = _particlePos,
            Hashes = _hashes,
        }.Schedule(_particleCount.Value, batchCount).Complete();

        _hashes.Slice(0, _particleCount.Value).SortJob(new Int2Comparer()).Schedule().Complete();
        
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
        }.Schedule(_range.Length, batchCount).Complete();
        
        new SetBlockLevelJob()
        {
            Range = _range,
            BlockLevel = _mbg.BlockLevel
        }.Schedule(bCount, 1).Complete();
        
        cellCount = _mbg.AllocateBaseCells();
        
        new SetCellTypesJob()
        {
            Range = _range,
            Lut = _mbg.GridInfos,
            LutOld = _mbg.GridInfosOld,
            SDF = _mbg.SDF,
            GridTypes = _mbg.GridTypes,
        }.Schedule(bCount, 1).Complete();
        
        new SetNeighborTypesJob()
        {
            Lut = _mbg.GridInfos,
            GridTypes = _mbg.GridTypes,
            GridSDF = _mbg.SDF
        }.Schedule(bCount, 1).Complete();
        
        _mbg.IterateCellSDF();
        
        new ParticlesCounterJob(_mbg.GridInfos, _mbg.SDF, _range, _particleID, _particleCount).Run();
        Debug.Log("Resample particle count: " + _particleCount.Value);
        
        new ResampleParticlesJob()
        {
            GridLut = _mbg.GridInfos,
            PosRaw = _particlePos,
            VelRaw = _particleVelocity,
            PosNew = _particlePosCopy,
            VelNew = _particleVelocityCopy,
            Ids = _particleID,
            GridVelocity = _mbg.GridVelocity,
            Ranges = _range,
        }.Schedule(bCount, batchCount).Complete();

        (_particlePos, _particlePosCopy) = (_particlePosCopy, _particlePos);
        (_particleVelocity, _particleVelocityCopy) = (_particleVelocityCopy, _particleVelocity);

        for (int i = 0; i < _particleVelocity.Length; i++)
        {
            _particleVelocity[i] = new float2(0, -1);
        }
        
        new ParticleToGridJob
        {
            Ranges = _range,
            GridLut = _mbg.GridInfos,
            GridSDF = _mbg.SDF,
            ParticlePos = _particlePos,
            ParticleVel = _particleVelocity,
            GridVelocity = _mbg.GridVelocityAlt,
        }.Schedule(gWidth * gWidth, batchCount).Complete();
        
        _mbg.GridVelocity.CopyFrom(_mbg.GridVelocityAlt);
        
        new AddForceJob
        {
            GridVelocity = _mbg.GridVelocity,
            GridTypes = _mbg.GridTypes,
            Gravity = new float2(0, gravity),
        }.Schedule(cellCount, batchCount).Complete();
        
        _mbg.Solve();
        
        new GridToParticleJob
        {
            GridVelocityOld = _mbg.GridVelocityAlt,
            GridVelocityNew = _mbg.GridVelocity,
            ParticleVel = _particleVelocity,
            ParticlePos = _particlePos,
            Flipness = flipness,
        }.Schedule(_particleCount.Value, batchCount).Complete();
        
        // new ParticlesAdvectionJob
        // {
        //     GridLut =  _mbg.GridInfos,
        //     GridVelocity = _mbg.GridVelocity,
        //     ParticlePos = _particlePos,
        // }.Schedule(_particleCount.Value, batchCount).Complete();
        //
        // new GridsAdvectionJob()
        // {
        //     GridVelocity = _mbg.GridVelocity,
        //     GridVelocityAlt = _mbg.GridVelocityAlt,
        //     GridTypes = _mbg.GridTypes,
        // }.Schedule(cellCount, batchCount).Complete();
    }

    private void Step()
    {
        int batchCount = 32;
        int cellCount = _mbg.CellCount;

        Profiler.BeginSample("Clear Grid");
        new ClearGridJob()
        {
            Start = _start,
            End = _end,
            Range = _range,
        }.Schedule(cellCount, batchCount).Complete();
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
        }.Schedule(cellCount, batchCount).Complete();
        
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

        new SetBlockLevelJob()
        {
            Range = _range,
            BlockLevel = _mbg.BlockLevel
        }.Schedule(bCount, 1).Complete();
        cellCount = _mbg.AllocateBaseCells();
        
        new SetCellTypesJob()
        {
            Range = _range,
            Lut = _mbg.GridInfos,
            LutOld = _mbg.GridInfosOld,
            SDF = _mbg.SDF,
            GridTypes = _mbg.GridTypes,
        }.Schedule(bCount, 1).Complete();
        
        new SetNeighborTypesJob()
        {
            Lut = _mbg.GridInfos,
            GridTypes = _mbg.GridTypes,
            GridSDF = _mbg.SDF
        }.Schedule(bCount, 1).Complete();
        
        new ParticlesCounterJob(_mbg.GridInfos, _mbg.SDF, _range, _particleID, _particleCount).Run();
        
        new ResampleParticlesJob()
        {
            GridLut = _mbg.GridInfos,
            PosRaw = _particlePos,
            VelRaw = _particleVelocity,
            PosNew = _particlePosCopy,
            VelNew = _particleVelocityCopy,
            Ids = _particleID,
            GridVelocity = _mbg.GridVelocity,
            Ranges = _range,
        }.Schedule(bCount, batchCount).Complete();

        (_particlePos, _particlePosCopy) = (_particlePosCopy, _particlePos);
        (_particleVelocity, _particleVelocityCopy) = (_particleVelocityCopy, _particleVelocity);
        
        Profiler.EndSample();
    
        Profiler.BeginSample("P2G");
        
        new ParticleToGridJob
        {
            Ranges = _range,
            GridLut = _mbg.GridInfos,
            ParticlePos = _particlePos,
            ParticleVel = _particleVelocity,
            GridVelocity = _mbg.GridVelocityAlt,
        }.Schedule(gWidth * gWidth, batchCount).Complete();
        
        _mbg.GridVelocity.CopyFrom(_mbg.GridVelocityAlt);
        
        new AddForceJob
        {
            GridVelocity = _mbg.GridVelocity,
            GridTypes = _mbg.GridTypes,
            Gravity = new float2(0, gravity),
        }.Schedule(cellCount, batchCount).Complete();
        Profiler.EndSample();
    
        Profiler.BeginSample("Solve Pressure");
        _mbg.Solve();
        Profiler.EndSample();
    
        Profiler.BeginSample("G2P");
        new GridToParticleJob
        {
            GridVelocityOld = _mbg.GridVelocityAlt,
            GridVelocityNew = _mbg.GridVelocity,
            ParticleVel = _particleVelocity,
            ParticlePos = _particlePos,
            Flipness = flipness,
        }.Schedule(_particleCount.Value, batchCount).Complete();
    
        Profiler.EndSample();
    
        Profiler.BeginSample("Advection");
        new ParticlesAdvectionJob
        {
            GridLut =  _mbg.GridInfos,
            GridVelocity = _mbg.GridVelocity,
            ParticlePos = _particlePos,
        }.Schedule(_particleCount.Value, batchCount).Complete();
        
        new GridsAdvectionJob()
        {
            GridVelocity = _mbg.GridVelocity,
            GridVelocityAlt = _mbg.GridVelocityAlt,
            GridTypes = _mbg.GridTypes,
        }.Schedule(cellCount, batchCount).Complete();
        Profiler.EndSample();
    }
    
    [BurstCompile]
    private struct ClearGridJob : IJobParallelFor
    {
        [WriteOnly] public NativeArray<int2> Range;
        [WriteOnly] public NativeArray<int> Start;
        [WriteOnly] public NativeArray<int> End;

        public void Execute(int i)
        {
            Start[i] = 0;
            End[i] = 0;
            Range[i] = int2.zero;
        }
    }

    [BurstCompile]
    private struct HashJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float4> Ps;
        [WriteOnly] public NativeArray<int2> Hashes;

        public void Execute(int i)
        {
            int2 coord = GetCoord(Ps[i].xy);
            int hash = coord.x + coord.y * gWidth * bWidth;
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
    private struct SetBlockLevelJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int2> Range;
        public NativeArray<int> BlockLevel;

        public void Execute(int i)
        {
            int oldLevel = BlockLevel[i];
            int level = oldLevel < 1 ? -1 : 2;
            if (level < 0)
            {
                int2 coord = Idx2Coord(i);
                int2 haloStart = math.max(0, coord * bWidth - 1);
                int2 haloEnd = math.min(bWidth * gWidth, coord * bWidth + bWidth + 1);
                for (int y = haloStart.y; y < haloEnd.y; y++)
                for (int x = haloStart.x; x < haloEnd.x; x++)
                {
                    int2 range = Range[x + y * gWidth * bWidth];
                    if (range.y > range.x)
                    {
                        level = 0;
                        break;
                    }
                }
            }
            BlockLevel[i] = level;
        }
    }

    [BurstCompile]
    private struct SetCellTypesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int2> Range;
        [ReadOnly] public NativeArray<int2> Lut;
        [ReadOnly] public NativeArray<int2> LutOld;
        [ReadOnly] public NativeArray<float> SDF;
        [NativeDisableParallelForRestriction, WriteOnly]
        public NativeArray<uint> GridTypes;

        public void Execute(int i)
        {
            int2 coord = Idx2Coord(i);
            int2 info = Lut[i];
            int level = info.x;
            if (level < 0) return;
            int ptr = info.y;
            int width = BlockWidth(level);
            int2 oldInfo = LutOld[i];
            int oldLevel = oldInfo.x;
            int oldPtr = oldInfo.y;
            for (int y = 0; y < width; y++)
            for (int x = 0; x < width; x++)
            {
                int localIdx = BlockCoord2Idx(x, y, width);
                int idx = ptr + localIdx;
                uint type = AIR;
                if (level == 0)
                {
                    int2 range = Range[(coord.x * bWidth + x) + (coord.y * bWidth + y) * gWidth * bWidth];
                    if (range.y > range.x) type = FLUID;
                    else if (oldLevel >= 0)
                    {
                        float oldSDF = SDF[oldPtr + localIdx];
                        if (oldSDF > Band1) type = FLUID; // always true when CFL < Band1
                    }
                }
                else type = FLUID; // inside block

                GridTypes[idx] = type;
            }
        }
    }

    [BurstCompile]
    private struct SetNeighborTypesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int2> Lut;
        [NativeDisableParallelForRestriction] public NativeArray<uint> GridTypes;
        [NativeDisableParallelForRestriction] public NativeArray<float> GridSDF;

        public void Execute(int i)
        {
            int2 coord = Idx2Coord(i);
            int2 info = Lut[i];
            int level = info.x;
            if (level < 0) return;
            int ptr = info.y;
            int width = BlockWidth(level);
            int haloWidth = width + 2;
            var haloBlock = new NativeArray<uint>(haloWidth * haloWidth, Allocator.Temp);
            FillHaloBlock(GridTypes, Lut, haloBlock, coord);
            for (int y = 1; y <= width; y++)
            for (int x = 1; x <= width; x++)
            {
                uint c = haloBlock[BlockCoord2Idx(x, y, haloWidth)];
                uint l = haloBlock[BlockCoord2Idx(x - 1, y, haloWidth)];
                uint r = haloBlock[BlockCoord2Idx(x + 1, y, haloWidth)];
                uint b = haloBlock[BlockCoord2Idx(x, y - 1, haloWidth)];
                uint t = haloBlock[BlockCoord2Idx(x, y + 1, haloWidth)];

                int idx = ptr + BlockCoord2Idx(x - 1, y - 1, width);
                GridTypes[idx] = PackGridTypes(c, l, r, b, t);

                float sdf = -1;
                if (c == FLUID)
                    sdf = level == 0 && math.any(new uint4(l, r, b, t) == AIR) ? 0 : 8;
                
                GridSDF[idx] = sdf;
            }

            haloBlock.Dispose();
        }
        private static void FillHaloBlock(NativeArray<uint> v, NativeArray<int2> infos, NativeArray<uint> block, int2 coord)
        {
            int2 info = infos[Coord2Idx(coord)];
            int level = info.x;
            int ptr = info.y;
            int blockWidth = BlockWidth(level);
            int haloBlockWidth = blockWidth + 2;
            for (int by = 0; by < blockWidth; by++)
            for (int bx = 0; bx < blockWidth; bx++)
            {
                int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
                int physicsIdx = ptr + BlockCoord2Idx(bx, by, blockWidth);
                    
                block[localIdx] = v[physicsIdx] & 3u;
            }
            int4 ox = new int4(-1, 1, 0, 0);
            int4 oy = new int4(0, 0, -1, 1);
            
            for (int n = 0; n < 4; n++)
            {
                int2 dir = new int2(ox[n], oy[n]);
                int2 curr = coord + dir;
                if (curr.x < 0 || curr.y < 0 || curr.x >= gWidth || curr.y >= gWidth)
                {
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = SOLID;
                    }
                    continue;
                }
                
                int2 nInfo = infos[Coord2Idx(curr)];
                int nLevel = nInfo.x;
                if (nLevel < 0)
                {
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = AIR;
                    }
                    continue;
                }
                
                int phn = nInfo.y;
                if (nLevel == level)
                {
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 nCoord = math.select(math.select(c, 0, dir > 0), blockWidth - 1, dir < 0);
                        int nLocalIdx = BlockCoord2Idx(nCoord, blockWidth);
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = v[phn + nLocalIdx] & 3;
                    }
                }
                else // always fluid cell inside
                {
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = FLUID;
                    }
                }
            }
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
    private struct ResampleParticlesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int2> GridLut;
        [ReadOnly] public NativeArray<int> Ids;
        [ReadOnly] public NativeArray<int2> Ranges;
        [ReadOnly] public NativeArray<float2> GridVelocity;

        [ReadOnly] public NativeArray<float4> PosRaw;
        [ReadOnly] public NativeArray<float2> VelRaw;

        [NativeDisableContainerSafetyRestriction, WriteOnly] public NativeArray<float4> PosNew;
        [NativeDisableContainerSafetyRestriction, WriteOnly] public NativeArray<float2> VelNew;

        public void Execute(int bid)
        {
            var info = GridLut[bid];
            var level = info.x;
            if (level != 0) return;
            int width = BlockWidth(level);
            int haloWidth = width + 2;
            var haloBlock = new NativeArray<float2>(haloWidth * haloWidth, Allocator.Temp);
            var coord = Idx2Coord(bid);
            FillHaloBlock(GridVelocity, GridLut, haloBlock, coord);
            float2 haloOrigin = coord * bWidth - 1;
            for (int y = 0; y < width; y++)
            for (int x = 0; x < width; x++)
            {
                var cCoord = new int2(coord.x * bWidth + x, coord.y * bWidth + y);
                int2 range = Ranges[(cCoord.x) + (cCoord.y) * gWidth * bWidth];
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
                
                float2 origin = ((float2)cCoord + 0.25f) * CellSize;
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
                    var v = ReadGridFaceBilinear(selectedPos - haloOrigin, haloBlock);
                    int p = range.x + existCount;
                    PosNew[p] = new float4(selectedPos, 0, math.length(v));
                    VelNew[p] = v;
                    posArr[existCount] = selectedPos;
                    existCount++;
                }
            }
        }
    
        private float2 ReadGridFaceBilinear(float2 pos, NativeArray<float2> block)
        {
            ReadGridFaceBilinear(pos * InvCellSize + new float2(0, -0.5f), 0, block, out var vx);
            ReadGridFaceBilinear(pos * InvCellSize + new float2(-0.5f, 0), 1, block, out var vy);
            return new float2(vx, vy);
        }
        
        private void ReadGridFaceBilinear(float2 uv, int axis, NativeArray<float2> block, out float v)
        {
            uv = math.clamp(uv, 1e-3f, bWidth + 2 - 1e-3f);
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

        private static float2 ReadGrid(int2 coord, NativeArray<float2> block)
        {
            return block[Coord2Idx(math.clamp(coord, 0, bWidth + 1))];
        }
        private static void FillHaloBlock(NativeArray<float2> v, NativeArray<int2> infos, NativeArray<float2> block, int2 coord)
        {
            int2 info = infos[Coord2Idx(coord)];
            int level = info.x;
            int ptr = info.y;
            int blockWidth = BlockWidth(level);
            int haloBlockWidth = blockWidth + 2;
            for (int by = 0; by < blockWidth; by++)
            for (int bx = 0; bx < blockWidth; bx++)
            {
                int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
                int physicsIdx = ptr + BlockCoord2Idx(bx, by, blockWidth);
                    
                block[localIdx] = v[physicsIdx];
            }
            int4 ox = new int4(-1, 1, 0, 0);
            int4 oy = new int4(0, 0, -1, 1);
            
            for (int n = 0; n < 4; n++)
            {
                int2 dir = new int2(ox[n], oy[n]);
                int2 curr = coord + dir;
                if (curr.x < 0 || curr.y < 0 || curr.x >= gWidth || curr.y >= gWidth)
                    continue;
                
                int2 nInfo = infos[Coord2Idx(curr)];
                int nLevel = nInfo.x;
                if (nLevel < 0)
                    continue;
                
                int phn = nInfo.y;
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
                    int nBlockWidth = BlockWidth(nLevel);
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
                    int nBlockWidth = BlockWidth(nLevel);
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
    private struct ParticlesCounterJob : IJob
    {
        [ReadOnly] private NativeArray<int2> _gridLut;
        [ReadOnly] private NativeArray<float> _gridSDF;
        private NativeArray<int2> _range;
        [WriteOnly] private NativeArray<int> _particleIDs;
        [WriteOnly] private NativeReference<int> _pCount;
            
        public ParticlesCounterJob(NativeArray<int2> lut, NativeArray<float> sdf, NativeArray<int2> range, NativeArray<int> particleIDs, NativeReference<int> pCount)
        {
            _gridLut = lut;
            _gridSDF = sdf;
            _range = range;
            _particleIDs = particleIDs;
            _pCount = pCount;
        }

        public void Execute()
        {
            int pCounter = 0;
            for (int i = 0; i < bCount; i++)
            {
                var info = _gridLut[i];
                var level = info.x;
                if (level != 0) continue;
                int ptr = info.y;
                int width = BlockWidth(level);
                var coord = Idx2Coord(i);
                for (int y = 0; y < width; y++)
                for (int x = 0; x < width; x++)
                {
                    int idx = ptr + BlockCoord2Idx(x, y, width);
                    var sdf = _gridSDF[idx];
                    int cid = (coord.x * bWidth + x) + (coord.y * bWidth + y) * gWidth * bWidth;
                    if (sdf < 0 || sdf >= Band2)
                    {
                        _range[cid] = int2.zero;
                        continue;
                    }

                    int2 range = _range[cid];
                    int count = range.y - range.x;
                    int expect = sdf < 1 ? count : math.max(4, count);
                    for (int j = 0; j < expect; j++)
                        _particleIDs[pCounter + j] = j < count ? range.x + j : -1; // if need add particle, set -1
                    
                    _range[cid] = new int2(pCounter, pCounter + expect);
                    pCounter += expect;
                }
            }
            _pCount.Value = pCounter;
        }
    }
        
    [BurstCompile]
    private struct ParticleToGridJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int2> Ranges;
        [ReadOnly] public NativeArray<int2> GridLut;
        [ReadOnly] public NativeArray<float> GridSDF;
        [ReadOnly] public NativeArray<float4> ParticlePos;
        [ReadOnly] public NativeArray<float2> ParticleVel;
        [NativeDisableParallelForRestriction] public NativeArray<float2> GridVelocity;

        public void Execute(int i)
        {
            int2 info = GridLut[i];
            int level = info.x;
            if (level != 0) 
                return;

            int ptr = info.y;
            
            int2 blockCoord = Idx2Coord(i);

            int width = BlockWidth(level);
            
            int haloWidth = width + 2;
            var haloBlock = new NativeArray<int2>(haloWidth * haloWidth, Allocator.Temp);

            int2 start = blockCoord * bWidth - 1;
            int2 haloStart = math.max(0, start);
            int2 haloEnd = math.min(bWidth * gWidth, blockCoord * bWidth + bWidth + 1);
            for (int y = haloStart.y; y < haloEnd.y; y++)
            for (int x = haloStart.x; x < haloEnd.x; x++)
            {
                haloBlock[BlockCoord2Idx(x - start.x, y - start.y, haloWidth)] = Ranges[x + y * gWidth * bWidth];
            }
            
            for (int y = 0; y < width; y++)
            for (int x = 0; x < width; x++)
            {
                int idx = ptr + BlockCoord2Idx(x, y, width);
                if (GridSDF[idx] > Band1) continue;
                
                int2 coord = blockCoord * bWidth + new int2(x, y);
                float2 velocity = float2.zero;
                float2 sum = 0;
                float2 cellCenter = ((float2)coord + 0.5f) * CellSize;
                float2 positionVx = cellCenter + new float2(-0.5f * CellSize, 0.0f);
                float2 positionVy = cellCenter + new float2(0.0f, -0.5f * CellSize);

                var nStart = math.max(coord - 1, 0);
                var nEnd = math.min(coord + 1, gWidth * bWidth - 1);
                for (int xx = nStart.x; xx <= nEnd.x; ++xx)
                for (int yy = nStart.y; yy <= nEnd.y; ++yy)
                {
                    int2 localCoord = new int2(xx, yy) - start;
                    int2 range = haloBlock[BlockCoord2Idx(localCoord.x, localCoord.y, haloWidth)];
                    for (int j = range.x; j < range.y; j++)
                    {
                        float4 p = ParticlePos[j];
                        float2 n_x = p.xy;
                        var n_v = ParticleVel[j];
                        
                        float2 weight = new float2(
                            GetWeight(positionVx - n_x, InvCellSize),
                            GetWeight(positionVy - n_x, InvCellSize));
                        
                        sum += weight;
                        
                        velocity.x += weight.x * n_v.x;
                        velocity.y += weight.y * n_v.y;
                    }
                }

                // 无粒子落到位时保留该 face 上一帧速度（读自己的 cell，避免跨线程竞态）
                float2 oldVelocity = GridVelocity[idx];
                velocity = math.select(oldVelocity, velocity / sum, sum > 1e-4f);
                GridVelocity[idx] = velocity;
            }
        }
        
        private static float GetWeight(float2 delta_pos, float grid_inv_spacing)
        {
            float2 dist = math.abs(delta_pos * grid_inv_spacing);
            float2 weight = math.saturate(GetQuadraticWeight(dist));
            return weight.x * weight.y;
        }

        private static float2 GetQuadraticWeight(float2 abs_x)
        {
            float2 dst = math.saturate(1.5f - abs_x);
            return math.select(0.5f * dst * dst, 0.75f - abs_x * abs_x, abs_x < 0.5f);
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
                velocity += Gravity * DeltaTime;

            GridVelocity[i] = EnforceBoundaryCondition(velocity, gridType);
        }
    }
    
    [BurstCompile]
    private struct GridToParticleJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float4> ParticlePos;
        [ReadOnly] public NativeArray<float2> GridVelocityOld;
        [ReadOnly] public NativeArray<float2> GridVelocityNew;
        [ReadOnly] public NativeArray<float2> GridVelocityOldDS;
        [ReadOnly] public NativeArray<float2> GridVelocityNewDS;
        [ReadOnly] public NativeArray<int2> GridLut;
        public NativeArray<float2> ParticleVel;
        public float Flipness;
    
        public void Execute(int i)
        {
            float2 pos = ParticlePos[i].xy;
            float2 vel = ParticleVel[i];

            SampleGridFaceBilinear(0, pos, GridVelocityOld, GridVelocityOldDS, GridVelocityNew, GridVelocityNewDS, GridLut, 
                out var gOriginVelX, out var gVelX);
            SampleGridFaceBilinear(1, pos, GridVelocityOld, GridVelocityOldDS, GridVelocityNew, GridVelocityNewDS, GridLut, 
                out var gOriginVelY, out var gVelY);
            
            float2 gOriginVel = new float2(gOriginVelX, gOriginVelY);
            float2 gVel = new float2(gVelX, gVelY);

            ParticleVel[i] = math.lerp(gVel, vel + (gVel - gOriginVel), Flipness);
        }
        
        private void SampleGridFaceBilinear(int axis, float2 pos, NativeArray<float2> vf1, NativeArray<float2> vc1, 
            NativeArray<float2> vf2, NativeArray<float2> vc2, NativeArray<int2> lut, out float u1, out float u2)
        {
            u1 = 0;
            u2 = 0;
            const int baseBlockWidth = bWidth;
            const int gridSize = gWidth * bWidth;
            float2 basePos = pos / cWidth;
            int2 baseCoord = (int2)math.floor(basePos);
            if (math.any(baseCoord < 0) || math.any(baseCoord >= gridSize))
                return;
            
            int2 blockCoord = baseCoord / baseBlockWidth;
            int2 info = lut[Coord2Idx(blockCoord)];
            int blockLevel = info.x;
            if (blockLevel != 0)
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
                int ptr = info.y;

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

            // edge between same level blocks
            {
                bool2 selector = weight > 0.5f;
                selector[axis] = false;
                int2 c0 = baseCoord - math.select(int2.zero, 1, selector);
                int2 c1 = c0 + 1;
                SamplePointFine(vf1, vf2, lut, c0.x, c0.y, out float2 lb1, out float2 lb2);
                SamplePointFine(vf1, vf2, lut, c1.x, c0.y, out float2 rb1, out float2 rb2);
                SamplePointFine(vf1, vf2, lut, c0.x, c1.y, out float2 lt1, out float2 lt2);
                SamplePointFine(vf1, vf2, lut, c1.x, c1.y, out float2 rt1, out float2 rt2);
                u1 = LerpBilinear(weight, lb1[axis], rb1[axis], lt1[axis], rt1[axis]);
                u2 = LerpBilinear(weight, lb2[axis], rb2[axis], lt2[axis], rt2[axis]);
            }
        }

        private static void SamplePointFine(NativeArray<float2> v1, NativeArray<float2> v2, NativeArray<int2> lut,
            int x, int y, out float2 r1, out float2 r2)
        {
            var baseCoord = math.clamp(new int2(x, y), 0, gWidth * bWidth - 1);
            const int baseBlockWidth = bWidth;
            int2 blockCoord = baseCoord / baseBlockWidth;
            int blockIdx = Coord2Idx(blockCoord);
            int2 info = lut[blockIdx];
            int level = info.x;
            r1 = float2.zero;
            r2 = float2.zero;
            if (level < 0)
                return;
            int blockWidth = baseBlockWidth >> level;
            int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
            r1 = v1[info.y + BlockCoord2Idx(localCoord, blockWidth)];
            r2 = v2[info.y + BlockCoord2Idx(localCoord, blockWidth)];
        }
    }

    [BurstCompile]
    private struct ParticlesAdvectionJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int2> GridLut;
        [ReadOnly] public NativeArray<float2> GridVelocity;
        [ReadOnly] public NativeArray<float2> GridVelocityDS;
        public NativeArray<float4> ParticlePos;
    
        public void Execute(int i)
        {
            float4 particle = ParticlePos[i];
            float2 pos = particle.xy;
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
            particle.xy = ClampPosition(pos);
            ParticlePos[i] = particle;
        }
        
        private void SampleGridFaceBilinear(int axis, float2 pos, NativeArray<float2> vf1, NativeArray<float2> vc1, 
            NativeArray<int2> lut, out float u1)
        {
            u1 = 0;
            const int baseBlockWidth = bWidth;
            const int gridSize = gWidth * bWidth;
            float2 basePos = pos / cWidth;
            int2 baseCoord = (int2)math.floor(basePos);
            if (math.any(baseCoord < 0) || math.any(baseCoord >= gridSize))
                return;
            
            int2 blockCoord = baseCoord / baseBlockWidth;
            int2 info = lut[Coord2Idx(blockCoord)];
            int blockLevel = info.x;
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
                int ptr = info.y;

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
                int2 infoLB = lut[Coord2Idx(lbBlockCoord)];
                int4 neighborLevel = blockLevel;
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
            int2 infoFace = lut[Coord2Idx(faceBlockCoord)];
        
            blockLevel = infoFace.x;
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
                
                int4 neighborLevelCur = blockLevel;
                int4 neighborLevelLB = blockLevel;
                int levelR = neighborLevelCur.y;
                int levelT = neighborLevelCur.z;
                int levelRT = neighborLevelCur.w;
                int levelL = neighborLevelLB.z;
                int levelB = neighborLevelLB.y;
                int levelLB = neighborLevelLB.x;
                int levelLT = lut[Coord2Idx(math.clamp(faceBlockCoord + new int2(-1, 1), 0, gWidth - 1))].x;
                int levelRB = lut[Coord2Idx(math.clamp(faceBlockCoord + new int2(1, -1), 0, gWidth - 1))].x;
                float dstToCoarse = 1;
                localPos = fineUV + new float2(1, 0.5f);
                float2 subPos = BlockWidth(fineLevel) - fineUV - new float2(0.5f, 0.5f);
                // return math.cmin(localPos);
                int levelC = infoFace.x;
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
        
        private static void SamplePointFine(NativeArray<float2> v1,  NativeArray<int2> lut,
            int x, int y, out float2 r1)
        {
            var baseCoord = math.clamp(new int2(x, y), 0, gWidth * bWidth - 1);
            const int baseBlockWidth = bWidth;
            int2 blockCoord = baseCoord / baseBlockWidth;
            int blockIdx = Coord2Idx(blockCoord);
            int2 info = lut[blockIdx];
            int level = info.x;
            r1 = float2.zero;
            if (level < 0)
                return;
            int blockWidth = baseBlockWidth >> level;
            int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
            r1 = v1[info.y + BlockCoord2Idx(localCoord, blockWidth)];
        }

        private static void SamplePointLevel(NativeArray<float2> vf1, NativeArray<float2> vc1, 
            NativeArray<int2> lut, int x, int y, int targetLevel, out float2 r1)
        {
            var baseCoord = math.clamp(new int2(x, y), 0, gWidth * bWidth - 1);
            const int baseBlockWidth = bWidth;
            int2 blockCoord = baseCoord / baseBlockWidth;
            int blockIdx = Coord2Idx(blockCoord);
            int2 info = lut[blockIdx];
            int level = info.x;
            
            r1 = float2.zero;
            if (level < 0) return;
            
            if (level == targetLevel)
            {
                int blockWidth = baseBlockWidth >> level;
                int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
                r1 = vf1[info.y + BlockCoord2Idx(localCoord, blockWidth)];
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
            uv = math.clamp(uv, 1e-3f, gWidth - 1e-3f);
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
        private static float2 ReadGrid(int2 coord, NativeArray<float2> block)
        {
            return block[Coord2Idx(math.clamp(coord, 0, gWidth - 1))];
        }
    }

    public void Test()
    {
        if (_mbg != null) _mbg.Dispose();
        _mbg = new MultiresBlockGridSolver();
        int counter = 0;
        int surface = Mathf.RoundToInt(gWidth * 2f / 3f); 
        for (int y = 0; y < gWidth; y++)
        for (int x = 0; x < gWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = -1;
            if (y < surface)
            {
                level = 0;
                if (y < surface - 1)
                {
                    level = 1;
                    if (y < surface - 2)
                        level = 2;
                }
            }

            _mbg.GridInfos[i] = new int2(level, counter);
            counter += level < 0 ? 0 : BlockWidth(level) * BlockWidth(level);
        }
        Debug.Log("allocate cells " + counter);
        _mbg.SetCellCount(counter);
        
        _mbg.FillMatrix(_mbg.GridLaplacian, _mbg.GridInfos);
        
        var rnd = new Random(123456);
        for (int y = 0; y < gWidth; y++)
        for (int x = 0; x < gWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int2 info = _mbg.GridInfos[i];
            int level = info.x;
            int ptr = info.y;
            int width = BlockWidth(level);
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int idx = ptr + BlockCoord2Idx(xx, yy, width);
                var vel = rnd.NextFloat2();
                if (x == 0 && xx == 0) vel.x = 0;
                if (y == 0 && yy == 0) vel.y = 0;
                _mbg.GridVelocity[idx] = vel;
            }
        }

        int count = gWidth * gWidth * 64;
        var v0 = new NativeArray<float>(count, Allocator.Temp);
        var v1 = new NativeArray<float>(count, Allocator.Temp);
        var b0 = new NativeArray<float>(count, Allocator.Temp);
        var b1 = new NativeArray<float>(count, Allocator.Temp);
        
        float sum1 = 0, sum2 = 0;
        for (int i = 0; i < counter; i++)
        {
            b1[i] = rnd.NextFloat(-1f, 1f);
            b0[i] = rnd.NextFloat(-1f, 1f);
            sum1 += b1[i];
            sum2 += b0[i];
        }

        for (int i = 0; i < counter; i++)
        {
            b1[i] -= sum1 / counter;
            b0[i] -= sum2 / counter;
        }

        sum1 = 0;
        sum2 = 0;
        for (int i = 0; i < counter; i++)
        {
            sum1 += b1[i];
            sum2 += b0[i];
        }
        
        Debug.Log("b1 sum: " + sum1 + ", b2 sum: " + sum2);
        
        _mbg.Flux.CopyFrom(b0);
        Solve();
        _mbg._pressure.CopyTo(v0);
        _mbg._pressure.CopyFrom(v1);
        _mbg.Flux.CopyFrom(b1);
        Solve();
        _mbg._pressure.CopyTo(v1);
        float dot1 = Dot(b1, v0, counter);
        float dot2 = Dot(b0, v1, counter);
        Debug.Log($"Symmetry check: (v1, Av2)={dot1}, (Av1, v2)={dot2}, diff={Math.Abs(dot1-dot2)}");
    }

    // Tests symmetry of the preconditioner M^{-1} directly (one V-cycle from x=0),
    // which IS a linear operator. Unlike Test() -> SolveMGPCG, this is a valid symmetry
    // check: for M^{-1} symmetric we must have <b0, M^{-1} b1> == <b1, M^{-1} b0>.
    public void TestVCycleSymmetry()
    {
        if (_mbg != null) _mbg.Dispose();
        _mbg = new MultiresBlockGridSolver();
        int counter = 0;
        int surface = Mathf.RoundToInt(gWidth * 2f / 3f);
        for (int y = 0; y < gWidth; y++)
        for (int x = 0; x < gWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = -1;
            if (y < surface)
            {
                level = 0;
                if (y < surface - 1)
                {
                    level = 1;
                    if (y < surface - 2)
                        level = 2;
                }
            }

            _mbg.GridInfos[i] = new int2(level, counter);
            counter += level < 0 ? 0 : BlockWidth(level) * BlockWidth(level);
        }
        _mbg.SetCellCount(counter);
        _mbg.FillMatrix(_mbg.GridLaplacian, _mbg.GridInfos);

        int count = gWidth * gWidth * 64;
        var v0 = new NativeArray<float>(count, Allocator.Temp);
        var v1 = new NativeArray<float>(count, Allocator.Temp);
        var b0 = new NativeArray<float>(count, Allocator.Temp);
        var b1 = new NativeArray<float>(count, Allocator.Temp);

        var rnd = new Random(123456);
        float sum0 = 0, sum1 = 0;
        for (int i = 0; i < counter; i++)
        {
            b0[i] = rnd.NextFloat(-1f, 1f);
            b1[i] = rnd.NextFloat(-1f, 1f);
            sum0 += b0[i];
            sum1 += b1[i];
        }
        for (int i = 0; i < counter; i++)
        {
            b0[i] -= sum0 / counter;
            b1[i] -= sum1 / counter;
        }

        // v0 = M^{-1} b0, v1 = M^{-1} b1 — each a single V-cycle from x = 0.
        _mbg.ApplyVCycle(b0, v0);
        _mbg.ApplyVCycle(b1, v1);

        float dot1 = Dot(b0, v1, counter);
        float dot2 = Dot(b1, v0, counter);
        Debug.Log($"V-cycle symmetry: <b0, M^-1 b1>={dot1}, <b1, M^-1 b0>={dot2}, " +
                  $"diff={Math.Abs(dot1 - dot2)}, rel={Math.Abs(dot1 - dot2) / Math.Max(1e-20f, Math.Abs(dot1))}");

        v0.Dispose();
        v1.Dispose();
        b0.Dispose();
        b1.Dispose();
    }

    private void Solve()
    {
        switch (solverType)
        {
            case SolverType.CG:
                _mbg.SolveCG(144);
                break;
            case SolverType.MG:
                _mbg.SolveMG(12);
                break;
            case SolverType.MGPCG:
                _mbg.SolveMGPCG(10);
                break;
        }
    }
    private float Dot(NativeArray<float> a, NativeArray<float> b, int count)
    {
        float dot = 0;
        for (int i = 0; i < count; i++)
        {
            dot += a[i] * b[i];
        }

        return dot;
    }

    private void OnDestroy()
    {
        Clear();
    }

    private void OnDisable()
    {
        Clear();
    }

    private void Clear()
    {
        _start.Dispose();
        _end.Dispose();
        _range.Dispose();
        _particleID.Dispose();
        _particlePos.Dispose();
        _particleVelocity.Dispose();
        _particlePosCopy.Dispose();
        _particleVelocityCopy.Dispose();
        _hashes.Dispose();
        _particleCount.Dispose();
        _mbg?.Dispose();
        _posBuffer.Release();
    }

    private void OnDrawGizmos()
    {
        if (_mbg == null)
            return;
        
        if (false)
        {
            for (int y = 0; y < gWidth; y++)
            for (int x = 0; x < gWidth; x++)
            {
                int i = Coord2Idx(x, y);
                int2 info = _mbg.GridInfos[i];
                int level = info.x;
                if (level < 0) continue;
                int ptr = info.y;
                int width = BlockWidth(level);
                float h = GetH(level);
                float half = h * 0.5f;
                float2 posBase = new float2(x * 8, y * 8);
                for (int yy = 0; yy < width; yy++)
                for (int xx = 0; xx < width; xx++)
                {
                    var center = new Vector3(posBase.x + xx * h + half, posBase.y + yy * h + half, 0f);
                    int idx = ptr + BlockCoord2Idx(xx, yy, width);

                    // float2 vel = _mbg.GridVelocity[idx];
                    float div = _mbg.Flux[idx] * 0.2f;
                    // float div = _mbg._pressure[idx];
                    Gizmos.color = new Color(math.max(0, div), math.max(0, -div), 0);
                    // Gizmos.DrawLine(center, center + new Vector3(vel.x, vel.y, 0f));
                    Gizmos.DrawCube(center, new Vector3(h, h, 0.01f));

                }
            }
        }
        
        Gizmos.color = Color.white;
        for (int y = 0; y < gWidth; y++)
        for (int x = 0; x < gWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int2 info = _mbg.GridInfos[i];
            int level = info.x;
            if (level < 0) continue;
            int ptr = info.y;
            int width = BlockWidth(level);
            float h = GetH(level);
            float half = h * 0.5f;
            float2 posBase = new float2(x * 8, y * 8);
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                var center = new Vector3(posBase.x + xx * h + half, posBase.y + yy * h + half, 0f);
                int idx = ptr + BlockCoord2Idx(xx, yy, width);
                float t = half * 0.8f;
                float d = _mbg.SDF[idx];
                Handles.Label(center, $"{d}");
                float2 vel = _mbg.GridVelocity[idx];
                Gizmos.DrawLine(center, center + new Vector3(vel.x, vel.y, 0));
                // float3 ps = _mbg.GridLaplacian[idx];
                // if ((_mbg.GridTypes[idx] & 3) != 0) continue;
                // Handles.Label(center, $"{ps.x:F2}");
                // Handles.Label(center + new Vector3(-t,0,0.01f), $"{ps.y:F2}");
                // Handles.Label(center + new Vector3(0,-t,0.01f), $"{ps.z:F2}");
                
                Gizmos.DrawWireCube(center, new Vector3(h, h, 0f));
            }
        }
    }

    private const float CellSize = 1;
    private const float InvCellSize = 1;
    
    private struct Int2Comparer : System.Collections.Generic.IComparer<int2>
    {
        public int Compare(int2 lhs, int2 rhs) => lhs.x - rhs.x;
    }

    private static int2 GetCoord(float2 pos) => (int2)math.floor(pos);
    private static int Coord2Idx(int x, int y)=> x + (y * gWidth);
    private static int BlockWidth(int level) => 1 << (3 - level);
    private static int BlockCoord2Idx(int2 coord, int res) => coord.x + coord.y * res;
    private static int BlockCoord2Idx(int x, int y, int res) => x + y * res;
    private static int2 Idx2Coord(int idx) => new int2(idx % gWidth, idx / gWidth);
    private static int Coord2Idx(int2 coord) => coord.x + coord.y * gWidth;
        
    private static float GetH(int level) => 1 << level;
    private const uint SOLID = 2;
    private const uint AIR = 1;
    private const uint FLUID = 0;
    private static bool IsFluidCell(uint gridTypes)
    {
        return (gridTypes & 3u) == FLUID;
    }
    private static bool IsSolidCell(uint gridTypes)
    {
        return (gridTypes & 3u) == SOLID;
    }
    private static bool2 IsSolidCell(uint2 gridTypes)
    {
        return (gridTypes & 3u) == SOLID;
    }
    private static uint2 NeighborGridTypeLB(uint gridTypes)
    {
        return new uint2((gridTypes >> 2) & 3u, (gridTypes >> 6) & 3u);
    }
    private static uint4 NeighborGridTypes( uint gridTypes)
    {
        return new uint4((gridTypes >> 2) & 3u, (gridTypes >> 4) & 3u, (gridTypes >> 6) & 3u, (gridTypes >> 8) & 3u);
    }
    private static uint PackGridTypes(uint c, uint l, uint r, uint b, uint t)
    {
        return c | (l << 2)  | (r << 4) | (b << 6) | (t << 8);
    }
    private static float2 EnforceBoundaryCondition(float2 velocity, uint gridTypes)
    {
        if (IsSolidCell(gridTypes))
            return float2.zero;
        return math.select(velocity, 0, IsSolidCell(NeighborGridTypeLB(gridTypes)));
    }
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
    private static float2 ClampPosition(float2 pos) =>
        math.clamp(pos, 0, 
            (gWidth * bWidth));

}

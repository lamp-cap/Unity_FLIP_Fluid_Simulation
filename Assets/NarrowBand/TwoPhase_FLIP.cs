using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;

namespace NarrowBand
{
    // Two-phase (water + air) variant of NarrowBand_FLIP.
    // - the whole domain is filled with fluid: left half water, right half air
    // - particlePos.z carries the phase density (water 100 / air 1)
    // - the signed distance field covers the whole domain:
    //   0,1,2 = water particle band, -1,-2,-3 = air particle band
    // - cell phase is classified from the P2G cell (mass) density
    // - pressure solve uses variable coefficients:
    //   face mobility = harmonic mean of the two mobilities, 2/(rhoA + rhoB)
    public class TwoPhase_FLIP : MonoBehaviour
    {
        private const uint SOLID = 2;
        private const uint AIR = 1;
        private const uint FLUID = 0;

        private const float InvDeltaTime = 60f;
        private const float DeltaTime = 1.0f / InvDeltaTime;
        private const float CellSize = 0.5f;
        private const float InvCellSize = 1.0f / CellSize;

        private const float TargetDensity = 5;

        private const float WaterDensity = 1000f;
        private const float AirDensity = 1f;
        // geometric mean of the two phase densities, splits mixed cells
        private const float PhaseThreshold = 31.62f; // sqrt(1000)
        // a cell holding at most this many water particles is spray, not a
        // water body (the band keeps >= 4 per real cell) — its droplets
        // free-fall instead of inheriting the air flow
        private const int SprayThreshold = 2;

        [Range(-10, 10)] public float gravity = -9;
        [Range(0, 1)] public float flipness = 0.95f;
        // 1:1000 mobility contrast needs more PCG iterations than single-phase;
        // tune live — the solver early-exits once the relative residual is low
        [Range(1, 32)] public int pcgIterations = 12;
        public Mesh mesh;
        public Material mat;     // water particles
        public Material airMat;  // air particles
        private float _rs;
        // first 600 frames of solver residuals, absolute + relative, summed
        // up in OnDestroy — the absolute value conflates problem scale (|b|
        // grows as gravity accelerates the flow), the relative residual is
        // the actual convergence quality
        private readonly float[] _absLog = new float[600];
        private readonly float[] _relLog = new float[600];
        private int _residualFrame;


        public const int NumParticles = 256 * 1024;
        public const int GridRes = 256;
        public const int NumGrid = GridRes * GridRes;
        private const int Band1 = 2;
        private const int Band2 = 3;
        // hard cap per band cell: max(4, count) never trims by itself, and a
        // shredded water-air interface turns whole regions into band cells
        private const int MaxParticlesPerCell = 8;

        private NativeArray<float2> _gridVelocity;
        private NativeArray<float2> _gridVelocityCopy;
        private NativeArray<float> _gridDensity;
        private NativeArray<float> _gridDivergence;
        private NativeArray<float> _gridPressure;
        private NativeArray<uint> _gridType;
        private NativeArray<float3> _gridLaplacian;
        private NativeArray<float> _gridSDF;
        private NativeArray<float> _gridRho;
        // water particles per cell (capped at MaxParticlesPerCell, fits a
        // byte) — any cell holding a water particle classifies water, so
        // only the count separates real water from isolated spray
        private NativeArray<byte> _gridWaterCount;
        private NativeArray<int> _start;
        private NativeArray<int> _end;

        private NativeArray<float4> _particlePos;
        private NativeArray<float2> _particleVelocity;
        private NativeArray<float4> _particlePosCopy;
        private NativeArray<float2> _particleVelocityCopy;
        private NativeArray<int> _particleID;
        private NativeArray<int2> _hashes;
        private NativeArray<int2> _range;
        // 1 = free-falling droplet: written by G2P, read by advection so
        // both agree on which particles skip the (air-dominated) grid
        private NativeArray<byte> _ballistic;

        private NativeReference<int> _particleCount;
        private NativeReference<int> _waterCount;

        private ComputeBuffer _posBuffer;
        private ComputeBuffer _airPosBuffer;
        private Bounds _bounds;
        private TwoPhaseMGSolver _mgPressureSolver;

        private float2 _oldMousePos;
        private float2 _oldMouseVec;
        private Camera _camera;

        void Awake()
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 60;
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
            _gridRho = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridWaterCount = new NativeArray<byte>(NumGrid, Allocator.Persistent);
            _particleID = new NativeArray<int>(NumParticles, Allocator.Persistent);
            _particlePos = new NativeArray<float4>(NumParticles, Allocator.Persistent);
            _particleVelocity = new NativeArray<float2>(NumParticles, Allocator.Persistent);
            _particlePosCopy = new NativeArray<float4>(NumParticles, Allocator.Persistent);
            _particleVelocityCopy = new NativeArray<float2>(NumParticles, Allocator.Persistent);
            _hashes = new NativeArray<int2>(NumParticles, Allocator.Persistent);
            _ballistic = new NativeArray<byte>(NumParticles, Allocator.Persistent);
            _particleCount = new NativeReference<int>(Allocator.Persistent);
            _waterCount = new NativeReference<int>(Allocator.Persistent);
            _mgPressureSolver = new TwoPhaseMGSolver(_gridLaplacian,_gridPressure, _gridDivergence, GridRes, CellSize);

            // left half water, right half air; every cell inside the domain is fluid
            int mid = GridRes / 2;
            for (int y = 0; y < GridRes; y++)
            for (int x = 0; x < GridRes; x++)
            {
                int i = y * GridRes + x;
                bool water = x < mid;
                bool boundary = x == 0 || x == GridRes - 1 || y == 0 || y == GridRes - 1
                             || x == mid - 1 || x == mid;
                _gridSDF[i] = water ? (boundary ? 0f : GridRes) : (boundary ? -1f : -GridRes);
                _gridRho[i] = water ? WaterDensity : AirDensity;
            }

            new SetGridTypeJob
            {
                GridType = _gridType,
            }.Schedule(NumGrid, 32).Complete();

            new ComputeDistanceFieldJob(_gridSDF).Run();
            new ParticlesCounterInitJob(_gridSDF, _gridType, _range, _particleID, _particleCount).Run();
            new ResampleParticlesJob()
            {
                PosRaw = _particlePosCopy,
                VelRaw = _particleVelocityCopy,
                PosNew = _particlePos,
                VelNew = _particleVelocity,
                Ids = _particleID,
                GridVelocity = _gridVelocityCopy,
                GridRho = _gridRho,
                Ranges = _range,
            }.Schedule(NumGrid, 32).Complete();

            _posBuffer = new ComputeBuffer(NumParticles, sizeof(float) * 4);
            mat.SetBuffer("_ParticleBuffer", _posBuffer);
            _airPosBuffer = new ComputeBuffer(NumParticles, sizeof(float) * 4);
            if (airMat != null)
                airMat.SetBuffer("_ParticleBuffer", _airPosBuffer);
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

            Debug.Log($"TwoPhase_FLIP initialized, particle num: {NumParticles}, grid res: {GridRes}x{GridRes}, " +
                      $"density ratio {AirDensity}:{WaterDensity}.");

            Test();
        }

        void Update()
        {
            Simulate();

            if (_residualFrame < _absLog.Length)
            {
                _absLog[_residualFrame] = _rs;
                _relLog[_residualFrame] = _mgPressureSolver.LastRelResidual;
                _residualFrame++;
            }

            // the compacted array is split by phase: [0, waterCount) water,
            // [waterCount, count) air — one draw per phase with its own material
            int waterCount = _waterCount.Value;
            int airCount = _particleCount.Value - waterCount;
            if (waterCount > 0)
            {
                _posBuffer.SetData(_particlePos, 0, 0, waterCount);
                Graphics.DrawMeshInstancedProcedural(mesh, 0, mat, _bounds, waterCount);
            }

            if (airMat != null && airCount > 0)
            {
                _airPosBuffer.SetData(_particlePos, waterCount, 0, airCount);
                Graphics.DrawMeshInstancedProcedural(mesh, 0, airMat, _bounds, airCount);
            }
        }

        private void OnDestroy()
        {
            AnalyzeResiduals();

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
            _gridRho.Dispose();
            _gridWaterCount.Dispose();

            _particleID.Dispose();
            _particlePos.Dispose();
            _particleVelocity.Dispose();
            _particlePosCopy.Dispose();
            _particleVelocityCopy.Dispose();
            _hashes.Dispose();
            _ballistic.Dispose();
            _posBuffer.Dispose();
            _airPosBuffer.Dispose();
            _mgPressureSolver.Dispose();

            _particleCount.Dispose();
            _waterCount.Dispose();
        }

        private static string Describe(float[] log, int n)
        {
            var sorted = new float[n];
            System.Array.Copy(log, sorted, n);
            System.Array.Sort(sorted);

            float Mean(int from, int to)
            {
                float sum = 0;
                for (int i = from; i < to; i++) sum += log[i];
                return sum / math.max(1, to - from);
            }

            int spikes = 0;
            for (int i = 0; i < n; i++)
                if (log[i] > 10f * sorted[n / 2]) spikes++;

            int third = math.max(1, n / 3);
            return $"mean {Mean(0, n):G4}, median {sorted[n / 2]:G4}, p90 {sorted[(int)(n * 0.9f)]:G4}, " +
                   $"p99 {sorted[math.min(n - 1, (int)(n * 0.99f))]:G4}, max {sorted[n - 1]:G4}\n" +
                   $"  前1/3 {Mean(0, third):G4}, 后1/3 {Mean(n - third, n):G4}, 尖峰帧 {spikes}";
        }

        // summary of the collected per-frame residuals, absolute (problem
        // scale included) and relative (true convergence quality)
        private void AnalyzeResiduals()
        {
            int n = math.min(_residualFrame, _absLog.Length);
            if (n == 0) return;

            Debug.Log($"[残差统计] frames: {n}, pcgIters: {pcgIterations}, " +
                      $"冷启动: {_mgPressureSolver.ColdStarts}\n" +
                      $"绝对残差: {Describe(_absLog, n)}\n" +
                      $"相对残差: {Describe(_relLog, n)}");
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
            GUI.Label(new Rect(0, 180, 100, 36),
                $"water: {_waterCount.Value} / air: {_particleCount.Value - _waterCount.Value}", _labelStyle);
            GUI.Label(new Rect(0, 216, 420, 36),
                _residualFrame >= _absLog.Length
                    ? $"残差收集完成 {_residualFrame}/{_absLog.Length}，退出 Play 查看 Console 统计"
                    : $"残差收集中 {_residualFrame}/{_absLog.Length}",
                _labelStyle);
        }

        private void Simulate()
        {
            int batchCount = 32;

            Profiler.BeginSample("Clear Grid");
            // Pressure is deliberately NOT cleared anymore: Solve_MGPCG
            // warm-starts from the previous frame's solution
            new ClearGridJob()
            {
                Start = _start,
                End = _end,
                Range = _range,
                Density = _gridDensity,
            }.Schedule(NumGrid, batchCount).Complete();
            Profiler.EndSample();

            Profiler.BeginSample("Build Lut");
            // counting sort, O(n + grid): rank particles inside their cell with
            // atomic adds, prefix-sum the cell counts, then scatter — replaces
            // the serial O(n log n) comparison sort
            new HashCountJob
            {
                Ps = _particlePos,
                Hashes = _hashes,
                Counts = _end,
            }.Schedule(_particleCount.Value, batchCount).Complete();

            new PrefixSumJob
            {
                Counts = _end,
                StartIndices = _start,
                Range = _range,
            }.Run();

            new ScatterJob
            {
                Hashes = _hashes,
                StartIndices = _start,
                PosRaw = _particlePos,
                VelRaw = _particleVelocity,
                PosNew = _particlePosCopy,
                VelNew = _particleVelocityCopy
            }.Schedule(_particleCount.Value, batchCount).Complete();

            (_particlePos, _particlePosCopy) = (_particlePosCopy, _particlePos);
            (_particleVelocity, _particleVelocityCopy) = (_particleVelocityCopy, _particleVelocity);

            // classify each cell's phase from its particles (mass density),
            // cells without particles inherit the previous distance field's sign
            new ClassifyGridPhaseJob
            {
                Range = _range,
                ParticlePos = _particlePos,
                PrevLevel = _gridSDF,
                GridRho = _gridRho,
                GridWaterCount = _gridWaterCount,
            }.Schedule(NumGrid, batchCount).Complete();

            Profiler.EndSample();

            Profiler.BeginSample("Resample");

            new SetGridTypeJob
            {
                GridType = _gridType,
            }.Schedule(NumGrid, batchCount).Complete();

            new SetGridLevelJob(_gridSDF, _gridType, _gridRho).Schedule(NumGrid, batchCount).Complete();

            new ComputeDistanceFieldJob(_gridSDF).Run();

            new ParticlesCounterJob(_gridSDF, _gridType, _range, _particleID, _particleCount, _waterCount).Run();

            new ResampleParticlesJob()
            {
                PosRaw = _particlePos,
                VelRaw = _particleVelocity,
                PosNew = _particlePosCopy,
                VelNew = _particleVelocityCopy,
                Ids = _particleID,
                GridVelocity = _gridVelocityCopy,
                GridRho = _gridRho,
                Ranges = _range,
            }.Schedule(NumGrid, batchCount).Complete();

            (_particlePos, _particlePosCopy) = (_particlePosCopy, _particlePos);
            (_particleVelocity, _particleVelocityCopy) = (_particleVelocityCopy, _particleVelocity);

            Profiler.EndSample();

            Profiler.BeginSample("P2G");

            new ComputeLaplacianJob
            {
                GridTypes = _gridType,
                GridRho = _gridRho,
                GridLaplacian = _gridLaplacian,
            }.Schedule(NumGrid, batchCount).Complete();

            new ParticleToGridJob
            {
                ParticlePos = _particlePos,
                ParticleVel = _particleVelocity,
                Range = _range,
                GridVelocity = _gridVelocityCopy,
                GridDensity = _gridDensity,
                GridLevel = _gridSDF,
                GridRho = _gridRho,
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
            new CalcDivergenceJob
            {
                GridVelocity = _gridVelocity,
                GridTypes = _gridType,
                GridDensity = _gridDensity,
                GridDivergence = _gridDivergence,
            }.Schedule(_gridVelocity.Length, batchCount).Complete();

            _mgPressureSolver.Solve_MGPCG(pcgIterations, out _rs);

            new UpdateVelocity
            {
                GridTypes = _gridType,
                GridPressure = _gridPressure,
                GridRho = _gridRho,
                GridVelocity = _gridVelocity,
            }.Schedule(NumGrid, batchCount).Complete();
            Profiler.EndSample();

            Profiler.BeginSample("G2P");
            new GridToParticleJob
            {
                GridVelocityOld = _gridVelocityCopy,
                GridVelocityNew = _gridVelocity,
                ParticleVel = _particleVelocity,
                ParticlePos = _particlePos,
                GridWaterCount = _gridWaterCount,
                Ballistic = _ballistic,
                Gravity = new float2(0, gravity),
                Flipness = flipness,
            }.Schedule(_particleCount.Value, batchCount).Complete();

            Profiler.EndSample();

            Profiler.BeginSample("Advection");
            new ParticlesAdvectionJob
            {
                GridVelocity = _gridVelocity,
                Ballistic = _ballistic,
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
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.white;
            Gizmos.DrawWireCube(_bounds.center, _bounds.size);
            if (!Application.isPlaying) return;
            for (int y = 0; y < GridRes; y++)
            for (int x = 0; x < GridRes; x++)
            {
                int idx = Coord2Idx(x, y);
                if (!IsFluidCell(_gridType[idx])) continue;
                float v = _gridPressure[idx] * 0.15f + 0.1f;
                bool water = _gridRho[idx] > PhaseThreshold;
                Gizmos.color = water
                    ? new Color(v * 0.4f, v * 0.6f, v, 0.5f)
                    : new Color(v, v, v * 0.3f, 0.5f);
                Gizmos.DrawCube(new Vector3((x + 0.5f) * CellSize * 0.1f, (y + 0.5f) * CellSize*0.1f, -0.1f), new Vector3(CellSize*0.1f, CellSize*0.1f, 0));
            }
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
        // harmonic mean of the two phase mobilities (1/rho):
        // same-phase faces give 1/rho, the water-air interface gives 2/(rhoW + rhoA)
        private static float FaceMobility(float rhoA, float rhoB)
        {
            return 2f / (rhoA + rhoB);
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
            [WriteOnly] public NativeArray<float> Density;

            public void Execute(int i)
            {
                Start[i] = 0;
                End[i] = 0;
                Range[i] = int2.zero;
                Density[i] = 0;
            }
        }

        // counting sort pass 1: hash each particle into its cell and claim a
        // rank in that cell's bucket with an atomic add on the cell counter
        [BurstCompile]
        private unsafe struct HashCountJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float4> Ps;
            [WriteOnly] public NativeArray<int2> Hashes;
            [NativeDisableParallelForRestriction] public NativeArray<int> Counts;

            public void Execute(int i)
            {
                int cell = Coord2Idx(math.clamp(GetCoord(Ps[i].xy), 0, GridRes - 1));
                int rank = System.Threading.Interlocked.Add(
                    ref UnsafeUtility.ArrayElementAsRef<int>(Counts.GetUnsafePtr(), cell), 1) - 1;
                Hashes[i] = new int2(cell, rank);
            }
        }

        // counting sort pass 2 (serial): exclusive prefix sum over the cell
        // counts turns each (cell, rank) into its final sorted slot
        [BurstCompile]
        private struct PrefixSumJob : IJob
        {
            [ReadOnly] public NativeArray<int> Counts;
            [WriteOnly] public NativeArray<int> StartIndices;
            [WriteOnly] public NativeArray<int2> Range;

            public void Execute()
            {
                int sum = 0;
                for (int i = 0; i < Counts.Length; i++)
                {
                    int count = Counts[i];
                    StartIndices[i] = sum;
                    Range[i] = new int2(sum, sum + count);
                    sum += count;
                }
            }
        }

        // counting sort pass 3: move the particle data straight into its slot
        [BurstCompile]
        private struct ScatterJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> Hashes;
            [ReadOnly] public NativeArray<int> StartIndices;
            [ReadOnly] public NativeArray<float4> PosRaw;
            [ReadOnly] public NativeArray<float2> VelRaw;

            [NativeDisableContainerSafetyRestriction, WriteOnly] public NativeArray<float4> PosNew;
            [NativeDisableContainerSafetyRestriction, WriteOnly] public NativeArray<float2> VelNew;

            public void Execute(int i)
            {
                int2 hash = Hashes[i];
                int dst = StartIndices[hash.x] + hash.y;
                PosNew[dst] = PosRaw[i];
                VelNew[dst] = VelRaw[i];
            }
        }

        // classifies every cell as water or air from the mass density of the
        // particles it currently contains; empty cells keep the sign of the
        // previous distance field
        [BurstCompile]
        private struct ClassifyGridPhaseJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int2> Range;
            [ReadOnly] public NativeArray<float4> ParticlePos;
            [ReadOnly] public NativeArray<float> PrevLevel;
            [WriteOnly] public NativeArray<float> GridRho;
            [WriteOnly] public NativeArray<byte> GridWaterCount;

            public void Execute(int i)
            {
                int2 range = Range[i];
                float mass = 0;
                int waterCount = 0;
                for (int j = range.x; j < range.y; j++)
                {
                    float rho = ParticlePos[j].z;
                    mass += rho;
                    if (rho > PhaseThreshold) waterCount++;
                }

                int count = range.y - range.x;
                GridWaterCount[i] = (byte)waterCount;
                GridRho[i] = count > 0
                    ? (mass > PhaseThreshold * count ? WaterDensity : AirDensity)
                    : (PrevLevel[i] >= 0 ? WaterDensity : AirDensity);
            }
        }

        [BurstCompile]
        private struct SetGridLevelJob : IJobParallelFor
        {
            [ReadOnly] private NativeArray<uint> _gridTypes;
            [ReadOnly] private NativeArray<float> _gridRho;
            [WriteOnly] private NativeArray<float> _gridLevel;

            public SetGridLevelJob(NativeArray<float> level, NativeArray<uint> gridTypes, NativeArray<float> gridRho)
            {
                _gridLevel = level;
                _gridTypes = gridTypes;
                _gridRho = gridRho;
            }

            public void Execute(int i)
            {
                uint gridType = _gridTypes[i];
                if (!IsFluidCell(gridType))
                {
                    // solid cells sit on the interface, they seed distance 0
                    _gridLevel[i] = 0;
                    return;
                }

                int2 coord = Idx2Coord(i);
                float rho = _gridRho[i];
                bool water = rho > PhaseThreshold;
                bool boundary = IsInterface(coord - new int2(1, 0), rho)
                             || IsInterface(coord + new int2(1, 0), rho)
                             || IsInterface(coord - new int2(0, 1), rho)
                             || IsInterface(coord + new int2(0, 1), rho);

                // the interface line belongs to the water side: water -> 0, air -> -1
                _gridLevel[i] = water ? (boundary ? 0f : GridRes) : (boundary ? -1f : -GridRes);
            }

            private bool IsInterface(int2 coord, float rho)
            {
                if (math.any(coord < 0) || math.any(coord > GridRes - 1)) return true; // solid wall
                uint gridType = _gridTypes[Coord2Idx(coord)];
                if (IsSolidCell(gridType)) return true;
                return _gridRho[Coord2Idx(coord)] != rho; // opposite phase
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
                    if (level == 0) continue;
                    int2 coord = Idx2Coord(i);
                    if (level > 0)
                    {
                        if (coord.x > 0)
                            level = math.min(level, 1 + math.max(_gridLevel[Coord2Idx(coord - offset.xy)], 0));
                        if (coord.y > 0)
                            level = math.min(level, 1 + math.max(_gridLevel[Coord2Idx(coord - offset.yx)], 0));
                    }
                    else
                    {
                        if (coord.x > 0)
                            level = math.max(level, -1 + math.min(_gridLevel[Coord2Idx(coord - offset.xy)], 0));
                        if (coord.y > 0)
                            level = math.max(level, -1 + math.min(_gridLevel[Coord2Idx(coord - offset.yx)], 0));
                    }

                    _gridLevel[i] = level;
                }

                for (int i = NumGrid - 1; i >= 0; i--)
                {
                    float level = _gridLevel[i];
                    if (level == 0) continue;
                    int2 coord = Idx2Coord(i);
                    if (level > 0)
                    {
                        if (coord.x < rightBound)
                            level = math.min(level, 1 + math.max(_gridLevel[Coord2Idx(coord + offset.xy)], 0));
                        if (coord.y < rightBound)
                            level = math.min(level, 1 + math.max(_gridLevel[Coord2Idx(coord + offset.yx)], 0));
                    }
                    else
                    {
                        if (coord.x < rightBound)
                            level = math.max(level, -1 + math.min(_gridLevel[Coord2Idx(coord + offset.xy)], 0));
                        if (coord.y < rightBound)
                            level = math.max(level, -1 + math.min(_gridLevel[Coord2Idx(coord + offset.yx)], 0));
                    }

                    _gridLevel[i] = level;
                }
            }
        }

        [BurstCompile]
        private struct ParticlesCounterInitJob : IJob
        {
            [ReadOnly] private NativeArray<float> _gridLevel;
            [ReadOnly] private NativeArray<uint> _gridTypes;
            private NativeArray<int2> _range;
            [WriteOnly] private NativeArray<int> _particleIDs;
            [WriteOnly] private NativeReference<int> _pCount;

            public ParticlesCounterInitJob(NativeArray<float> level, NativeArray<uint> gridTypes,
                NativeArray<int2> range, NativeArray<int> particleIDs, NativeReference<int> pCount)
            {
                _gridLevel = level;
                _gridTypes = gridTypes;
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
                    if (!IsFluidCell(_gridTypes[i]) || level < -Band2 || level >= Band2) continue;
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
            [ReadOnly] private NativeArray<uint> _gridTypes;
            private NativeArray<int2> _range;
            [WriteOnly] private NativeArray<int> _particleIDs;
            [WriteOnly] private NativeReference<int> _pCount;
            [WriteOnly] private NativeReference<int> _waterCount;

            public ParticlesCounterJob(NativeArray<float> level, NativeArray<uint> gridTypes,
                NativeArray<int2> range, NativeArray<int> particleIDs,
                NativeReference<int> pCount, NativeReference<int> waterCount)
            {
                _gridLevel = level;
                _gridTypes = gridTypes;
                _range = range;
                _particleIDs = particleIDs;
                _pCount = pCount;
                _waterCount = waterCount;
            }

            public void Execute()
            {
                // water band cells first, then air band cells, so the compacted
                // array splits into [0, waterCount) and [waterCount, count)
                int ptr = 0;
                for (int i = 0; i < NumGrid; i++)
                {
                    if (IsWaterBand(i)) ptr = AssignSlots(i, ptr);
                    else if (!IsAirBand(i)) _range[i] = int2.zero;
                }
                _waterCount.Value = ptr;

                for (int i = 0; i < NumGrid; i++)
                {
                    if (IsAirBand(i)) ptr = AssignSlots(i, ptr);
                }
                _pCount.Value = ptr;
            }

            private bool IsWaterBand(int i)
            {
                float level = _gridLevel[i];
                return IsFluidCell(_gridTypes[i]) && level >= 0 && level < Band2;
            }

            private bool IsAirBand(int i)
            {
                float level = _gridLevel[i];
                return IsFluidCell(_gridTypes[i]) && level < 0 && level >= -Band2;
            }

            private int AssignSlots(int i, int ptr)
            {
                var range = _range[i];
                // an inverted range (End unwritten) must never leak a negative
                // count into expect, or ptr walks backwards below zero
                int count = math.max(0, range.y - range.x);
                // level 0 (water side of the interface) keeps its particle count,
                // every other band cell maintains at least 4 particles;
                // cap per cell and clamp to the pool so the count cannot run away
                int expect = _gridLevel[i] == 0 ? count : math.max(4, count);
                expect = math.clamp(expect, 0, MaxParticlesPerCell);
                expect = math.min(expect, NumParticles - ptr);
                int keep = math.min(count, expect);
                for (int j = 0; j < expect; j++)
                {
                    _particleIDs[ptr + j] = j < keep ? range.x + j : - 1; // if need add particle, set -1
                }
                _range[i] = new int2(ptr, ptr + expect);
                return ptr + expect;
            }
        }

        [BurstCompile]
        private struct ResampleParticlesJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<int> Ids;
            [ReadOnly] public NativeArray<int2> Ranges;
            [ReadOnly] public NativeArray<float2> GridVelocity;
            [ReadOnly] public NativeArray<float> GridRho;

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
                // a cell can hold more than 4 particles, size the scratch arrays accordingly
                int capacity = math.max(4, range.y - range.x);
                var posArr = new NativeArray<float2>(capacity, Allocator.Temp);
                var velArr = new NativeArray<float2>(capacity, Allocator.Temp);
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
                    // newly spawned particles take the phase density of their cell
                    PosNew[p] = new float4(selectedPos, GridRho[gid], math.length(v));
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

        // with both phases simulated the whole domain interior is fluid,
        // only the domain boundary is solid
        [BurstCompile]
        private struct SetGridTypeJob :IJobParallelFor
        {
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
                return FLUID;
            }
        }

        [BurstCompile]
        private struct ParticleToGridJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float4> ParticlePos;
            [ReadOnly] public NativeArray<float2> ParticleVel;
            [ReadOnly] public NativeArray<int2> Range;
            [ReadOnly] public NativeArray<float> GridLevel;
            [ReadOnly] public NativeArray<float> GridRho;
            public NativeArray<float2> GridVelocity;
            [WriteOnly] public NativeArray<float> GridDensity;

            public void Execute(int i)
            {
                int2 coord = Idx2Coord(i);
                float2 cellCenter = ((float2)coord + 0.5f) * CellSize;
                float2 sdf = ReadGridFacesBilinear(cellCenter, GridLevel);
                // deep interior of either phase: keep the advected velocity,
                // density is a full cell of TargetDensity particles
                if (math.all(sdf > Band1) || math.all(sdf < -Band1))
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
                        // count density, phase-blind: mass weighting would let
                        // neighbouring water particles blow up air cells at 100:1
                        density += GetPoly6Weight(dist * InvCellSize);
                    }
                }

                velocity = math.select(GridVelocity[i], velocity / sum, sum > 1e-4f);
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

                    // disabled per the paper: the pressure path should only use
                    // the set phase densities (GridRho = 100/1 from classification),
                    // never a density projected from the particles. P2G still
                    // computes GridDensity (count form) in case we re-enable this
                    // float deltaDensity = GridDensity[i] - TargetDensity;
                    // deltaDensity = math.max(-0.1f, deltaDensity);
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
            [ReadOnly] public NativeArray<float> GridRho;
            public NativeArray<float2> GridVelocity;

            public void Execute(int i)
            {
                int2 cellIdx = Idx2Coord(i);

                float2 velocity = GridVelocity[i];
                uint grid_types = GridTypes[i];
                float pressure = GridPressure[i];
                float rho = GridRho[i];

                uint2 lbType = NeighborGridTypeLB(grid_types);
                int c_id_xp = IsSolidCell(lbType.x) ? i : Coord2Idx(cellIdx + new int2(-1, 0));
                int c_id_yp = IsSolidCell(lbType.y) ? i : Coord2Idx(cellIdx + new int2(0, -1));

                // same face mobilities as the pressure matrix, so the
                // projected field is divergence free for the mixed phases
                float kL = IsSolidCell(lbType.x) ? 0 : FaceMobility(rho, GridRho[c_id_xp]);
                float kB = IsSolidCell(lbType.y) ? 0 : FaceMobility(rho, GridRho[c_id_yp]);

                velocity.x -= kL * InvCellSize * (pressure - GridPressure[c_id_xp]);
                velocity.y -= kB * InvCellSize * (pressure - GridPressure[c_id_yp]);

                GridVelocity[i] = EnforceBoundaryCondition(velocity, grid_types);
            }
        }

        [BurstCompile]
        private struct GridToParticleJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float4> ParticlePos;
            [ReadOnly] public NativeArray<float2> GridVelocityOld;
            [ReadOnly] public NativeArray<float2> GridVelocityNew;
            [ReadOnly] public NativeArray<byte> GridWaterCount;
            // 1 = this particle free-falls (set here, consumed by advection)
            public NativeArray<byte> Ballistic;
            public float2 Gravity;
            public NativeArray<float2> ParticleVel;
            public float Flipness;

            public void Execute(int i)
            {
                float2 pos = ParticlePos[i].xy;
                float2 vel = ParticleVel[i];
                int2 coord = GetCoord(pos);

                // spray test: a cell holding ANY water particle classifies as
                // water (one 1000-mass particle outweighs the threshold), so
                // phase alone can never spot an isolated droplet — the water
                // COUNT can. No cell in the sample window holds a real water
                // body (all <= SprayThreshold) => free fall instead of
                // inheriting the air flow
                bool freeFall = ParticlePos[i].z > PhaseThreshold;
                if (freeFall)
                {
                    int maxWater = 0;
                    for (int x = coord.x - 1; x <= coord.x + 2; ++x)
                    for (int y = coord.y - 1; y <= coord.y + 2; ++y)
                    {
                        int idx = Coord2Idx(math.clamp(new int2(x, y), 0, GridRes - 1));
                        maxWater = math.max(maxWater, GridWaterCount[idx]);
                    }
                    freeFall = maxWater <= SprayThreshold;
                }

                Ballistic[i] = freeFall ? (byte)1 : (byte)0;
                if (freeFall)
                {
                    ParticleVel[i] = vel + Gravity * DeltaTime;
                    return;
                }

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
            [ReadOnly] public NativeArray<byte> Ballistic;
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
                
                if (Ballistic[i] == 1)
                    velocity = math.lerp(ParticleVel[i], velocity, 0.01f);
                
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

        // variable-coefficient laplacian: face mobility = 2/(rhoA + rhoB);
        // a solid face is a zero-gradient (Neumann) wall — the mirrored
        // pressure cancels the face term, so it adds NOTHING to the center
        [BurstCompile]
        private struct ComputeLaplacianJob: IJobParallelFor
        {
            [ReadOnly] public NativeArray<uint> GridTypes;
            [ReadOnly] public NativeArray<float> GridRho;
            [WriteOnly] public NativeArray<float3> GridLaplacian;

            public void Execute(int index)
            {
                uint gridType = GridTypes[index];
                uint2 xAxisType = NeighborGridTypeAxis(0, gridType);
                uint2 yAxisType = NeighborGridTypeAxis(1, gridType);
                int2 coord = Idx2Coord(index);
                float rho = GridRho[index];

                // counting a solid face's mobility in the center pins the wall
                // pressure to zero (Dirichlet); the projection then drains
                // volume out through the walls and the sealed box leaks
                float kL = IsSolidCell(xAxisType.x) ? 0 : FaceMobility(rho, GridRho[Coord2Idx(coord - new int2(1, 0))]);
                float kR = IsSolidCell(xAxisType.y) ? 0 : FaceMobility(rho, GridRho[Coord2Idx(coord + new int2(1, 0))]);
                float kB = IsSolidCell(yAxisType.x) ? 0 : FaceMobility(rho, GridRho[Coord2Idx(coord - new int2(0, 1))]);
                float kT = IsSolidCell(yAxisType.y) ? 0 : FaceMobility(rho, GridRho[Coord2Idx(coord + new int2(0, 1))]);

                float3 a = float3.zero;
                if (IsFluidCell(gridType))
                {
                    a = new float3(kL + kR + kB + kT,
                        IsFluidCell(xAxisType.x) ? -kL : 0,
                        IsFluidCell(yAxisType.x) ? -kB : 0);
                }

                GridLaplacian[index] = a;
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

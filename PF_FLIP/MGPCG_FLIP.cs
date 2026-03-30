using System.Collections.Generic;
using Sirenix.OdinInspector;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;

namespace PF_FLIP
{
    public class MGPCG_FLIP : MonoBehaviour
    {
        private const uint SOLID = 2;
        private const uint AIR = 1;
        private const uint FLUID = 0;
    
        private const float InvDeltaTime = 60f;
        private const float DeltaTime = 1.0f / InvDeltaTime;
        private const float CellSize = 0.5f;
        private const float InvCellSize = 1.0f / CellSize;

        private const float TargetDensity = 5;

        [Range(-10, 10)] public float gravity = -9;
        [Range(0.5f, 1)] public float flipness = 0.95f;
        public Mesh mesh;
        public Material mat;
        private float _rs;
        

        public const int NumParticles = 256 * 256;
        public const int GridRes = 256;
        public const int NumGrid = GridRes * GridRes;

        private NativeArray<float2> _gridVelocity;
        private NativeArray<float2> _gridVelocityCopy;
        private NativeArray<float> _gridDensity;
        private NativeArray<float> _gridWeight;
        private NativeArray<float> _gridDivergence;
        private NativeArray<float> _gridPressure;
        private NativeArray<float> _gridPressureCopy;
        private NativeArray<float> _gridDPressure;
        private NativeArray<float> _gridDPressureCopy;
        private NativeArray<uint> _gridType;
        private NativeArray<float3> _gridLaplacian;
        private NativeArray<float2> _gridDeltaPos;
        private NativeArray<int> _start;
        private NativeArray<int> _end;
        
        private NativeArray<float4> _particlePos;
        private NativeArray<float2> _particleVelocity;
        private NativeArray<float4> _particlePosCopy;
        private NativeArray<float2> _particleVelocityCopy;
        private NativeArray<float2x2> _particleAffine;
        private NativeArray<int2> _hashes;
        private NativeArray<int2> _range;

        private ComputeBuffer _posBuffer;
        private Bounds _bounds;
        private Neumann_UAAMGSolver _mgPressureSolver;

        private float2 _oldMousePos;
        private float2 _oldMouseVec;
        private Camera _camera;

        void Start()
        {
            _camera = Camera.main;
            _gridVelocity = new NativeArray<float2>(NumGrid, Allocator.Persistent);
            _gridVelocityCopy = new NativeArray<float2>(NumGrid, Allocator.Persistent);
            _gridDivergence = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridPressure = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridPressureCopy = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridDPressure = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridDPressureCopy = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridType = new NativeArray<uint>(NumGrid, Allocator.Persistent);
            _start = new NativeArray<int>(NumGrid, Allocator.Persistent);
            _end = new NativeArray<int>(NumGrid, Allocator.Persistent);
            _range = new NativeArray<int2>(NumGrid, Allocator.Persistent);
            _gridDensity = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridWeight = new NativeArray<float>(NumGrid, Allocator.Persistent);
            _gridDeltaPos = new NativeArray<float2>(NumGrid, Allocator.Persistent);
            _gridLaplacian = new NativeArray<float3>(NumGrid, Allocator.Persistent);
        
            _particlePos = new NativeArray<float4>(NumParticles, Allocator.Persistent);
            _particleVelocity = new NativeArray<float2>(NumParticles, Allocator.Persistent);
            _particlePosCopy = new NativeArray<float4>(NumParticles, Allocator.Persistent);
            _particleVelocityCopy = new NativeArray<float2>(NumParticles, Allocator.Persistent);
            _particleAffine = new NativeArray<float2x2>(NumParticles, Allocator.Persistent);
            _hashes = new NativeArray<int2>(NumParticles, Allocator.Persistent);
            
            _mgPressureSolver = new Neumann_UAAMGSolver(_gridLaplacian,_gridPressure, _gridDivergence, GridRes, CellSize);

            for (int y = 0; y < GridRes; y++)
            for (int x = 0; x < GridRes; x++)
            {
                int id = y * GridRes + x;
                float2 pos = new float2(x, y) * 0.5f + new float2(120, 1);
                _particlePos[id] = new float4(pos * CellSize, 0, 1);
            }

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
        }

        void Update()
        {
            Simulate();
            _posBuffer.SetData(_particlePos);
            Graphics.DrawMeshInstancedProcedural(mesh, 0, mat, _bounds, NumParticles);
        }

        private void OnDestroy()
        {
            var shortCut = ScriptableObject.CreateInstance<GridShortCut>();
            shortCut.laplacian = _gridLaplacian.ToArray();
            shortCut.divergence = _gridDivergence.ToArray();
            UnityEditor.AssetDatabase.CreateAsset(shortCut,
                $"Assets/PF_FLIP/ShortCut_{GridRes}_{NumParticles}.asset");
            UnityEditor.AssetDatabase.Refresh();
            
            _gridVelocity.Dispose();
            _gridVelocityCopy.Dispose();
            _gridDivergence.Dispose();
            _gridPressure.Dispose();
            _gridPressureCopy.Dispose();
            _gridDPressure.Dispose();
            _gridDPressureCopy.Dispose();
            _gridType.Dispose();
            _start.Dispose();
            _end.Dispose();
            _range.Dispose();
            _gridDensity.Dispose();
            _gridWeight.Dispose();
            _gridDeltaPos.Dispose();
            _gridLaplacian.Dispose();
            
            _particlePos.Dispose();
            _particleVelocity.Dispose();
            _particlePosCopy.Dispose();
            _particleVelocityCopy.Dispose();
            _particleAffine.Dispose();
            _hashes.Dispose();
            _posBuffer.Dispose();
            _mgPressureSolver.Dispose();
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
        }

        private void Simulate()
        {
            int batchCount = 64;
            Profiler.BeginSample("Clear Grid");
            new ClearGridJob()
            {
                Start = _start,
                End = _end,
                Range = _range,
                Pressure = _gridPressure,
                Density = _gridDensity,
            }.Schedule(_gridVelocity.Length, batchCount).Complete();
            Profiler.EndSample();
            
            Profiler.BeginSample("Build Lut");
            new HashJob
            {
                Ps = _particlePos,
                Hashes = _hashes,
            }.Schedule(_particlePos.Length, batchCount).Complete();

            Profiler.BeginSample("Sort");
            _hashes.SortJob(new Int2Comparer()).Schedule().Complete();
            Profiler.EndSample();
        
            new BuildLutJob
            {
                Hashes = _hashes,
                StartIndices = _start,
                EndIndices = _end
            }.Schedule(_hashes.Length, batchCount).Complete();
        
            new CombineLutJob
            {
                StartIndices = _start,
                EndIndices = _end,
                Range = _range,
            }.Schedule(_range.Length, batchCount).Complete();
            
            new ShuffleJob
            {
                Hashes = _hashes,
                PosRaw = _particlePos,
                PosNew = _particlePosCopy,
                VelRaw = _particleVelocity,
                VelNew = _particleVelocityCopy,
            }.Schedule(_particlePos.Length, batchCount).Complete();

            (_particlePos, _particlePosCopy) = (_particlePosCopy, _particlePos);
            (_particleVelocity, _particleVelocityCopy) = (_particleVelocityCopy, _particleVelocity);
            Profiler.EndSample();
        
            Profiler.BeginSample("P2G");
            new SetGridTypeJob
            {
                Range = _range,
                GridType = _gridType,
            }.Schedule(_range.Length, batchCount).Complete();

            new ComputeLaplacianJob
            {
                GridTypes = _gridType,
                GridLaplacian = _gridLaplacian,
            }.Schedule(_gridType.Length, batchCount).Complete();
        
            new ParticleToGridJob
            {
                ParticlePos = _particlePos,
                ParticleVel = _particleVelocity,
                ParticleAffine = _particleAffine,
                Range = _range,
                GridVelocity = _gridVelocityCopy,
                GridDensity = _gridDensity,
            }.Schedule(_gridVelocity.Length, batchCount).Complete();
            
            _gridVelocity.CopyFrom(_gridVelocityCopy);

            var mouseVec = float2.zero;
            if (_camera != null)
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
            
            // _mgPressureSolver.Solve_Jacobi(128, out _rs);
            // _mgPressureSolver.Solve_MG(6, out _rs);
            _mgPressureSolver.Solve_MGPCG(3, out _rs);
            // _mgPressureSolver.Solve_Jacobi(8, out _rs);
            // _mgPressureSolver.Solve_CG(8, out _, out _rs);
            
            new UpdateVelocity
            {
                GridTypes = _gridType,
                GridPressure = _gridPressure,
                GridVelocity = _gridVelocity,
            }.Schedule(_gridVelocity.Length, batchCount).Complete();
            Profiler.EndSample();
        
            Profiler.BeginSample("G2P");
            new GridToParticleJob
            {
                GridVelocityOld = _gridVelocityCopy,
                GridVelocityNew = _gridVelocity,
                ParticleAffine = _particleAffine,
                ParticleVel = _particleVelocity,
                ParticlePos = _particlePos,
                Flipness = flipness,
            }.Schedule(_particlePos.Length, batchCount).Complete();
        
            new AdvectionJob
            {
                GridVelocity = _gridVelocity,
                ParticlePos = _particlePos,
                ParticleVel = _particleVelocity
            }.Schedule(_particlePos.Length, batchCount).Complete();
            Profiler.EndSample();
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.white;
            Gizmos.DrawWireCube(_bounds.center, _bounds.size);
            // Gizmos.DrawLine(new Vector3(-3.2f, -3.1f, 0), new Vector3(3.1f, -3.1f, 0));
            // Gizmos.DrawLine(new Vector3(3.0f, -3.2f, 0), new Vector3(3.0f, 3.1f, 0));
        }

        #region Utils

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
        private static uint2 NeighborGridTypeAxis(int axis, uint gridTypes)
        {
            return new uint2((gridTypes >> (axis * 4 + 2)) & 3u,
                (gridTypes >> (axis * 4 + 4)) & 3u);
        }
        private static float2 EnforceBoundaryCondition(float2 velocity, uint gridTypes)
        {
            if (IsSolidCell(gridTypes))
                return float2.zero;
            return math.select(velocity, math.max(velocity, -velocity), IsSolidCell(NeighborGridTypeLB(gridTypes)));
        }
        
        private static float2 ClampPosition(float2 pos)
        {
            return math.clamp(pos, 0.05f*CellSize, (GridRes - 0.5f) * CellSize);
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
            [Unity.Collections.ReadOnly] public NativeArray<float4> Ps;
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
            [Unity.Collections.ReadOnly] public NativeArray<int2> Hashes;
            [NativeDisableParallelForRestriction] public NativeArray<int> StartIndices;
            [NativeDisableParallelForRestriction] public NativeArray<int> EndIndices;

            public void Execute(int i)
            {
                int prev = i == 0 ? NumParticles - 1 : i - 1;
                int next = i == NumParticles - 1 ? 0 : i + 1;
                int currID = Hashes[i].x;
                int prevID = Hashes[prev].x;
                int nextID = Hashes[next].x;
                if (currID != prevID) StartIndices[currID] = i;
                if (currID != nextID) EndIndices[currID] = i + 1;
            }
        }
    
        [BurstCompile]
        private struct CombineLutJob : IJobParallelFor
        {
            [Unity.Collections.ReadOnly] public NativeArray<int> StartIndices;
            [Unity.Collections.ReadOnly] public NativeArray<int> EndIndices;
            [WriteOnly] public NativeArray<int2> Range;

            public void Execute(int i)
            {
                Range[i] = new int2(StartIndices[i], EndIndices[i]);
            }
        }
        
        [BurstCompile]
        private struct ShuffleJob : IJobParallelFor
        {
            [Unity.Collections.ReadOnly] public NativeArray<int2> Hashes;
            [Unity.Collections.ReadOnly] public NativeArray<float4> PosRaw;
            [Unity.Collections.ReadOnly] public NativeArray<float2> VelRaw;
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
        private struct SetGridTypeJob :IJobParallelFor
        {
            [Unity.Collections.ReadOnly] public NativeArray<int2> Range;
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
                int2 range = Range[Coord2Idx(coord)];
                return range.y > range.x ? FLUID : AIR;
            }
        }

        [BurstCompile]
        private struct ParticleToGridJob : IJobParallelFor
        {
            [Unity.Collections.ReadOnly] public NativeArray<float4> ParticlePos;
            [Unity.Collections.ReadOnly] public NativeArray<float2> ParticleVel;
            [Unity.Collections.ReadOnly] public NativeArray<float2x2> ParticleAffine;
            [Unity.Collections.ReadOnly] public NativeArray<int2> Range;
            [WriteOnly] public NativeArray<float2> GridVelocity;
            [WriteOnly] public NativeArray<float> GridDensity;

            public void Execute(int i)
            {
                int2 coord = Idx2Coord(i);
                
                float2 cellCenter = ((float2)coord + 0.5f) * CellSize;

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
                        var n_c = ParticleAffine[j];
                        var n_v = ParticleVel[j];
                        
                        float2 weight = new float2(
                            GetWeight(position_vx - n_x, InvCellSize),
                            GetWeight(position_vy - n_x, InvCellSize));
                        
                        sum += weight;
                        
                        velocity.x += weight.x * (n_v.x + math.mul(n_c, position_vx - n_x).x);
                        velocity.y += weight.y * (n_v.y + math.mul(n_c, position_vy - n_x).y);
                        
                        float2 dist = n_x - cellCenter;
                        density += GetPoly6Weight(dist * InvCellSize);
                    }
                }

                velocity = math.select(float2.zero, velocity / sum, sum > 1e-4f);
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
        }
    
        [BurstCompile]
        private struct AddForceJob : IJobParallelFor
        {
            [Unity.Collections.ReadOnly] public NativeArray<uint> GridTypes;
            public NativeArray<float2> GridVelocity;
            public float2 Gravity;
            public float2 MousePos;
            public float2 MouseVec;

            public void Execute(int i)
            {
                float2 velocity = GridVelocity[i];
                uint gridType = GridTypes[i];
                velocity += Gravity * DeltaTime;
                float2 coord = (float2)Idx2Coord(i) + 0.5f;
                float dst = math.distance(coord, MousePos);
                if (dst < 8)
                    velocity += MouseVec * (8 - dst) * 10 * DeltaTime;

                GridVelocity[i] = EnforceBoundaryCondition(velocity, gridType);
            }
        }
    
        [BurstCompile]
        private struct CalcDivergenceJob : IJobParallelFor
        {
            [Unity.Collections.ReadOnly] public NativeArray<float2> GridVelocity;
            [Unity.Collections.ReadOnly] public NativeArray<uint> GridTypes;
            [Unity.Collections.ReadOnly] public NativeArray<float> GridDensity;
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

                    float deltaDensity = math.max(0, GridDensity[i] - TargetDensity);
                    // deltaDensity = (math.any(cellIdx <= 2) || math.any(cellIdx >= 61)) 
                    //     ? math.max(0, deltaDensity)
                    //     : math.max(-0.1f, deltaDensity);
                    divergence -= 0.01f * deltaDensity;
                }

                GridDivergence[i] = -divergence;
            }
        }
        
        [BurstCompile]
        private struct UpdateVelocity : IJobParallelFor
        {
            [Unity.Collections.ReadOnly] public NativeArray<uint> GridTypes;
            [Unity.Collections.ReadOnly] public NativeArray<float> GridPressure;
            public NativeArray<float2> GridVelocity;
        
            public void Execute(int i)
            {
                int2 cellIdx = Idx2Coord(i);

                float2 velocity = GridVelocity[i];
                uint grid_types = GridTypes[i];
                float pressure = GridPressure[i];

                uint2 lbType = NeighborGridTypeLB(grid_types);
                int c_id_xp = IsSolidCell(lbType.x) ? i : Coord2Idx(cellIdx + new int2(-1, 0));
                int c_id_yp = IsSolidCell(lbType.y) ? i : Coord2Idx(cellIdx + new int2(0, -1));

                // float pressure = GridPressure[i];

                velocity.x -= InvCellSize * (pressure - GridPressure[c_id_xp]);
                velocity.y -= InvCellSize * (pressure - GridPressure[c_id_yp]);
                
                GridVelocity[i] = EnforceBoundaryCondition(velocity, grid_types);
            }
        }
    
        [BurstCompile]
        private struct GridToParticleJob : IJobParallelFor
        {
            [Unity.Collections.ReadOnly] public NativeArray<float4> ParticlePos;
            [Unity.Collections.ReadOnly] public NativeArray<float2> GridVelocityOld;
            [Unity.Collections.ReadOnly] public NativeArray<float2> GridVelocityNew;
            public NativeArray<float2> ParticleVel;
            public NativeArray<float2x2> ParticleAffine;
            public float Flipness;
        
            public void Execute(int i)
            {
                float2 pos = ParticlePos[i].xy;
                float2 vel = ParticleVel[i];
                int2 coord = GetCoord(pos);

                // ReadGridFaceBilinear(pos, GridVelocityOld, GridVelocityNew, 
                //     out float2 gOriginVel, out float2 gVel);
                //
                // float2 p_pic_vel = gVel;
                // float2 p_flip_vel = vel + (gVel - gOriginVel);
                // ParticleVel[i] = gVel;

                // 在 G2P 阶段，更新粒子的 C 矩阵
                float2x2 new_C = float2x2.zero;
                float2 new_v = float2.zero;
                float2 old_v = float2.zero;
                
                for (int x = coord.x - 1; x <= coord.x + 2; ++x)
                for (int y = coord.y - 1; y <= coord.y + 2; ++y)
                {
                    int2 nCoord = new int2(x, y);
                    int idx = Coord2Idx(math.clamp(nCoord, 0, GridRes - 1));
                    
                    float2 g_vel = GridVelocityNew[idx];
                    
                    float2 pos_u = (nCoord + new float2(0, 0.5f)) * CellSize;
                    float2 pos_v = (nCoord + new float2(0.5f, 0)) * CellSize;
                    
                    float2 weights = new float2(GetWeight(pos_u - pos, InvCellSize),
                        GetWeight(pos_v - pos, InvCellSize)); 

                    float2 weightedNewV = weights * g_vel;
                    float2 weightedOldV = weights * GridVelocityOld[idx];
                    
                    // 对应公式：C += w * v * dist^T
                    float2 dx_u = pos_u - pos;
                    float2 dx_v = pos_v - pos;

                    var term = math.float2x2(
                        weightedNewV * new float2(dx_u.x, dx_v.x),
                        weightedNewV * new float2(dx_u.y, dx_v.y));
                    
                    new_C += term;
                    new_v += weightedNewV;
                    old_v += weightedOldV;
                }


                ParticleAffine[i] = new_C * (3 * InvCellSize * InvCellSize);
                // ParticleVel[i] = new_v;
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
        private struct AdvectionJob : IJobParallelFor
        {
            [Unity.Collections.ReadOnly] public NativeArray<float2> GridVelocity;
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
                velocity = math.select(velocity, math.max(velocity, -velocity), pos <= 0); 
                velocity = math.select(velocity, math.min(velocity, -velocity), pos >= GridRes);
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
        private struct ComputeLaplacianJob: IJobParallelFor
        {
            [Unity.Collections.ReadOnly] public NativeArray<uint> GridTypes;
            [WriteOnly] public NativeArray<float3> GridLaplacian;

            public void Execute(int index)
            {
                uint gridType = GridTypes[index];
                uint2 xAxisType = NeighborGridTypeAxis(0, gridType);
                uint2 yAxisType = NeighborGridTypeAxis(1, gridType);
                
                float center = 4;
                if (IsSolidCell(xAxisType.x)) center -= 1;
                if (IsSolidCell(xAxisType.y)) center -= 1;
                if (IsSolidCell(yAxisType.x)) center -= 1;
                if (IsSolidCell(yAxisType.y)) center -= 1;
                
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

        [Button]
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

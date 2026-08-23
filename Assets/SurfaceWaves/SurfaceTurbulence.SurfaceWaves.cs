using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;

namespace SurfaceWaves
{
    // Surface Turbulence (Mercier et al. SIGGRAPH Asia 2015), reduced to 2D and
    // integrated with the narrow-band FLIP in SurfaceTurbulence.cs.
    //
    // 3D -> 2D reductions used throughout:
    //   - surface points lie on a 1D curve (seeded on the box free surface at init);
    //   - normals are 2D vectors, tangent is the single perpendicular direction;
    //   - the tangent-plane least-squares fit becomes a 1D linear regression;
    //   - the surface Laplacian becomes a 1D second derivative along the tangent.
    //
    // The constraint level-set is the paper's Gaussian sum over coarse particles
    // (computeConstraintLevel/Gradient), NOT _gridSDF (which is a chamfer distance
    // band valid only inside the fluid and would not place points outside the surface).
    public partial class SurfaceTurbulence
    {
        // ===== parameters (world units; CellSize = 0.5) =====
        [Header("Surface Turbulence")]
        public float outerRadius = 1.0f;
        public int surfaceDensity = 20;
        public int nbSurfaceMaintenanceIterations = 4;
        public float waveSpeed = 8f;              // 降低波速，提高稳定性
        public float waveDamping = 2.0f;          // 添加阻尼，抑制振荡
        public float waveSeedFrequency = 4f;
        public float waveMaxAmplitude = 0.1f;     // 大幅降低振幅上限
        public float waveMaxFrequency = 400f;     // 降低频率上限
        public float waveMaxSeedingAmplitude = 0.3f; // 减少播种强度
        public float waveSeedingCurvatureThresholdRegionCenter = 0.08f;  // 提高阈值，减少播种点
        public float waveSeedingCurvatureThresholdRegionRadius = 0.02f;
        public float waveSeedStepSizeRatioOfMax = 0.01f; // 大幅减慢播种增长速度
        public Material surfMat;

        [Header("Surface Debug")]
        public bool drawSurfaceGizmos = false;

        // derived
        private float _innerRadius;
        private float _meanFineDistance;
        private float _constraintA;
        private float _normalRadius;
        private float _tangentRadius;
        private float2 _domainMin;
        private float2 _domainMax;

        // fluid box (grid coords, matches Start() init in SurfaceTurbulence.cs)
        private static readonly int2 SurfBoxMin = new int2(0, 0);
        private static readonly int2 SurfBoxMax = new int2(150, 150);

        private const int MaxSurfacePoints = 16384;

        // surface point data (packed, count = _surfCount.Value)
        private NativeArray<float2> _surfPos;
        private NativeArray<float2> _surfNormal;
        private NativeArray<float> _surfH;
        private NativeArray<float> _surfDtH;
        private NativeArray<float> _surfSeed;
        private NativeArray<float> _surfSeedAmp;
        private NativeArray<float> _surfSource;

        // temporaries
        private NativeArray<float> _surfDensity;
        private NativeArray<float2> _surfDisp;
        private NativeArray<float2> _surfNormalTemp;
        private NativeArray<float> _surfCurvature;
        private NativeArray<float> _surfLaplacian;
        private NativeArray<float> _surfWaveSlope;
        private NativeArray<byte> _surfAlive;
        private NativeArray<byte> _surfHasCandidate;
        private NativeArray<float2> _surfAddPos;
        private NativeArray<float4> _surfRender;

        private NativeReference<int> _surfCount;
        private NativeReference<int> _surfNewStart;

        // coarse previous positions (snapshot before coarse advection)
        private NativeArray<float2> _particlePosPrev;

        // uniform-grid buckets (linked list per cell) for neighbor queries
        private NativeArray<int> _coarseHead;   // NumGrid
        private NativeArray<int> _coarseNext;   // NumParticles
        private NativeArray<int> _prevHead;     // NumGrid
        private NativeArray<int> _prevNext;     // NumParticles
        private NativeArray<int> _surfHead;     // NumGrid
        private NativeArray<int> _surfNext;     // MaxSurfacePoints

        private ComputeBuffer _surfBuffer;
        private int _frameCount = 0;

        // ===== lifecycle =====

        private void InitSurfaceTurbulence()
        {
            _innerRadius = outerRadius / 2f;
            _meanFineDistance = math.PI * (outerRadius + _innerRadius) / surfaceDensity;
            _constraintA = math.log(2f / (1f + WeightCoarseDensity(outerRadius + _innerRadius, outerRadius)))
                           / (math.pow((outerRadius + _innerRadius) / 2f, 2f) - _innerRadius * _innerRadius);
            _normalRadius = 0.5f * (outerRadius + _innerRadius);
            _tangentRadius = 2.1f * _meanFineDistance;
            _domainMin = new float2(2f * CellSize, 2f * CellSize);
            _domainMax = new float2(GridRes * CellSize - 2f * CellSize, GridRes * CellSize - 2f * CellSize);

            _surfPos = new NativeArray<float2>(MaxSurfacePoints, Allocator.Persistent);
            _surfNormal = new NativeArray<float2>(MaxSurfacePoints, Allocator.Persistent);
            _surfH = new NativeArray<float>(MaxSurfacePoints, Allocator.Persistent);
            _surfDtH = new NativeArray<float>(MaxSurfacePoints, Allocator.Persistent);
            _surfSeed = new NativeArray<float>(MaxSurfacePoints, Allocator.Persistent);
            _surfSeedAmp = new NativeArray<float>(MaxSurfacePoints, Allocator.Persistent);
            _surfSource = new NativeArray<float>(MaxSurfacePoints, Allocator.Persistent);

            _surfDensity = new NativeArray<float>(MaxSurfacePoints, Allocator.Persistent);
            _surfDisp = new NativeArray<float2>(MaxSurfacePoints, Allocator.Persistent);
            _surfNormalTemp = new NativeArray<float2>(MaxSurfacePoints, Allocator.Persistent);
            _surfCurvature = new NativeArray<float>(MaxSurfacePoints, Allocator.Persistent);
            _surfLaplacian = new NativeArray<float>(MaxSurfacePoints, Allocator.Persistent);
            _surfWaveSlope = new NativeArray<float>(MaxSurfacePoints, Allocator.Persistent);
            _surfAlive = new NativeArray<byte>(MaxSurfacePoints, Allocator.Persistent);
            _surfHasCandidate = new NativeArray<byte>(MaxSurfacePoints, Allocator.Persistent);
            _surfAddPos = new NativeArray<float2>(MaxSurfacePoints, Allocator.Persistent);
            _surfRender = new NativeArray<float4>(MaxSurfacePoints, Allocator.Persistent);

            _surfCount = new NativeReference<int>(Allocator.Persistent);
            _surfNewStart = new NativeReference<int>(Allocator.Persistent);

            _particlePosPrev = new NativeArray<float2>(NumParticles, Allocator.Persistent);

            _coarseHead = new NativeArray<int>(NumGrid, Allocator.Persistent);
            _coarseNext = new NativeArray<int>(NumParticles, Allocator.Persistent);
            _prevHead = new NativeArray<int>(NumGrid, Allocator.Persistent);
            _prevNext = new NativeArray<int>(NumParticles, Allocator.Persistent);
            _surfHead = new NativeArray<int>(NumGrid, Allocator.Persistent);
            _surfNext = new NativeArray<int>(MaxSurfacePoints, Allocator.Persistent);

            _surfBuffer = new ComputeBuffer(MaxSurfacePoints, sizeof(float) * 4);
            if (surfMat != null)
                surfMat.SetBuffer("_ParticleBuffer", _surfBuffer);

            SeedSurfacePoints();
        }

        private void DisposeSurfaceTurbulence()
        {
            if (!_surfPos.IsCreated) return;
            _surfPos.Dispose();
            _surfNormal.Dispose();
            _surfH.Dispose();
            _surfDtH.Dispose();
            _surfSeed.Dispose();
            _surfSeedAmp.Dispose();
            _surfSource.Dispose();
            _surfDensity.Dispose();
            _surfDisp.Dispose();
            _surfNormalTemp.Dispose();
            _surfCurvature.Dispose();
            _surfLaplacian.Dispose();
            _surfWaveSlope.Dispose();
            _surfAlive.Dispose();
            _surfHasCandidate.Dispose();
            _surfAddPos.Dispose();
            _surfRender.Dispose();
            _surfCount.Dispose();
            _surfNewStart.Dispose();
            _particlePosPrev.Dispose();
            _coarseHead.Dispose();
            _coarseNext.Dispose();
            _prevHead.Dispose();
            _prevNext.Dispose();
            _surfHead.Dispose();
            _surfNext.Dispose();
            _surfBuffer.Release();
        }
        
        private void OnSTGUI()
        {
            int count = _surfCount.Value;
            if (count == 0) return;

            // 统计波浪振幅
            float minH = float.MaxValue;
            float maxH = float.MinValue;
            float avgH = 0f;
            float avgAbsH = 0f;
            int nonZeroCount = 0;
            int clampedCount = 0;

            for (int i = 0; i < count; i++)
            {
                float h = _surfH[i];
                minH = Mathf.Min(minH, h);
                maxH = Mathf.Max(maxH, h);
                avgH += h;
                avgAbsH += Mathf.Abs(h);
                if (Mathf.Abs(h) > 0.001f) nonZeroCount++;
                if (Mathf.Abs(h) >= waveMaxAmplitude * 0.99f) clampedCount++;
            }
            avgH /= count;
            avgAbsH /= count;

            // 统计曲率
            float minCurv = float.MaxValue;
            float maxCurv = float.MinValue;
            float avgCurv = 0f;
            int highCurvCount = 0;
            if (_surfCurvature.Length > 0)
            {
                for (int i = 0; i < count; i++)
                {
                    float c = _surfCurvature[i];
                    minCurv = Mathf.Min(minCurv, c);
                    maxCurv = Mathf.Max(maxCurv, c);
                    avgCurv += c;
                    if (c > waveSeedingCurvatureThresholdRegionCenter) highCurvCount++;
                }
                avgCurv /= count;
            }

            // 统计 seed amplitude
            float minSeedAmp = float.MaxValue;
            float maxSeedAmp = float.MinValue;
            int activeSeedCount = 0;
            if (_surfSeedAmp.Length > 0)
            {
                for (int i = 0; i < count; i++)
                {
                    float amp = _surfSeedAmp[i];
                    minSeedAmp = Mathf.Min(minSeedAmp, amp);
                    maxSeedAmp = Mathf.Max(maxSeedAmp, amp);
                    if (amp > 0.01f) activeSeedCount++;
                }
            }

            // 显示统计信息
            GUIStyle style = new GUIStyle();
            style.normal.textColor = Color.yellow;
            style.fontSize = 20;

            int yOffset = 240;
            const int lineHeight = 40;
            GUI.Label(new Rect(10, yOffset, 600, 20), $"=== Wave Diagnostics (Frame {_frameCount}) ===", style);
            yOffset += lineHeight;

            GUI.Label(new Rect(10, yOffset, 600, 20), $"Surface Points: {count}", style);
            yOffset += lineHeight;

            style.normal.textColor = clampedCount > count * 0.5f ? Color.red : Color.yellow;
            GUI.Label(new Rect(10, yOffset, 600, 20),
                $"Wave H: min={minH:F4}, max={maxH:F4}, avg={avgH:F4}, avgAbs={avgAbsH:F4}", style);
            yOffset += lineHeight;

            GUI.Label(new Rect(10, yOffset, 600, 20),
                $"Non-zero: {nonZeroCount} ({100f * nonZeroCount / count:F1}%), Clamped: {clampedCount} ({100f * clampedCount / count:F1}%)", style);
            yOffset += lineHeight;

            style.normal.textColor = Color.cyan;
            GUI.Label(new Rect(10, yOffset, 600, 20),
                $"Curvature: min={minCurv:F5}, max={maxCurv:F5}, avg={avgCurv:F5}", style);
            yOffset += lineHeight;

            style.normal.textColor = highCurvCount > count * 0.5f ? Color.red : Color.cyan;
            GUI.Label(new Rect(10, yOffset, 600, 20),
                $"High curvature (>{waveSeedingCurvatureThresholdRegionCenter:F4}): {highCurvCount} ({100f * highCurvCount / count:F1}%)", style);
            yOffset += lineHeight;

            style.normal.textColor = Color.green;
            GUI.Label(new Rect(10, yOffset, 600, 20),
                $"Seed Amp: min={minSeedAmp:F4}, max={maxSeedAmp:F4}, active={activeSeedCount}", style);
            yOffset += lineHeight;
        }

        // ===== seeding (box free surface: top + right edges; left/bottom are walls) =====

        private void SeedSurfacePoints()
        {
            int count = 0;
            float2 minW = (float2)SurfBoxMin * CellSize;
            float2 maxW = (float2)SurfBoxMax * CellSize;

            // right edge (x = maxW.x, outward normal +x)
            for (float t = minW.y; t <= maxW.y + 1e-6f; t += _meanFineDistance)
            {
                float2 pos = new float2(maxW.x, t) + new float2(1f, 0f) * outerRadius;
                WriteSurfacePoint(count++, pos, new float2(1f, 0f));
            }
            // top edge (y = maxW.y, outward normal +y)
            for (float t = minW.x; t <= maxW.x + 1e-6f; t += _meanFineDistance)
            {
                float2 pos = new float2(t, maxW.y) + new float2(0f, 1f) * outerRadius;
                WriteSurfacePoint(count++, pos, new float2(0f, 1f));
            }

            _surfCount.Value = count;
            Debug.Log($"Surface turbulence seeded {count} surface points.");
        }

        private void WriteSurfacePoint(int i, float2 pos, float2 normal)
        {
            _surfPos[i] = pos;
            _surfNormal[i] = normal;
            _surfH[i] = 0f;
            _surfDtH[i] = 0f;
            _surfSeed[i] = 0f;
            _surfSeedAmp[i] = 0f;
            _surfSource[i] = 0f;
        }

        // ===== weight kernels =====

        private static float TriangularWeight(float d, float r) => 1f - d / r;

        private static float ExponentialWeight(float d, float r, float falloff)
        {
            if (d > r) return 0f;
            float t = d / r;
            return math.exp(-falloff * t * t);
        }

        private static float WeightCoarseDensity(float d, float outerRadius) => ExponentialWeight(d, outerRadius, 2f);
        private static float WeightAdvection(float d, float outerRadius) => d > 2f * outerRadius ? 0f : TriangularWeight(d, 2f * outerRadius);
        private static float WeightSurfaceNormal(float d, float normalRadius) => d > normalRadius ? 0f : TriangularWeight(d, normalRadius);
        private static float WeightSurfaceTangent(float d, float tangentRadius) => d > tangentRadius ? 0f : TriangularWeight(d, tangentRadius);

        private static bool IsInDomain(float2 pos, float2 min, float2 max) => math.all(pos >= min) && math.all(pos <= max);

        private static float Smoothstep(float l, float r, float v)
        {
            float x = math.saturate((v - l) / (r - l));
            return x * x * (3f - 2f * x);
        }

        // ===== constraint level-set (Gaussian sum over coarse particles) =====

        private static float ConstraintLevel(NativeArray<float4> coarse, NativeArray<int> head, NativeArray<int> next,
            float2 pos, float outerRadius, float innerRadius, float constraintA)
        {
            float lvl = 0f;
            float rad = 1.5f * outerRadius;
            int2 minC = math.clamp(GetCoord(pos - rad), 0, GridRes - 1);
            int2 maxC = math.clamp(GetCoord(pos + rad), 0, GridRes - 1);
            for (int cy = minC.y; cy <= maxC.y; cy++)
            for (int cx = minC.x; cx <= maxC.x; cx++)
            {
                int id = head[Coord2Idx(cx, cy)];
                while (id >= 0)
                {
                    lvl += math.exp(-constraintA * math.lengthsq(coarse[id].xy - pos));
                    id = next[id];
                }
            }
            if (lvl > 1f) lvl = 1f;
            if (lvl <= 0f) return -1e30f;
            return (math.sqrt(-math.log(lvl) / constraintA) - innerRadius) / (outerRadius - innerRadius);
        }

        private static float2 ConstraintGradient(NativeArray<float4> coarse, NativeArray<int> head, NativeArray<int> next,
            float2 pos, float outerRadius, float constraintA)
        {
            float2 grad = float2.zero;
            float rad = 1.5f * outerRadius;
            int2 minC = math.clamp(GetCoord(pos - rad), 0, GridRes - 1);
            int2 maxC = math.clamp(GetCoord(pos + rad), 0, GridRes - 1);
            for (int cy = minC.y; cy <= maxC.y; cy++)
            for (int cx = minC.x; cx <= maxC.x; cx++)
            {
                int id = head[Coord2Idx(cx, cy)];
                while (id >= 0)
                {
                    float2 np = coarse[id].xy;
                    float e = math.exp(-constraintA * math.lengthsq(np - pos));
                    grad += 2f * constraintA * e * (pos - np);
                    id = next[id];
                }
            }
            return math.normalizesafe(grad);
        }

        // ===== neighbor predicates over the buckets =====

        private static bool HasNeighborF2(float2 pos, float r, NativeArray<int> head, NativeArray<int> next, NativeArray<float2> positions)
        {
            int2 minC = math.clamp(GetCoord(pos - r), 0, GridRes - 1);
            int2 maxC = math.clamp(GetCoord(pos + r), 0, GridRes - 1);
            for (int cy = minC.y; cy <= maxC.y; cy++)
            for (int cx = minC.x; cx <= maxC.x; cx++)
            {
                int id = head[Coord2Idx(cx, cy)];
                while (id >= 0)
                {
                    if (math.length(positions[id] - pos) <= r) return true;
                    id = next[id];
                }
            }
            return false;
        }

        // Sequential-kill semantics matching the original C++ (points are processed in index
        // order and killed points stop counting as neighbors for points processed afterward,
        // so a dense cluster is thinned down to one survivor instead of wiped out entirely).
        // Points with id > self haven't been decided yet, so they're treated as still alive.
        private static bool HasLiveNeighborOtherF2(int self, float2 pos, float r, NativeArray<int> head, NativeArray<int> next,
            NativeArray<float2> positions, NativeArray<byte> alive)
        {
            int2 minC = math.clamp(GetCoord(pos - r), 0, GridRes - 1);
            int2 maxC = math.clamp(GetCoord(pos + r), 0, GridRes - 1);
            for (int cy = minC.y; cy <= maxC.y; cy++)
            for (int cx = minC.x; cx <= maxC.x; cx++)
            {
                int id = head[Coord2Idx(cx, cy)];
                while (id >= 0)
                {
                    if (id != self && (id > self || alive[id] != 0) && math.length(positions[id] - pos) <= r) return true;
                    id = next[id];
                }
            }
            return false;
        }

        private static bool HasCoarseNeighbor(float2 pos, float r, NativeArray<int> head, NativeArray<int> next, NativeArray<float4> coarse)
        {
            int2 minC = math.clamp(GetCoord(pos - r), 0, GridRes - 1);
            int2 maxC = math.clamp(GetCoord(pos + r), 0, GridRes - 1);
            for (int cy = minC.y; cy <= maxC.y; cy++)
            for (int cx = minC.x; cx <= maxC.x; cx++)
            {
                int id = head[Coord2Idx(cx, cy)];
                while (id >= 0)
                {
                    if (math.length(coarse[id].xy - pos) <= r) return true;
                    id = next[id];
                }
            }
            return false;
        }

        // ===== bucket build jobs =====

        [BurstCompile]
        private struct BuildCoarseBucketsJob : IJob
        {
            [ReadOnly] public NativeArray<float4> Pos;
            public int Count;
            public NativeArray<int> Head;
            public NativeArray<int> Next;

            public void Execute()
            {
                for (int i = 0; i < Head.Length; i++) Head[i] = -1;
                for (int i = 0; i < Count; i++)
                {
                    int2 c = GetCoord(Pos[i].xy);
                    if (math.any(c < 0) || math.any(c >= GridRes)) continue;
                    int cell = Coord2Idx(c);
                    Next[i] = Head[cell];
                    Head[cell] = i;
                }
            }
        }

        [BurstCompile]
        private struct BuildF2BucketsJob : IJob
        {
            [ReadOnly] public NativeArray<float2> Pos;
            public int Count;
            public NativeArray<int> Head;
            public NativeArray<int> Next;

            public void Execute()
            {
                for (int i = 0; i < Head.Length; i++) Head[i] = -1;
                for (int i = 0; i < Count; i++)
                {
                    int2 c = GetCoord(Pos[i]);
                    if (math.any(c < 0) || math.any(c >= GridRes)) continue;
                    int cell = Coord2Idx(c);
                    Next[i] = Head[cell];
                    Head[cell] = i;
                }
            }
        }

        private void BuildCoarseBucket()
        {
            new BuildCoarseBucketsJob { Pos = _particlePos, Count = _particleCount.Value, Head = _coarseHead, Next = _coarseNext }.Run();
        }

        private void BuildPrevBucket()
        {
            new BuildF2BucketsJob { Pos = _particlePosPrev, Count = _particleCount.Value, Head = _prevHead, Next = _prevNext }.Run();
        }

        private void BuildSurfBucket()
        {
            new BuildF2BucketsJob { Pos = _surfPos, Count = _surfCount.Value, Head = _surfHead, Next = _surfNext }.Run();
        }

        // ===== snapshot / advection =====

        [BurstCompile]
        private struct SnapshotPrevPosJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float4> Pos;
            [WriteOnly] public NativeArray<float2> Prev;
            public void Execute(int i) { Prev[i] = Pos[i].xy; }
        }

        private void SnapshotPrevPos()
        {
            new SnapshotPrevPosJob { Pos = _particlePos, Prev = _particlePosPrev }
                .Schedule(_particleCount.Value, 64).Complete();
        }

        [BurstCompile]
        private struct AdvectSurfacePointsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float4> CoarsePos;   // current
            [ReadOnly] public NativeArray<float2> CoarsePrev;  // previous
            [ReadOnly] public NativeArray<int> PrevHead;
            [ReadOnly] public NativeArray<int> PrevNext;
            public float OuterRadius;
            public NativeArray<float2> SurfPos;

            public void Execute(int idx)
            {
                float2 p = SurfPos[idx];
                float2 avg = float2.zero;
                float total = 0f;
                float rad = 2f * OuterRadius;
                int2 minC = math.clamp(GetCoord(p - rad), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(p + rad), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = PrevHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        float2 prevPos = CoarsePrev[id];
                        float d = math.length(prevPos - p);
                        if (d <= rad)
                        {
                            float w = WeightAdvection(d, OuterRadius);
                            avg += w * (CoarsePos[id].xy - prevPos);
                            total += w;
                        }
                        id = PrevNext[id];
                    }
                }
                if (total > 1e-6f) avg /= total;
                SurfPos[idx] = p + avg;
            }
        }

        // ===== normals =====

        [BurstCompile]
        private struct ComputeSurfaceNormalsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<float4> CoarsePos;
            [ReadOnly] public NativeArray<int> CoarseHead;
            [ReadOnly] public NativeArray<int> CoarseNext;
            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            public float OuterRadius;
            public float NormalRadius;
            public float ConstraintA;
            public float2 DomainMin;
            public float2 DomainMax;
            [WriteOnly] public NativeArray<float2> SurfNormal;

            public void Execute(int idx)
            {
                float2 pos = SurfPos[idx];
                float2 gradient = ConstraintGradient(CoarsePos, CoarseHead, CoarseNext, pos, OuterRadius, ConstraintA);
                if (math.lengthsq(gradient) < 1e-8f)
                {
                    SurfNormal[idx] = new float2(0f, 1f);
                    return;
                }
                float2 n = gradient;
                float2 t1 = new float2(-n.y, n.x);

                float sw = 0f, swx = 0f, swx2 = 0f, swz = 0f, swxz = 0f;
                int2 minC = math.clamp(GetCoord(pos - NormalRadius), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(pos + NormalRadius), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = SurfHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        float2 neighborPos = SurfPos[id];

                        // Process with ghost/mirror reflections (inline for Burst compatibility)
                        // Ghost left boundary
                        if (neighborPos.x - DomainMin.x <= NormalRadius)
                        {
                            float2 gPos = new float2(2f * DomainMin.x - neighborPos.x, neighborPos.y);
                            float2 off = gPos - pos;
                            float d = math.length(off);
                            if (d <= NormalRadius)
                            {
                                float x = math.dot(off, t1);
                                float z = math.dot(off, n);
                                float w = WeightSurfaceNormal(d, NormalRadius);
                                sw += w; swx += w * x; swx2 += w * x * x; swz += w * z; swxz += w * x * z;
                            }
                        }
                        // Ghost right boundary
                        if (DomainMax.x - neighborPos.x <= NormalRadius)
                        {
                            float2 gPos = new float2(2f * DomainMax.x - neighborPos.x, neighborPos.y);
                            float2 off = gPos - pos;
                            float d = math.length(off);
                            if (d <= NormalRadius)
                            {
                                float x = math.dot(off, t1);
                                float z = math.dot(off, n);
                                float w = WeightSurfaceNormal(d, NormalRadius);
                                sw += w; swx += w * x; swx2 += w * x * x; swz += w * z; swxz += w * x * z;
                            }
                        }
                        // Ghost bottom boundary
                        if (neighborPos.y - DomainMin.y <= NormalRadius)
                        {
                            float2 gPos = new float2(neighborPos.x, 2f * DomainMin.y - neighborPos.y);
                            float2 off = gPos - pos;
                            float d = math.length(off);
                            if (d <= NormalRadius)
                            {
                                float x = math.dot(off, t1);
                                float z = math.dot(off, n);
                                float w = WeightSurfaceNormal(d, NormalRadius);
                                sw += w; swx += w * x; swx2 += w * x * x; swz += w * z; swxz += w * x * z;
                            }
                        }
                        // Ghost top boundary
                        if (DomainMax.y - neighborPos.y <= NormalRadius)
                        {
                            float2 gPos = new float2(neighborPos.x, 2f * DomainMax.y - neighborPos.y);
                            float2 off = gPos - pos;
                            float d = math.length(off);
                            if (d <= NormalRadius)
                            {
                                float x = math.dot(off, t1);
                                float z = math.dot(off, n);
                                float w = WeightSurfaceNormal(d, NormalRadius);
                                sw += w; swx += w * x; swx2 += w * x * x; swz += w * z; swxz += w * x * z;
                            }
                        }
                        // Real position
                        {
                            float2 off = neighborPos - pos;
                            float d = math.length(off);
                            if (d <= NormalRadius)
                            {
                                float x = math.dot(off, t1);
                                float z = math.dot(off, n);
                                float w = WeightSurfaceNormal(d, NormalRadius);
                                sw += w; swx += w * x; swx2 += w * x * x; swz += w * z; swxz += w * x * z;
                            }
                        }

                        id = SurfNext[id];
                    }
                }

                float det = sw * swx2 - swx * swx;
                float2 normal;
                if (math.abs(det) < 1e-6f) normal = n;
                else
                {
                    float a = (sw * swxz - swx * swz) / det;
                    normal = math.normalizesafe(n - a * t1);
                }
                if (math.dot(gradient, normal) < 0f) normal = -normal;
                SurfNormal[idx] = normal;
            }
        }

        [BurstCompile]
        private struct ComputeAveragedNormalsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<float2> SurfNormal;
            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            public float NormalRadius;
            [WriteOnly] public NativeArray<float2> Out;

            public void Execute(int idx)
            {
                float2 pos = SurfPos[idx];
                float2 acc = float2.zero;
                int2 minC = math.clamp(GetCoord(pos - NormalRadius), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(pos + NormalRadius), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = SurfHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        float d = math.length(SurfPos[id] - pos);
                        if (d <= NormalRadius)
                            acc += WeightSurfaceNormal(d, NormalRadius) * SurfNormal[id];
                        id = SurfNext[id];
                    }
                }
                Out[idx] = math.normalizesafe(acc);
            }
        }

        [BurstCompile]
        private struct AssignNormalsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> Src;
            [WriteOnly] public NativeArray<float2> Dst;
            public void Execute(int i) { Dst[i] = Src[i]; }
        }

        // ===== add / delete =====

        // Phase 1: compute candidate addition positions (parallel, read-only)
        [BurstCompile]
        private struct ComputeAdditionsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<float4> CoarsePos;
            [ReadOnly] public NativeArray<int> CoarseHead;
            [ReadOnly] public NativeArray<int> CoarseNext;
            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            public float OuterRadius;
            public float TangentRadius;
            public float MeanFineDistance;
            public float ConstraintA;
            public float2 DomainMin;
            public float2 DomainMax;
            [WriteOnly] public NativeArray<float2> CandidatePos;
            [WriteOnly] public NativeArray<byte> HasCandidate;

            public void Execute(int i)
            {
                HasCandidate[i] = 0;
                float2 pos = SurfPos[i];
                float2 gradient = ConstraintGradient(CoarsePos, CoarseHead, CoarseNext, pos, OuterRadius, ConstraintA);
                if (math.lengthsq(gradient) < 1e-8f) return;

                float2 tanDisp = float2.zero;
                int2 minC = math.clamp(GetCoord(pos - TangentRadius), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(pos + TangentRadius), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = SurfHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        if (id != i)
                        {
                            float2 dir = pos - SurfPos[id];
                            float len = math.length(dir);
                            if (len > 1e-6f)
                            {
                                dir /= len;
                                float2 dn = math.dot(dir, gradient) * gradient;
                                float2 dt = dir - dn;
                                float w = WeightSurfaceTangent(len, TangentRadius);
                                tanDisp += w * dt;
                            }
                        }
                        id = SurfNext[id];
                    }
                }
                if (math.lengthsq(tanDisp) > 0f)
                {
                    tanDisp = math.normalize(tanDisp);
                    float2 creationPos = pos + MeanFineDistance * tanDisp;
                    if (IsInDomain(creationPos, DomainMin, DomainMax) &&
                        !HasNeighborF2(creationPos, MeanFineDistance - 1e-6f, SurfHead, SurfNext, SurfPos))
                    {
                        CandidatePos[i] = creationPos;
                        HasCandidate[i] = 1;
                    }
                }
            }
        }

        // Phase 2b/2c: mark deletions for coarse-neighbor and constraint-level checks (parallel)
        [BurstCompile]
        private struct MarkDeletionsParallelJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<float4> CoarsePos;
            [ReadOnly] public NativeArray<int> CoarseHead;
            [ReadOnly] public NativeArray<int> CoarseNext;
            public NativeArray<byte> SurfAlive;
            public float OuterRadius;
            public float InnerRadius;
            public float ConstraintA;

            public void Execute(int i)
            {
                if (SurfAlive[i] == 0) return;
                if (!HasCoarseNeighbor(SurfPos[i], 2f * OuterRadius, CoarseHead, CoarseNext, CoarsePos))
                {
                    SurfAlive[i] = 0;
                    return;
                }
                float level = ConstraintLevel(CoarsePos, CoarseHead, CoarseNext, SurfPos[i], OuterRadius, InnerRadius, ConstraintA);
                if (level < -0.2f || level > 1.2f) SurfAlive[i] = 0;
            }
        }

        [BurstCompile]
        private struct AddDeleteSurfacePointsJob : IJob
        {
            public NativeArray<float2> SurfPos;
            public NativeArray<float2> SurfNormal;
            public NativeArray<float> SurfH;
            public NativeArray<float> SurfDtH;
            public NativeArray<float> SurfSeed;
            public NativeArray<float> SurfSeedAmp;
            public NativeArray<float> SurfSource;
            [ReadOnly] public NativeArray<float2> CandidatePos;
            [ReadOnly] public NativeArray<byte> HasCandidate;
            public NativeArray<byte> SurfAlive;
            public NativeReference<int> SurfCount;
            public NativeReference<int> SurfNewStart;

            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            [ReadOnly] public NativeArray<float> GridSDF;

            public float MeanFineDistance;
            public float2 DomainMin;
            public float2 DomainMax;

            public void Execute()
            {
                int n = SurfCount.Value;

                // Phase 2a: mark deletions (domain + density clustering, must be sequential for cascade)
                for (int i = 0; i < n; i++) SurfAlive[i] = 1;
                for (int i = 0; i < n; i++)
                {
                    float2 pos = SurfPos[i];
                    bool deepInterior = ReadGrid(GetCoord(pos), GridSDF) >= Band2;
                    bool kill = deepInterior || !IsInDomain(pos, DomainMin, DomainMax) ||
                        HasLiveNeighborOtherF2(i, pos, 0.67f * MeanFineDistance, SurfHead, SurfNext, SurfPos, SurfAlive);
                    if (kill) SurfAlive[i] = 0;
                }
                // Phase 2b/2c are now done by MarkDeletionsParallelJob (before this job runs)

                // Phase 3: compact survivors + append additions
                int write = 0;
                for (int i = 0; i < n; i++)
                {
                    if (SurfAlive[i] == 0) continue;
                    if (write != i)
                    {
                        SurfPos[write] = SurfPos[i];
                        SurfNormal[write] = SurfNormal[i];
                        SurfH[write] = SurfH[i];
                        SurfDtH[write] = SurfDtH[i];
                        SurfSeed[write] = SurfSeed[i];
                        SurfSeedAmp[write] = SurfSeedAmp[i];
                        SurfSource[write] = SurfSource[i];
                    }
                    write++;
                }
                int newStart = write;

                // Append valid candidates from parallel addition phase
                for (int i = 0; i < n && write < MaxSurfacePoints; i++)
                {
                    if (HasCandidate[i] != 0)
                    {
                        SurfPos[write] = CandidatePos[i];
                        SurfNormal[write] = new float2(0f, 1f);
                        SurfH[write] = 0f;
                        SurfDtH[write] = 0f;
                        SurfSeed[write] = 0f;
                        SurfSeedAmp[write] = 0f;
                        SurfSource[write] = 0f;
                        write++;
                    }
                }
                SurfCount.Value = write;
                SurfNewStart.Value = newStart;
            }
        }

        // ===== regularization =====

        [BurstCompile]
        private struct ComputeSurfaceDensitiesJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            public float NormalRadius;
            public float2 DomainMin;
            public float2 DomainMax;
            [WriteOnly] public NativeArray<float> Density;

            public void Execute(int idx)
            {
                float2 pos = SurfPos[idx];
                float density = 0f;
                int2 minC = math.clamp(GetCoord(pos - NormalRadius), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(pos + NormalRadius), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = SurfHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        float2 neighborPos = SurfPos[id];

                        // Process with ghost/mirror reflections (inline for Burst compatibility)
                        // Ghost left
                        if (neighborPos.x - DomainMin.x <= NormalRadius)
                        {
                            float2 gPos = new float2(2f * DomainMin.x - neighborPos.x, neighborPos.y);
                            float d = math.length(gPos - pos);
                            if (d <= NormalRadius) density += WeightSurfaceNormal(d, NormalRadius);
                        }
                        // Ghost right
                        if (DomainMax.x - neighborPos.x <= NormalRadius)
                        {
                            float2 gPos = new float2(2f * DomainMax.x - neighborPos.x, neighborPos.y);
                            float d = math.length(gPos - pos);
                            if (d <= NormalRadius) density += WeightSurfaceNormal(d, NormalRadius);
                        }
                        // Ghost bottom
                        if (neighborPos.y - DomainMin.y <= NormalRadius)
                        {
                            float2 gPos = new float2(neighborPos.x, 2f * DomainMin.y - neighborPos.y);
                            float d = math.length(gPos - pos);
                            if (d <= NormalRadius) density += WeightSurfaceNormal(d, NormalRadius);
                        }
                        // Ghost top
                        if (DomainMax.y - neighborPos.y <= NormalRadius)
                        {
                            float2 gPos = new float2(neighborPos.x, 2f * DomainMax.y - neighborPos.y);
                            float d = math.length(gPos - pos);
                            if (d <= NormalRadius) density += WeightSurfaceNormal(d, NormalRadius);
                        }
                        // Real position
                        {
                            float d = math.length(neighborPos - pos);
                            if (d <= NormalRadius) density += WeightSurfaceNormal(d, NormalRadius);
                        }

                        id = SurfNext[id];
                    }
                }
                Density[idx] = density;
            }
        }

        [BurstCompile]
        private struct ComputeSurfaceDisplacementsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<float2> SurfNormal;
            [ReadOnly] public NativeArray<float> Density;
            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            public float NormalRadius;
            public float MeanFineDistance;
            public float2 DomainMin;
            public float2 DomainMax;
            [WriteOnly] public NativeArray<float2> Disp;

            public void Execute(int idx)
            {
                float2 pos = SurfPos[idx];
                float2 n = SurfNormal[idx];

                float2 dispNormal = float2.zero;
                float2 dispTangent = float2.zero;
                float wTotal = 0f;

                int2 minC = math.clamp(GetCoord(pos - NormalRadius), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(pos + NormalRadius), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = SurfHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        float2 neighborPos = SurfPos[id];
                        float2 neighborNormal = SurfNormal[id];

                        if (Density[id] > 0f)
                        {
                            // Process with ghost/mirror reflections (inline for Burst compatibility)
                            // Ghost left
                            if (neighborPos.x - DomainMin.x <= NormalRadius)
                            {
                                float2 gPos = new float2(2f * DomainMin.x - neighborPos.x, neighborPos.y);
                                float2 gNormal = new float2(-neighborNormal.x, neighborNormal.y);
                                ProcessDisplacement(pos, n, gPos, gNormal, NormalRadius, Density[id],
                                    ref dispNormal, ref dispTangent, ref wTotal);
                            }
                            // Ghost right
                            if (DomainMax.x - neighborPos.x <= NormalRadius)
                            {
                                float2 gPos = new float2(2f * DomainMax.x - neighborPos.x, neighborPos.y);
                                float2 gNormal = new float2(-neighborNormal.x, neighborNormal.y);
                                ProcessDisplacement(pos, n, gPos, gNormal, NormalRadius, Density[id],
                                    ref dispNormal, ref dispTangent, ref wTotal);
                            }
                            // Ghost bottom
                            if (neighborPos.y - DomainMin.y <= NormalRadius)
                            {
                                float2 gPos = new float2(neighborPos.x, 2f * DomainMin.y - neighborPos.y);
                                float2 gNormal = new float2(neighborNormal.x, -neighborNormal.y);
                                ProcessDisplacement(pos, n, gPos, gNormal, NormalRadius, Density[id],
                                    ref dispNormal, ref dispTangent, ref wTotal);
                            }
                            // Ghost top
                            if (DomainMax.y - neighborPos.y <= NormalRadius)
                            {
                                float2 gPos = new float2(neighborPos.x, 2f * DomainMax.y - neighborPos.y);
                                float2 gNormal = new float2(neighborNormal.x, -neighborNormal.y);
                                ProcessDisplacement(pos, n, gPos, gNormal, NormalRadius, Density[id],
                                    ref dispNormal, ref dispTangent, ref wTotal);
                            }
                            // Real position
                            ProcessDisplacement(pos, n, neighborPos, neighborNormal, NormalRadius, Density[id],
                                ref dispNormal, ref dispTangent, ref wTotal);
                        }

                        id = SurfNext[id];
                    }
                }
                if (wTotal > 0f)
                {
                    dispNormal /= wTotal;
                    dispTangent /= wTotal;
                }
                dispNormal *= 0.75f;
                dispTangent *= 0.25f * MeanFineDistance;
                Disp[idx] = dispNormal + dispTangent;
            }

            private static void ProcessDisplacement(float2 pos, float2 n, float2 gPos, float2 gNormal,
                float normalRadius, float density, ref float2 dispNormal, ref float2 dispTangent, ref float wTotal)
            {
                float2 dir = pos - gPos;
                float len = math.length(dir);
                if (len > 1e-6f)
                {
                    float2 dn = math.dot(dir, n) * n;
                    float2 dt = dir - dn;

                    float w = WeightSurfaceNormal(len, normalRadius) / density;

                    float2 gN = gNormal;
                    if (math.dot(gN, n) < 0f) gN = -gN;
                    float2 nn = n + gN;
                    float denom = math.dot(n, nn);
                    if (math.abs(denom) > 1e-6f)
                    {
                        float2 dnCorr = -(math.dot(nn, dir) / denom) * n;
                        dispNormal += w * dnCorr;
                        dispTangent += w * math.normalizesafe(dt);
                        wTotal += w;
                    }
                }
            }
        }

        [BurstCompile]
        private struct ApplySurfaceDisplacementsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> Disp;
            public NativeArray<float2> SurfPos;
            public void Execute(int i) { SurfPos[i] = SurfPos[i] + Disp[i]; }
        }

        // ===== constraint =====

        [BurstCompile]
        private struct ConstrainSurfaceJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float4> CoarsePos;
            [ReadOnly] public NativeArray<int> CoarseHead;
            [ReadOnly] public NativeArray<int> CoarseNext;
            public float OuterRadius;
            public float InnerRadius;
            public float ConstraintA;
            public NativeArray<float2> SurfPos;

            public void Execute(int idx)
            {
                float2 pos = SurfPos[idx];
                float level = ConstraintLevel(CoarsePos, CoarseHead, CoarseNext, pos, OuterRadius, InnerRadius, ConstraintA);
                float scale = OuterRadius - InnerRadius;
                if (level > 1f)
                    SurfPos[idx] = pos - scale * (level - 1f) * ConstraintGradient(CoarsePos, CoarseHead, CoarseNext, pos, OuterRadius, ConstraintA);
                else if (level < 0f)
                    SurfPos[idx] = pos - scale * level * ConstraintGradient(CoarsePos, CoarseHead, CoarseNext, pos, OuterRadius, ConstraintA);
            }
        }

        // ===== wave data interpolation for new points =====

        [BurstCompile]
        private struct InterpolateNewWaveDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            // written at NewStart + idx (outside the scheduled [0, length) range);
            // read region [0, NewStart) is disjoint from write region [NewStart, count), so this is safe.
            [NativeDisableParallelForRestriction] public NativeArray<float> SurfH;
            [NativeDisableParallelForRestriction] public NativeArray<float> SurfDtH;
            [NativeDisableParallelForRestriction] public NativeArray<float> SurfSeed;
            [NativeDisableParallelForRestriction] public NativeArray<float> SurfSeedAmp;
            public float TangentRadius;
            public int NewStart;

            public void Execute(int idx)
            {
                int i = NewStart + idx;
                float2 pos = SurfPos[i];
                float h = 0f, dtH = 0f, seed = 0f, seedAmp = 0f, wTotal = 0f;
                int2 minC = math.clamp(GetCoord(pos - TangentRadius), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(pos + TangentRadius), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = SurfHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        if (id < NewStart) // only use pre-existing points
                        {
                            float d = math.length(SurfPos[id] - pos);
                            if (d <= TangentRadius)
                            {
                                float w = WeightSurfaceTangent(d, TangentRadius);
                                h += w * SurfH[id];
                                dtH += w * SurfDtH[id];
                                seed += w * SurfSeed[id];
                                seedAmp += w * SurfSeedAmp[id];
                                wTotal += w;
                            }
                        }
                        id = SurfNext[id];
                    }
                }
                if (wTotal > 0f)
                {
                    SurfH[i] = h / wTotal;
                    SurfDtH[i] = dtH / wTotal;
                    SurfSeed[i] = seed / wTotal;
                    SurfSeedAmp[i] = seedAmp / wTotal;
                }
            }
        }

        // ===== wave evolution =====

        [BurstCompile]
        private struct AddSeedJob : IJobParallelFor
        {
            public NativeArray<float> SurfH;
            [ReadOnly] public NativeArray<float> SurfSeed;
            public void Execute(int i) { SurfH[i] += SurfSeed[i]; }
        }

        [BurstCompile]
        private struct ComputeSurfaceWaveSlopeJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<float2> SurfNormal;
            [ReadOnly] public NativeArray<float> SurfH;
            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            public float TangentRadius;
            [WriteOnly] public NativeArray<float> Slope;

            public void Execute(int idx)
            {
                float2 pos = SurfPos[idx];
                float2 n = math.normalizesafe(SurfNormal[idx]);
                float2 t1 = new float2(-n.y, n.x);

                float sw = 0f, swx = 0f, swx2 = 0f, swz = 0f, swxz = 0f;
                int2 minC = math.clamp(GetCoord(pos - TangentRadius), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(pos + TangentRadius), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = SurfHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        float2 off = SurfPos[id] - pos;
                        float d = math.length(off);
                        if (d <= TangentRadius)
                        {
                            float x = math.dot(off, t1);
                            float z = SurfH[id];
                            float w = WeightSurfaceTangent(d, TangentRadius);
                            sw += w; swx += w * x; swx2 += w * x * x; swz += w * z; swxz += w * x * z;
                        }
                        id = SurfNext[id];
                    }
                }
                float det = sw * swx2 - swx * swx;
                Slope[idx] = math.abs(det) < 1e-6f ? 0f : (sw * swxz - swx * swz) / det;
            }
        }

        [BurstCompile]
        private struct ComputeSurfaceWaveLaplaciansJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<float2> SurfNormal;
            [ReadOnly] public NativeArray<float> SurfH;
            [ReadOnly] public NativeArray<float> Slope;
            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            public float TangentRadius;
            [WriteOnly] public NativeArray<float> Laplacian;

            public void Execute(int idx)
            {
                float2 pPos = SurfPos[idx];
                float2 n = math.normalizesafe(SurfNormal[idx]);
                float2 t1 = new float2(-n.y, n.x);
                float slope = Slope[idx];
                float ph = SurfH[idx];

                float laplacian = 0f;
                float wTotal = 0f;
                int2 minC = math.clamp(GetCoord(pPos - TangentRadius), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(pPos + TangentRadius), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = SurfHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        float2 dir = SurfPos[id] - pPos;
                        float len = math.length(dir);
                        if (len > 1e-5f)
                        {
                            float2 tangentDir = len * math.normalizesafe(dir - math.dot(dir, n) * n);
                            float dirX = math.dot(tangentDir, t1);
                            float dz = SurfH[id] - ph - slope * dirX;
                            float w = WeightSurfaceTangent(len, TangentRadius);
                            wTotal += w;
                            laplacian += math.clamp(w * 4f * dz / (len * len), -100f, 100f);
                        }
                        id = SurfNext[id];
                    }
                }
                Laplacian[idx] = wTotal > 0f ? laplacian / wTotal : 0f;
            }
        }

        [BurstCompile]
        private struct EvolveWaveJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float> Laplacian;
            [ReadOnly] public NativeArray<float> SurfSeed;
            public NativeArray<float> SurfH;
            public NativeArray<float> SurfDtH;
            public float WaveSpeed;
            public float WaveDamping;
            public float Dt;
            public float WaveMaxAmplitude;
            public float WaveMaxFrequency;

            public void Execute(int i)
            {
                float dtH = SurfDtH[i];
                dtH += WaveSpeed * WaveSpeed * Dt * Laplacian[i];
                dtH /= (1f + Dt * WaveDamping);
                float h = SurfH[i] + Dt * dtH;
                h /= (1f + Dt * WaveDamping);
                h -= SurfSeed[i];

                dtH = math.clamp(dtH, -WaveMaxFrequency * WaveMaxAmplitude, WaveMaxFrequency * WaveMaxAmplitude);
                h = math.clamp(h, -WaveMaxAmplitude, WaveMaxAmplitude);
                SurfDtH[i] = dtH;
                SurfH[i] = h;
            }
        }

        [BurstCompile]
        private struct ComputeSurfaceCurvatureJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<float2> SurfNormal;
            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            public float NormalRadius;
            public float2 DomainMin;
            public float2 DomainMax;
            [WriteOnly] public NativeArray<float> Curvature;

            public void Execute(int idx)
            {
                float2 pPos = SurfPos[idx];
                float2 pNormal = SurfNormal[idx];
                float curv = 0f, wTotal = 0f;

                int2 minC = math.clamp(GetCoord(pPos - NormalRadius), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(pPos + NormalRadius), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = SurfHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        float2 neighborPos = SurfPos[id];
                        float2 neighborNormal = SurfNormal[id];

                        // Process with ghost/mirror reflections (inline for Burst compatibility)
                        // Ghost left
                        if (neighborPos.x - DomainMin.x <= NormalRadius)
                        {
                            float2 gPos = new float2(2f * DomainMin.x - neighborPos.x, neighborPos.y);
                            float2 gNormal = new float2(-neighborNormal.x, neighborNormal.y);
                            ProcessCurvature(pPos, pNormal, gPos, gNormal, NormalRadius, ref curv, ref wTotal);
                        }
                        // Ghost right
                        if (DomainMax.x - neighborPos.x <= NormalRadius)
                        {
                            float2 gPos = new float2(2f * DomainMax.x - neighborPos.x, neighborPos.y);
                            float2 gNormal = new float2(-neighborNormal.x, neighborNormal.y);
                            ProcessCurvature(pPos, pNormal, gPos, gNormal, NormalRadius, ref curv, ref wTotal);
                        }
                        // Ghost bottom
                        if (neighborPos.y - DomainMin.y <= NormalRadius)
                        {
                            float2 gPos = new float2(neighborPos.x, 2f * DomainMin.y - neighborPos.y);
                            float2 gNormal = new float2(neighborNormal.x, -neighborNormal.y);
                            ProcessCurvature(pPos, pNormal, gPos, gNormal, NormalRadius, ref curv, ref wTotal);
                        }
                        // Ghost top
                        if (DomainMax.y - neighborPos.y <= NormalRadius)
                        {
                            float2 gPos = new float2(neighborPos.x, 2f * DomainMax.y - neighborPos.y);
                            float2 gNormal = new float2(neighborNormal.x, -neighborNormal.y);
                            ProcessCurvature(pPos, pNormal, gPos, gNormal, NormalRadius, ref curv, ref wTotal);
                        }
                        // Real position
                        ProcessCurvature(pPos, pNormal, neighborPos, neighborNormal, NormalRadius, ref curv, ref wTotal);

                        id = SurfNext[id];
                    }
                }
                Curvature[idx] = wTotal > 0f ? math.abs(curv / wTotal) : 0f;
            }

            private static void ProcessCurvature(float2 pPos, float2 pNormal, float2 gPos, float2 gNormal,
                float normalRadius, ref float curv, ref float wTotal)
            {
                if (math.dot(pNormal, gNormal) < 0f) return; // backfacing
                float2 dir = pPos - gPos;
                float dist = math.length(dir);
                if (dist < normalRadius / 100f) return;

                float distn = math.dot(dir, pNormal);
                float w = WeightSurfaceNormal(dist, normalRadius);
                curv += w * distn;
                wTotal += w;
            }
        }

        [BurstCompile]
        private struct SmoothCurvatureJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<float> Curvature;
            [ReadOnly] public NativeArray<int> SurfHead;
            [ReadOnly] public NativeArray<int> SurfNext;
            public float NormalRadius;
            public float2 DomainMin;
            public float2 DomainMax;
            [WriteOnly] public NativeArray<float> Source;

            public void Execute(int idx)
            {
                float2 pPos = SurfPos[idx];
                float curv = 0f, wTotal = 0f;
                int2 minC = math.clamp(GetCoord(pPos - NormalRadius), 0, GridRes - 1);
                int2 maxC = math.clamp(GetCoord(pPos + NormalRadius), 0, GridRes - 1);
                for (int cy = minC.y; cy <= maxC.y; cy++)
                for (int cx = minC.x; cx <= maxC.x; cx++)
                {
                    int id = SurfHead[Coord2Idx(cx, cy)];
                    while (id >= 0)
                    {
                        float2 neighborPos = SurfPos[id];

                        // Process with ghost/mirror reflections (inline for Burst compatibility)
                        // Ghost left
                        if (neighborPos.x - DomainMin.x <= NormalRadius)
                        {
                            float2 gPos = new float2(2f * DomainMin.x - neighborPos.x, neighborPos.y);
                            float d = math.length(gPos - pPos);
                            if (d <= NormalRadius)
                            {
                                float w = WeightSurfaceNormal(d, NormalRadius);
                                curv += w * Curvature[id];
                                wTotal += w;
                            }
                        }
                        // Ghost right
                        if (DomainMax.x - neighborPos.x <= NormalRadius)
                        {
                            float2 gPos = new float2(2f * DomainMax.x - neighborPos.x, neighborPos.y);
                            float d = math.length(gPos - pPos);
                            if (d <= NormalRadius)
                            {
                                float w = WeightSurfaceNormal(d, NormalRadius);
                                curv += w * Curvature[id];
                                wTotal += w;
                            }
                        }
                        // Ghost bottom
                        if (neighborPos.y - DomainMin.y <= NormalRadius)
                        {
                            float2 gPos = new float2(neighborPos.x, 2f * DomainMin.y - neighborPos.y);
                            float d = math.length(gPos - pPos);
                            if (d <= NormalRadius)
                            {
                                float w = WeightSurfaceNormal(d, NormalRadius);
                                curv += w * Curvature[id];
                                wTotal += w;
                            }
                        }
                        // Ghost top
                        if (DomainMax.y - neighborPos.y <= NormalRadius)
                        {
                            float2 gPos = new float2(neighborPos.x, 2f * DomainMax.y - neighborPos.y);
                            float d = math.length(gPos - pPos);
                            if (d <= NormalRadius)
                            {
                                float w = WeightSurfaceNormal(d, NormalRadius);
                                curv += w * Curvature[id];
                                wTotal += w;
                            }
                        }
                        // Real position
                        {
                            float d = math.length(neighborPos - pPos);
                            if (d <= NormalRadius)
                            {
                                float w = WeightSurfaceNormal(d, NormalRadius);
                                curv += w * Curvature[id];
                                wTotal += w;
                            }
                        }

                        id = SurfNext[id];
                    }
                }
                Source[idx] = wTotal > 0f ? curv / wTotal : 0f;
            }
        }

        [BurstCompile]
        private struct SeedWavesJob : IJobParallelFor
        {
            public NativeArray<float> Source; // read + written (overwritten for display, matches paper)
            public NativeArray<float> SurfSeed;
            public NativeArray<float> SurfSeedAmp;
            public float WaveSeedFrequency;
            public float WaveSpeed;
            public float WaveMaxAmplitude;
            public float WaveMaxSeedingAmplitude;
            public float ThresholdCenter;
            public float ThresholdRadius;
            public float StepSizeRatio;
            public float Dt;
            public int FrameCount;

            public void Execute(int i)
            {
                float source = Smoothstep(ThresholdCenter - ThresholdRadius, ThresholdCenter + ThresholdRadius, Source[i]) * 2f - 1f;
                float theta = Dt * FrameCount * WaveSpeed * WaveSeedFrequency;
                float costheta = math.cos(theta);
                float maxSeedAmp = WaveMaxSeedingAmplitude * WaveMaxAmplitude;

                float amp = math.clamp(SurfSeedAmp[i] + source * StepSizeRatio * maxSeedAmp, 0f, maxSeedAmp);
                SurfSeedAmp[i] = amp;
                SurfSeed[i] = amp * costheta;
                Source[i] = source >= 0f ? 1f : 0f; // overwrite source (write-only here; we alias _surfSource)
            }
        }

        [BurstCompile]
        private struct WriteSurfBufferJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float2> SurfPos;
            [ReadOnly] public NativeArray<float2> SurfNormal;
            [ReadOnly] public NativeArray<float> SurfH;
            [WriteOnly] public NativeArray<float4> Out;

            public void Execute(int i)
            {
                float2 p = SurfPos[i] + SurfNormal[i] * SurfH[i];
                Out[i] = new float4(p, 0f, SurfH[i]);
            }
        }

        // ===== orchestration =====

        private void SurfaceMaintenance(int iterations)
        {
            for (int it = 0; it < iterations; it++)
            {
                BuildSurfBucket();
                int n = _surfCount.Value;

                // Phase 1: compute additions in parallel
                new ComputeAdditionsJob
                {
                    SurfPos = _surfPos,
                    CoarsePos = _particlePos,
                    CoarseHead = _coarseHead,
                    CoarseNext = _coarseNext,
                    SurfHead = _surfHead,
                    SurfNext = _surfNext,
                    OuterRadius = outerRadius,
                    TangentRadius = _tangentRadius,
                    MeanFineDistance = _meanFineDistance,
                    ConstraintA = _constraintA,
                    DomainMin = _domainMin,
                    DomainMax = _domainMax,
                    CandidatePos = _surfAddPos,
                    HasCandidate = _surfHasCandidate,
                }.Schedule(n, 64).Complete();

                // Phase 2b/2c: mark deletions (parallel, coarse neighbor + constraint level checks)
                new MarkDeletionsParallelJob
                {
                    SurfPos = _surfPos,
                    CoarsePos = _particlePos,
                    CoarseHead = _coarseHead,
                    CoarseNext = _coarseNext,
                    SurfAlive = _surfAlive,
                    OuterRadius = outerRadius,
                    InnerRadius = _innerRadius,
                    ConstraintA = _constraintA,
                }.Schedule(n, 64).Complete();

                // Phase 2a + 3: serial cascade deletion (domain + density) + compact
                new AddDeleteSurfacePointsJob
                {
                    SurfPos = _surfPos,
                    SurfNormal = _surfNormal,
                    SurfH = _surfH,
                    SurfDtH = _surfDtH,
                    SurfSeed = _surfSeed,
                    SurfSeedAmp = _surfSeedAmp,
                    SurfSource = _surfSource,
                    CandidatePos = _surfAddPos,
                    HasCandidate = _surfHasCandidate,
                    SurfAlive = _surfAlive,
                    SurfCount = _surfCount,
                    SurfNewStart = _surfNewStart,
                    SurfHead = _surfHead,
                    SurfNext = _surfNext,
                    GridSDF = _gridSDF,
                    MeanFineDistance = _meanFineDistance,
                    DomainMin = _domainMin,
                    DomainMax = _domainMax,
                }.Run();
                if (_surfCount.Value == 0) return;

                BuildSurfBucket();
                new ComputeSurfaceNormalsJob
                {
                    SurfPos = _surfPos,
                    CoarsePos = _particlePos,
                    CoarseHead = _coarseHead,
                    CoarseNext = _coarseNext,
                    SurfHead = _surfHead,
                    SurfNext = _surfNext,
                    OuterRadius = outerRadius,
                    NormalRadius = _normalRadius,
                    ConstraintA = _constraintA,
                    DomainMin = _domainMin,
                    DomainMax = _domainMax,
                    SurfNormal = _surfNormal,
                }.Schedule(_surfCount.Value, 64).Complete();

                new ComputeAveragedNormalsJob
                {
                    SurfPos = _surfPos,
                    SurfNormal = _surfNormal,
                    SurfHead = _surfHead,
                    SurfNext = _surfNext,
                    NormalRadius = _normalRadius,
                    Out = _surfNormalTemp,
                }.Schedule(_surfCount.Value, 64).Complete();
                new AssignNormalsJob { Src = _surfNormalTemp, Dst = _surfNormal }.Schedule(_surfCount.Value, 64).Complete();

                new ComputeSurfaceDensitiesJob
                {
                    SurfPos = _surfPos,
                    SurfHead = _surfHead,
                    SurfNext = _surfNext,
                    NormalRadius = _normalRadius,
                    DomainMin = _domainMin,
                    DomainMax = _domainMax,
                    Density = _surfDensity,
                }.Schedule(_surfCount.Value, 64).Complete();

                new ComputeSurfaceDisplacementsJob
                {
                    SurfPos = _surfPos,
                    SurfNormal = _surfNormal,
                    Density = _surfDensity,
                    SurfHead = _surfHead,
                    SurfNext = _surfNext,
                    NormalRadius = _normalRadius,
                    MeanFineDistance = _meanFineDistance,
                    DomainMin = _domainMin,
                    DomainMax = _domainMax,
                    Disp = _surfDisp,
                }.Schedule(_surfCount.Value, 64).Complete();
                new ApplySurfaceDisplacementsJob { Disp = _surfDisp, SurfPos = _surfPos }.Schedule(_surfCount.Value, 64).Complete();

                // Skip BuildSurfBucket: displacements are tiny (~0.01*meanFineDistance), points rarely cross cells
                new ConstrainSurfaceJob
                {
                    CoarsePos = _particlePos,
                    CoarseHead = _coarseHead,
                    CoarseNext = _coarseNext,
                    OuterRadius = outerRadius,
                    InnerRadius = _innerRadius,
                    ConstraintA = _constraintA,
                    SurfPos = _surfPos,
                }.Schedule(_surfCount.Value, 64).Complete();

                // Skip BuildSurfBucket: constraint moves are small, interpolation uses stale bucket (acceptable)
                int newStart = _surfNewStart.Value;
                if (newStart < _surfCount.Value)
                {
                    new InterpolateNewWaveDataJob
                    {
                        SurfPos = _surfPos,
                        SurfHead = _surfHead,
                        SurfNext = _surfNext,
                        SurfH = _surfH,
                        SurfDtH = _surfDtH,
                        SurfSeed = _surfSeed,
                        SurfSeedAmp = _surfSeedAmp,
                        TangentRadius = _tangentRadius,
                        NewStart = newStart,
                    }.Schedule(_surfCount.Value - newStart, 64).Complete();
                }
            }
        }

        private void SurfaceWavesStep()
        {
            BuildSurfBucket();

            new AddSeedJob { SurfH = _surfH, SurfSeed = _surfSeed }.Schedule(_surfCount.Value, 64).Complete();

            new ComputeSurfaceWaveSlopeJob
            {
                SurfPos = _surfPos,
                SurfNormal = _surfNormal,
                SurfH = _surfH,
                SurfHead = _surfHead,
                SurfNext = _surfNext,
                TangentRadius = _tangentRadius,
                Slope = _surfWaveSlope,
            }.Schedule(_surfCount.Value, 64).Complete();

            new ComputeSurfaceWaveLaplaciansJob
            {
                SurfPos = _surfPos,
                SurfNormal = _surfNormal,
                SurfH = _surfH,
                Slope = _surfWaveSlope,
                SurfHead = _surfHead,
                SurfNext = _surfNext,
                TangentRadius = _tangentRadius,
                Laplacian = _surfLaplacian,
            }.Schedule(_surfCount.Value, 64).Complete();

            new EvolveWaveJob
            {
                Laplacian = _surfLaplacian,
                SurfSeed = _surfSeed,
                SurfH = _surfH,
                SurfDtH = _surfDtH,
                WaveSpeed = waveSpeed,
                WaveDamping = waveDamping,
                Dt = DeltaTime,
                WaveMaxAmplitude = waveMaxAmplitude,
                WaveMaxFrequency = waveMaxFrequency,
            }.Schedule(_surfCount.Value, 64).Complete();

            new ComputeSurfaceCurvatureJob
            {
                SurfPos = _surfPos,
                SurfNormal = _surfNormal,
                SurfHead = _surfHead,
                SurfNext = _surfNext,
                NormalRadius = _normalRadius,
                DomainMin = _domainMin,
                DomainMax = _domainMax,
                Curvature = _surfCurvature,
            }.Schedule(_surfCount.Value, 64).Complete();

            new SmoothCurvatureJob
            {
                SurfPos = _surfPos,
                Curvature = _surfCurvature,
                SurfHead = _surfHead,
                SurfNext = _surfNext,
                NormalRadius = _normalRadius,
                DomainMin = _domainMin,
                DomainMax = _domainMax,
                Source = _surfSource,
            }.Schedule(_surfCount.Value, 64).Complete();

            new SeedWavesJob
            {
                Source = _surfSource,
                SurfSeed = _surfSeed,
                SurfSeedAmp = _surfSeedAmp,
                WaveSeedFrequency = waveSeedFrequency,
                WaveSpeed = waveSpeed,
                WaveMaxAmplitude = waveMaxAmplitude,
                WaveMaxSeedingAmplitude = waveMaxSeedingAmplitude,
                ThresholdCenter = waveSeedingCurvatureThresholdRegionCenter,
                ThresholdRadius = waveSeedingCurvatureThresholdRegionRadius,
                StepSizeRatio = waveSeedStepSizeRatioOfMax,
                Dt = DeltaTime,
                FrameCount = _frameCount,
            }.Schedule(_surfCount.Value, 64).Complete();
        }

        private void SurfaceTurbulenceStep()
        {
            if (_surfCount.Value == 0) return;

            Profiler.BeginSample("Surface Turbulence");
            BuildPrevBucket();
            BuildCoarseBucket();

            new AdvectSurfacePointsJob
            {
                CoarsePos = _particlePos,
                CoarsePrev = _particlePosPrev,
                PrevHead = _prevHead,
                PrevNext = _prevNext,
                OuterRadius = outerRadius,
                SurfPos = _surfPos,
            }.Schedule(_surfCount.Value, 64).Complete();

            SurfaceMaintenance(nbSurfaceMaintenanceIterations);
            if (_surfCount.Value > 0)
                SurfaceWavesStep();
            Profiler.EndSample();

            _frameCount++;
        }

        private void DrawSurfacePoints()
        {
            if (surfMat == null || mesh == null || _surfCount.Value == 0) return;

            new WriteSurfBufferJob
            {
                SurfPos = _surfPos,
                SurfNormal = _surfNormal,
                SurfH = _surfH,
                Out = _surfRender,
            }.Schedule(_surfCount.Value, 64).Complete();

            _surfBuffer.SetData(_surfRender, 0, 0, _surfCount.Value);
            Graphics.DrawMeshInstancedProcedural(mesh, 0, surfMat, _bounds, _surfCount.Value);
        }

        private void DrawSurfaceDebugGizmos()
        {
            if (!drawSurfaceGizmos) return;
            int count = _surfCount.Value;
            for (int i = 0; i < count; i++)
            {
                float2 p = _surfPos[i];
                float2 n = _surfNormal[i];
                float h = _surfH[i];
                Vector3 pv = new Vector3(p.x * 0.1f, p.y * 0.1f, 0f);
                Vector3 nv = new Vector3(n.x, n.y, 0f) * 0.5f;
                Gizmos.color = h >= 0f ? Color.red : Color.blue;
                Gizmos.DrawSphere(pv + nv * h * 0.1f, 0.02f);
                Gizmos.DrawLine(pv, pv + nv * h * 0.1f);
            }
        }
    }
}

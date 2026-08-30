using Abecombe.GPUUtil;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace SurfaceWaves
{
    // =====================================================================
    // GPU surface turbulence (Mercier et al., SIGGRAPH Asia 2015), hosted on
    // top of the narrow-band FLIP solver that lives in the other partial.
    //
    // The full pipeline is wired: advection, regularization, add/delete with
    // over-density thinning, compaction, and wave seeding/evolution. The live point
    // count is GPU-resident (_surfCountBuf) and every per-point pass dispatches
    // indirectly off it, so the only CPU readbacks are the one at seeding and a
    // ~1 Hz diagnostic poll.
    //
    // Two deliberate deviations from the reference, both documented at their site:
    //   - one compaction per maintenance iteration instead of two, so new points
    //     are density-checked in the following iteration rather than the same one
    //   - the sort and prefix scan run at fixed capacity, which is what lets the
    //     count stay on the GPU
    //
    // Rendering is a debug point-sprite view (position + wave height as color), not
    // the displaced surface the paper builds.
    // =====================================================================
    public partial class FLIPWithSurfaceTurbulence
    {
        [Header("Surface Turbulence")]
        // enabled via surfaceDetail == SurfaceTurbulence on the main file
        public ComputeShader surfaceTurbulenceCs;
        public ComputeShader deviceRadixSortCs;
        public ComputeShader prefixScanCs; // Assets/MarchingCubesGPU/PrefixScan.compute
        private Material surfaceMat;
        public bool drawSurfacePoints = true;

        // 1 Hz diagnostic readback toggle. Off by default: the poll below is a
        // SYNCHRONOUS GetData on buffers the sim wrote this frame, so each poll
        // stalls the whole GPU pipeline — a once-a-second multi-ms hitch that
        // is easy to misread as "the simulation is slow". While off, the GUI
        // live-count / overflow numbers go stale (and capacity-saturation
        // warnings are silenced); nothing on the GPU path depends on them.
        public bool pollDiagnostics = false;

        // World units. outerRadius = 2 * GridSpacing puts the coarse kernel
        // support at the solver's particle spacing, like the 2D port does.
        public float outerRadius = 2f * GridSpacing;
        public int surfaceDensity = 20;
        [Range(0, 4)]
        public int nbSurfaceMaintenanceIterations = 2;

        // Over-density thinning (C++ hasNeighborOtherThanItself). Toggle to A/B
        // whether it is culling too aggressively.
        public bool enableDensityThinning = true;

        public float waveSpeed = 8f;
        public float waveDamping = 2f;
        public float waveSeedFrequency = 4f;
        public float waveMaxAmplitude = 0.1f;
        public float waveMaxFrequency = 400f;
        public float waveMaxSeedingAmplitude = 0.3f;
        public float waveSeedingCurvatureThresholdRegionCenter = 0.08f;
        public float waveSeedingCurvatureThresholdRegionRadius = 0.02f;
        public float waveSeedStepSizeRatioOfMax = 0.01f;

        // Capacity of every per-surface-point buffer. ~172 B/point once the
        // compaction ping-pong is counted, so 1<<17 is ~22 MB of point state (plus
        // ~86 MB of fixed overhead for _surfRange and _particlesPrev).
        //
        // NOTE: this is a serialized public field, so the value in the scene wins
        // over this default. Check the startup log for the figure actually in use.
        public int maxSurfacePoints = 1 << 17;

        // Hard ceiling on surface-turbulence VRAM. Exceeding it clamps
        // maxSurfacePoints with an error rather than letting a silent
        // ComputeBuffer allocation failure crash the graphics thread later.
        public int surfaceMemoryBudgetMB = 512;

        // ===== derived parameters (C++ initFines preamble) =====
        private float _innerRadius;
        private float _meanFineDistance;
        private float _constraintA;
        private float _normalRadius;
        private float _tangentRadius;
        private float3 _bndMin;
        private float3 _bndMax;

        // ===== surface point state =====
        // Double-buffered: compaction gathers survivors from Read into Write and
        // swaps. Read is always the live set.
        private readonly GPUDoubleBuffer<float3> _surfPos = new();
        private readonly GPUDoubleBuffer<float3> _surfNormal = new();
        private readonly GPUDoubleBuffer<float> _surfWaveH = new();
        private readonly GPUDoubleBuffer<float> _surfWaveDtH = new();
        private readonly GPUDoubleBuffer<float> _surfWaveSeed = new();
        private readonly GPUDoubleBuffer<float> _surfWaveSeedAmp = new();
        private readonly GPUDoubleBuffer<float> _surfWaveSource = new();

        private readonly GPUBuffer<float3> _surfTempVec3 = new();
        private readonly GPUBuffer<float> _surfTempFloat = new();

        private readonly GPUBuffer<uint> _surfAlive = new();
        private readonly GPUDoubleBuffer<uint> _surfStatus = new();
        private readonly GPUBuffer<uint> _surfKill = new();

        // ===== compaction / indirect-dispatch state =====
        private readonly GPUBuffer<uint> _scanBuf = new();
        private ComputeBuffer _scanTotal;
        private ComputeBuffer _survivorCount;
        private ComputeBuffer _scanPartitionCounter;
        private ComputeBuffer _scanPartitionDescriptors;

        // Live count, GPU-resident. The CPU never needs to read it.
        private ComputeBuffer _surfCountBuf;
        private ComputeBuffer _surfDispatchArgs;
        private ComputeBuffer _surfDrawArgs;

        // Diagnostics, polled off the hot path (see PollSurfaceDiagnostics).
        private ComputeBuffer _surfOverflow;
        private ComputeBuffer _seedReject;
        private readonly uint[] _diagScratch = new uint[1];
        private int _liveCountApprox;
        private uint _overflowTotal;
        private float _nextDiagPoll;

        private int _kScanInit, _kScanDecoupled;
        private const int ScanPartitionSize = 512;
        private int _scanNumPartitions;

        private readonly GPUBuffer<float3> _surfCandidatePos = new();
        private readonly GPUBuffer<uint> _surfHasCandidate = new();

        private readonly GPUBuffer<uint> _surfHash = new();
        private readonly GPUBuffer<uint> _surfID = new();
        private readonly GPUBuffer<uint2> _surfRange = new();

        private readonly GPUBuffer<float4> _surfRender = new();
        private ComputeBuffer _surfCounter;

        // pre-advection snapshot of the coarse particles; shares the index space
        // of _particles.Read, which is what AdvectSurfacePoints needs.
        private readonly GPUBuffer<Particle> _particlesPrev = new();

        // ===== radix sort scratch (Assets/Sort/DeviceRadixSort.compute) =====
        private const int RadixPartitionSize = 3840;
        private const int Radix = 256;
        private const int RadixPasses = 4;
        private readonly GPUBuffer<uint> _sortAltHash = new();
        private readonly GPUBuffer<uint> _sortAltID = new();
        private ComputeBuffer _sortGlobalHist;
        private ComputeBuffer _sortPassHist;

        private int _kSortInit, _kSortUpSweep, _kSortScan, _kSortDownSweep;

        private int _kClearSurfaceRange, _kMakeSurfacePair, _kSetSurfaceRange;
        private int _kInitSurfacePoints, _kAdvectSurfacePoints;
        private int _kComputeSurfaceNormals, _kComputeAveragedNormals, _kAssignNormals;
        private int _kComputeSurfaceDensities, _kComputeSurfaceDisplacements, _kApplySurfaceDisplacements;
        private int _kConstrainSurface, _kInterpolateNewWaveData;
        private int _kComputeAdditions, _kMarkDeletions;
        private int _kMarkDeletionsDensity, _kApplyDensityKills;
        private int _kSnapshotCoarse;
        private int _kStageAliveForScan, _kStageCandidatesForScan;
        private int _kGatherSurvivors, _kAppendCandidates, _kFinalizeSurfaceCount;
        private int _kResetAliveAfterCompaction, _kClearSurfaceStatus, _kWriteSurfaceDispatchArgs;
        private int _kAddSeed, _kComputeSurfaceWaveNormal, _kComputeSurfaceWaveLaplacians, _kEvolveWave;
        private int _kComputeSurfaceCurvature, _kSmoothCurvature, _kSeedWaves, _kDisplaceForRender;

        // Whether the buffers were actually allocated. Disposal must key off this,
        // not surfaceDetail: a failed init leaves the mode set, and toggling
        // modes at runtime would otherwise leak every buffer.
        private bool _surfaceInited;

        // Seeded count only. The live count is GPU-resident in _surfCountBuf; this
        // is kept for the "did seeding produce anything" guard and the GUI label.
        private int _surfaceCount;
        private int _stFrameCount;
        private MaterialPropertyBlock _surfMpb;

        // ===== lifecycle =====

        private void InitSurfaceTurbulence()
        {
            surfaceMat = new Material(Shader.Find("ParticleRendering/ParticleInstance"));
            // The scan reads whole partitions without a bound check
            // (PrefixScan.compute:50), and its total is the sum over every slot it
            // touches. Rounding capacity to a partition multiple means the staging
            // kernels can zero the entire buffer, so no padding tail can leak into
            // the total and no read runs off the end.
            int rounded = DivRoundUp(maxSurfacePoints, ScanPartitionSize) * ScanPartitionSize;
            if (rounded != maxSurfacePoints)
            {
                Debug.Log($"Surface turbulence: maxSurfacePoints {maxSurfacePoints} rounded up to " +
                          $"{rounded} (prefix-scan partition multiple).");
                maxSurfacePoints = rounded;
            }
            _scanNumPartitions = maxSurfacePoints / ScanPartitionSize;

            // Refuse to allocate past the budget. A failed ComputeBuffer allocation
            // is SILENT — the crash surfaces later as BufferD3D12::BeginWrite when
            // SetData writes into the invalid handle, which points nowhere near the
            // real cause. Clamping here turns that into a readable message.
            long projected = ProjectedSurfaceBytes();
            long budget = (long)surfaceMemoryBudgetMB * 1024 * 1024;
            if (projected > budget)
            {
                int affordable = (int)((budget
                                        - (long)NumGrids * 8
                                        - (long)ParticlesBufferSize * 16)
                                       / BytesPerSurfacePoint);
                affordable = Mathf.Max(ScanPartitionSize,
                    affordable / ScanPartitionSize * ScanPartitionSize);
                Debug.LogError(
                    $"Surface turbulence: maxSurfacePoints={maxSurfacePoints} needs " +
                    $"{projected / (1024 * 1024)} MB, over the {surfaceMemoryBudgetMB} MB budget. " +
                    $"Clamping to {affordable} ({ProjectedBytesFor(affordable) / (1024 * 1024)} MB). " +
                    $"Raise surfaceMemoryBudgetMB if you have the VRAM.");
                maxSurfacePoints = affordable;
                _scanNumPartitions = maxSurfacePoints / ScanPartitionSize;
            }
            else
            {
                Debug.Log($"Surface turbulence: maxSurfacePoints={maxSurfacePoints}, " +
                          $"~{projected / (1024 * 1024)} MB of buffers.");
            }

            _innerRadius = outerRadius * 0.5f;
            _meanFineDistance = math.PI * (outerRadius + _innerRadius) / surfaceDensity;
            _constraintA = math.log(2f / (1f + ExponentialWeight(outerRadius + _innerRadius, outerRadius, 2f)))
                           / (math.pow((outerRadius + _innerRadius) * 0.5f, 2f) - _innerRadius * _innerRadius);
            _normalRadius = 0.5f * (outerRadius + _innerRadius);
            _tangentRadius = 2.1f * _meanFineDistance;
            _bndMin = 2f * GridSpacing;
            _bndMax = (float3)GridSize * GridSpacing - 2f * GridSpacing;

            _surfPos.Init(maxSurfacePoints);
            _surfNormal.Init(maxSurfacePoints);
            _surfWaveH.Init(maxSurfacePoints);
            _surfWaveDtH.Init(maxSurfacePoints);
            _surfWaveSeed.Init(maxSurfacePoints);
            _surfWaveSeedAmp.Init(maxSurfacePoints);
            _surfWaveSource.Init(maxSurfacePoints);
            _surfTempVec3.Init(maxSurfacePoints);
            _surfTempFloat.Init(maxSurfacePoints);
            _surfAlive.Init(maxSurfacePoints);
            _surfStatus.Init(maxSurfacePoints);
            _surfKill.Init(maxSurfacePoints);
            _surfCandidatePos.Init(maxSurfacePoints);
            _surfHasCandidate.Init(maxSurfacePoints);
            _surfHash.Init(maxSurfacePoints);
            _surfID.Init(maxSurfacePoints);
            _surfRange.Init(NumGrids);
            _surfRender.Init(maxSurfacePoints);
            _particlesPrev.Init(ParticlesBufferSize);

            // Seeding-only. InitSurfacePoints is the sole kernel that appends
            // through this atomic counter, and it runs once. Per-frame appends go
            // through the prefix scan instead (CompactSurfacePoints), so this never
            // needs resetting.
            _surfCounter = new ComputeBuffer(1, sizeof(uint));
            _surfCounter.SetData(new uint[] { 0 });

            _scanBuf.Init(maxSurfacePoints);
            _scanTotal = new ComputeBuffer(1, sizeof(uint));
            _survivorCount = new ComputeBuffer(1, sizeof(uint));
            _scanPartitionCounter = new ComputeBuffer(1, sizeof(uint));
            // +1: the scan broadcasts into slot pid+1 (PrefixScan.compute:72)
            _scanPartitionDescriptors = new ComputeBuffer(_scanNumPartitions + 1, sizeof(uint));

            _surfCountBuf = new ComputeBuffer(1, sizeof(uint));
            _surfCountBuf.SetData(new uint[] { 0 });
            _surfOverflow = new ComputeBuffer(1, sizeof(uint));
            _surfOverflow.SetData(new uint[] { 0 });
            _seedReject = new ComputeBuffer(5, sizeof(uint));
            _seedReject.SetData(new uint[5]);
            _surfDispatchArgs = new ComputeBuffer(3, sizeof(uint), ComputeBufferType.IndirectArguments);
            _surfDispatchArgs.SetData(new uint[] { 0, 1, 1 });
            _surfDrawArgs = new ComputeBuffer(5, sizeof(uint), ComputeBufferType.IndirectArguments);
            _surfDrawArgs.SetData(new uint[] { 0, 1, 0, 0, 0 });

            _kScanInit = prefixScanCs.FindKernel("Init");
            _kScanDecoupled = prefixScanCs.FindKernel("DecoupledLookbackScan");

            int maxBlocks = DivRoundUp(maxSurfacePoints, RadixPartitionSize);
            _sortAltHash.Init(maxSurfacePoints);
            _sortAltID.Init(maxSurfacePoints);
            _sortGlobalHist = new ComputeBuffer(Radix * RadixPasses, sizeof(uint));
            _sortPassHist = new ComputeBuffer(Radix * maxBlocks * RadixPasses, sizeof(uint));

            var cs = surfaceTurbulenceCs;
            _kClearSurfaceRange = cs.FindKernel("ClearSurfaceRange");
            _kMakeSurfacePair = cs.FindKernel("MakeSurfacePair");
            _kSetSurfaceRange = cs.FindKernel("SetSurfaceRange");
            _kInitSurfacePoints = cs.FindKernel("InitSurfacePoints");
            _kAdvectSurfacePoints = cs.FindKernel("AdvectSurfacePoints");
            _kComputeSurfaceNormals = cs.FindKernel("ComputeSurfaceNormals");
            _kComputeAveragedNormals = cs.FindKernel("ComputeAveragedNormals");
            _kAssignNormals = cs.FindKernel("AssignNormals");
            _kComputeSurfaceDensities = cs.FindKernel("ComputeSurfaceDensities");
            _kComputeSurfaceDisplacements = cs.FindKernel("ComputeSurfaceDisplacements");
            _kApplySurfaceDisplacements = cs.FindKernel("ApplySurfaceDisplacements");
            _kConstrainSurface = cs.FindKernel("ConstrainSurface");
            _kInterpolateNewWaveData = cs.FindKernel("InterpolateNewWaveData");
            _kComputeAdditions = cs.FindKernel("ComputeAdditions");
            _kMarkDeletions = cs.FindKernel("MarkDeletions");
            _kMarkDeletionsDensity = cs.FindKernel("MarkDeletionsDensity");
            _kApplyDensityKills = cs.FindKernel("ApplyDensityKills");
            _kSnapshotCoarse = cs.FindKernel("SnapshotCoarse");
            _kStageAliveForScan = cs.FindKernel("StageAliveForScan");
            _kStageCandidatesForScan = cs.FindKernel("StageCandidatesForScan");
            _kGatherSurvivors = cs.FindKernel("GatherSurvivors");
            _kAppendCandidates = cs.FindKernel("AppendCandidates");
            _kFinalizeSurfaceCount = cs.FindKernel("FinalizeSurfaceCount");
            _kResetAliveAfterCompaction = cs.FindKernel("ResetAliveAfterCompaction");
            _kClearSurfaceStatus = cs.FindKernel("ClearSurfaceStatus");
            _kWriteSurfaceDispatchArgs = cs.FindKernel("WriteSurfaceDispatchArgs");
            _kAddSeed = cs.FindKernel("AddSeed");
            _kComputeSurfaceWaveNormal = cs.FindKernel("ComputeSurfaceWaveNormal");
            _kComputeSurfaceWaveLaplacians = cs.FindKernel("ComputeSurfaceWaveLaplacians");
            _kEvolveWave = cs.FindKernel("EvolveWave");
            _kComputeSurfaceCurvature = cs.FindKernel("ComputeSurfaceCurvature");
            _kSmoothCurvature = cs.FindKernel("SmoothCurvature");
            _kSeedWaves = cs.FindKernel("SeedWaves");
            _kDisplaceForRender = cs.FindKernel("DisplaceForRender");

            _kSortInit = deviceRadixSortCs.FindKernel("InitDeviceRadixSort");
            _kSortUpSweep = deviceRadixSortCs.FindKernel("UpSweep");
            _kSortScan = deviceRadixSortCs.FindKernel("Scan");
            _kSortDownSweep = deviceRadixSortCs.FindKernel("DownSweep");

            _surfMpb = new MaterialPropertyBlock();
            _surfMpb.SetBuffer("_ParticleRenderingBuffer", _surfRender);
            _surfMpb.SetFloat("_Radius", _meanFineDistance);
            _surfMpb.SetFloat("_NearClipPlane", _cam.nearClipPlane);
            _surfMpb.SetFloat("_FarClipPlane", _cam.farClipPlane);
            _surfMpb.SetVector("_SlowColor", new Color(0.5f, 0, 0, 1f));
            _surfMpb.SetVector("_FastColor", new Color(0, 0.5f, 0.92f, 1f));
            // _SurfaceRender.w is the wave height, which the particle shader feeds
            // through _VelocityRange to pick a color: troughs blue, crests light.
            _surfMpb.SetVector("_VelocityRange", new Vector2(-waveMaxAmplitude, waveMaxAmplitude));
            _surfMpb.SetFloat("_FresnelPower", 0.3f);

            _surfaceInited = true;
            SeedSurfacePoints();
        }

        // Per-surface-point bytes across every buffer sized by maxSurfacePoints.
        // Double-buffered (compaction ping-pong): pos 12*2, normal 12*2, waveH 4*2,
        // waveDtH 4*2, waveSeed 4*2, waveSeedAmp 4*2, waveSource 4*2, status 4*2
        //   = 48*2 = 96
        // Single: tempVec3 12, tempFloat 4, alive 4, kill 4, candidatePos 12,
        //   hasCandidate 4, hash 4, id 4, render 16, scanBuf 4, sortAltHash 4,
        //   sortAltID 4 = 76
        private const int BytesPerSurfacePoint = 96 + 76;

        private long ProjectedBytesFor(int points)
        {
            return (long)points * BytesPerSurfacePoint
                   + (long)NumGrids * 8
                   + (long)ParticlesBufferSize * 16
                   + (long)Radix * DivRoundUp(points, RadixPartitionSize) * RadixPasses * 4;
        }

        private long ProjectedSurfaceBytes()
        {
            long perPoint = (long)maxSurfacePoints * BytesPerSurfacePoint;
            long range = (long)NumGrids * 8;                       // _surfRange uint2
            long prevParticles = (long)ParticlesBufferSize * 16;    // _particlesPrev
            long sortHist = (long)Radix * DivRoundUp(maxSurfacePoints, RadixPartitionSize)
                            * RadixPasses * 4;
            return perPoint + range + prevParticles + sortHist;
        }

        private static float ExponentialWeight(float d, float r, float falloff)
        {
            if (d > r) return 0f;
            float t = d / r;
            return math.exp(-falloff * t * t);
        }

        private void DisposeSurfaceTurbulence()
        {
            if (!_surfaceInited) return;
            _surfaceInited = false;

            _surfPos.Dispose();
            _surfNormal.Dispose();
            _surfWaveH.Dispose();
            _surfWaveDtH.Dispose();
            _surfWaveSeed.Dispose();
            _surfWaveSeedAmp.Dispose();
            _surfWaveSource.Dispose();
            _surfTempVec3.Dispose();
            _surfTempFloat.Dispose();
            _surfAlive.Dispose();
            _surfStatus.Dispose();
            _surfKill.Dispose();
            _surfCandidatePos.Dispose();
            _surfHasCandidate.Dispose();
            _surfHash.Dispose();
            _surfID.Dispose();
            _surfRange.Dispose();
            _surfRender.Dispose();
            _particlesPrev.Dispose();

            _sortAltHash.Dispose();
            _sortAltID.Dispose();
            _scanBuf.Dispose();

            _scanTotal?.Dispose();
            _survivorCount?.Dispose();
            _scanPartitionCounter?.Dispose();
            _scanPartitionDescriptors?.Dispose();
            _surfCountBuf?.Dispose();
            _surfDispatchArgs?.Dispose();
            _surfDrawArgs?.Dispose();
            _surfOverflow?.Dispose();
            _seedReject?.Dispose();

            _surfCounter?.Dispose();
            _sortGlobalHist?.Dispose();
            _sortPassHist?.Dispose();
        }
        // ===== parameter / buffer binding =====

        private void SetSurfaceParams(CommandBuffer cmd)
        {
            var cs = surfaceTurbulenceCs;
            SetParams(cmd, cs); // _GridSize / _CellSize / _InvCellSize / _DeltaTime / ...

            cmd.SetComputeFloatParam(cs, "_OuterRadius", outerRadius);
            cmd.SetComputeFloatParam(cs, "_InnerRadius", _innerRadius);
            cmd.SetComputeFloatParam(cs, "_MeanFineDistance", _meanFineDistance);
            cmd.SetComputeFloatParam(cs, "_ConstraintA", _constraintA);
            cmd.SetComputeFloatParam(cs, "_NormalRadius", _normalRadius);
            cmd.SetComputeFloatParam(cs, "_TangentRadius", _tangentRadius);

            cmd.SetComputeFloatParam(cs, "_WaveSpeed", waveSpeed);
            cmd.SetComputeFloatParam(cs, "_WaveDamping", waveDamping);
            cmd.SetComputeFloatParam(cs, "_WaveSeedFrequency", waveSeedFrequency);
            cmd.SetComputeFloatParam(cs, "_WaveMaxAmplitude", waveMaxAmplitude);
            cmd.SetComputeFloatParam(cs, "_WaveMaxFrequency", waveMaxFrequency);
            cmd.SetComputeFloatParam(cs, "_WaveMaxSeedingAmplitude", waveMaxSeedingAmplitude);
            cmd.SetComputeFloatParam(cs, "_SeedThresholdCenter", waveSeedingCurvatureThresholdRegionCenter);
            cmd.SetComputeFloatParam(cs, "_SeedThresholdRadius", waveSeedingCurvatureThresholdRegionRadius);
            cmd.SetComputeFloatParam(cs, "_SeedStepSizeRatio", waveSeedStepSizeRatioOfMax);

            cmd.SetComputeIntParam(cs, "_FrameCount", _stFrameCount);
            cmd.SetComputeIntParam(cs, "_MaxSurfacePoints", maxSurfacePoints);
            // _SurfaceCount is no longer a uniform — it is a macro over
            // _SurfaceCountBuf so compaction can update it GPU-side.
            cmd.SetComputeVectorParam(cs, "_BndMin", new Vector4(_bndMin.x, _bndMin.y, _bndMin.z, 0f));
            cmd.SetComputeVectorParam(cs, "_BndMax", new Vector4(_bndMax.x, _bndMax.y, _bndMax.z, 0f));
        }

        // Binds every buffer the surface shader declares. SetComputeBufferParam is
        // a no-op for names a kernel does not reference, so one helper covers all
        // kernels and keeps the dispatch sites readable.
        //
        // outIsRead: AppendCandidates runs AFTER the gather swap, so its `*Out`
        // targets must be the buffers the gather just filled — which by then are
        // Read, not Write. Every other kernel wants Write.
        private void BindSurfaceBuffers(CommandBuffer cmd, int kernel, bool outIsRead = false)
        {
            var cs = surfaceTurbulenceCs;

            // Both range params point at the same LUT: the surface step runs after
            // G2P while _gridParticleRange still holds the pre-advection layout,
            // which is exactly the layout _particlesPrev was snapshotted in.
            cmd.SetComputeBufferParam(cs, kernel, "_CoarseParticles", _particles.Read);
            cmd.SetComputeBufferParam(cs, kernel, "_CoarsePrevParticles", _particlesPrev);
            cmd.SetComputeBufferParam(cs, kernel, "_CoarseRange", _gridParticleRange.Read);
            cmd.SetComputeBufferParam(cs, kernel, "_CoarsePrevRange", _gridParticleRange.Read);

            cmd.SetComputeBufferParam(cs, kernel, "_SurfacePos", _surfPos.Read);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceNormal", _surfNormal.Read);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceWaveH", _surfWaveH.Read);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceWaveDtH", _surfWaveDtH.Read);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceWaveSeed", _surfWaveSeed.Read);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceWaveSeedAmp", _surfWaveSeedAmp.Read);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceWaveSource", _surfWaveSource.Read);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceStatus", _surfStatus.Read);

            cmd.SetComputeBufferParam(cs, kernel, "_SurfacePosOut", outIsRead ? _surfPos.Read : _surfPos.Write);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceNormalOut", outIsRead ? _surfNormal.Read : _surfNormal.Write);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceWaveHOut", outIsRead ? _surfWaveH.Read : _surfWaveH.Write);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceWaveDtHOut", outIsRead ? _surfWaveDtH.Read : _surfWaveDtH.Write);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceWaveSeedOut", outIsRead ? _surfWaveSeed.Read : _surfWaveSeed.Write);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceWaveSeedAmpOut", outIsRead ? _surfWaveSeedAmp.Read : _surfWaveSeedAmp.Write);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceWaveSourceOut", outIsRead ? _surfWaveSource.Read : _surfWaveSource.Write);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceStatusOut", outIsRead ? _surfStatus.Read : _surfStatus.Write);

            cmd.SetComputeBufferParam(cs, kernel, "_TempVec3", _surfTempVec3);
            cmd.SetComputeBufferParam(cs, kernel, "_TempFloat", _surfTempFloat);

            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceAlive", _surfAlive);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceKill", _surfKill);

            cmd.SetComputeBufferParam(cs, kernel, "_ScanBuf", _scanBuf);
            cmd.SetComputeBufferParam(cs, kernel, "_ScanTotal", _scanTotal);
            cmd.SetComputeBufferParam(cs, kernel, "_SurvivorCount", _survivorCount);

            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceCountBuf", _surfCountBuf);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceOverflow", _surfOverflow);
            cmd.SetComputeBufferParam(cs, kernel, "_SeedReject", _seedReject);

            // Narrow-band SDF, for IsDeepInterior. Same texture the solver's
            // GetType reads, so the Band1 threshold means the same thing here.
            cmd.SetComputeTextureParam(cs, kernel, "_GridSDFR", _gridSDF);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceDispatchArgs", _surfDispatchArgs);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceDrawArgs", _surfDrawArgs);

            cmd.SetComputeBufferParam(cs, kernel, "_CandidatePos", _surfCandidatePos);
            cmd.SetComputeBufferParam(cs, kernel, "_HasCandidate", _surfHasCandidate);

            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceHash", _surfHash);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceID", _surfID);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceRange", _surfRange);

            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceRender", _surfRender);
            cmd.SetComputeBufferParam(cs, kernel, "_SurfaceCounter", _surfCounter);
        }

        // Per-point kernel over the live set. Group count comes from
        // _surfDispatchArgs, written on the GPU, so the CPU never needs the count.
        private void DispatchSurfaceIndirect(CommandBuffer cmd, int kernel, bool outIsRead = false)
        {
            BindSurfaceBuffers(cmd, kernel, outIsRead);
            cmd.DispatchCompute(surfaceTurbulenceCs, kernel, _surfDispatchArgs, 0);
        }

        // Fixed-size pass (capacity or grid). Count is a compile-time-known CPU
        // value, so a direct dispatch is correct and cheaper than a round trip.
        private void DispatchSurfaceDirect(CommandBuffer cmd, int kernel, int groups)
        {
            if (groups <= 0) return;
            BindSurfaceBuffers(cmd, kernel);
            cmd.DispatchCompute(surfaceTurbulenceCs, kernel, groups, 1, 1);
        }

        private int CapGroups => DivRoundUp(maxSurfacePoints, 128);

        // ===== surface neighbor bucket =====
        //
        // ClearSurfaceRange -> MakeSurfacePair -> DeviceRadixSort(hash, id) ->
        // SetSurfaceRange, which is the layout the shader's LOOP_SURFACE_NEIGHBORS
        // macro expects (range indexes the sorted id payload, surface records stay
        // put so ids remain stable).
        private void BuildSurfaceBucket(CommandBuffer cmd)
        {
            cmd.BeginSample("ST.BuildBucket");
            BuildSurfaceBucketInner(cmd);
            cmd.EndSample("ST.BuildBucket");
        }

        private void BuildSurfaceBucketInner(CommandBuffer cmd)
        {
            DispatchSurfaceDirect(cmd, _kClearSurfaceRange, DivRoundUp(NumGrids, 128));
            DispatchSurfaceDirect(cmd, _kMakeSurfacePair, CapGroups);
            SortSurfacePairs(cmd);
            DispatchSurfaceDirect(cmd, _kSetSurfaceRange, CapGroups);
        }

        // Sorts the FULL capacity, not the live count. DeviceRadixSort takes its key
        // count and thread-block count as CPU-side values, so sizing it to the live
        // count would force a readback and defeat the indirect dispatch. Slots past
        // the count hash to SURFACE_HASH_INVALID and sort to the tail, which
        // SetSurfaceRange skips. Cost is a fixed 4-pass sort over maxSurfacePoints
        // per bucket build; a conservative high-water mark could shrink it later.
        private void SortSurfacePairs(CommandBuffer cmd)
        {
            var cs = deviceRadixSortCs;
            int numKeys = maxSurfacePoints;
            int threadBlocks = DivRoundUp(numKeys, RadixPartitionSize);

            cmd.SetComputeIntParam(cs, "e_numKeys", numKeys);
            cmd.SetComputeIntParam(cs, "e_threadBlocks", threadBlocks);

            cmd.SetComputeBufferParam(cs, _kSortInit, "b_globalHist", _sortGlobalHist);
            cmd.SetComputeBufferParam(cs, _kSortUpSweep, "b_passHist", _sortPassHist);
            cmd.SetComputeBufferParam(cs, _kSortUpSweep, "b_globalHist", _sortGlobalHist);
            cmd.SetComputeBufferParam(cs, _kSortScan, "b_passHist", _sortPassHist);
            cmd.SetComputeBufferParam(cs, _kSortDownSweep, "b_passHist", _sortPassHist);
            cmd.SetComputeBufferParam(cs, _kSortDownSweep, "b_globalHist", _sortGlobalHist);

            cmd.DispatchCompute(cs, _kSortInit, 1, 1, 1);

            GraphicsBuffer keys = _surfHash;
            GraphicsBuffer payload = _surfID;
            GraphicsBuffer altKeys = _sortAltHash;
            GraphicsBuffer altPayload = _sortAltID;

            // 4 LSD passes over a 32-bit key: even pass count, so the sorted result
            // lands back in _surfHash / _surfID.
            for (int radixShift = 0; radixShift < 32; radixShift += 8)
            {
                cmd.SetComputeIntParam(cs, "e_radixShift", radixShift);

                cmd.SetComputeBufferParam(cs, _kSortUpSweep, "b_sort", keys);
                cmd.DispatchCompute(cs, _kSortUpSweep, threadBlocks, 1, 1);

                cmd.DispatchCompute(cs, _kSortScan, Radix, 1, 1);

                cmd.SetComputeBufferParam(cs, _kSortDownSweep, "b_sort", keys);
                cmd.SetComputeBufferParam(cs, _kSortDownSweep, "b_sortPayload", payload);
                cmd.SetComputeBufferParam(cs, _kSortDownSweep, "b_alt", altKeys);
                cmd.SetComputeBufferParam(cs, _kSortDownSweep, "b_altPayload", altPayload);
                cmd.DispatchCompute(cs, _kSortDownSweep, threadBlocks, 1, 1);

                (keys, altKeys) = (altKeys, keys);
                (payload, altPayload) = (altPayload, payload);
            }
        }

        // ===== prefix scan (Assets/MarchingCubesGPU/PrefixScan.compute) =====
        //
        // Turns _scanBuf into exclusive destination slots in place and writes the
        // total to totalBuf. Runs at fixed capacity: numPartitions must equal the
        // dispatch group count exactly, because the total is written by the group
        // that draws pid == numPartitions - 1 (PrefixScan.compute:114).
        private void ScanFlags(CommandBuffer cmd, ComputeBuffer totalBuf)
        {
            var cs = prefixScanCs;
            cmd.SetComputeIntParam(cs, "numElements", maxSurfacePoints);
            cmd.SetComputeIntParam(cs, "numPartitions", _scanNumPartitions);

            cmd.SetComputeBufferParam(cs, _kScanInit, "partitionCounter", _scanPartitionCounter);
            cmd.SetComputeBufferParam(cs, _kScanInit, "partitionDescriptors", _scanPartitionDescriptors);
            cmd.DispatchCompute(cs, _kScanInit, DivRoundUp(_scanNumPartitions, 128), 1, 1);

            cmd.SetComputeBufferParam(cs, _kScanDecoupled, "toScan", _scanBuf);
            cmd.SetComputeBufferParam(cs, _kScanDecoupled, "totalVertCount", totalBuf);
            cmd.SetComputeBufferParam(cs, _kScanDecoupled, "partitionCounter", _scanPartitionCounter);
            cmd.SetComputeBufferParam(cs, _kScanDecoupled, "partitionDescriptors", _scanPartitionDescriptors);
            cmd.DispatchCompute(cs, _kScanDecoupled, _scanNumPartitions, 1, 1);
        }

        // ===== compaction (C++ doCompress + insertBufferedParticles) =====
        private void CompactSurfacePoints(CommandBuffer cmd)
        {
            cmd.BeginSample("SurfaceCompaction");

            // survivors: alive flags -> destination slots
            DispatchSurfaceDirect(cmd, _kStageAliveForScan, CapGroups);
            ScanFlags(cmd, _survivorCount);
            DispatchSurfaceIndirect(cmd, _kGatherSurvivors);

            _surfPos.Swap();
            _surfNormal.Swap();
            _surfWaveH.Swap();
            _surfWaveDtH.Swap();
            _surfWaveSeed.Swap();
            _surfWaveSeedAmp.Swap();
            _surfWaveSource.Swap();
            _surfStatus.Swap();

            // candidates: appended after the survivors. outIsRead because the
            // gather's output is now Read.
            DispatchSurfaceDirect(cmd, _kStageCandidatesForScan, CapGroups);
            ScanFlags(cmd, _scanTotal);
            DispatchSurfaceIndirect(cmd, _kAppendCandidates, outIsRead: true);

            // count := survivors + appended, then refresh the indirect args. Both
            // single-threaded; still dispatched so the value never leaves the GPU.
            DispatchSurfaceDirect(cmd, _kFinalizeSurfaceCount, 1);
            DispatchSurfaceDirect(cmd, _kWriteSurfaceDispatchArgs, 1);

            // everything below the new count is alive; clear the staging flags
            DispatchSurfaceDirect(cmd, _kResetAliveAfterCompaction, CapGroups);

            cmd.EndSample("SurfaceCompaction");
        }

        // ===== seeding (C++ initFines) =====

        private void SeedSurfacePoints()
        {
            var cmd = CommandBufferPool.Get("SurfaceTurbulence.Seed");
            cmd.Clear();

            _surfaceCount = 0;

            // Depends on InitParticles having run: it fills _particlesCount and
            // leaves _gridParticleRange.Read / _particles.Read consistent, which
            // is the coarse bucket InitSurfacePoints walks.
            if (_particlesCount[0] <= 0)
            {
                Debug.LogError("Surface turbulence: no live coarse particles at seeding time. " +
                               "InitSurfaceTurbulence must run after InitParticles.");
                CommandBufferPool.Release(cmd);
                return;
            }

            SetSurfaceParams(cmd);

            // InitSurfacePoints walks the LIVE coarse particles, not the whole
            // capacity that SetParams advertises.
            cmd.SetComputeIntParam(surfaceTurbulenceCs, "_NumParticles", _particlesCount[0]);

            BindSurfaceBuffers(cmd, _kInitSurfacePoints);
            cmd.DispatchCompute(surfaceTurbulenceCs, _kInitSurfacePoints,
                DivRoundUp(_particlesCount[0], 128), 1, 1);

            Graphics.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);

            // The only readback in the system, and only because InitSurfacePoints
            // appends through an atomic counter whose result has to be published to
            // _surfCountBuf. From here on the count stays on the GPU.
            var counter = new uint[1];
            _surfCounter.GetData(counter);
            int seeded = (int)counter[0];
            _surfaceCount = Mathf.Min(seeded, maxSurfacePoints);

            if (seeded > maxSurfacePoints)
                Debug.LogWarning($"Surface turbulence: seeding produced {seeded} points, " +
                                 $"clamped to maxSurfacePoints={maxSurfacePoints}. Raise the capacity " +
                                 $"or lower surfaceDensity.");

            _surfCountBuf.SetData(new uint[] { (uint)_surfaceCount });

            if (_surfaceCount > 0)
            {
                cmd = CommandBufferPool.Get("SurfaceTurbulence.SeedFinish");
                cmd.Clear();
                SetSurfaceParams(cmd);

                // Derive the indirect args from the freshly published count, before
                // anything dispatches off them.
                DispatchSurfaceDirect(cmd, _kWriteSurfaceDispatchArgs, 1);

                // Seeding marks every point PNEW. Clear it here so the first frame's
                // InterpolateNewWaveData does not blank the whole set; from then on
                // AppendCandidates sets it and ClearSurfaceStatus retires it each
                // frame.
                DispatchSurfaceDirect(cmd, _kClearSurfaceStatus, CapGroups);

                // Alive flags start unset for slots seeding skipped.
                DispatchSurfaceDirect(cmd, _kResetAliveAfterCompaction, CapGroups);

                // Fill the render buffer so the points are visible before the first
                // step runs (e.g. if the sim starts paused).
                DispatchSurfaceIndirect(cmd, _kDisplaceForRender);

                Graphics.ExecuteCommandBuffer(cmd);
                cmd.Clear();
                CommandBufferPool.Release(cmd);
            }

            Debug.Log($"Surface turbulence: seeded {_surfaceCount} points " +
                      $"(capacity {maxSurfacePoints}), meanFineDistance={_meanFineDistance:F4}, " +
                      $"outerRadius={outerRadius:F3}, normalRadius={_normalRadius:F3}, " +
                      $"tangentRadius={_tangentRadius:F4}, constraintA={_constraintA:F4}");

            // Attribute the outcome to a rejection stage. The interesting ratio is
            // notNearSurface vs the live coarse count: if almost every particle is
            // rejected there, the problem is the nearSurface proxy, not the sphere
            // sampling below it.
            var rej = new uint[5];
            _seedReject.GetData(rej);
            Debug.Log($"Surface turbulence seeding stages: " +
                      $"coarse={_particlesCount[0]}, notNearSurface={rej[0]}, " +
                      $"nearSurface={(uint)_particlesCount[0] - rej[0]}, " +
                      $"candidates={rej[1] + rej[2] + rej[3] + rej[4]} " +
                      $"(outOfDomain={rej[1]}, bandInterior={rej[4]}, " +
                      $"occluded={rej[2]}, accepted={rej[3]})");
        }
        // ===== per-frame step (C++ surfaceTurbulence) =====

        private void SurfaceTurbulenceStep(CommandBuffer cmd)
        {
            // _surfaceCount is the SEEDED count and never updates after that — the
            // live count lives in _surfCountBuf. This is only a "did seeding produce
            // anything" guard; if the live count later reaches zero the indirect
            // dispatches simply issue zero groups.
            if (_surfaceDetail != SurfaceDetailMode.SurfaceTurbulence || _surfaceCount <= 0) return;

            cmd.BeginSample("SurfaceTurbulence");
            SetSurfaceParams(cmd);

            // advection uses (curPos - prevPos) of the coarse neighbors
            cmd.BeginSample("ST.Advect");
            DispatchSurfaceIndirect(cmd, _kAdvectSurfacePoints);
            cmd.EndSample("ST.Advect");

            cmd.BeginSample("ST.Maintenance");
            SurfaceMaintenance(cmd, nbSurfaceMaintenanceIterations);
            cmd.EndSample("ST.Maintenance");

            cmd.BeginSample("ST.Waves");
            SurfaceWavesStep(cmd);
            cmd.EndSample("ST.Waves");

            DispatchSurfaceIndirect(cmd, _kDisplaceForRender);

            cmd.EndSample("SurfaceTurbulence");
            _stFrameCount++;
        }

        // C++ regularizeSurfacePoints + addDeleteSurfacePoints, iterated.
        private void SurfaceMaintenance(CommandBuffer cmd, int iterations)
        {
            for (int it = 0; it < iterations; it++)
            {
                BuildSurfaceBucket(cmd);

                // Candidate creation positions, consumed by CompactSurfacePoints
                // below. Candidates are mutually blind by design — the C++ buffers
                // them and only inserts after the loop (surfaceturbulence.cpp:599),
                // so two may land closer than meanFineDistance; the next
                // iteration's density pass cleans that up.
                //
                // Only on the last iteration: _HasCandidate is overwritten wholesale
                // each time this runs, so every earlier iteration's output was
                // discarded unread. Costs ~600 coarse visits per point per
                // iteration for nothing.
                if (it == iterations - 1)
                    DispatchSurfaceIndirect(cmd, _kComputeAdditions);

                // Over-density thinning, staged into _SurfaceKill. Must run BEFORE
                // MarkDeletions: the C++ ORs the density and domain criteria in one
                // index-ordered loop, so a point that is itself doomed by the
                // domain test is still alive when it kills its lower-index
                // neighbor. Applying domain kills first would spare those.
                if (enableDensityThinning)
                    DispatchSurfaceIndirect(cmd, _kMarkDeletionsDensity);

                // Per-point criteria: out-of-domain, no coarse neighbor in the
                // advection radius, off the constraint level set.
                DispatchSurfaceIndirect(cmd, _kMarkDeletions);

                // Fold the staged density kills in. Separated from the marking pass
                // because that one reads neighbors' _SurfaceAlive — mutating it in
                // flight would race and break the equivalence with the C++ cascade.
                if (enableDensityThinning)
                    DispatchSurfaceIndirect(cmd, _kApplyDensityKills);

                // Repack survivors and append the candidates.
                //
                // DEVIATION: the C++ compacts TWICE per iteration — once between
                // the add and delete halves (surfaceturbulence.cpp:605) and once at
                // the end (:639) — so a freshly inserted point is density-checked in
                // the same iteration that created it. Here the single compaction
                // runs after the deletion marks, so new points are not checked until
                // the NEXT iteration. They do get checked (maintenance iterates), so
                // the point set is transiently denser than the reference rather than
                // wrong. Matching it exactly would mean two compactions plus two
                // extra bucket rebuilds per iteration, roughly doubling the cost of
                // the most expensive part of the frame.
                CompactSurfacePoints(cmd);

                // Compaction renumbers every point, so the sorted id payload the
                // bucket holds is stale. The geometry passes below need neighbor
                // queries, hence a second build per iteration.
                BuildSurfaceBucket(cmd);

                // normals: fit, smooth, assign (temp vec3 carries the smoothed set)
                DispatchSurfaceIndirect(cmd, _kComputeSurfaceNormals);
                DispatchSurfaceIndirect(cmd, _kComputeAveragedNormals);
                DispatchSurfaceIndirect(cmd, _kAssignNormals);

                // regularization: density -> displacement -> apply
                DispatchSurfaceIndirect(cmd, _kComputeSurfaceDensities);
                DispatchSurfaceIndirect(cmd, _kComputeSurfaceDisplacements);
                DispatchSurfaceIndirect(cmd, _kApplySurfaceDisplacements);

                DispatchSurfaceIndirect(cmd, _kConstrainSurface);

                // Interpolates wave data onto the PNEW points AppendCandidates just
                // wrote, averaging from their established neighbors.
                DispatchSurfaceIndirect(cmd, _kInterpolateNewWaveData);

                // Retire PNEW inside the iteration that set it. Leaving it latched
                // until frame end would make the NEXT iteration re-interpolate
                // these same points — against geometry that regularization has
                // since moved, and excluding that iteration's own new points.
                DispatchSurfaceDirect(cmd, _kClearSurfaceStatus, CapGroups);
            }
        }

        // C++ surfaceWaves
        private void SurfaceWavesStep(CommandBuffer cmd)
        {
            BuildSurfaceBucket(cmd);

            DispatchSurfaceIndirect(cmd, _kAddSeed);

            // wave normal -> _TempVec3, laplacian -> _TempFloat, then evolve
            DispatchSurfaceIndirect(cmd, _kComputeSurfaceWaveNormal);
            DispatchSurfaceIndirect(cmd, _kComputeSurfaceWaveLaplacians);
            DispatchSurfaceIndirect(cmd, _kEvolveWave);

            // curvature -> _TempFloat, smoothed -> _SurfaceWaveSource, then seed
            DispatchSurfaceIndirect(cmd, _kComputeSurfaceCurvature);
            DispatchSurfaceIndirect(cmd, _kSmoothCurvature);
            DispatchSurfaceIndirect(cmd, _kSeedWaves);
        }

        // Snapshot of the coarse particles before the solver advects them; taken
        // inside GridToParticle between the G2P and Advection dispatches.
        private void SnapshotCoarseParticles(CommandBuffer cmd)
        {
            if (_surfaceDetail != SurfaceDetailMode.SurfaceTurbulence || !_surfaceInited) return;

            // Compute copy, not cmd.CopyBuffer: GPUUtil allocates with
            // GraphicsBuffer.Target.Structured only, and CopyBuffer demands
            // Target.CopySource on the source, so that call throws.
            SetSurfaceParams(cmd);
            BindSurfaceBuffers(cmd, _kSnapshotCoarse);
            cmd.SetComputeBufferParam(surfaceTurbulenceCs, _kSnapshotCoarse,
                "_CoarsePrevOut", _particlesPrev);
            cmd.DispatchCompute(surfaceTurbulenceCs, _kSnapshotCoarse,
                DivRoundUp(ParticlesBufferSize, 128), 1, 1);
        }

        // Once a second, off the simulation path. The live count is GPU-resident, so
        // without this there is no way to see whether compaction is working or
        // whether the buffers are saturated. Deliberately NOT used to drive any
        // dispatch — everything that matters reads the count on the GPU.
        private void PollSurfaceDiagnostics()
        {
            if (!pollDiagnostics) return;
            if (_surfaceDetail != SurfaceDetailMode.SurfaceTurbulence || _surfaceCount <= 0) return;
            if (Time.unscaledTime < _nextDiagPoll) return;
            _nextDiagPoll = Time.unscaledTime + 1f;

            _surfCountBuf.GetData(_diagScratch);
            _liveCountApprox = (int)_diagScratch[0];

            _surfOverflow.GetData(_diagScratch);
            if (_diagScratch[0] != 0)
            {
                _overflowTotal += _diagScratch[0];
                _surfOverflow.SetData(new uint[] { 0 });
                Debug.LogWarning($"Surface turbulence: dropped {_diagScratch[0]} candidate points " +
                                 $"(total {_overflowTotal}) — maxSurfacePoints={maxSurfacePoints} is " +
                                 $"saturated. Raise it or lower surfaceDensity.");
            }
        }

        private void DrawSurfacePoints()
        {
            if (_surfaceDetail != SurfaceDetailMode.SurfaceTurbulence || !drawSurfacePoints) return;
            if (surfaceMat == null || _surfaceCount <= 0) return;

            // Vertex count comes from _surfDrawArgs, written on the GPU alongside the
            // dispatch args, so the draw tracks compaction without a readback.
            Graphics.DrawProceduralIndirect(surfaceMat, _bounds, MeshTopology.Points,
                _surfDrawArgs, 0, null, _surfMpb, ShadowCastingMode.Off, false);
        }
    }
}

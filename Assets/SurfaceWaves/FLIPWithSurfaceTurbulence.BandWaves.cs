using Abecombe.GPUUtil;
using UnityEngine;
using UnityEngine.Rendering;

namespace SurfaceWaves
{
    // =====================================================================
    // Band waves — the cheap surface-detail alternative to the surface
    // turbulence point cloud. Same wave PDE, but carried by the solver's SDF
    // narrow band: two full-grid dispatches per frame instead of ~50 indirect
    // kernels over up to 1M points. See BandWaves.compute for the method.
    //
    // Reuses the serialized tuning knobs from the "Surface Turbulence" block:
    //   waveSpeed      — propagation speed, silently CFL-capped here (3D
    //                    staggered scheme needs dt*speed/cellSize <= 1/sqrt(3);
    //                    waveSpeed 8 at dt 1/60, cellSize 0.2 caps to 6.0)
    //   waveDamping    — same damping form as EvolveWave
    //   waveMaxAmplitude — clamp on h (world units, later the mesh displacement)
    // Don't run this and the point system at the same time — independent
    // features, but both on top of the same frame budget.
    // =====================================================================
    public partial class FLIPWithSurfaceTurbulence
    {
        [Header("Band Waves (SDF-band wave equation)")]
        // enabled via surfaceDetail == BandWaves on the main file
        public ComputeShader bandWavesCs;

        // Surface-velocity forcing gain for the source term. 0 = no waves.
        [Range(0f, 4f)]
        public float bandWaveSourceGain = 0.5f;

        // Band thickness in SDF levels (cells): surface layer is 0, fluid
        // grows inward. 4 gives the Laplacian a 3D tube to travel in before
        // the projection collapses it back onto the surface.
        [Range(1f, 8f)]
        public float bandWaveBandMax = 4f;

        // Air-side slab of the band (cells). The MC density iso sits ~0.5-1.5
        // cells into the air (density bleeds past the outer particles), so h
        // must extend there or the mesh samples nothing. 2 matches the
        // solver band (Band1).
        [Range(0f, 4f)]
        public float bandWaveAirBand = 2f;

        // Phase 2: displace the MC mesh vertices along the SDF normal by h
        // and tilt the shading normal by the ripple slope. A/B toggle:
        // BuildMesh rewrites the vertex buffer every frame, so flipping this
        // off restores the undisplaced mesh on the very next frame.
        public bool bandWaveDisplaceMesh = true;

        [Range(0f, 4f)]
        public float bandWaveDisplaceScale = 1f;

        // Slope-to-normal gain. Negative inverts the lighting tilt if it
        // reads backwards on your setup.
        [Range(-4f, 4f)]
        public float bandWaveDisplaceNormalGain = 1f;

        private readonly GPUDoubleTexture3D _bandWaveH = new();
        private readonly GPUDoubleTexture3D _bandWaveDtH = new();
        private ComputeBuffer _bandWaveStats;
        private readonly uint[] _bandWaveStatsScratch = new uint[4];
        private string _bandWaveStatsText = "Band waves: not initialized";
        private float _bandWaveNextPoll;

        private int _kBandReset, _kBandStep, _kBandProject, _kBandStats, _kBandDisplace;
        private bool _bandWavesInited;

        private void InitBandWaves()
        {
            if (bandWavesCs == null)
            {
                Debug.LogError("Band waves: bandWavesCs is not assigned. " +
                               "Assign Assets/SurfaceWaves/BandWaves.compute on the component.");
                return;
            }

            _bandWaveH.Init(GridSize, RenderTextureFormat.RHalf);
            _bandWaveDtH.Init(GridSize, RenderTextureFormat.RHalf);
            _bandWaveStats = new ComputeBuffer(4, sizeof(uint));
            _bandWaveStats.SetData(new uint[4]);

            _kBandReset = bandWavesCs.FindKernel("ResetBandWaves");
            _kBandStep = bandWavesCs.FindKernel("BandWaveStep");
            _kBandProject = bandWavesCs.FindKernel("BandWaveProject");
            _kBandStats = bandWavesCs.FindKernel("BandWaveStats");
            _kBandDisplace = bandWavesCs.FindKernel("BandWaveDisplaceMesh");

            var cmd = CommandBufferPool.Get("BandWaves.Init");
            cmd.Clear();
            SetParams(cmd, bandWavesCs);
            for (int i = 0; i < 2; i++) // clear both slots of both buffers
            {
                cmd.SetComputeTextureParam(bandWavesCs, _kBandReset, "_WaveHW", _bandWaveH.Write);
                cmd.SetComputeTextureParam(bandWavesCs, _kBandReset, "_WaveDtHW", _bandWaveDtH.Write);
                cmd.DispatchCompute(bandWavesCs, _kBandReset,
                    _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
                _bandWaveH.Swap();
                _bandWaveDtH.Swap();
            }
            Graphics.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);

            _bandWavesInited = true;
            Debug.Log($"Band waves: init {GridSize}, effective waveSpeed = {EffectiveBandWaveSpeed():F2} " +
                      $"(raw {waveSpeed}, CFL-capped), bandMax={bandWaveBandMax}, gain={bandWaveSourceGain}");
        }

        // CFL for the staggered wave scheme in 3D is dt*speed/cellSize <=
        // 1/sqrt(3); clamp with margin rather than let the field diverge.
        private float EffectiveBandWaveSpeed()
        {
            float dt = 1f / (_slowDown ? 600f : 60f);
            return Mathf.Min(waveSpeed, 0.5f * GridSpacing / dt);
        }

        private void BandWavesStep(CommandBuffer cmd)
        {
            if (_surfaceDetail != SurfaceDetailMode.BandWaves || !_bandWavesInited) return;

            cmd.BeginSample("BandWaves");
            var cs = bandWavesCs;
            SetParams(cmd, cs); // _GridSize / _CellSize / _InvCellSize / _DeltaTime

            float speed = EffectiveBandWaveSpeed();
            cmd.SetComputeFloatParam(cs, "_BandWaveCoeff",
                speed * speed * (1f / (_slowDown ? 600f : 60f)) / (GridSpacing * GridSpacing));
            cmd.SetComputeFloatParam(cs, "_BandWaveDamping", waveDamping);
            cmd.SetComputeFloatParam(cs, "_BandWaveMaxAmp", waveMaxAmplitude);
            cmd.SetComputeFloatParam(cs, "_BandWaveBandMax", bandWaveBandMax);
            cmd.SetComputeFloatParam(cs, "_BandWaveAirBand", bandWaveAirBand);
            cmd.SetComputeFloatParam(cs, "_BandWaveSourceGain", bandWaveSourceGain);

            cmd.SetComputeTextureParam(cs, _kBandStep, "_GridSDFR", _gridSDF);
            cmd.SetComputeTextureParam(cs, _kBandStep, "_VelocityR", _gridVelocity);
            cmd.SetComputeTextureParam(cs, _kBandStep, "_WaveHR", _bandWaveH.Read);
            cmd.SetComputeTextureParam(cs, _kBandStep, "_WaveHW", _bandWaveH.Write);
            cmd.SetComputeTextureParam(cs, _kBandStep, "_WaveDtHR", _bandWaveDtH.Read);
            cmd.SetComputeTextureParam(cs, _kBandStep, "_WaveDtHW", _bandWaveDtH.Write);
            cmd.DispatchCompute(cs, _kBandStep, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
            _bandWaveH.Swap();
            _bandWaveDtH.Swap();

            // closest-point projection: h := h(surface), so the Cartesian
            // Laplacian above acts as the SURFACE Laplacian
            cmd.SetComputeTextureParam(cs, _kBandProject, "_GridSDFR", _gridSDF);
            cmd.SetComputeTextureParam(cs, _kBandProject, "_WaveHR", _bandWaveH.Read);
            cmd.SetComputeTextureParam(cs, _kBandProject, "_WaveHW", _bandWaveH.Write);
            cmd.DispatchCompute(cs, _kBandProject, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
            _bandWaveH.Swap();

            // keep the DrawStructuredBuffer wave-debug tint bound to the
            // current read slot (the ping-pong swaps every frame)
            if (meshMat != null)
                meshMat.SetTexture("_WaveTex", _bandWaveH.Read);

            cmd.EndSample("BandWaves");
        }

        // Runs inside PrepareForRenderingMesh, right after BuildMesh: rewrites
        // the freshly generated vertices in place, displaced by the wave field.
        // Order matters — the wave step ran earlier in the same command buffer,
        // so _bandWaveH.Read holds this frame's h.
        private void BandWaveDisplaceMeshPass(CommandBuffer cmd)
        {
            if (_surfaceDetail != SurfaceDetailMode.BandWaves || !_bandWavesInited || !bandWaveDisplaceMesh) return;

            var cs = bandWavesCs;
            SetParams(cmd, cs);
            cmd.SetComputeFloatParam(cs, "_BandWaveDisplaceScale", bandWaveDisplaceScale);
            cmd.SetComputeFloatParam(cs, "_BandWaveDisplaceNormalGain", bandWaveDisplaceNormalGain);
            cmd.SetComputeTextureParam(cs, _kBandDisplace, "_GridSDFR", _gridSDF);
            cmd.SetComputeTextureParam(cs, _kBandDisplace, "_GridTypesR", _gridTypes);
            cmd.SetComputeTextureParam(cs, _kBandDisplace, "_DensityR", _gridOldVelocity);
            cmd.SetComputeTextureParam(cs, _kBandDisplace, "_WaveHR", _bandWaveH.Read);
            cmd.SetComputeBufferParam(cs, _kBandDisplace, "_MeshVertexCount", _argsBuffer);
            cmd.SetComputeBufferParam(cs, _kBandDisplace, "_MeshVertices", _verticesBuffer);
            // fixed capacity dispatch; threads past the GPU-side vertex count
            // (read from the indirect args buffer) exit immediately
            cmd.DispatchCompute(cs, _kBandDisplace, DivRoundUp(_vertBufferSize, 128), 1, 1);
        }

        // ~1 Hz, behind the same pollDiagnostics toggle as the surface
        // turbulence poll (sync GetData stalls the pipeline — off by default).
        private void PollBandWaveStats()
        {
            if (_surfaceDetail != SurfaceDetailMode.BandWaves) return;

            if (bandWavesCs == null)
            {
                _bandWaveStatsText = "Band waves: assign BandWaves.compute to bandWavesCs";
                return;
            }
            // init only ever happens in Start (mode is frozen); !_bandWavesInited
            // here means it aborted on the missing shader above
            if (!_bandWavesInited) return;
            if (!pollDiagnostics)
            {
                _bandWaveStatsText = "Band waves: enable pollDiagnostics to see stats";
                return;
            }
            if (Time.unscaledTime < _bandWaveNextPoll) return;
            _bandWaveNextPoll = Time.unscaledTime + 1f;

            _bandWaveStats.SetData(new uint[4]);
            var cmd = CommandBufferPool.Get("BandWaves.Stats");
            cmd.Clear();
            SetParams(cmd, bandWavesCs);
            cmd.SetComputeFloatParam(bandWavesCs, "_BandWaveBandMax", bandWaveBandMax);
            cmd.SetComputeFloatParam(bandWavesCs, "_BandWaveAirBand", bandWaveAirBand);
            cmd.SetComputeTextureParam(bandWavesCs, _kBandStats, "_GridSDFR", _gridSDF);
            cmd.SetComputeTextureParam(bandWavesCs, _kBandStats, "_WaveHR", _bandWaveH.Read);
            cmd.SetComputeBufferParam(bandWavesCs, _kBandStats, "_BandWaveStats", _bandWaveStats);
            cmd.DispatchCompute(bandWavesCs, _kBandStats,
                _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
            Graphics.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);

            _bandWaveStats.GetData(_bandWaveStatsScratch);
            // [0] holds asuint(max|h|); |h| >= 0 so the bit pattern is a
            // non-negative float and the round-trip is exact.
            float maxH = System.BitConverter.Int32BitsToSingle(unchecked((int)_bandWaveStatsScratch[0]));
            uint bandCells = _bandWaveStatsScratch[2];
            float avgH = bandCells > 0
                ? (_bandWaveStatsScratch[1] / 256f) / bandCells
                : 0f;
            _bandWaveStatsText = $"Band waves: max|h|={maxH:F4}, avg|h|={avgH:F5}, " +
                                 $"band cells={bandCells}, active={_bandWaveStatsScratch[3]}";
        }

        private void DisposeBandWaves()
        {
            if (!_bandWavesInited) return;
            _bandWavesInited = false;
            _bandWaveH.Dispose();
            _bandWaveDtH.Dispose();
            _bandWaveStats?.Dispose();
        }
    }
}

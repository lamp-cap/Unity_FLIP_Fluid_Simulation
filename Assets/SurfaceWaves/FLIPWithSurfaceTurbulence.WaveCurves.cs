using System.Runtime.InteropServices;
using Abecombe.GPUUtil;
using UnityEngine;
using UnityEngine.Rendering;

namespace SurfaceWaves
{
    // =====================================================================
    // Wave curves — Lagrangian dispersive ripples on the free surface, after
    // Skrivan et al. 2019 (see WaveCurves.compute for the paper-equation map
    // and the list of deviations kept from the Houdini VEX reference).
    //
    // One fixed-size dispatch chain per frame; nothing reads back:
    //   StepWaveCurves — reset per-frame counters, scatter seeds (full grid),
    //                    evolve ALL slots in place, generate new curves into
    //                    contiguous reservations inside this frame's segment
    //   SwapWaveCurves — swap the curve read/write slots
    //
    // Storage model: Evolve rewrites every slot each frame (survivor or dead
    // marker), so the buffers stay coherent and no count exists at all.
    // Space is reclaimed by a ring of generations: frame f allocates only in
    // segment f mod G (G = lifetime frames + margin), whose previous
    // occupants are all dead markers by the time it recycles. A birth budget
    // (WcSeedBudget) fits each frame's reservations inside the segment.
    //
    // Rendering does NOT touch the mesh: the curves are drawn directly as
    // camera-facing ribbons (WaveCurveLines.shader) over the undisplaced MC
    // water. Segment i joins points i and i+1 when curveId matches (ids are
    // frame-salted — seed indices are reused every frame), so connectivity
    // rides in the point data and no index buffer or readback is needed.
    //
    // Call sites (wired into the frame flow, gated on the mode frozen at Start):
    //   StepWaveCurves — Simulation(), after SurfaceTurbulenceStep/
    //                    BandWavesStep (post pressure solve + SDF)
    //   DrawWaveCurves — Update(), next to DrawSurfacePoints (post-swap, so
    //                    it reads this frame's freshest slot)
    //   SwapWaveCurves — Simulation(), after the Rendering block
    //
    // Draft simplification: dispatches and the draw are fixed-size over
    // buffer capacity with in-shader guards (the point system's
    // WriteDispatchArgs / DrawProceduralIndirect pattern is the upgrade).
    // =====================================================================
    public partial class FLIPWithSurfaceTurbulence
    {
        [Header("Wave Curves (Lagrangian wave packets)")]
        // enabled via surfaceDetail == WaveCurves on the main file
        public ComputeShader waveCurvesCs;

        [Range(0f, 1f)] public float waveCurveGrowthRate = 0.01f;   // x beta(k)=3.6/k
        [Range(0f, 1f)] public float waveCurveDamping = 0.0001f;    // exp(-d*k*dt)
        public float waveCurveMaxAge = 1f;                           // s, VEX ships 1.0
        public float waveCurveGStarFloor = 0.1f;                     // no RTI
        [Tooltip("0: g* from finite-difference accel (VEX parity). 1: g* = -N·grad p (paper Eq. 26) — verify MGPCG pressure units first.")]
        [Range(0f, 1f)] public float waveCurveUsePressureGravity = 0f;

        [Tooltip("Wavelengths (m) of the three curve bands. Paper: 0.20/0.10/0.05/0.025/0.0125.")]
        public Vector3 waveCurveBandLambdas = new Vector3(0.2f, 0.1f, 0.05f);
        public float waveCurveCycles = 8f;        // curve length in wavelengths
        public float waveCurveMarchRes = 0.05f;   // geodesic step (m)
        public float waveCurveInitAction = 1e-4f;

        [Range(0f, 10f)] public float waveCurveSeedRate = 1f;   // seeds/cell per unit G
        // keep below waveCurveNoiseFloor: the noise term maxes at
        // noiseFloor*1.0, so a threshold at/above it means calm water never
        // seeds and curves only appear where -div(U) is compressing
        public float waveCurveGThreshold = 0.015f;
        public float waveCurveNoiseFloor = 0.02f;

        // ribbon rendering (WaveCurveLines.shader)
        [Range(0f, 0.2f)] public float waveCurveLineWidth = 0.02f;  // half-width, m
        [Range(0f, 8f)] public float waveCurveAmpGain = 1f;         // steepness brightness

        public int waveCurveMaxPoints = 200_000;
        // keep >= worst-case surface-layer cell count (GridSize.x * GridSize.z
        // * ~3 levels): the atomic seed cap fills in grid-scan order, so a
        // smaller cap concentrates seeds — and therefore every curve — into
        // a z-slab at one side of the domain (8192 capped at z < 4 m here)
        public int waveCurveMaxSeeds = 65_536;

        // ---- GPU state ----
        [StructLayout(LayoutKind.Sequential)]
        private struct WaveCurvePointGpu // mirrors WaveCurvePoint (80 B)
        {
            public Vector3 pos; public float theta;
            public Vector3 k; public float action;
            public float age, radius, amp, band;
            public Vector3 prevVel; public float gStar;
            public uint curveId; public float pad0; public Vector2 pad1;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct WaveSeedGpu // mirrors WaveSeed (96 B)
        {
            public Vector3 pos; public float G;
            public Vector3 normal; public float band;
            public Vector3 kHat; public float pad0;
            public Vector3 vel; public float pad1;
        }

        private ComputeBuffer[] _wcCurves;      // [2], stride 80
        private ComputeBuffer _wcSeeds;         // stride 96
        private ComputeBuffer _wcSeedCount;     // [0] seed count, [1] births
                                                // this frame, [2] segment
                                                // alloc cursor
        private Material _wcMat;                // WaveCurveLines, ribbon rendering
        private bool _wcReadIndex;
        private int _wcFrame;
        private int _wcGenerations;             // ring segments (>= lifetime frames)
        private int _wcSegSize;                 // slots per generation segment

        private readonly uint[] _wcStatsScratch = new uint[3];
        private string _wcStatsText = "Wave curves: not initialized";
        private float _wcNextPoll;

        private int _kWcReset, _kWcScatter, _kWcEvolve, _kWcGenerate;
        private bool _wcInited;

        /// Generation ring: G segments, frame f allocates in f mod G. A curve
        /// lives at most maxAge, so with G = lifetime frames + margin every
        /// slot is long-dead (a marker) by the time its segment recycles —
        /// reclaim is free, no GC kernel.
        private int WcGenerations => Mathf.CeilToInt(waveCurveMaxAge * 60f) + 8;

        /// Max curve births per frame: worst-case points per curve (longest
        /// band, both sides + centre) must fit the generation segment.
        private int WcSeedBudget()
        {
            float lambdaMax = Mathf.Max(waveCurveBandLambdas.x,
                Mathf.Max(waveCurveBandLambdas.y, waveCurveBandLambdas.z));
            int stepsWorst = Mathf.Max(4,
                (int)(lambdaMax * waveCurveCycles / Mathf.Max(waveCurveMarchRes, 1e-4f)));
            int worstPts = 2 * stepsWorst + 1;
            return Mathf.Max(1, _wcSegSize / worstPts);
        }

        private ComputeBuffer WcCurveRead => _wcCurves[_wcReadIndex ? 0 : 1];
        private ComputeBuffer WcCurveWrite => _wcCurves[_wcReadIndex ? 1 : 0];

        private void InitWaveCurves()
        {
            if (_wcInited) return;
            if (waveCurvesCs == null)
            {
                Debug.LogError("Wave curves: waveCurvesCs is not assigned. " +
                               "Assign Assets/SurfaceWaves/WaveCurves.compute on the component.");
                return;
            }

            _wcCurves = new ComputeBuffer[2];
            for (int i = 0; i < 2; i++)
                _wcCurves[i] = new ComputeBuffer(waveCurveMaxPoints, 80, ComputeBufferType.Structured);
            _wcSeeds = new ComputeBuffer(waveCurveMaxSeeds, 96, ComputeBufferType.Structured);
            _wcSeedCount = new ComputeBuffer(3, 4, ComputeBufferType.Structured);
            _wcSeedCount.SetData(new uint[3]);

            var shader = Shader.Find("SurfaceWaves/WaveCurveLines");
            if (shader == null)
            {
                Debug.LogError("Wave curves: WaveCurveLines.shader not found — " +
                               "feature disabled, buffers released.");
                DisposeWaveCurves(); // don't hold ~38 MB for a dead feature
                return;
            }
            _wcMat = new Material(shader);

            _wcGenerations = WcGenerations;
            _wcSegSize = Mathf.Max(1, waveCurveMaxPoints / _wcGenerations);

            _kWcReset = waveCurvesCs.FindKernel("WaveCurvesResetCounters");
            _kWcScatter = waveCurvesCs.FindKernel("WaveCurvesScatterSeeds");
            _kWcEvolve = waveCurvesCs.FindKernel("WaveCurvesEvolve");
            _kWcGenerate = waveCurvesCs.FindKernel("WaveCurvesGenerate");

            _wcInited = true;
            Debug.Log($"Wave curves: init {waveCurveMaxPoints} pts, {waveCurveMaxSeeds} seeds, " +
                      $"gen ring {_wcGenerations}x{_wcSegSize}, budget {WcSeedBudget()}/frame, " +
                      $"bands {waveCurveBandLambdas}");
        }

        private void SetWaveCurveParams(CommandBuffer cmd, ComputeShader cs, int kernel)
        {
            SetParams(cmd, cs); // common solver params (_DeltaTime, _GridSize, ...)

            cmd.SetComputeFloatParam(cs, "_WcGrowthRate", waveCurveGrowthRate);
            cmd.SetComputeFloatParam(cs, "_WcDamping", waveCurveDamping);
            cmd.SetComputeFloatParam(cs, "_WcGStarFloor", waveCurveGStarFloor);
            cmd.SetComputeFloatParam(cs, "_WcMaxAge", waveCurveMaxAge);
            cmd.SetComputeFloatParam(cs, "_WcMinSteepness", 0.01f * 2f * Mathf.PI); // A/lambda < 0.01
            cmd.SetComputeFloatParam(cs, "_WcUsePressureGravity", waveCurveUsePressureGravity);
            cmd.SetComputeFloatParam(cs, "_WcSeedRate", waveCurveSeedRate);
            cmd.SetComputeFloatParam(cs, "_WcGThreshold", waveCurveGThreshold);
            cmd.SetComputeFloatParam(cs, "_WcNoiseFloor", waveCurveNoiseFloor);
            cmd.SetComputeFloatParam(cs, "_WcCycles", waveCurveCycles);
            cmd.SetComputeFloatParam(cs, "_WcLengthMult", 1f);
            cmd.SetComputeFloatParam(cs, "_WcMarchRes", waveCurveMarchRes);
            cmd.SetComputeFloatParam(cs, "_WcInitAction", waveCurveInitAction);
            cmd.SetComputeVectorParam(cs, "_WcBandLambdas", waveCurveBandLambdas);
            cmd.SetComputeIntParam(cs, "_WcMaxPoints", waveCurveMaxPoints);
            cmd.SetComputeIntParam(cs, "_WcMaxSeeds", waveCurveMaxSeeds);
            cmd.SetComputeIntParam(cs, "_WcSeedBudget", WcSeedBudget());
            cmd.SetComputeIntParam(cs, "_WcFrame", _wcFrame);
        }

        /// Sim-side dispatch chain. Gated on the mode frozen at Start; init
        /// happens only there, so !_wcInited means the shader was missing.
        private void StepWaveCurves(CommandBuffer cmd)
        {
            if (_surfaceDetail != SurfaceDetailMode.WaveCurves || !_wcInited) return;
            _wcFrame++;

            // 0. reset per-frame counters (seed count / births / segment
            // alloc cursor). Segment params persist on the command buffer
            // for Generate's reservation limit below.
            int segBase = (_wcFrame % _wcGenerations) * _wcSegSize;
            cmd.SetComputeIntParam(waveCurvesCs, "_WcSegBase", segBase);
            cmd.SetComputeIntParam(waveCurvesCs, "_WcSegEnd", segBase + _wcSegSize);
            cmd.SetComputeBufferParam(waveCurvesCs, _kWcReset, "_WcSeedCountBuf", _wcSeedCount);
            cmd.DispatchCompute(waveCurvesCs, _kWcReset, 1, 1, 1);

            // 1. seeding analysis (full grid)
            SetWaveCurveParams(cmd, waveCurvesCs, _kWcScatter);
            cmd.SetComputeTextureParam(waveCurvesCs, _kWcScatter, "_GridSDFR", _gridSDF);
            cmd.SetComputeTextureParam(waveCurvesCs, _kWcScatter, "_VelocityR", _gridVelocity);
            cmd.SetComputeTextureParam(waveCurvesCs, _kWcScatter, "_DensityR", _gridOldVelocity);
            cmd.SetComputeBufferParam(waveCurvesCs, _kWcScatter, "_WcSeeds", _wcSeeds);
            cmd.SetComputeBufferParam(waveCurvesCs, _kWcScatter, "_WcSeedCountBuf", _wcSeedCount);
            cmd.DispatchCompute(waveCurvesCs, _kWcScatter,
                _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);

            // 2. evolve + compact read -> write
            SetWaveCurveParams(cmd, waveCurvesCs, _kWcEvolve);
            cmd.SetComputeTextureParam(waveCurvesCs, _kWcEvolve, "_GridSDFR", _gridSDF);
            cmd.SetComputeTextureParam(waveCurvesCs, _kWcEvolve, "_VelocityR", _gridVelocity);
            cmd.SetComputeTextureParam(waveCurvesCs, _kWcEvolve, "_DensityR", _gridOldVelocity);
            // TODO(units): MGPCG _x is the pressure solve target — confirm it
            // is kinematic (m^2/s^2) before trusting the Eq. 26 route.
            cmd.SetComputeTextureParam(waveCurvesCs, _kWcEvolve, "_PressureR", _gridPressurePymaid[0]);
            cmd.SetComputeBufferParam(waveCurvesCs, _kWcEvolve, "_WcCurveR", WcCurveRead);
            cmd.SetComputeBufferParam(waveCurvesCs, _kWcEvolve, "_WcCurveW", WcCurveWrite);
            cmd.DispatchCompute(waveCurvesCs, _kWcEvolve, DivRoundUp(waveCurveMaxPoints, 128), 1, 1);

            // 3. generate new curves into the write buffer
            SetWaveCurveParams(cmd, waveCurvesCs, _kWcGenerate);
            cmd.SetComputeTextureParam(waveCurvesCs, _kWcGenerate, "_GridSDFR", _gridSDF);
            cmd.SetComputeTextureParam(waveCurvesCs, _kWcGenerate, "_VelocityR", _gridVelocity);
            cmd.SetComputeTextureParam(waveCurvesCs, _kWcGenerate, "_DensityR", _gridOldVelocity);
            cmd.SetComputeBufferParam(waveCurvesCs, _kWcGenerate, "_WcSeeds", _wcSeeds);
            cmd.SetComputeBufferParam(waveCurvesCs, _kWcGenerate, "_WcSeedCountBuf", _wcSeedCount);
            cmd.SetComputeBufferParam(waveCurvesCs, _kWcGenerate, "_WcCurveW", WcCurveWrite);
            cmd.DispatchCompute(waveCurvesCs, _kWcGenerate, DivRoundUp(waveCurveMaxSeeds, 64), 1, 1);
        }

        /// Ribbon draw. Call from Update after Simulation — post-swap, so
        /// WcCurveRead holds this frame's freshest points. The draw covers
        /// the full capacity; markers park their segments outside the frustum
        /// in the vertex shader (see WaveCurveLines.hlsl), so there is no
        /// count to read back.
        private void DrawWaveCurves()
        {
            if (_surfaceDetail != SurfaceDetailMode.WaveCurves || !_wcInited || _wcMat == null) return;

            _wcMat.SetBuffer("_WcCurveBuf", WcCurveRead);
            _wcMat.SetFloat("_Width", waveCurveLineWidth);
            _wcMat.SetFloat("_AmpGain", waveCurveAmpGain);
            _wcMat.SetFloat("_MaxAge", waveCurveMaxAge);

            // 6 vertices per segment (2 triangles), one segment per point pair
            // (same overload shape as GPU_FLIP's volume draw)
            Graphics.DrawProcedural(_wcMat, _bounds, MeshTopology.Triangles,
                6 * (waveCurveMaxPoints - 1), 1, null, null, ShadowCastingMode.Off, false);
        }

        /// End of frame: flip the curve read/write slots. There is no count
        /// to carry — Evolve rewrites every slot, so the swapped-in buffer is
        /// fully coherent by construction.
        private void SwapWaveCurves()
        {
            if (_surfaceDetail != SurfaceDetailMode.WaveCurves || !_wcInited) return;
            _wcReadIndex = !_wcReadIndex;
        }

        // ~1 Hz, behind the same pollDiagnostics toggle as the other polls
        // (GetData stalls the pipeline — off by default). Reads this frame's
        // counters straight off _wcSeedCount; seeds/births/cursor.
        private void PollWaveCurveStats()
        {
            if (_surfaceDetail != SurfaceDetailMode.WaveCurves || !_wcInited) return;
            if (!pollDiagnostics)
            {
                _wcStatsText = "Wave curves: enable pollDiagnostics to see stats";
                return;
            }
            if (Time.unscaledTime < _wcNextPoll) return;
            _wcNextPoll = Time.unscaledTime + 1f;

            _wcSeedCount.GetData(_wcStatsScratch);
            _wcStatsText = $"Wave curves: seeds {_wcStatsScratch[0]}, " +
                           $"born {_wcStatsScratch[1]} (budget {WcSeedBudget()}), " +
                           $"cursor {_wcStatsScratch[2]}/{_wcSegSize}";
        }

        private void DisposeWaveCurves()
        {
            foreach (var b in _wcCurves ?? System.Array.Empty<ComputeBuffer>()) b?.Dispose();
            _wcSeeds?.Dispose();
            _wcSeedCount?.Dispose();
            if (Application.isPlaying) Destroy(_wcMat);
            _wcInited = false;
        }
    }
}

#ifndef SURFACE_TURBULENCE_UTILS_HLSL
#define SURFACE_TURBULENCE_UTILS_HLSL

// Shared helpers + data interface for the 3D surface-turbulence compute port.
//
// This is a faithful HLSL translation of surfaceturbulence.cpp (Mercier et al.
// SIGGRAPH Asia 2015), matching the conventions of the GPU FLIP solver:
//   - packed `Particle { uint2 packedPosition; uint2 packedVelocity; }`
//   - Morton Z-order `Coord2Idx` cell indexing (reused from FlipUtils.hlsl)
//   - sorted-LUT neighbor buckets: `_*Range[cell] = uint2(start, end)`.
//
// NOTE on the coarse bucket layout: the GPU FLIP solver's counting sort
// (CountingSort.compute `Rearrange`) physically scatters the particle records
// into cell order, so `_CoarseRange[cell]` indexes `_CoarseParticles` DIRECTLY.
// There is no id-payload indirection on the coarse side (an earlier revision of
// this header assumed one). The surface bucket does keep an id payload, because
// DeviceRadixSort sorts (hash, id) pairs without moving the surface records.
//
// Data interface (the C# side fills these in later; nothing here touches C#):
//   coarse particles (owned by the FLIP solver, in GPU_FLIP packed format)
//   surface points (owned by this system: float3 pos/normal, float wave fields)

#include "../GPU_FLIP/FlipUtils.hlsl"

#ifndef PI
#define PI 3.14159265358979323846
#endif

// ===== surface turbulence parameters (world units, matching GPU_FLIP) =====
float _OuterRadius;
float _InnerRadius;
float _MeanFineDistance;
float _ConstraintA;
float _NormalRadius;
float _TangentRadius;

float _WaveSpeed;
float _WaveDamping;
float _WaveSeedFrequency;
float _WaveMaxAmplitude;
float _WaveMaxFrequency;
float _WaveMaxSeedingAmplitude;
float _SeedThresholdCenter;
float _SeedThresholdRadius;
float _SeedStepSizeRatio;
float _RenderDisplacementScale; // render-only multiplier on normal*h (1 = C++)

int _FrameCount;
int _MaxSurfacePoints; // capacity of the surface point buffers

// Live surface point count. Lives on the GPU so compaction can update it without
// a readback: element [0] is the count, and the per-point kernels are dispatched
// indirectly off args derived from it (WriteSurfaceDispatchArgs).
//
// Exposed as a macro so every kernel body can keep saying `_SurfaceCount`. Reads
// of an RWStructuredBuffer are fine; nothing but the compaction kernels writes it.
RWStructuredBuffer<uint> _SurfaceCountBuf;
#define _SurfaceCount ((int)_SurfaceCountBuf[0])

// domain bounds for ghost/mirror reflection (world units; C++ bndXm..bndZp)
float3 _BndMin;
float3 _BndMax;

// ===== coarse particles (owned by GPU FLIP; cell-sorted in place) =====
// _CoarseParticles and _CoarsePrevParticles share ONE index space: slot i of the
// prev buffer is the pre-advection state of slot i of the current buffer. C# gets
// that by snapshotting the particle buffer right before the advection dispatch.
StructuredBuffer<Particle> _CoarseParticles;      // post-advection positions
StructuredBuffer<Particle> _CoarsePrevParticles;  // pre-advection snapshot
RWStructuredBuffer<Particle> _CoarsePrevOut;      // write view, SnapshotCoarse only
StructuredBuffer<uint2>    _CoarseRange;          // cell -> [start,end) into _CoarseParticles
StructuredBuffer<uint2>    _CoarsePrevRange;      // cell ranges of the snapshot layout

// ===== surface points (owned by this system) =====
RWStructuredBuffer<float3> _SurfacePos;
RWStructuredBuffer<float3> _SurfaceNormal;
RWStructuredBuffer<float>  _SurfaceWaveH;
RWStructuredBuffer<float>  _SurfaceWaveDtH;
RWStructuredBuffer<float>  _SurfaceWaveSeed;
RWStructuredBuffer<float>  _SurfaceWaveSeedAmp;
RWStructuredBuffer<float>  _SurfaceWaveSource;

// temporaries (mapped 1:1 to C++ tempSurfaceVec3 / tempSurfaceFloat)
RWStructuredBuffer<float3> _TempVec3;
RWStructuredBuffer<float>  _TempFloat;

// alive/status flags (status bit 0 == PNEW, matching C++ ParticleBase::PNEW)
RWStructuredBuffer<uint> _SurfaceAlive;   // 1 = alive, 0 = killed
RWStructuredBuffer<uint> _SurfaceStatus;

// Deferred kill flags for the over-density thinning. That pass reads the alive
// flags of NEIGHBORS, so it must not mutate them in flight; it stages decisions
// here and ApplyDensityKills folds them into _SurfaceAlive afterwards.
RWStructuredBuffer<uint> _SurfaceKill;

// candidate addition slots (one per surface point; compaction is deferred)
RWStructuredBuffer<float3> _CandidatePos;
RWStructuredBuffer<uint>   _HasCandidate;

// surface bucket (sorted LUT, stable-id payload like the coarse bucket)
// Slots past the live count hash to this and sort to the tail.
#define SURFACE_HASH_INVALID 0xFFFFFFFFu
RWStructuredBuffer<uint>  _SurfaceHash;
RWStructuredBuffer<uint>  _SurfaceID;
RWStructuredBuffer<uint2> _SurfaceRange;

// ===== compaction scratch =====
// Ping-pong destinations for the survivor gather. The 8 fields here are the
// persistent per-point state; temps are rebuilt each pass and the flag/bucket
// arrays are reset or rebuilt, so they do not need permuting.
RWStructuredBuffer<float3> _SurfacePosOut;
RWStructuredBuffer<float3> _SurfaceNormalOut;
RWStructuredBuffer<float>  _SurfaceWaveHOut;
RWStructuredBuffer<float>  _SurfaceWaveDtHOut;
RWStructuredBuffer<float>  _SurfaceWaveSeedOut;
RWStructuredBuffer<float>  _SurfaceWaveSeedAmpOut;
RWStructuredBuffer<float>  _SurfaceWaveSourceOut;
RWStructuredBuffer<uint>   _SurfaceStatusOut;

// Scan working set. _ScanBuf is staged from a flag array then turned into
// exclusive destinations in place by PrefixScan.compute; _ScanTotal receives that
// scan's total. _SurvivorCount is a copy of the survivor total, kept because the
// candidate scan overwrites _ScanTotal.
RWStructuredBuffer<uint> _ScanBuf;
RWStructuredBuffer<uint> _ScanTotal;
RWStructuredBuffer<uint> _SurvivorCount;

// Diagnostic: candidates dropped because the buffers were full. Accumulates until
// the host reads and resets it.
RWStructuredBuffer<uint> _SurfaceOverflow;

// Seeding rejection tally, so "seeded 0 points" can be attributed to a stage
// instead of guessed at. Slots:
//   [0] coarse particles that failed the nearSurface test
//   [1] sphere candidates rejected by IsInDomain
//   [2] sphere candidates rejected by the `valid` (no other coarse particle
//       within outerRadius) test
//   [3] candidates accepted
//   [4] sphere candidates rejected as narrow-band interior (IsDeepInterior)
RWStructuredBuffer<uint> _SeedReject;

// Indirect args, written on the GPU by WriteSurfaceDispatchArgs.
RWStructuredBuffer<uint> _SurfaceDispatchArgs; // (groupsX, 1, 1)
RWStructuredBuffer<uint> _SurfaceDrawArgs;     // (vertexCount, instanceCount, 0, 0, 0)

// render output: xyz = displaced position (pos + normal*H), w = H
RWStructuredBuffer<float4> _SurfaceRender;

// atomic append counter for InitSurfacePoints (element [0] = count)
RWStructuredBuffer<uint> _SurfaceCounter;

// ===== weighting kernels (C++ weightingKernels) =====

float TriangularWeight(float distance, float radius)
{
    return 1.0 - distance / radius;
}

float ExponentialWeight(float distance, float radius, float falloff)
{
    if (distance > radius) return 0.0;
    float tmp = distance / radius;
    return exp(-falloff * tmp * tmp);
}

float WeightAdvection(float distance)
{
    return distance > 2.0 * _OuterRadius ? 0.0 : TriangularWeight(distance, 2.0 * _OuterRadius);
}

float WeightCoarseDensity(float distance)
{
    return ExponentialWeight(distance, _OuterRadius, 2.0);
}

float WeightSurfaceNormal(float distance)
{
    return distance > _NormalRadius ? 0.0 : TriangularWeight(distance, _NormalRadius);
}

float WeightSurfaceTangent(float distance)
{
    return distance > _TangentRadius ? 0.0 : TriangularWeight(distance, _TangentRadius);
}

float Smoothstep(float edgeLeft, float edgeRight, float val)
{
    float x = saturate((val - edgeLeft) / (edgeRight - edgeLeft));
    return x * x * (3.0 - 2.0 * x);
}

bool IsInDomain(float3 pos)
{
    return all(pos >= _BndMin) && all(pos <= _BndMax);
}

// Narrow-band SDF, same texture and convention the solver uses: the value grows
// inward and `>= Band1` means definitely fluid (ResampleParticles.compute
// GetType). Only the band carries coarse particles; deeper than that is grid-only.
//
// Band1 lives in NarrowBand/NBFlipUtils.hlsl, which this header does not include
// (it includes GPU_FLIP/FlipUtils.hlsl instead). Duplicated rather than pulling in
// a second copy of the whole utils header — keep in sync with NBFlipUtils.hlsl:11.
#ifndef Band1
#define Band1 2
#endif
Texture3D<float> _GridSDFR;

// True when pos sits in the grid-only interior. The coarse particle set stops at
// Band1, so the "an empty neighbour cell exists" proxy that InitSurfacePoints and
// ComputeAdditions rely on fires on the band's INNER edge too, manufacturing a
// second surface inside the liquid. The SDF is what distinguishes "empty because
// air" from "empty because past the band".
bool IsDeepInterior(float3 pos)
{
    int3 c = clamp((int3)floor(pos * _InvCellSize), int3(0, 0, 0), _GridSize - int3(1, 1, 1));
    return _GridSDFR[c] >= Band1;
}

float3 SafeNormalize(float3 v)
{
    float l = length(v);
    return l > 1e-8 ? v / l : float3(0, 0, 0);
}

// Tangent frame from a (not necessarily unit) normal: t1 perpendicular to n,
// t2 = n x t1. Mirrors C++ computeSurfaceNormals.
void GetTangentFrame(float3 n, out float3 t1, out float3 t2)
{
    float3 nn = SafeNormalize(n);
    float3 vx = float3(1, 0, 0);
    float3 vy = float3(0, 1, 0);
    float dotX = dot(nn, vx);
    float dotY = dot(nn, vy);
    t1 = SafeNormalize(abs(dotX) < abs(dotY) ? cross(nn, vx) : cross(nn, vy));
    t2 = SafeNormalize(cross(nn, t1));
}

// ===== neighbor loops (sorted-LUT buckets, C++ LOOP_NEIGHBORS_*) =====
//
// The three loop families (coarse / coarse-prev / surface) used to be macros
// here; they are now hand-inlined at each use site in SurfaceTurbulence.compute
// so they can be tuned per kernel. Shared shape: clamp the cell range covering
// [center - radius, center + radius] to the grid, walk each cell's sorted range,
// then every particle in it.

// The coarse bucket is built from the PRE-advection positions (C# runs the
// surface step after G2P, reusing the cell ranges the solver already had), so a
// particle may have moved out of the cell its range entry belongs to.
//
// One extra cell ring covers a drift of up to cellSize, i.e. |v|*dt < cellSize.
// The solver runs a fixed dt = 1/60 at cellSize = 0.2, so this holds up to
// |v| = 12 m/s. A tall dam break can exceed that, and past it the
// CURRENT-position loops start missing particles that drifted more than one cell
// (the prev-position loops are exact regardless, being bucketed on the snapshot).
//
// COST: cubic in the pad, and this is the single most expensive thing in the
// system. At radius 1.5*outerRadius the scan is 343 cells at pad=0 vs 729 at
// pad=1 (2.13x); at 2*outerRadius it is 729 vs 1331 (1.83x). Dropping to 0 is
// roughly a 2x frame-time win.
//
// What pad=0 actually costs in accuracy: a particle that drifted out of its
// bucketed cell is missed by the level-set sum, where its Gaussian contribution
// at that distance is exp(-constraintA * r^2) with r >= 1.5*outerRadius, i.e.
// ~0.007 of a nearby particle's weight. The level-set is a sum over ~600
// neighbours, so one missed far-field term is noise. Raise to 1 if fast splashes
// show artifacts.
#ifndef COARSE_CELL_PAD
#define COARSE_CELL_PAD 0
#endif

// Buffer per family, for finding the inline copies: coarse reads _CoarseRange +
// _CoarseParticles (current positions, pad with COARSE_CELL_PAD); coarse-prev
// reads _CoarsePrevRange + _CoarsePrevParticles (snapshot layout, exact, no
// pad); surface reads _SurfaceRange + _SurfaceID and skips dead points.

// ===== ghost / mirror reflection (C++ LOOP_GHOSTS_*) =====
//
// Also inlined. For a neighbor near a domain boundary, emit up to 6 mirrored
// positions (one per boundary) followed by the real position; the +normal
// variant also flips the copied normal on the mirrored axis. The gf counter
// both selects the mirror and advances the state: each branch tests gf < k and
// sets gf = k, the final else emits the real point (gf = 6) and the while
// condition ends the loop. `continue` inside the body skips to the next ghost.

// ===== constraint level-set (C++ computeConstraintLevel / Gradient) =====
//
// Gaussian sum over coarse particles, NOT the grid SDF. Faithful to the C++:
// lvl is clamped to <= 1 so -log(lvl) stays defined; a point with no coarse
// neighbors in the query radius yields lvl=0 -> level = +inf (deleted later).

// Query-radius truncation, in units of outerRadius. The C++ scans 1.5x, but
// the tail that buys is thin while the sweep is cubic in the radius: with
// innerRadius = outerRadius/2, A*outerRadius^2 = 2.22 at ANY radius, so the
// per-particle cutoff is scale-invariant — 1.5x keeps exp(-A r^2) down to
// ~0.7% per particle while sweeping 343 cells (outerRadius = 2*GridSpacing
// here), 1.375x keeps ~1.5% over 125-216 cells (~1.8x cheaper).
//
// Why not smaller: a point on the outer constraint boundary (level == 1)
// holds lvl ~= 0.1, so far-field crumbs are a real fraction of its sum. At
// 1.375x the truncation biases that point to level ~1.1 — still inside the
// 1.2 kill margin. Around 1.25x the bias reaches ~+0.3 and MarkDeletions
// starts retiring points that belong on the outer boundary.
#ifndef CONSTRAINT_QUERY_SCALE
#define CONSTRAINT_QUERY_SCALE 1.375
#endif

float ConstraintLevel(float3 pos)
{
    float lvl = 0.0;
    // Expanded from LOOP_COARSE_NEIGHBORS_BEGIN(pos, CONSTRAINT_QUERY_SCALE * _OuterRadius).
    {
        const float queryRadius = CONSTRAINT_QUERY_SCALE * _OuterRadius;
        int3 minCell = max((int3)floor((pos - queryRadius) * _InvCellSize) - COARSE_CELL_PAD, int3(0, 0, 0));
        int3 maxCell = min((int3)floor((pos + queryRadius) * _InvCellSize) + COARSE_CELL_PAD, _GridSize - int3(1, 1, 1));
        for (int cz = minCell.z; cz <= maxCell.z; ++cz)
        for (int cy = minCell.y; cy <= maxCell.y; ++cy)
        for (int cx = minCell.x; cx <= maxCell.x; ++cx)
        {
            uint2 rng = _CoarseRange[Coord2Idx((uint)cx, (uint)cy, (uint)cz)];
            for (uint pid = rng.x; pid < rng.y; ++pid)
            {
                float3 npos = DecodePosition(_CoarseParticles[pid].packedPosition);
                lvl += exp(-_ConstraintA * dot(npos - pos, npos - pos));
            }
        }
    }
    if (lvl > 1.0) lvl = 1.0;
    return (sqrt(-log(lvl) / _ConstraintA) - _InnerRadius) / (_OuterRadius - _InnerRadius);
}

float3 ConstraintGradient(float3 pos)
{
    float3 gradient = float3(0, 0, 0);
    // Expanded from LOOP_COARSE_NEIGHBORS_BEGIN(pos, CONSTRAINT_QUERY_SCALE * _OuterRadius).
    {
        const float queryRadius = CONSTRAINT_QUERY_SCALE * _OuterRadius;
        int3 minCell = max((int3)floor((pos - queryRadius) * _InvCellSize) - COARSE_CELL_PAD, int3(0, 0, 0));
        int3 maxCell = min((int3)floor((pos + queryRadius) * _InvCellSize) + COARSE_CELL_PAD, _GridSize - int3(1, 1, 1));
        for (int cz = minCell.z; cz <= maxCell.z; ++cz)
        for (int cy = minCell.y; cy <= maxCell.y; ++cy)
        for (int cx = minCell.x; cx <= maxCell.x; ++cx)
        {
            uint2 rng = _CoarseRange[Coord2Idx((uint)cx, (uint)cy, (uint)cz)];
            for (uint pid = rng.x; pid < rng.y; ++pid)
            {
                float3 npos = DecodePosition(_CoarseParticles[pid].packedPosition);
                gradient += 2.0 * _ConstraintA * exp(-_ConstraintA * dot(npos - pos, npos - pos)) * (pos - npos);
            }
        }
    }
    return SafeNormalize(gradient);
}

// Level and gradient are the same Gaussian sum, so callers needing both
// (ConstrainSurface: level picks the branch, gradient is the projection
// direction) make one neighborhood sweep instead of two.
void ConstraintLevelGradient(float3 pos, out float level, out float3 gradient)
{
    float lvl = 0.0;
    float3 grad = float3(0, 0, 0);
    // Expanded from LOOP_COARSE_NEIGHBORS_BEGIN(pos, CONSTRAINT_QUERY_SCALE * _OuterRadius).
    {
        const float queryRadius = CONSTRAINT_QUERY_SCALE * _OuterRadius;
        int3 minCell = max((int3)floor((pos - queryRadius) * _InvCellSize) - COARSE_CELL_PAD, int3(0, 0, 0));
        int3 maxCell = min((int3)floor((pos + queryRadius) * _InvCellSize) + COARSE_CELL_PAD, _GridSize - int3(1, 1, 1));
        for (int cz = minCell.z; cz <= maxCell.z; ++cz)
        for (int cy = minCell.y; cy <= maxCell.y; ++cy)
        for (int cx = minCell.x; cx <= maxCell.x; ++cx)
        {
            uint2 rng = _CoarseRange[Coord2Idx((uint)cx, (uint)cy, (uint)cz)];
            for (uint pid = rng.x; pid < rng.y; ++pid)
            {
                float3 npos = DecodePosition(_CoarseParticles[pid].packedPosition);
                float e = exp(-_ConstraintA * dot(npos - pos, npos - pos));
                lvl += e;
                grad += 2.0 * _ConstraintA * e * (pos - npos);
            }
        }
    }
    if (lvl > 1.0) lvl = 1.0;
    level = (sqrt(-log(lvl) / _ConstraintA) - _InnerRadius) / (_OuterRadius - _InnerRadius);
    gradient = SafeNormalize(grad);
}

// Existence predicate over the coarse bucket (C++ hasNeighbor).
bool HasCoarseNeighbor(float3 pos, float radius)
{
    // Expanded from LOOP_COARSE_NEIGHBORS_BEGIN(pos, radius).
    {
        int3 minCell = max((int3)floor((pos - radius) * _InvCellSize) - COARSE_CELL_PAD, int3(0, 0, 0));
        int3 maxCell = min((int3)floor((pos + radius) * _InvCellSize) + COARSE_CELL_PAD, _GridSize - int3(1, 1, 1));
        for (int cz = minCell.z; cz <= maxCell.z; ++cz)
        for (int cy = minCell.y; cy <= maxCell.y; ++cy)
        for (int cx = minCell.x; cx <= maxCell.x; ++cx)
        {
            uint2 rng = _CoarseRange[Coord2Idx((uint)cx, (uint)cy, (uint)cz)];
            for (uint pid = rng.x; pid < rng.y; ++pid)
            {
                float3 npos = DecodePosition(_CoarseParticles[pid].packedPosition);
                if (length(npos - pos) <= radius) return true;
            }
        }
    }
    return false;
}

#endif // SURFACE_TURBULENCE_UTILS_HLSL

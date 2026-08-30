// =============================================================================
// Wave curve ribbon rendering — shared by the two passes of
// WaveCurveLines.shader. No connectivity buffer: Generate appends each curve
// as one contiguous run of points, so segment i connects points i and i+1
// whenever their curveId matches (mismatched pairs park outside the frustum).
//
// Topology is Triangles, 6 vertices per segment; the vertex shader maps
// SV_VertexID -> (segment, corner) and expands the pair into a camera-facing
// ribbon quad. Fixed-size draw over the buffer capacity — dead segments are
// clipped by position, no readback.
// =============================================================================

#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

// mirrors WaveCurvePoint in WaveCurves.compute / WaveCurvePointGpu in
// FLIPWithSurfaceTurbulence.WaveCurves.cs (80 B)
struct WaveCurvePoint
{
    float3 pos;      // grid-local world, same frame as the solver
    float  theta;
    float3 k;
    float  action;
    float  age, radius, amp, band;
    float3 prevVel;
    float  gStar;
    uint   curveId;  // unique per curve generation (frame-salted)
    float  _pad0;
    float2 _pad1;
};

StructuredBuffer<WaveCurvePoint> _WcCurveBuf;

// dead-slot marker id, matches WC_DEAD_CURVE in WaveCurves.compute
#define WC_DEAD_CURVE 0xffffffffu

float _Width;    // ribbon half-width, world units
float _AmpGain;  // steepness -> brightness gain
float _MaxAge;   // age fade denominator

struct Varyings
{
    float4 positionCS : SV_POSITION;
    float4 color      : COLOR0;
};

// dead segments park fully outside the frustum (homogeneous far corner);
// every quad is built from its own segment's vertices, so no live triangle
// ever shares one of these
static const float4 WC_CLIP_DEAD = float4(1e8, 1e8, 1e8, 1.0);

// band hues: blue = longest lambda, yellow = mid, magenta = shortest
float3 WcBandColor(float band)
{
    return (band < 0.5) ? float3(0.10, 0.85, 1.00)
         : (band < 1.5) ? float3(1.00, 0.85, 0.10)
                        : float3(1.00, 0.30, 0.90);
}

Varyings Vertex(uint id : SV_VertexID)
{
    Varyings o;
    o.color = 0;

    uint seg    = id / 6;
    uint corner = id % 6;

    // no count read: the fixed dispatch (6 * (capacity-1) vertices) bounds
    // every index, and everything that is not a live same-curve pair —
    // markers, unwritten slots, run boundaries — parks outside the frustum
    WaveCurvePoint a = _WcCurveBuf[seg];
    WaveCurvePoint b = _WcCurveBuf[seg + 1u];
    if (a.curveId != b.curveId
        || a.curveId == WC_DEAD_CURVE
        || b.curveId == WC_DEAD_CURVE) { o.positionCS = WC_CLIP_DEAD; return o; }

    float3 tangent = b.pos - a.pos;
    float tl = length(tangent);
    if (tl < 1e-5) { o.positionCS = WC_CLIP_DEAD; return o; }
    tangent /= tl;

    float3 viewDir = normalize(_WorldSpaceCameraPos - a.pos);
    float3 side = cross(tangent, viewDir);
    float sl = length(side);
    if (sl < 1e-4) { o.positionCS = WC_CLIP_DEAD; return o; } // looking along the tangent
    side = side / sl * _Width;

    // triangles (0,1,2) and (3,4,5): corners 0/3/5 take endpoint a,
    // corners 0/1/3 take the -side edge
    float3 wp = (corner == 0 || corner == 3 || corner == 5) ? a.pos : b.pos;
    wp += ((corner == 0 || corner == 1 || corner == 3) ? -1.0 : 1.0) * side;
    // lift toward the camera so the ribbon does not z-fight the water mesh
    wp += viewDir * (_Width + 5e-4);

    o.positionCS = TransformWorldToHClip(wp);

    // brightness by steepness (amp*k, Appendix C scale); fresh curves carry a
    // dim floor so they are visible before the action grows. Fade with age.
    float steep = saturate(a.amp * length(a.k) * 0.5 * _AmpGain);
    float ageFade = 1.0 - saturate(a.age / max(_MaxAge, 1e-3));
    o.color = float4(WcBandColor(a.band) * (0.35 + 1.3 * steep),  ageFade);

    return o;
}

half4 Fragment(Varyings input) : SV_Target
{
    return input.color;
}

void FragDepth(Varyings input) {}

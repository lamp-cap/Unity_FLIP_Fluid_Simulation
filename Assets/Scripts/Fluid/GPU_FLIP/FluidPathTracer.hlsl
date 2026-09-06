#ifndef FLUID_PATH_TRACER_INCLUDED
#define FLUID_PATH_TRACER_INCLUDED

// Shared liquid-metal path tracer for the FLIP fluid.
// Included by both DrawVolume.shader (MC mesh primary hit) and
// DrawVolumeBox.shader (bounding-box + raymarched primary hit).

#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

TEXTURE3D(_Density);        SAMPLER(sampler_Density);
TEXTURECUBE(_Cube);         SAMPLER(sampler_Cube);
float3 _Size;
float4 _Color;
float _Range;
float _Threshold;
float _Step;
float _Offset;
float4 _MetalColor;
float _Roughness;
float _Bounces;
float _Samples;
float _LightIntensity;
float _LightRadius;
float _LightDistance;

#define PI 3.1415926535
#define TWO_PI 6.28318530718
#define MAX_DIST 1e10

float sqr(float x) { return x * x; }
float max3(float3 a) { return max(a.x, max(a.y, a.z)); }
float3 safeNorm(float3 a)
{
    float l2 = dot(a, a);
    return (l2 < 1e-12) ? float3(1, 0, 0) : a * rsqrt(l2);
}

// ---- PCG hash RNG (matches Common.glsl pcg2d, used only when _Roughness>0) ----
uint2 pcg2d(uint2 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}
// internal RNG state, seeded per pixel/sample
static uint2 s0;
float  rand()  { s0 = pcg2d(s0); return float(s0.x) / float(0xffffffffu); }
float2 rand2() { s0 = pcg2d(s0); return float2(s0) / float(0xffffffffu); }

// Box slab intersection in world space, returns (tNear, tFar)
float2 insect(float3 ro, float3 rd, float3 p0, float3 p1)
{
    float3 inv = 1.0 / rd;
    float3 t0 = (p0 - ro) * inv;
    float3 t1 = (p1 - ro) * inv;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float dstA = max(max(tmin.x, tmin.y), tmin.z);
    float dstB = min(tmax.x, min(tmax.y, tmax.z));
    return float2(dstA, dstB);
}

float SampleDistance(float3 pos)
{
    float3 uvw = pos / _Size;
    if (any(uvw < -0.01) || any(uvw > 1.01)) return -1;
    return SAMPLE_TEXTURE3D_LOD(_Density, sampler_Density, uvw, 0).a;
}
// Normal from density gradient. Outside the field we read -1 (empty, matching
// SampleDistance) instead of the -100 sentinel the MC compute uses to CAP the
// mesh -- that sentinel explodes the gradient near walls and snaps the normal to
// a flat axis-aligned mirror. -1 keeps a natural magnitude so wall-hugging fluid
// still shows its real curvature.
float SampleDensityDelta(int axis, float3 coord)
{
    float3 off = 0; off[axis] = _Offset;
    float3 uvw0 = (coord + off) / float3(256, 128, 128);
    float3 uvw1 = (coord - off) / float3(256, 128, 128);
    float d0 = (uvw0[axis] > 1.00001f) ? -1 : SAMPLE_TEXTURE3D_LOD(_Density, sampler_Density, uvw0, 0).a;
    float d1 = (uvw1[axis] < -0.00001f) ? -1 : SAMPLE_TEXTURE3D_LOD(_Density, sampler_Density, uvw1, 0).a;
    return d1 - d0;
}
float3 SurfaceNormal(float3 pos)
{
    float3 coord = pos * 5;
    float dx = SampleDensityDelta(0, coord);
    float dy = SampleDensityDelta(1, coord);
    float dz = SampleDensityDelta(2, coord);
    return safeNorm(float3(dx, dy, dz));
}

// Ray-march the density field for the iso-surface. Returns t (<0 = miss).
// Bracketed linear refinement between the last two samples (deterministic).
float RaycastFluid(float3 ro, float3 rd, float tMin, float tMax, bool inside)
{
    float span = tMax - tMin;
    if (span <= 0) return -1;
    const int STEPS = 96;
    float stepSize = span / STEPS;
    float sig = inside ? -1.0 : 1.0;
    float prev = inside ? 1e3 : -1e3;
    [loop]
    for (int i = 0; i < STEPS; i++)
    {
        float t = tMin + stepSize * (i + 1);
        if (t > tMax) break;
        float d = SampleDistance(ro + rd * t);
        if (sig * d > sig * _Threshold)
        {
            // linear interpolate crossing between prev and d
            float tt = t - stepSize * (d - _Threshold) / (d - prev);
            return clamp(tt, tMin, tMax);
        }
        prev = d;
    }
    return -1;
}

// HDR-boosted cubemap background (ports BufferB Background())
float3 Background(float3 rd)
{
    float3 col = SAMPLE_TEXTURECUBE_LOD(_Cube, sampler_Cube, rd, 0).rgb;
    const float fakeHDR = 1.0;
    return pow(col, 2.0) + fakeHDR * col * clamp(exp(15.0 * (length(col) - 1.45)), 0.0, 2.0);
}

// Orthonormal tangent frame (ports BufferB basis())
void basis(float3 n, out float3 f, out float3 r)
{
    if (n.z < -0.999999) { f = float3(0, -1, 0); r = float3(-1, 0, 0); }
    else
    {
        float a = 1.0 / (1.0 + n.z);
        float b = -n.x * n.y * a;
        f = float3(1.0 - n.x * n.x * a, b, -n.x);
        r = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }
}

// GGX NDF half-vector sample around normal n (only used when _Roughness>0)
float3 SampleGGXNormal(float3 n, float alpha)
{
    if (alpha < 1e-4) return n;
    float3 f, r; basis(n, f, r);
    float2 xi = rand2();
    float phi = TWO_PI * xi.x;
    float cosT = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float3 h = float3(sinT * cos(phi), sinT * sin(phi), cosT);
    return safeNorm(h.x * f + h.y * r + h.z * n);
}

// Schlick fresnel with metallic F0 tint (gives edge->white brightening)
float3 FresnelMetal(float3 F0, float cosTheta)
{
    float m = pow(saturate(1.0 - cosTheta), 5.0);
    return F0 + (1.0 - F0) * m;
}

// Emissive key lights the metal reflects. Directions from the box center;
// distance scaled by _LightDistance so they sit well outside a large domain.
static const float3 LIGHT_DIR[4] = {
    float3( 0.6,  0.6, 0.5), float3(-0.6, -0.6, 0.5),
    float3(-0.6,  0.6, 0.5), float3( 0.6, -0.6, 0.5)
};
// Ray-sphere: nearest positive hit distance, MAX_DIST on miss.
float iSphere(float3 ro, float3 rd, float3 center, float radius)
{
    float3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return MAX_DIST;
    float d = -b - sqrt(h);
    return d > 0.0 ? d : MAX_DIST;
}

struct Hit
{
    float  t;        // distance along ray
    float3 normal;   // surface normal at hit
    float3 emission; // > 0 for lights
    bool   isFluid;  // true if the fluid iso-surface was hit
};

// Intersect the whole scene: fluid iso-surface (inside the box) + key lights.
Hit TraceScene(float3 ro, float3 rd, bool inside)
{
    Hit h; h.t = MAX_DIST; h.normal = 0; h.emission = 0; h.isFluid = false;

    // fluid: clip ray to the density box, then march
    float2 tb = insect(ro, rd, 0.0, _Size);
    float tN = max(tb.x, 0.0);
    if (tb.y > tN)
    {
        float td = RaycastFluid(ro, rd, tN, tb.y, inside);
        if (td > 0.0)
        {
            h.t = td;
            h.normal = SurfaceNormal(ro + rd * td);
            h.isFluid = true;
        }
    }

#if defined(ENABLE_LIGHTS)
    float dist = _LightDistance * length(_Size);
    float r = _LightRadius * ((_Size.x + _Size.y + _Size.z) / 3.0);
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        float3 c = _Size * 0.5 + normalize(LIGHT_DIR[i]) * dist;
        float td = iSphere(ro, rd, c, r);
        if (td < h.t)
        {
            h.t = td;
            h.normal = safeNorm(ro + rd * td - c);
            h.emission = _LightIntensity;
            h.isFluid = false;
        }
    }
#endif
    return h;
}

// Multi-bounce path trace. The primary fluid hit point + normal are supplied
// by the caller (rasterized surface for the mesh, or a box-march for the volume).
float3 PathTrace(float3 ro, float3 rd, float3 firstNormal)
{
    float3 radiance = 0.0;
    float3 throughput = _MetalColor.rgb; // metal has no diffuse albedo, pure specular tint
    float  alpha = _Roughness * _Roughness;

    float3 normal = firstNormal;
    int maxB = (int)_Bounces;
    [loop]
    for (int b = 0; b < maxB; b++)
    {
        // orient normal against the incoming ray
        if (dot(normal, rd) > 0.0) normal = -normal;

        float3 m = SampleGGXNormal(normal, alpha);
        float  cosTheta = saturate(dot(-rd, m));
        float3 F = FresnelMetal(_MetalColor.rgb, cosTheta);

        // metal: always reflect, weighted by fresnel
        rd = safeNorm(reflect(rd, m));
        ro = ro + normal * _Step * 2.0;
        throughput *= F;

        Hit h = TraceScene(ro, rd, false);
        if (h.t >= MAX_DIST)
        {
            radiance += throughput * Background(rd);
            break;
        }
        if (!h.isFluid) // hit a light
        {
            radiance += throughput * h.emission;
            break;
        }
        // bounced back onto the fluid: continue from the new surface point
        ro = ro + rd * h.t;
        normal = h.normal;

        if (max3(throughput) < 0.02) break;
    }
    return radiance;
}

float3 tonemap(float3 c)
{
    return tanh(pow(max(c, 0.0), 1.0 / 2.2));
}

// Shared multi-sample driver. worldSeed keeps per-pixel RNG decorrelated.
float3 RenderFluid(float3 ro, float3 rd, float3 firstNormal, float2 worldSeed)
{
    int spp = (int)_Samples;
    float3 col = 0.0;
    [loop]
    for (int s = 0; s < spp; s++)
    {
        s0 = uint2(asuint(worldSeed.x) ^ (uint)(s * 9781u),
                   asuint(worldSeed.y) ^ asuint(_Time.y) ^ (uint)(s * 6151u));
        col += PathTrace(ro, rd, firstNormal);
    }
    col /= max(1, spp);
    return tonemap(col);
}
#endif

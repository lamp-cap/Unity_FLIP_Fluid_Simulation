#ifndef FLIP_UTILS_HLSL
#define FLIP_UTILS_HLSL

#define GROUP_THREADS_X 128
#define GROUP_THREADS_D3 8, 4, 4

#define AIR 0
#define FLUID 1
#define SOLID 2

struct Particle
{
    uint2 packedPosition;
    uint2 packedVelocity;
};

int3 _GridSize;
float3 _GridMin;
int _NumParticles;
int _NumCells;
float _CellSize;
float _InvCellSize;
float _DeltaTime;

float2 OctWrap( float2 v )
{
    return ( 1.0 - abs(v.yx) ) * ( v >= 0.0 ? 1.0 : -1.0 );
}
 
float2 EncodeNormal(float3 n)
{
    n /= ( abs( n.x ) + abs( n.y ) + abs( n.z ) );
    n.xy = n.z >= 0.0 ? n.xy : OctWrap( n.xy );
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}
 
float3 DecodeNormal( float2 f )
{
    f = f * 2.0 - 1.0;
 
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    float3 n = float3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
    float t = saturate( -n.z );
    n.xy += n.xy >= 0.0 ? -t : t;
    return normalize( n );
}

inline uint Morton3DGetThirdBits(uint num) {
    uint x = num        & 0x49249249;
    x = (x ^ (x >> 2))  & 0xc30c30c3;
    x = (x ^ (x >> 4))  & 0x0f00f00f;
    x = (x ^ (x >> 8))  & 0xff0000ff;
    x = (x ^ (x >> 16)) & 0x0000ffff;
    return x;
}

inline uint3 MortonD3Decode(uint code)
{
    return uint3(Morton3DGetThirdBits(code), Morton3DGetThirdBits(code >> 1), Morton3DGetThirdBits(code >> 2));
}

inline uint Morton3DSplitBy3Bits(uint num) 
{
    uint x = num & 1023u;
    x = (x | (x << 16)) & 0xff0000ff;
    x = (x | (x << 8))  & 0x0f00f00f;
    x = (x | (x << 4))  & 0xc30c30c3;
    x = (x | (x << 2))  & 0x49249249;
    return x;
}

inline uint Morton3DEncode(uint x, uint y, uint z)
{
    return Morton3DSplitBy3Bits(x) | (Morton3DSplitBy3Bits(y) << 1) | (Morton3DSplitBy3Bits(z) << 2);
}
inline uint Morton3DEncode(uint3 v)
{
    return Morton3DSplitBy3Bits(v.x) | (Morton3DSplitBy3Bits(v.y) << 1) | (Morton3DSplitBy3Bits(v.z) << 2);
}

inline uint Coord2Idx(uint x, uint y, uint z)
{
    return Morton3DEncode(x, y, z);
    // return z * _GridSize.x * _GridSize.y + y * _GridSize.x + x;
}
inline uint Coord2Idx(uint3 coord)
{
    // return Morton3DEncode(coord.x, coord.y, coord.z);
    return Coord2Idx(coord.x, coord.y, coord.z);
}

inline uint PackUint3(uint3 v)
{
    return v.x | (v.y << 10) | (v.z << 20);
}

inline uint3 UnpackUint3(uint v)
{
    return uint3(v & 1023u, (v >> 10) & 1023u, (v >> 20) & 1023u);
}

inline uint2 EncodePosition(float3 pos)
{
    float3 cellPos = pos * _InvCellSize;
    return uint2(Morton3DEncode((uint3)floor(cellPos)), PackUint3(round(frac(cellPos) * 1023)));
}

inline uint3 PositionCoord(uint2 packedPos)
{
    return MortonD3Decode(packedPos.x);
}

inline float3 DecodePosition(uint2 packedPos)
{
    float3 coord = MortonD3Decode(packedPos.x);
    float3 localPos = UnpackUint3(packedPos.y) / 1023.0;
    return (coord + localPos) * _CellSize;
}

inline uint PackUNorm2(float2 v)
{
    uint2 coord = (uint2)round(v * 65535) & 65535u;
    return coord.x | (coord.y << 16);
}

inline float2 UnpackUNorm2(uint packed)
{
    return float2((packed & 65535) / 65535.0, (packed >> 16) / 65535.0);
}

inline float2 EncodeVelocity(float3 vel)
{
    float len = length(vel);
    return float2(asuint(len), len > 1e-8 ? PackUNorm2(EncodeNormal(vel / len)) : 0);
}

inline float3 DecodeVelocity(uint2 packedVel)
{
    float len = asfloat(packedVel.x);
    return len > 1e-8 ? len * DecodeNormal(UnpackUNorm2(packedVel.y)) : 0;
}

// inline uint3 Idx2Coord(uint id)
// {
//     return MortonD3Decode(id);
//     // return uint3(id % _GridSize.x, (id / _GridSize.x) % _GridSize.y, id / (_GridSize.x * _GridSize.y));
// }
inline float3 GetLinearWeight(float3 abs_x)
{
    return saturate(1.0f - abs_x);
}

inline float3 GetQuadraticWeight(float3 abs_x)
{
    return abs_x < 0.5f ? 0.75f - abs_x * abs_x : 0.5f * saturate(1.5f - abs_x) * saturate(1.5f - abs_x);
}

// #define USE_LINEAR_KERNEL

inline float GetWeight(float3 p_pos, float3 c_pos, float grid_inv_spacing)
{
    const float3 dist = abs((p_pos - c_pos) * grid_inv_spacing);

    #if defined(USE_LINEAR_KERNEL)
    const float3 weight = GetLinearWeight(dist);
    #else // defined(USE_QUADRATIC_KERNEL)
    const float3 weight = GetQuadraticWeight(dist);
    #endif

    return weight.x * weight.y * weight.z;
}

inline bool IsFluidCell(uint grid_type)
{
    return grid_type == FLUID;
}

inline bool IsSolidCell(uint grid_type)
{
    return grid_type == SOLID;
}
inline bool3 IsSolidCell(uint3 gridTypes)
{
    return gridTypes == SOLID;
}

inline uint GetMyType (uint grid_types) { return grid_types & 3; }
inline uint GetXPrevType (uint gridTypes) { return (gridTypes >>  2) & 3; }
inline uint GetXNextType (uint gridTypes) { return (gridTypes >>  4) & 3; }
inline uint GetYPrevType (uint gridTypes) { return (gridTypes >>  6) & 3; }
inline uint GetYNextType (uint gridTypes) { return (gridTypes >>  8) & 3; }
inline uint GetZPrevType (uint gridTypes) { return (gridTypes >> 10) & 3; }
inline uint GetZNextType (uint gridTypes) { return (gridTypes >> 12) & 3; }

inline uint3 GetPrevTypes (uint grid_types) { return uint3(GetXPrevType(grid_types), GetYPrevType(grid_types), GetZPrevType(grid_types)); }
inline uint3 GetNextTypes (uint grid_types) { return ((uint3)grid_types >> int3(4, 8, 12)) & 3; }

inline uint PackGridTyped(uint type, uint t_xp, uint t_xn, uint t_yp, uint t_yn, uint t_zp, uint t_zn)
{
    return type | (t_xp << 2) | (t_xn << 4) | (t_yp << 6) | (t_yn << 8) | (t_zp << 10) | (t_zn << 12);
}

inline bool IsActive(float x) { return abs(x) > 1e-5f; } 
inline bool3 IsActive(float3 x) { return abs(x) > 1e-5f; } 

inline float3 EnforceBoundaryCondition(float3 velocity, float4 coeff)
{
    if (!IsActive(coeff.x))
        return 0;
    return IsActive(coeff.yzw) ? velocity : max(0, velocity);
}

inline float3 EnforceBoundaryCondition(float3 velocity, uint gridTypes)
{
    if (IsSolidCell(GetMyType(gridTypes)))
        return 0;
    return IsSolidCell(GetPrevTypes(gridTypes)) ? max(0, velocity) : velocity;
}

#endif // FLIP_UTILS_HLSL
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragCoord = floor(fragCoord);
    Initialize(fragCoord, iFrame, iResolution.xy);
    
    vec3 pos = dim3from2(fragCoord);
    fragColor.zw = unpackvec3(floatBitsToUint(voxel(ch0, pos).x)).zy;
    
    vec3 pos4x = pos * float(BLOCK_SIZE);
    
    //Compute surface mask
    if(any(greaterThanEqual(pos4x, size3d))) 
    {
        return;
    }
    
    uvec2 hasSurface = uvec2(0, 0);
    loop(i, BLOCK_SIZE) loop(j, BLOCK_SIZE) loop(k, BLOCK_SIZE)
    {
        vec3 p = pos4x + vec3(i, j, k);
        uint surface = 0u;
        loop(ii, 2) loop(jj, 2) loop(kk, 2)
        {
            vec3 pp = p + vec3(ii, jj, kk);
            float density = unpackvec3(floatBitsToUint(voxel(ch0, pp).x)).z;
            surface |= (density >= IsoValue) ? 2u : 1u;
        }
        uint id = uint((i * BLOCK_SIZE + j) * BLOCK_SIZE + k);
        uint id0 = id / 32u;
        uint id1 = id - id0 * 32u;
        if(surface == 3u) { // Has both higher and lower than iso value, so it's a surface
           if(id0 == 0u) hasSurface.x |= 1u << id1;
           if(id0 == 1u) hasSurface.y |= 1u << id1;
        }
    }

    fragColor.xy = uintBitsToFloat(hasSurface);
}

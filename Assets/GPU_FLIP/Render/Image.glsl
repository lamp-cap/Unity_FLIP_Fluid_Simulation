vec3 tonemap(vec3 c)
{
    return tanh(1.0 * pow(c, vec3(1.0 / 2.2)));
}

#define DENOISE_R 4
vec3 denoiser(vec2 fragCoord)
{
    vec4 col = vec4(0.0);
    vec4 data0 = LOAD(ch0, fragCoord);
    vec3 c0 =unpackvec3(floatBitsToUint(data0.w));
    vec2 d0 = data0.yz;
    range(i, -DENOISE_R, DENOISE_R) range(j, -DENOISE_R, DENOISE_R)
    {
        vec2 dp = vec2(i, j);
        vec2 p = fragCoord + dp;
        vec4 data = LOAD(ch0, p);
        vec3 c = unpackvec3(floatBitsToUint(data.w));
        vec2 d = data.yz;
        float dist0 = length(dp) / float(DENOISE_R);
        float dist1 = distance(c0, c)*0.2;
        float dist2 = distance(d0, d)*0.2;
        float weight = exp(-sqr(dist0 + dist1 + dist2));
        col += vec4(c, 1.0) * weight;        
    }
    return col.xyz / col.w;
}

void mainImage( out vec4 col, in vec2 fragCoord )
{    
    fragCoord = floor(fragCoord);
    //col.xyz = unpackvec3(floatBitsToUint(LOAD(ch0, fragCoord).w));
    col.xyz = denoiser(fragCoord);
    //col.xyz = LOAD(ch0, fragCoord).yzz/300.0;
    col.xyz = tonemap(col.xyz);
}
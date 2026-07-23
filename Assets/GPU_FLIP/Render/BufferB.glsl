//antialiasing
#define AA ivec2(2,2)

#define LIQUID_METAL

#define METAL_COLOR vec3(0.800,0.604,0.180)
#define METAL_F0 1.0
#define METAL_ROUGHNESS 0.0

#define WATER_COLOR vec3(1,1,1)
#define WATER_ROUGHNESS 0.0
#define WATER_F0 0.05

#define TRANSP_ABSORB vec3(0.780,0.933,1.000)
#define TRANSP_IOR 1.35

#define FOV 2.5
#define MaxBounces 16

float checkerBoard(vec2 p) {
   return mod(floor(p.x) + floor(p.y), 2.);
}

vec3 safeNorm(vec3 a)
{
    float l2 = dot(a,a);
    return (l2 < 1e-6) ? vec3(1,0,0) : (a * inversesqrt(l2)); 
}

void basis(in vec3 n, out vec3 f, out vec3 r)
{
    if(n.z < -0.999999) {
        f = vec3(0 , -1, 0);
        r = vec3(-1, 0, 0);
    } else {
    	float a = 1./(1. + n.z);
    	float b = -n.x*n.y*a;
    	f = vec3(1. - n.x*n.x*a, b, -n.x);
    	r = vec3(b, 1. - n.y*n.y*a , -n.y);
    }
}

mat3 mat3FromNormal(in vec3 n)
{
    vec3 x; vec3 y;
    basis(n, x, y);
    return mat3(x,y,n);
}


vec3 ggxSample(vec3 wi, float alphax, float alphay, vec2 xi)
{   
    xi = mix(vec2(0.00), vec2(0.75), xi); //remove outliers, they don't look very nice
    //stretch view
    vec3 v = normalize(vec3(wi.x * alphax, wi.y * alphay, wi.z));

    //orthonormal basis
    vec3 t1 = (v.z < 0.9999) ? safeNorm(cross(v, vec3(0.0, 0.0, 1.0))) : vec3(1.0, 0.0, 0.0);
    vec3 t2 = cross(t1, v);

    //sample point with polar coordinates
    float a = 1.0 / (1.0 + v.z);
    float r = sqrt(xi.x);
    float phi = (xi.y < a) ? xi.y / a*PI : PI + (xi.y - a) / (1.0 - a) * PI;
    float p1 = r*cos(phi);
    float p2 = r*sin(phi)*((xi.y < a) ? 1.0 : v.z);

    //compute normal
    vec3 n = p1*t1 + p2*t2 + v*sqrt(1.0 - p1*p1 - p2*p2);

    //unstretch
    return normalize(vec3(n.x * alphax, n.y * alphay, n.z));
}

vec4 rand4blue()
{
    return texelFetch(ch2, ivec2((uvec2(pixel) + pcg2d(s1))%1024u), 0);
}


vec2 rand2blue()
{
    return rand4blue().xy;
}


vec3 getRay(vec2 angles, vec2 pos)
{
    mat3 camera = getCamera(angles);
    return normalize(transpose(camera)*vec3(FOV*pos.x, 1., FOV*pos.y));
}

float Density(vec3 p)
{
    return trilinear(ch1, p).z;
}

float DensityY(vec3 p)
{
    return trilinear(ch1, p).w;
}

vec3 DensityNormal(vec3 pos)
{
    const float h = 1.0;
    float d211 = Density(pos + vec3(h, 0, 0));
    float d121 = Density(pos + vec3(0, h, 0));
    float d112 = Density(pos + vec3(0, 0, h));
    float d011 = Density(pos + vec3(-h, 0, 0));
    float d101 = Density(pos + vec3(0, -h, 0));
    float d110 = Density(pos + vec3(0, 0, -h));
    return normalize(vec3(d211 - d011, d121 - d101, d112 - d110));
}

vec3 Background(vec3 rd)
{
    vec3 col = texture(iChannel3,  rd.yzx).xyz;
    const float fakeHDR = 1.0;
    const float brightness = 1.0;
    //return vec3(0.0);
    return brightness*(pow(col, vec3(2.0)) + fakeHDR*col*clamp(exp(15.0*(length(col) - 1.45)), 0.0, 2.0));
}

vec4 GetVoxelSamples(vec3 ro, vec3 rd, float mint, float deltat)
{   
    float y0 = Density(ro += rd * mint);
    float y1 = Density(ro += rd * deltat);
    float y2 = Density(ro += rd * deltat);
    float y3 = Density(ro += rd * deltat);
    return vec4(y0, y1, y2, y3) - IsoValue;
}

uvec2 getBlock(vec3 block)
{
    vec2 data = LOAD3D(ch1, block).xy;
    return floatBitsToUint(data);
}

bool getVoxel(vec3 pos, vec3 block, uvec2 blockData)
{
    vec3 blockPos = pos - block;
    uint id = uint((blockPos.x * float(BLOCK_SIZE) + blockPos.y) * float(BLOCK_SIZE) + blockPos.z);
    uint data = bool(id >> 5u) ? blockData.y : blockData.x;
    uint voxelFlag = (data >> (id & 31u)) & 1u;
    return bool(voxelFlag);
}

bool BlockRaycast(inout vec3 ro, vec3 rd, VoxelRayProps props, vec3 block, float tmax, uvec2 surfaceMask)
{
    bool blockEmpty = all(equal(surfaceMask, uvec2(0)));
    if(blockEmpty) return false;
    VoxelRay ray = CreateVoxelRay(ro, props);
    bool hit = false;
    for(int i = 0; i < 12; i++) 
    {
        vec4 next = ComputeNextVoxel(ray);
        bool hasSurface = getVoxel(ray.voxelPos, block, surfaceMask);
        if(hasSurface) { 
            float tdNew = ray.curTraveled;
            vec3 roNew = ro + rd * tdNew;

            //intersect the trilinear surface
            float mint = -5e-5, maxt = next.w - tdNew, deltat = (maxt - mint) / 3.0;
            vec4 ys = GetVoxelSamples(roNew, rd, mint, deltat);
            vec2 result = iIsoSurf4Samples(mint, deltat, ys);

            if (result.y > 0.5) {
                hit = true;
                ro = roNew + rd * result.x;
                break; 
            }
        } 
        StepVoxelRay(ray, next);
        if(ray.curTraveled >= tmax) break;
    }
    return hit;
}

bool GridRaycast(inout vec3 ro, vec3 rd, float maxt)
{
    ro = ro / float(BLOCK_SIZE);
    maxt = maxt / float(BLOCK_SIZE);

    VoxelRayProps props = CreateVoxelRayProps(rd);
    VoxelRay ray = CreateVoxelRay(ro, props);

    bool hit = false;
    
    for(int i = 0; i < 128; i++) 
    {   
        vec4 next = ComputeNextVoxel(ray);
        uvec2 surfaceMask = getBlock(ray.voxelPos);
        vec3 roNew = (ro + rd*ray.curTraveled) * float(BLOCK_SIZE);
        if(BlockRaycast(roNew, rd, ray.props, ray.voxelPos * float(BLOCK_SIZE), (min(next.w, maxt) - ray.curTraveled) * float(BLOCK_SIZE), surfaceMask)) {
            ro = roNew;
            hit = true;
            break;
        }
        StepVoxelRay(ray, next);
        if(ray.curTraveled > maxt) break;
    }
    return hit;
}

struct HitProp
{
    float td;
    bool opaque;
    vec3 albedo;
    vec3 normal;
    vec3 emission;
    float roughness;
    float F0;
};

HitProp iWater(vec3 ro, vec3 rd)
{
    vec3 ro0 = ro;
    vec2 tdBox = iBox(ro - (size3d-1.0) * 0.5, rd, 0.5*(size3d - 1.0) - 1e-3);
    float td = max(tdBox.x, 0.0);
    ro += td * rd;
    float maxt = tdBox.y - td;
    #ifdef LIQUID_METAL
    HitProp result = HitProp(MAX_DIST, true, METAL_COLOR, vec3(0.0), vec3(0.0), METAL_ROUGHNESS, METAL_F0);
    #else
    HitProp result = HitProp(MAX_DIST, false, WATER_COLOR, vec3(0.0), vec3(0.0), WATER_ROUGHNESS, WATER_F0);
    #endif
    if(tdBox.y < MAX_DIST) {
        if(GridRaycast(ro, rd, maxt)) {
            result.td = length(ro - ro0);
            result.normal = normalize(DensityNormal(ro));
            //result.emission = DensityY(ro) * vec3(1,1,1);
        } 
    }

    return result;
}

HitProp iPlane(vec3 ro, vec3 rd, vec3 planeNormal, float planeDist) {
    float a = dot(rd, planeNormal);
    float d = -(dot(ro, planeNormal)+planeDist)/a;
    HitProp result = HitProp(MAX_DIST, true, vec3(0.9), vec3(0.0), vec3(0.0), 0.15, 1.0);
    if (a < 0.0 && d > 0.0) {
        result.normal = -planeNormal;
    	result.td = d;
        float check = checkerBoard(0.05*(ro.xy + rd.xy*d));
        result.roughness = 0.005 + check*0.15;
        result.albedo = (check > 0.5) ? vec3(0.016,0.161,0.165) : vec3(0.655,0.773,0.867);
        
    }
    return result;
}

HitProp iSphere( in vec3 ro, in vec3 rd, float sphereRadius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - sphereRadius*sphereRadius;
    float h = b*b - c;
    HitProp result = HitProp(MAX_DIST, true, vec3(0.9), vec3(0.0), vec3(0.0), 0.15, 0.0);
    if (h >= 0.) {
	    h = sqrt(h);
        float d1 = -b-h;
        float d2 = -b+h;
        if (d1 >= 0.0) {
            result.td = d1;
            result.normal = normalize(ro + rd*d1);
            result.emission = vec3(12.0);
        }
    }
    return result;
}


HitProp Combine(HitProp a, HitProp b)
{
    if (a.td < b.td) return a; else return b;
}

HitProp TraceRay(inout vec3 ro, vec3 rd)
{
    HitProp result = iWater(ro, rd);
    result = Combine(result, iPlane(ro, rd, vec3(0,0,1), 0.0));
    //result = Combine(result, iPlane(ro, rd, vec3(0,0,-1), 100.));
    result = Combine(result, iSphere(ro - vec3(300,300,100), rd, 150.0));
    result = Combine(result, iSphere(ro - vec3(-150,-150,100), rd, 150.0));
    result = Combine(result, iSphere(ro - vec3(-150,300,100), rd, 150.0));
    result = Combine(result, iSphere(ro - vec3(300,-150,100), rd, 150.0));
    //result = Combine(result, iSphere(ro - size3d*0.5, rd, 15.0));
    ro += rd * result.td;
    return result;
}

struct Ray
{
    vec3 ro;
    vec3 rd;
    vec3 incoming;
    vec3 absorption;
    vec3 data;
    bool inside;
    float totalTraveled;
};

vec2 sampleDisk(vec2 xi)
{
	float theta = TWO_PI * xi.x;
	float r = sqrt(xi.y);
	return vec2(cos(theta), sin(theta)) * r;
}

vec3 cosineHemisphere(vec2 xi)
{
    vec2 disk = sampleDisk(xi);
	return vec3(disk.x, disk.y, sqrt(max(0.0, 1.0 - dot(disk, disk))));
}

vec3 clampVec(vec3 dir, vec3 normal)
{
    float a = dot(dir, normal);
    return (a < 0.0) ? -dir : dir;
}

void PathTrace(inout Ray ray)
{
    float cameraDensity = Density(ray.ro);
    ray.inside = (cameraDensity > IsoValue) && InsideSimDomain(ray.ro);
    ray.incoming      = vec3(0.0);
    ray.absorption    = vec3(1.0);
    ray.totalTraveled = 0.0;
    ray.data = vec3(0.0);
    int bounce = 0; 
    for(;bounce < MaxBounces; bounce++)
    {
        HitProp hit = TraceRay(ray.ro, ray.rd);
        ray.totalTraveled += hit.td;
        if(bounce == 0) ray.data.x = hit.td;
        if(bounce == 1) ray.data.y = hit.td;
        //if(bounce == 1) bounces.y = hit.td;
        if(hit.td < MAX_DIST)
        {
            vec3 normal = hit.normal;
            normal = ray.inside ?  normal : -normal;
            
            mat3 basis = mat3FromNormal(normal);
            mat3 inv = transpose(basis);
            vec3 rd_local = inv*ray.rd;
            
            vec4 rng = rand4blue();
            vec3 M = ggxSample(-rd_local, hit.roughness, hit.roughness, rng.xy);
 
            float n1 = ray.inside ? 1.0 : TRANSP_IOR;
            float n2 = ray.inside ? TRANSP_IOR : 1.0;

            if(ray.inside)
            {
                vec3 addedAbsorption = exp(- 0.25*(1.0 - TRANSP_ABSORB) * hit.td);
                ray.absorption *= addedAbsorption;
            }
            
            vec3 reflDir = reflect(rd_local, M);
            vec3 refrDir = refract(rd_local, M, n2 / n1);
            float kS = mix(fresnelFull(reflDir, refrDir, M, n2, n1), 1.0, hit.F0);
            float kD = 1.0 - kS;
            
            float reflProb = pow(kS, 0.5);
            bool doReflection = rng.z <= reflProb;
            
            ray.absorption *= hit.albedo;
            ray.incoming += hit.emission * ray.absorption;

            if(doReflection)
            {   // specular reflection
               ray.ro += 0.02 * normal;
               ray.rd = reflDir;
               ray.absorption *= kS / reflProb;
            }
            else
            {   
                if(hit.opaque) // scatter back outside in random dir after refraction
                {
                    ray.ro += 0.02 * normal;
                    ray.rd  = cosineHemisphere(rng.xy);
                }
                else // just refraction
                {
                    ray.ro -= 0.02 * normal;
                    ray.rd  = refrDir;
                    ray.inside = !ray.inside;
                }
                ray.absorption *= kD / (1.0 - reflProb);
            }
            ray.rd = safeNorm(basis*ray.rd);
            
            if(max(ray.absorption.x, max(ray.absorption.y, ray.absorption.z)) < 0.075) break;
        }
        else
        {
            ray.incoming += Background(ray.rd) * ray.absorption;
            break;
        }
    }
    //ray.incoming = vec3(bounce)/float(MaxBounces);
    //ray.incoming = ray.inside?vec3(1,1,1):vec3(0,0,0);
    //ray.incoming = bounces / 200.0;
}


Ray render(vec2 fragCoord, vec2 offset)
{
    Initialize(fragCoord * vec2(AA) + offset, iFrame, iResolution.xy);
    fragCoord += 1.0*rand4blue().xy;
    vec2 uv = (fragCoord - 0.5*R) / max(R.x, R.y);
    vec2 angles = vec2(2.*PI, PI)*(iMouse.xy/iResolution.xy - 0.5);

    if(iMouse.z <= 0.)
    {
        angles = vec2(0.04 + 0.2*iTime, -0.4);
    }
    vec3 rd = getRay(angles, uv);
    vec3 center_rd = getRay(angles, vec2(0.));
 
    float d = sqrt(dot(vec3(size3d), vec3(size3d)))*0.65;
    vec3 ro = vec3(size3d)*vec3(0.5, 0.5, 0.5) - center_rd*d;


    Ray ray;
    ray.ro = ro;
    ray.rd = rd;
    PathTrace(ray);
    
    return ray;
}


void AddDensity(inout Particle p, in Particle incoming, float rad)
{
    if(incoming.mass == 0u) return;
    float d = distance(p.pos, incoming.pos);
    float mass = float(incoming.mass);
    p.density += mass*GD(d,rad);
}

vec2 AddRenderDensity(vec3 pos, in Particle incoming, float rad)
{
    if(incoming.mass == 0u) return vec2(0.0);
    float d = distance(pos, incoming.pos);
    float mass = float(incoming.mass);
    float weight = mass*GD(d,rad);
    return vec2(1.0, smoothstep(0.7, 0.8, length(incoming.vel))) * weight;
}

//compute particle SPH densities
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragCoord = floor(fragCoord);
    Initialize(fragCoord, iFrame, iResolution.xy);
    vec3 pos = dim3from2(fragCoord);
    
    Particle p0, p1;
    vec2 dens = vec2(0.0);
    
    //load the particles
    vec4 packed = LOAD3D(ch0, pos);
    unpackParticles(packed, pos, p0, p1);
    
    range(i, -2, 2) range(j, -2, 2) range(k, -2, 2)
    {
        int dist = i*i + j*j + k*k;
        if(dist == 0 || dist > 16) continue;
        vec3 pos1 = pos + vec3(i, j, k);
        Particle p0_, p1_;
        unpackParticles(LOAD3D(ch0, pos1), pos1, p0_, p1_);

        if(p0.mass > 0u)
        {
            AddDensity(p0, p0_, KERNEL_RADIUS);
            AddDensity(p0, p1_, KERNEL_RADIUS);
        }
        if(p1.mass > 0u)
        {
            AddDensity(p1, p0_, KERNEL_RADIUS);
            AddDensity(p1, p1_, KERNEL_RADIUS);
        }
        dens += AddRenderDensity(pos, p0_, RENDER_KERNEL_RADIUS);
        dens += AddRenderDensity(pos, p1_, RENDER_KERNEL_RADIUS);
    }

    if(p0.mass > 0u)
    {
        AddDensity(p0, p0, KERNEL_RADIUS);
        AddDensity(p0, p1, KERNEL_RADIUS);
    }
    if(p1.mass > 0u)
    {
        AddDensity(p1, p0, KERNEL_RADIUS);
        AddDensity(p1, p1, KERNEL_RADIUS);
    }
    dens += AddRenderDensity(pos, p0, RENDER_KERNEL_RADIUS);
    dens += AddRenderDensity(pos, p1, RENDER_KERNEL_RADIUS);
    
    if(any(lessThan(pos, vec3(1.0))) || any(greaterThan(pos, size3d - 2.0))) dens = vec2(0.0);
    
    fragColor.x = uintBitsToFloat(packvec3(vec3(p0.density, p1.density, dens.x)));
    
    float blend = 0.7;
    vec3 col = vec3(0.0);
    vec3 data = vec3(0.0);
    
    loop(i, AA.x) loop(j, AA.y) {
        Ray r = render(fragCoord, vec2(i,j));
        col += r.incoming;
        data += r.data;
    }
    col /= float(AA.x * AA.y);
    data /= float(AA.x * AA.y);
    
    fragColor.w = uintBitsToFloat(packvec3(col.xyz));
    fragColor.yz = data.xy *vec2(1.0, 0.0);
}

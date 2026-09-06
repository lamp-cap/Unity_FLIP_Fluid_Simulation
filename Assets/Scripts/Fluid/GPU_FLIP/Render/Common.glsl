#define ch0 iChannel0
#define ch1 iChannel1
#define ch2 iChannel2
#define ch3 iChannel3

#define R iResolution.xy

#define LOAD(ch, pos) texelFetch(ch, ivec2(pos), 0)
#define LOAD3D(ch, pos) texelFetch(ch, ivec2(dim2from3(pos)), 0)

#define PI 3.1415926535
#define TWO_PI 6.28318530718

#define light_dir normalize(vec3(0.741,1.000,0.580))

#define initial_particle_density 1u
#define dt (isKeyPressed(KEY_P) ? 0.0 : 1.0)
#define rest_density 0.45
#define gravity 0.01

#define KERNEL_RADIUS 1.0
#define RENDER_KERNEL_RADIUS 1.1
#define PRESSURE 2.5
#define PRESSURE_RAD 0.8
#define VISCOSITY 0.65
#define SPIKE_KERNEL 0.5
#define SPIKE_RAD 0.75

#define force_mouse 0.005
#define force_mouse_rad 40.0
#define force_boundary 5.0
#define boundary_h 5.0
#define max_velocity 1.0
#define cooling 0.0

#define MAX_DIST 1e10
#define BLOCK_SIZE 4

#define IsoValue 0.243

//3d slice aspect ratio (approximate)
#define ar vec2(1.0,0.4)
vec2 SCALE;
vec2 size2d;
vec3 size3d;
vec2 iRes;

//internal RNG state 
uvec2 s0, s1; 
vec2 pixel;

void Initialize(vec2 p, int frame, vec2 iR)
{
    pixel = p;
    s0 = uvec2(p.x, uint(frame)*145175u + uint(p.y));
    s1 = uvec2(frame, 0);
    iRes = iR;
    iR = min(vec2(1280, 900), iR);
    size2d = iR;
    SCALE = floor(ar*pow(size2d.x*size2d.y,0.166666666))+1.0;
    size3d = vec3(floor(size2d.xy/SCALE)-2.0, SCALE.x*SCALE.y);
}

vec2 dim2from3(vec3 p3d)
{
    //p3d = clamp(p3d, vec3(0.0), size3d);
    float ny = float(uint(floor(p3d.z)) / uint(SCALE.x));
    float nx = floor(p3d.z) - ny*SCALE.x;
    return vec2(nx, ny)*vec2(size3d.xy) + p3d.xy;
}

vec3 dim3from2(vec2 p2d)
{
    vec2 tile = floor(p2d/size3d.xy);
    return vec3(p2d - size3d.xy*tile, tile.x + SCALE.x*tile.y);
}


#define pixel(a, p, s) texture(a, (p+0.5)/vec2(s))

vec4 voxel(sampler2D ch, vec3 p3d)
{
    return pixel(ch, dim2from3(p3d), iRes);
}

//trilinear interpolation = linear interp between layers
vec4 trilinear(sampler2D ch, vec3 p3d)
{
    return mix(voxel(ch, vec3(p3d.xy, floor(p3d.z))),voxel(ch, vec3(p3d.xy, ceil(p3d.z))), fract(p3d.z));
}


bool InsideSimDomain(vec3 pos)
{
    return all(greaterThan(pos, vec3(0.0))) && all(lessThan(pos, size3d));
}

float sqr(float x)
{
    return x * x;
}

float cub(float x)
{
    return x*x*x;
}

float Pressure(float rho)
{
    return (rho/rest_density - 1.0)/max(rho*rho, 0.001);
}

float Gaussian(float r, float d)
{
    float norm = 1.0/(cub(d)*sqrt(cub(TWO_PI)));
    return norm * exp(-0.5*sqr(r/d));
}

vec3 GaussianGrad(vec3 dx, float d)
{
    float norm = 1.0/(cub(d)*sqrt(cub(TWO_PI)));
    float r = length(dx);
    return - (norm/sqr(d)) * exp(-0.5*sqr(r/d)) * dx;
}

#define GD(x, R) Gaussian(length(x),R)
#define GGRAD(x, R) GaussianGrad(x, R)
#define GS(x) exp(-dot(x,x))
#define DIR(phi) vec2(cos(TWO_PI*phi),sin(TWO_PI*phi))

#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)

float sdBox( vec3 p, vec3 b )
{
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}


// https://www.pcg-random.org/
uvec2 pcg2d(inout uvec2 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v>>16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v>>16u);
    return v;
}

float rand(){ return float(pcg2d(s0).x)/float(0xffffffffu); }
vec2 rand2(){ return vec2(pcg2d(s0))/float(0xffffffffu); }

vec2 nrand2(float sigma, vec2 mean)
{
	vec2 Z = rand2();
    return mean + sigma * sqrt(-2.0 * log(Z.x)) * 
           vec2(cos(TWO_PI * Z.y),sin(TWO_PI * Z.y));
}

vec3 udir(vec2 rng)
{
    float phi = 2.*PI*rng.x;
    float ctheta = 2.*rng.y-1.;
    float stheta = sqrt(1.0-ctheta*ctheta);
    return vec3(cos(phi)*stheta, sin(phi)*stheta, ctheta);
}

struct Particle 
{
    uint mass;
    vec3 pos;
    vec3 vel;
    vec3 force;
    float density;
};

//5 bits for shared exponent, 9 bits for each component
uint packvec3(vec3 v)
{
    //get the exponent
    float maxv = max(abs(v.x), max(abs(v.y), abs(v.z)));
    int exp = clamp(int(ceil(log2(maxv))), -15, 15);
    float scale = exp2(-float(exp));
    uvec3 sv = uvec3(round(clamp(v*scale, -1.0, 1.0) * 255.0) + 255.0);
    uint packed = uint(exp + 15) | (sv.x << 5) | (sv.y << 14) | (sv.z << 23);
    return packed;
}

vec3 unpackvec3(uint packed)
{
    int exp = int(packed & 0x1Fu) - 15;
    vec3 sv = vec3((packed >> 5) & 0x1FFu, (packed >> 14) & 0x1FFu, (packed >> 23) & 0x1FFu);
    vec3 v = (sv - 255.0) / 255.0;
    v *= exp2(float(exp));
    return v;
}

uint packmasspos(uint mass, vec3 p0)
{
    uvec3 pos0 = uvec3(clamp(p0, 0.0, 1.0) * 255.0);
    return mass | (pos0.x << 8) | (pos0.y << 16) | (pos0.z << 24);
}

uint packMassPos(uint mass, vec3 pos)
{
    uvec3 pos0 = uvec3(clamp(pos, 0.0, 1.0) * 255.0); // Assuming pos range [0, 1] in a cell
    uint data1 = mass | (pos0.x << 8) | (pos0.y << 16) | (pos0.z << 24);
    return data1;
}

void unpackMassPos(uint packed, out uint mass, out vec3 pos)
{
    mass = packed & 0xFFu;
    uvec3 pos0 = uvec3((packed >> 8) & 0xFFu, (packed >> 16) & 0xFFu, (packed >> 24) & 0xFFu);
    pos = vec3(pos0) / 255.0;
}

vec4 packParticles(Particle p0, Particle p1, vec3 pos)
{
    p0.pos -= pos;
    p1.pos -= pos;

    uvec4 data = uvec4(
        packMassPos(p0.mass, p0.pos),
        packMassPos(p1.mass, p1.pos),
        packvec3(p0.vel),
        packvec3(p1.vel)
    );

    return uintBitsToFloat(data);
}

vec2 packForce(Particle p0, Particle p1)
{
    uvec2 data = uvec2(
        packvec3(p0.force),
        packvec3(p1.force)
    );

    return uintBitsToFloat(data);
}

void unpackParticles(vec4 packed, vec3 pos, out Particle p0, out Particle p1)
{
    uvec4 data = floatBitsToUint(packed);
    
    unpackMassPos(data.x, p0.mass, p0.pos);
    unpackMassPos(data.y, p1.mass, p1.pos);

    p0.pos += pos;
    p1.pos += pos;

    p0.vel = unpackvec3(data.z);
    p1.vel = unpackvec3(data.w);
}

void unpackForce(vec2 packed, inout Particle p0, inout Particle p1)
{
    uvec2 data = floatBitsToUint(packed);

    p0.force = unpackvec3(data.x);
    p1.force = unpackvec3(data.y);
}

float sdBox( in vec2 p, in vec2 b )
{
    vec2 d = abs(p)-b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

int ClosestCluster(Particle p0, Particle p1, Particle incoming)
{
    //first try to choose the particle with significantly smaller mass
    if(float(p0.mass) < 0.01*float(p1.mass) || float(p1.mass) < 0.01*float(p0.mass))
    {
        return p0.mass < p1.mass ? 0 : 1;
    }

    //otherwise choose the closest one
    float d0 = length(p0.pos - incoming.pos);
    float d1 = length(p1.pos - incoming.pos);
    return d0 < d1 ? 0 : 1;
}

void BlendParticle(inout Particle p, in Particle incoming)
{
    uint newMass = p.mass + incoming.mass;
    vec2 weight = vec2(p.mass, incoming.mass) / float(newMass);
    p.pos = p.pos*weight.x + incoming.pos*weight.y;
    p.vel = p.vel*weight.x + incoming.vel*weight.y;
    p.mass = newMass;
}

void IntegrateParticle(inout Particle p, float timeStep)
{
    p.vel += p.force * timeStep;

    //velocity limit
    float v = length(p.vel)/max_velocity;
    p.vel /= (v > 1.)?v:1.;
    
    p.pos += p.vel * timeStep;
}

void Clusterize(inout Particle p0, inout Particle p1, in Particle incoming, vec3 pos, float timeStep)
{
    if(incoming.mass == 0u) return;
    //move incoming particle forwards in time
    IntegrateParticle(incoming, timeStep);
    //check if the incoming particle is in the cell
    if(!all(equal(pos, floor(incoming.pos)))) return;
    
    int closest = ClosestCluster(p0, p1, incoming);
    if(closest == 0)
    {
        BlendParticle(p0, incoming);
    }
    else
    {
        BlendParticle(p1, incoming);
    }
}

void SplitParticle(inout Particle p1, inout Particle p2)
{
    vec3 pos = p1.pos;
    uint newMass = p1.mass;
    p1.mass = newMass/2u;
    p2.mass = newMass - p1.mass;
    vec3 dir = udir(rand2());
    p1.pos = pos - dir*5e-3;
    p2.pos = pos + dir*5e-3;
    p2.vel = p1.vel;
}

void ApplyForce(inout Particle p, in Particle incoming)
{
    if(incoming.mass == 0u) return;

    vec3 dx = incoming.pos - p.pos;
    vec3 dv = incoming.vel - p.vel;
    float d = length(dx);
    vec3 dir = dx / max(d, 1e-3);
    
    float rho0 = p.density;
    float rho1 = incoming.density;
    float mass0 = float(p.mass);
    float mass1 = float(incoming.mass);
  
    vec3 ggrad = GGRAD(dx, PRESSURE_RAD);
    float pressure = 0.5*p.density*(Pressure(p.density) + Pressure(incoming.density));
    vec3 F_SPH = - PRESSURE * pressure * ggrad;
    vec3 F_VISC = VISCOSITY * dot(dir, dv) * ggrad;
    vec3 F_SPIKE = SPIKE_KERNEL * GD(d, SPIKE_RAD) * dir;
   
    p.force += - (F_SPH + F_VISC + F_SPIKE) * mass1;
}

float minv(vec3 a)
{
    return min(min(a.x, a.y),a.z);
}

float maxv(vec3 a)
{
    return max(max(a.x, a.y),a.z);
}

float distance2border(vec3 p)
{
    vec3 a = vec3(size3d - 1.) - p;
    return min(minv(p),minv(a)) + 1.;
}

vec4 border_grad(vec3 p)
{
    const float dx = 0.001;
    const vec3 k = vec3(1,-1,0);
	return  (k.xyyx*distance2border(p + k.xyy*dx) +
			 k.yyxx*distance2border(p + k.yyx*dx) +
			 k.yxyx*distance2border(p + k.yxy*dx) +
			 k.xxxx*distance2border(p + k.xxx*dx))/vec4(4.*dx,4.*dx,4.*dx,4.);
}

void FinalizeForce(inout Particle p, float time)
{
    p.force = p.force;/// max(0.0001, float(p.mass));
    p.force += gravity*vec3(0.4*sin(1.5*time), 0.2*cos(0.75*time), -1.0); //gravity
    vec4 border = border_grad(p.pos);
    vec3 bound =1.*normalize(border.xyz)*exp(-0.4*border.w*border.w);
    p.force += force_boundary * bound;
}

vec3 hsv2rgb( in vec3 c )
{
    vec3 rgb = clamp( abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );

	rgb = rgb*rgb*(3.0-2.0*rgb); // cubic smoothing	

	return c.z * mix( vec3(1.0), rgb, c.y);
}

//Undefine to only use 2 newton steps
//#define FULL_PRECISION

float EvalCubic(float x, vec4 k)
{
    return ((k.x * x + k.y) * x + k.z) * x + k.w;
}

float EvalQuadratic(float x, vec3 k)
{
    return (k.x * x + k.y) * x + k.z;
}

float NewtonIteration(float x, vec4 k3, vec3 k2)
{
    return x - EvalCubic(x, k3) / EvalQuadratic(x, k2);
}

float LinearSolve(float x1, float x2, float y1, float y2)
{
    return x1 + (x2 - x1) * (0.0 - y1) / (y2 - y1);
}

vec2 SolveSegment(vec4 k3, vec3 k2, float mint, float maxt, float c0, float c1)
{
    float x = LinearSolve(mint, maxt, c0, c1);
    
#ifdef FULL_PRECISION
    //usually you dont need more than 1 newton step for actual surface inside the voxel, but still can get weird artifacts
    for(int i = 0; i < 16; i++) 
    {
        float xprev = x;
        x = clamp(NewtonIteration(x, k3, k2), mint, maxt);
        x = NewtonIteration(x, k3, k2);
        if(abs(x-xprev) < 1e-2) break;
    }
#else
    x = clamp(NewtonIteration(x, k3, k2), mint, maxt);
    x = NewtonIteration(x, k3, k2);
    //x = NewtonIteration(x, k3, k2);
#endif
    
    return vec2(x, 1.0);
}

vec2 solveCubicFirstRootApprox(vec4 k3, float mint, float maxt, float cmin, float cmax)
{
    vec3 k2 = vec3(3.0 * k3.x, 2.0 * k3.y, k3.z); // 3ax^2 + 2bx + c, derivative of cubic
    float D = k2.y * k2.y - 4.0 * k2.x * k2.z;
    vec4 segment = vec4(0.0);

    if(D <= 0.0) { // one potential segment with root
        if(cmin * cmax > 0.0) return vec2(0.0); // no root in this segment
        segment = vec4(mint, maxt, cmin, cmax);
    } else { // 3 potential segments with roots
        float r = 1.0 / (2.0 * k2.x);
        float b = sqrt(D) * r;
        float xa = - k2.y * r - b;
        float xb = - k2.y * r + b;
        float xmin = clamp(min(xa, xb), mint, maxt);
        float xmax = clamp(max(xa, xb), mint, maxt);

        float cmin0 = EvalCubic(xmin, k3);
        float cmax0 = EvalCubic(xmax, k3);
        
        if(cmin * cmin0 < 0.0) segment = vec4(mint, xmin, cmin, cmin0); //root in first segment
        else if(cmin0 * cmax0 < 0.0) segment = vec4(xmin, xmax, cmin0, cmax0); //root in second segment
        else if(cmax0 * cmax < 0.0) segment = vec4(xmax, maxt, cmax0, cmax); //root in third segment
        else return vec2(0.0); //no roots in any segment
    }

    return SolveSegment(k3, k2, segment.x, segment.y, segment.z, segment.w);
}

vec2 solveCubicFirstRootApprox(vec4 k3, float mint, float maxt)
{
    float cmin = EvalCubic(mint, k3);
    float cmax = EvalCubic(maxt, k3);
    return solveCubicFirstRootApprox(k3, mint, maxt, cmin, cmax);
}

bool isSameSign(vec4 ys)
{
    return ys.x*ys.w > 0.0;
    bvec4 sgn = greaterThan(ys, vec4(0.0));
    return (sgn.x == sgn.w) && (sgn.x == sgn.z) && (sgn.x == sgn.w);
}

// Intersection from 4 trilinear samples along the ray
vec2 iIsoSurf4Samples(float mint, float deltat, vec4 ys)
{
    if(isSameSign(ys)) return vec2(0.0);
    //get cubic from 4 samples along ray (at equal intervals)
    float t1 = 3.0 * ys.y + ys.w;
    float t2 = 3.0 * ys.z + ys.x;
    float c3 = (t1 - t2) / 6.0;
    float c2 = (ys.z + ys.x) / 2.0 - ys.y;
    float c1 = (ys.z - ys.x) / 2.0 - c3;
    float c0 = ys.y;
    vec2 sol = solveCubicFirstRootApprox(vec4(c3, c2, c1, c0), -1.0, 2.0, ys.x, ys.w);
    sol.x = (sol.x + 1.0) * deltat + mint; //remap sol.x to [mint, maxt]
    return sol;
}


mat3 getCamera(vec2 angles)
{
   mat3 theta_rot = mat3(1,   0,              0,
                          0,  cos(angles.y),  -sin(angles.y),
                          0,  sin(angles.y),  cos(angles.y)); 
        
   mat3 phi_rot = mat3(cos(angles.x),   sin(angles.x), 0.,
        		       -sin(angles.x),   cos(angles.x), 0.,
        		        0.,              0.,            1.); 
        
   return theta_rot*phi_rot;
}

vec2 iBox( in vec3 ro, in vec3 rd, in vec3 boxSize ) 
{
    vec3 m = sign(rd)/max(abs(rd), 1e-8);
    vec3 n = m*ro;
    vec3 k = abs(m)*boxSize;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

	float tN = max( max( t1.x, t1.y ), t1.z );
	float tF = min( min( t2.x, t2.y ), t2.z );
	
    if (tN > tF || tF <= 0.) {
        return vec2(MAX_DIST);
    } else {
        return vec2(tN, tF);
    }
}

struct VoxelRayProps
{
    vec3 rayInv;
    vec3 rayMask;
    vec3 rayStep;
};

VoxelRayProps CreateVoxelRayProps(vec3 rd)
{
    VoxelRayProps props;
    props.rayInv = sign(rd) / max(abs(rd), vec3(1e-6));
    props.rayMask = step(vec3(0.0), rd);
    props.rayStep = props.rayMask * 2.0 - 1.0;
    return props;
}

struct VoxelRay
{
    vec3 voxelPos;
    vec3 offset;
    float curTraveled;
    VoxelRayProps props;
};

VoxelRay CreateVoxelRay(vec3 ro, VoxelRayProps props)
{
    VoxelRay ray;
    ray.voxelPos = floor(ro);
    ray.offset = (props.rayMask - ro - props.rayStep*3e-6) * props.rayInv;
    ray.curTraveled = 0.0;
    ray.props = props;
    return ray;
}

vec4 ComputeNextVoxel(in VoxelRay ray)
{
    vec3 planeDistances = ray.voxelPos * ray.props.rayInv + ray.offset;
    float nextDistance = min(min(planeDistances.x, planeDistances.y), planeDistances.z); 
    vec3 nextVoxel = ray.voxelPos + vec3(equal(planeDistances, vec3(nextDistance))) * ray.props.rayStep;
    return vec4(nextVoxel, nextDistance);
}

void StepVoxelRay(inout VoxelRay ray, vec4 next)
{
    ray.voxelPos = next.xyz;
    ray.curTraveled = next.w;
}

float fresnelFull(vec3 Refl, vec3 Refr, vec3 Norm, float n1, float n2)
{
    float cosRefl = abs(dot(Refl, Norm));
    float cosRefr = abs(dot(Refr, Norm));
    float Rpar = (n1*cosRefl - n2*cosRefr) / (n1*cosRefl + n2*cosRefr);
    float Rper = (n1*cosRefr - n2*cosRefl) / (n1*cosRefr + n2*cosRefl);
    return 0.5*(Rpar*Rpar + Rper*Rper);
}

float NDF_ggx(vec3 m, vec3 n, float alpha)
{
    float alpha2 = alpha*alpha; 
    return alpha2/(PI*sqr( sqr(max(dot(n,m), 0.)) * (alpha2 - 1.0) + 1.0 ));
}

float G_ggx(float NdotV, float alpha)
{
    float alpha2 = alpha*alpha;
    return 2.0*NdotV/(NdotV + sqrt( mix(NdotV*NdotV, 1.0, alpha2) ));
}

const int KEY_SPACE = 32;
const int KEY_LEFT  = 37;
const int KEY_UP    = 38;
const int KEY_RIGHT = 39;
const int KEY_DOWN  = 40;
const int KEY_P = 80;

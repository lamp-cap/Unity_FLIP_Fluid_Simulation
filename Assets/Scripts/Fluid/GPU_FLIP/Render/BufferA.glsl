bool isKeyPressed(int KEY)
{
	return texelFetch( iChannel3, ivec2(KEY,2), 0 ).x > 0.5;
}

#define EMITTER_POS vec3(0.1,0.5,0.5)
#define EMITTER_RAD 4.0
#define EMITTER_VEL vec3(1.0, 0.0, 0.0)
#define EMITTER_NUM 1

#define VOID_POS vec3(0.8,0.5,0.1)
#define VOID_RAD 12.0

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragCoord = floor(fragCoord);
    Initialize(fragCoord, iFrame, iResolution.xy);
    vec3 pos = dim3from2(fragCoord);
    
    Particle p0, p1;

    //advect neighbors and accumulate + clusterize density if they fall into this cell
    range(i, -1, 1) range(j, -1, 1) range(k, -1, 1)
    {
        //load the particles 
        vec3 pos1 = pos + vec3(i, j, k);
        if(!all(lessThanEqual(pos1, size3d)) || !all(greaterThanEqual(pos1, vec3(0.0))))
        {
            continue;
        }
        Particle p0_, p1_;
        unpackParticles(LOAD3D(ch0, pos1), pos1, p0_, p1_);
        unpackForce(LOAD3D(ch1, pos1).xy, p0_, p1_);
        
        Clusterize(p0, p1, p0_, pos, dt);
        Clusterize(p0, p1, p1_, pos, dt);
    }
    
    if(p1.mass == 0u && p0.mass > 0u)
    {
        SplitParticle(p0, p1);
    }

    if(p0.mass == 0u && p1.mass > 0u)
    {
        SplitParticle(p1, p0);
    }
    
    if(isKeyPressed(KEY_UP))
    {
        float void_d = distance(p0.pos, size3d*VOID_POS);
        if(void_d < VOID_RAD)
        {
            p0.mass = 0u;
        }
    }

    if(!isKeyPressed(KEY_LEFT))
    {
        vec3 dx = normalize(p0.pos - size3d*0.5);
        p0.vel += vec3(dx.y, -dx.x, 0.0)*0.003;
    }

    if(isKeyPressed(KEY_RIGHT))
    {
        vec3 dx = normalize(p0.pos - size3d*0.5);
        p0.vel += vec3(-dx.y, dx.x, 0.0)*0.003;
    }
    
      if(iFrame < 10)
    {
        if(pos.x < 0.4*size3d.x && pos.x > 0.0*size3d.x && 
           pos.y < 0.85*size3d.y && pos.y > 0.15*size3d.y &&
           pos.z < 0.85*size3d.z && pos.z > 0.15*size3d.z)
        {
            p0.mass = initial_particle_density;
            p1.mass = 0u;
        }

        p0.pos = pos;
        p0.vel = vec3(0.0);
        p1.pos = pos;
        p1.vel = vec3(0.0);
    }

    if(all(equal(p0.pos, p1.pos)))
    {
        p1.pos += 1e-2;
    }
    
    if(isKeyPressed(KEY_SPACE))
    {
    	float emitter_d = distance(pos, size3d*EMITTER_POS);
        if(emitter_d < EMITTER_RAD && int(pos.y) % 2 == 0 && int(pos.z) % 2 == 0 && int(pos.x) % 2 == 0)
        {
            Particle emit;
            emit.pos = pos;
            emit.mass = 1u;
            emit.vel = EMITTER_VEL;
            
            BlendParticle(p0, emit);
        }
    }
    
    vec4 packed = packParticles(p0, p1, pos);
    fragColor = packed;
}
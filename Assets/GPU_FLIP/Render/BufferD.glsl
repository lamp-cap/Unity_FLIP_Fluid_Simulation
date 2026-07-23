void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragCoord = floor(fragCoord);
    Initialize(fragCoord, iFrame, iResolution.xy);
    vec3 pos = dim3from2(fragCoord);

    Particle p0, p1;
    
    //load the particles
    vec4 packed = LOAD3D(ch0, pos);
    unpackParticles(packed, pos, p0, p1);
    
    //load density
    vec2 densities = unpackvec3(floatBitsToUint(voxel(ch1, pos).x)).xy;
    p0.density = densities.x;
    p1.density = densities.y;
    
    if(p0.mass + p1.mass > 0u) 
    {
        range(i, -2, 2) range(j, -2, 2) range(k, -2, 2)
        {
            int dist = i*i + j*j + k*k;
            if(dist == 0 || dist >= 8) continue;
            vec3 pos1 = pos + vec3(i, j, k);
            Particle p0_, p1_;
            unpackParticles(LOAD3D(ch0, pos1), pos1, p0_, p1_);
            
            vec2 densities_ = unpackvec3(floatBitsToUint(voxel(ch1, pos1).x)).xy;
            p0_.density = densities_.x;
            p1_.density = densities_.y;

            //apply the force
            ApplyForce(p0, p0_);
            ApplyForce(p0, p1_);
            ApplyForce(p1, p0_);
            ApplyForce(p1, p1_);
        }

        ApplyForce(p0, p1);
        ApplyForce(p1, p0);

        FinalizeForce(p0, iTime);
        FinalizeForce(p1, iTime);
    }


    fragColor = vec4(packForce(p0, p1), 0, 0);
}
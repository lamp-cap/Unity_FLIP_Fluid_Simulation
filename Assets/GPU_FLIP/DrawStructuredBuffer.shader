Shader "Custom/DrawStructuredBuffer" 
{
    Properties 
    {
        _Cube ("Cube", Cube) = "" {}
    	_Color ("_Color", Color) = (1,1,1,1)
    	_Range ("Range", float) = 1
    	_Threshold ("_Threshold", Range(0, 3)) = 0.5
    	_Offset ("Normal Offset", Range(0.01, 1)) = 0.5
    	_Step ("Step", Range(0.005, 0.5)) = 0.05
    }
	SubShader 
	{
		HLSLINCLUDE
		
			float3 DecodeNormalOct( float2 f )
			{
				f = f * 2.0 - 1.0;
				// https://twitter.com/Stubbesaurus/status/937994790553227264
				float3 n = float3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
				float t = saturate( -n.z );
				n.xy += n.xy >= 0.0 ? -t : t;
				return normalize( n );
			}

			inline uint3 UnpackUint3(uint v)
			{
				return uint3(v & 1023u, (v >> 10) & 1023u, (v >> 20) & 1023u);
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

			inline float3 DecodePosition(uint2 packedPos)
			{
				float3 coord = MortonD3Decode(packedPos.x);
				float3 localPos = UnpackUint3(packedPos.y) / 1023.0;
				return (coord + localPos) * 0.1 - 10;
			}

			inline float2 UnpackUNorm2(uint2 packed)
			{
				return float2(asfloat(packed.x), asfloat(packed.y));
			}

			inline float3 DecodeNormal(uint2 packedNorm)
			{
				return DecodeNormalOct(UnpackUNorm2(packedNorm));
			}
		ENDHLSL
		Pass 
		{
			Tags {
	            "RenderPipeline" = "UniversalPipeline"
	            "IgnoreProjector" = "True"
	            "RenderType" = "Transparent"
				"Queue"="Transparent"
			}
			Cull Back
//			Blend SrcAlpha OneMinusSrcAlpha

			HLSLPROGRAM

			#pragma vertex vert
			#pragma fragment frag

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
		

			StructuredBuffer<uint4> _Buffer;

			struct a2v
			{
				float4 pos : POSITION;
				float3 normal : NORMAL;
			};
			
			struct v2f 
			{
				float4  pos : SV_POSITION;
				float3 normal : TEXCOORD0;
				float3 worldPos :TEXCOORD1;
			};

			// v2f vert(a2v i)
			// {
			// 	v2f o;
			// 	o.worldPos = TransformObjectToWorld(i.pos);
			// 	o.pos = TransformWorldToHClip(o.worldPos);
			// 	
			// 	return o;
			// }
			v2f vert(uint id : SV_VertexID)
			{
				uint4 vert = _Buffer[id];
			
				v2f o;
				o.normal = DecodeNormal(vert.zw);
				o.worldPos = float4(DecodePosition(vert.xy) + o.normal * 0.1, 1);
				o.pos = TransformWorldToHClip(o.worldPos);
				
				return o;
			}
			
            TEXTURE3D(_Density);        SAMPLER(sampler_Density);
            TEXTURECUBE(_Cube);         SAMPLER(sampler_Cube);
			float3 _Size;
			float4 _Color;
			float _Range;
			float _Threshold;
			float _Step;
			float _Offset;
			
            float2 insect(float3 ro,float3 rd,float3 p0,float3 p1)
            {
                float3 t0 = (p0 - ro) / rd;
                float3 t1 = (p1 - ro) / rd;
                float3 tmin = min(t0, t1);
                float3 tmax = max(t0, t1);

                float dstA = max(max(tmin.x, tmin.y), tmin.z);
                float dstB = min(tmax.x, min(tmax.y, tmax.z));
                return float2(max(dstA, 0), dstB);
            }

            float3 GetSkyColor(float3 n)
            {
                return min(SAMPLE_TEXTURECUBE_LOD(_Cube, sampler_Cube, n, 0).rgb, 20);
            }

            float SampleDistacne(float3 pos)
            {
                float3 uvw = pos / _Size;
                if (any(uvw < -0.01) || any(uvw > 1.01)) return -1;
            	// return 1;
                return SAMPLE_TEXTURE3D_LOD(_Density, sampler_Density, uvw, 0).a;
            }
			
			float RayMarching(float3 ro, float3 rd, bool inside)
			{
			    float2 bounds = insect(ro, rd, 0,_Size);
			    if (bounds.y > bounds.x)
			    {
			        float start = max(bounds.x, 0);
			    	float stepSize = (bounds.y - bounds.x)/50;
			    	float prev = inside ? 5 : 0;
			    	float sig = inside ?-1 : 1;
			        for (int iters = 0;iters< 64;iters++)
			        {
			            float t = start + stepSize * iters;
			        	if (t > bounds.y)
			        	{
			        		return inside ? bounds.y - 0.001 : -1;
			        	}
			            float3 pos = ro + rd * t;
			            float d = SampleDistacne(pos);
			            if (sig * d > sig * _Threshold)
			            {
			            	t = min(bounds.y-0.001, lerp(t - stepSize, t, (_Threshold - prev)/(d - prev)));
			                return t;
			            }
			        	prev = d;
			        }
			    }
				
				return -1.0;
			}
			// https://www.shadertoy.com/view/4djSRW
			float hash13(float3 p3){
				p3  = frac(p3 * .1031);
			    p3 += dot(p3, p3.zyx + 33.33);
			    return frac((p3.x + p3.y) * p3.z);
			}
            float SampleDensityDelta(int axis, float3 coord)
            {
                float3 off = 0; off[axis] = _Offset;
            	float3 uvw0 = (coord + off) / float3(256, 128, 128);
            	float3 uvw1 = (coord - off) / float3(256, 128, 128);
            	float d0 = uvw0[axis] > 1.00001f ? -100: SAMPLE_TEXTURE3D_LOD(_Density, sampler_Density, uvw0, 0).a;
            	float d1 = uvw1[axis] <-0.00001f ? -100: SAMPLE_TEXTURE3D_LOD(_Density, sampler_Density, uvw1, 0).a;
                return d1 - d0;
            }
            float3 normal(float3 pos)
            {
            	float3 coord = pos * 5;
                float dx = SampleDensityDelta(0, coord);
                float dy = SampleDensityDelta(1, coord);
                float dz = SampleDensityDelta(2, coord);
                
                return normalize(float3(dx, dy, dz));
            }
			struct refInfo{
			    float3 reflected;
			    float3 refracted;
			    float refFac;
			};

			refInfo getRefInfo(float3 dir, float3 norm, float n1, float n2){
			    float3 refl = reflect(dir,norm);
			    float3 refr = refract(dir,norm,n1/n2);
			    float incidence = dot(dir,-norm);
			    float transmission = dot(refr,-norm);
			    float refS = (n1*incidence - n2*transmission)/
			                 (n1*incidence + n2*transmission);
			    float refP = (n2*incidence - n1*transmission)/
			                 (n2*incidence + n1*transmission);
			    
			    float ref = lerp(refS*refS,refP*refP,0.5);
			    refInfo info;
				info.reflected = refl;
				info.refracted = refr;
				info.refFac = ref;
			    return info;
			}
			float4 Render(float3 ro, float3 rd)
			{
			    float3 col = 2.5;
			    float ior = 1.3; // IOR of the next material we'll enter divided by the IOR of the currrent material we're in
			    bool inside = false;
			    bool done = false;
			    float d0 = RayMarching(ro, rd, inside);
			    if (d0 <= 0.0)
			    	discard;
			    
			    float3 pos = ro + rd * d0;
			    float3 norm = normal(pos);
			    if (inside){
				    norm*=-1.0;
				    col *= pow((float3)(2.71828),-_Color.rgb*d0 * _Color.a * 2);
			    }
			    refInfo ref0 = getRefInfo(rd,norm,1.0,ior);
			    ior = 1.0/ior;
			    inside = !inside;
			    ro = pos+rd*_Step * 2;
			    rd = ref0.refracted;
			    float3 reflectDir = ref0.reflected;
			    for (int i=0; i<5; i++)
			    {
				    float d = RayMarching(ro, rd, inside);
			    	if (d>0.0)
			    	{
			    		pos = ro + rd * d;
			    		norm = normal(pos);
			    		if (inside){
			    			norm*=-1.0;
			    			col *= pow((float3)(2.71828),-_Color.rgb*d * _Color.a * 2);
			    		}
			    		refInfo ref = getRefInfo(rd,norm,1.0,ior);
			    		if (hash13(pos)>ref.refFac){
			    			ior = 1.0/ior;
			    			inside = !inside;
			    			ro = pos+rd*_Step * 2;
			    			rd = ref.refracted;
			    		}else{
			    			ro = pos-rd*_Step;
			    			rd = ref.reflected;
			    		}
			    	}
			    	else
			    	{
			    		col *= GetSkyColor(rd).rgb;
			    		done = true;
			    		break;
			    	}
			    }
			    if (!done){ // Ray did not escape
			        col *= pow((float3)(2.71828),-_Color.rgb * 5) * 0.03;
			    }
			    // float4 prev = texelFetch(iChannel0,ivec2(fragCoord),0);
			    // float fac = 1.0-(1.0/(prev.a+1.0));
			    // if (texelFetch(iChannel2,ivec2(32,0),0).r>0.5){
			    //     fac = 0.0;
			    //     prev.a = 0.0;
			    // }
				return float4(max(0, lerp(col, GetSkyColor(reflectDir), ref0.refFac)), 1);
			}
			
            float iBox( in float3 ro, in float3 rd, in float2 distBound, inout float3 normal, 
                        in float3 p0, in float3 p1) 
			{
                float3 t1 = (lerp(p1, p0, step(0., rd * sign(p1 - p0))) - ro) / rd;
                float3 t2 = (lerp(p0, p1, step(0., rd * sign(p1 - p0))) - ro) / rd;

                const float tN = max( max( t1.x, t1.y ), t1.z );
                const float tF = min( min( t2.x, t2.y ), t2.z );
				
                if (tN > tF || tF <= 0.) {
                    return 1000;
                }
                if (tN >= distBound.x && tN <= distBound.y) {
                    normal = -sign(rd)*step(t1.yzx,t1.xyz)*step(t1.zxy,t1.xyz);
                    return tN;
                }
                if (tF >= distBound.x && tF <= distBound.y) { 
                    normal = sign(rd)*step(t2.xyz,t2.yzx)*step(t2.xyz,t2.zxy);
                    return tF;
                }
                return 1000;
            }
			
			float4 frag(v2f i) : SV_Target
			{
				float3 posWS = i.worldPos;
				// return float4((posWS > 0?float3(1,1,1):float3(0,0,0)), 1);
                float3 V = normalize(posWS - _WorldSpaceCameraPos);

                float3 ro = posWS - V;
                float3 rd = V;
#if 1
				return min(Render(ro, rd), 100);
#else
				
                const float3 N = i.normal;
				// return float4(saturate(N), 1);

	            // // Refract the ray.
                const float ior = 0.75;
                const float3 refract1 = refract(V, N, ior);
                const float3 reflectColor1 = GetSkyColor(reflect(V, N));
                const float reflectProb1 = max(0.04, pow(abs(1.f + dot(V, N)), 5));
				// return reflectProb1;

                float3 outBoxN = 0;
                iBox(clamp(posWS, 0, _Size)+refract1*0.01, refract1, float2(0, 100), outBoxN,0, _Size );
				// return float4(max(0, outBoxN), 1);

                if(dot(refract1, outBoxN) > 0)
                {
                    outBoxN = -outBoxN;
                }
                const float3 refracted2 = refract(refract1, outBoxN, 1.f/ior);
                const float3 reflect2 = reflect(refract1, outBoxN);
                float reflectProb2 = max(0.04, pow(abs(1.f + dot(refract1, outBoxN)), 5.0));
                if (all(abs(refracted2) < 1e-5)) reflectProb2 = 1;

                // return reflectProb2;
                const float3 color2 = lerp(GetSkyColor(refracted2), GetSkyColor(normalize(reflect2*(1-outBoxN*0.3))), reflectProb2);
                float3 color1 = lerp(color2, reflectColor1, reflectProb1);
                return float4(color1, 1);
#endif
			}

			ENDHLSL

		}
		
        Pass
        {
            Name "ShadowCaster"
            Tags
            {
                "LightMode" = "ShadowCaster"
            }

            // -------------------------------------
            // Render State Commands
            ZWrite On
            ZTest LEqual
            ColorMask 0
            Cull Back

            HLSLPROGRAM

            // -------------------------------------
            // Shader Stages
            #pragma vertex Vertex
            #pragma fragment ShadowPassFragment

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local _ALPHATEST_ON
            #pragma shader_feature_local_fragment _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A
            #pragma shader_feature_local_vertex _ENABLE_WIND
            #pragma shader_feature_local_vertex _WINDCOLORCHANNELCONTROL_ON

            //--------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing
            #include_with_pragmas "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DOTS.hlsl"

            // -------------------------------------
            // Universal Pipeline keywords

            // -------------------------------------
            // Unity defined keywords
            #pragma multi_compile_fragment _ LOD_FADE_CROSSFADE

            // This is used during shadow map generation to differentiate between directional and punctual light shadows, as they use different formulas to apply Normal Bias
            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW

            // -------------------------------------
            // Includes
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/ShadowCasterPass.hlsl"
            
            
			StructuredBuffer<uint4> _Buffer;
            
			float4 MGetShadowPositionHClip(uint id)
			{
				uint4 vert = _Buffer[id];

			    float3 positionWS = float4(DecodePosition(vert.xy), 1);
			    float3 normalWS = DecodeNormal(vert.zw);

			#if _CASTING_PUNCTUAL_LIGHT_SHADOW
			    float3 lightDirectionWS = normalize(_LightPosition - positionWS);
			#else
			    float3 lightDirectionWS = _LightDirection;
			#endif

			    float4 positionCS = TransformWorldToHClip(ApplyShadowBias(positionWS, normalWS, lightDirectionWS));

			#if UNITY_REVERSED_Z
			    positionCS.z = min(positionCS.z, UNITY_NEAR_CLIP_VALUE);
			#else
			    positionCS.z = max(positionCS.z, UNITY_NEAR_CLIP_VALUE);
			#endif

			    return positionCS;
			}

			Varyings Vertex(uint id : SV_VertexID)
			{
			    Varyings output;
			    UNITY_SETUP_INSTANCE_ID(input);
			    UNITY_TRANSFER_INSTANCE_ID(input, output);

			    output.positionCS = MGetShadowPositionHClip(id);
			    return output;
			}
            ENDHLSL
        }
	}
}
Shader "Instanced/Particle2D" 
{
    Properties 
    {
        _FluidColor ("Fluid Color", COLOR) = (1, 1, 1, 1)
        _AirColor ("Air Color", COLOR) = (1, 1, 1, 1)
        _Size ("Size", float) = 0.035
        _VelScale ("vel Scale", Range(0, 2)) = 0.1
    }

    SubShader 
    {
        Pass 
        {
            Tags 
            { 
                "Queue" = "Transparent" 
                "RenderType" = "Transparent" 
                "IgnoreProjector" = "True"
            }
            
            ZTest Always
            ZWrite Off
//            Blend SrcAlpha OneMinusSrcAlpha
		    Lighting Off
            Cull Off

            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            
			#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

			struct appdata
            {
				float4 positionOS	: POSITION;
				float2 uv			: TEXCOORD0;
			};
            struct v2f
            {
				float2 uv			: TEXCOORD0;
                float4 pos : SV_POSITION;
            };

            CBUFFER_START(UnityPerMaterial)
            float _Size;
            float _VelScale;
            float4 _FluidColor;
            float4 _AirColor;
            CBUFFER_END

            StructuredBuffer<float4> _ParticleBuffer;

            v2f vert (appdata v, uint instanceID : SV_InstanceID)
            {
                v2f o;
                float4 particle = _ParticleBuffer[instanceID];
                float4 data = float4(particle.xy * 0.1, 0, 1.0);

                o.pos = TransformWorldToHClip(data.xyz + v.positionOS.xyz * _Size);
                o.uv = float2(particle.z, particle.w * _VelScale);
                
                return o;
            }

            float4 frag (v2f i) : SV_Target
            {
                return lerp(_AirColor, _FluidColor, exp2(-i.uv.y*i.uv.y));
            }

            ENDHLSL
        }
    }
}

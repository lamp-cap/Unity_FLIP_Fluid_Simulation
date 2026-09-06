Shader "Instanced/SurfaceWavePoint2D"
{
    Properties
    {
        _CrestColor ("Crest Color", COLOR) = (1, 0.4, 0.3, 1)
        _TroughColor ("Trough Color", COLOR) = (0.2, 0.5, 1, 1)
        _Size ("Size", float) = 0.05
        _HScale ("H Scale", Range(0, 40)) = 8
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
            Lighting Off
            Cull Off

            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct appdata
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };
            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 pos : SV_POSITION;
            };

            CBUFFER_START(UnityPerMaterial)
            float _Size;
            float _HScale;
            float4 _CrestColor;
            float4 _TroughColor;
            CBUFFER_END

            // xy = displaced surface point position, w = wave height H
            StructuredBuffer<float4> _ParticleBuffer;

            v2f vert (appdata v, uint instanceID : SV_InstanceID)
            {
                v2f o;
                float4 particle = _ParticleBuffer[instanceID];
                float4 data = float4(particle.xy * 0.1, 0, 1.0);

                o.pos = TransformWorldToHClip(data.xyz + v.positionOS.xyz * _Size);
                o.uv = float2(particle.z, particle.w * _HScale);

                return o;
            }

            float4 frag (v2f i) : SV_Target
            {
                float t = saturate(0.5 + i.uv.y);
                return lerp(_TroughColor, _CrestColor, t);
            }

            ENDHLSL
        }
    }
}

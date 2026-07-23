Shader "Custom/DrawVolume"
{
    Properties
    {
        _Cube ("Cube", Cube) = "" {}
        _Color ("_Color", Color) = (1,1,1,1)
        _Range ("Range", float) = 1
        _Threshold ("IsoValue", Range(0, 3)) = 0.5
        _Offset ("Normal Offset", Range(0.01, 1)) = 0.5
        _Step ("Step", Range(0.005, 0.5)) = 0.05
        [Header(Liquid Metal Path Tracer)]
        _MetalColor ("Metal Color (F0)", Color) = (0.8,0.604,0.18,1)
        _Roughness ("Metal Roughness", Range(0,0.5)) = 0.0
        _Bounces ("Max Bounces", Range(1,16)) = 8
        _Samples ("Samples Per Pixel", Range(1,8)) = 1
        [Toggle(ENABLE_LIGHTS)] _EnableLights ("Enable Key Lights", Float) = 1
        _LightIntensity ("Light Intensity", Range(0,50)) = 12
        _LightRadius ("Light Radius (x avg _Size)", Range(0.05,3)) = 0.6
        _LightDistance ("Light Distance (x |_Size|)", Range(1,20)) = 6
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

            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag
            #pragma shader_feature_local ENABLE_LIGHTS

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #include "FluidPathTracer.hlsl"

            StructuredBuffer<uint4> _Buffer;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 normal : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
            };

            v2f vert(uint id : SV_VertexID)
            {
                uint4 vert = _Buffer[id];
                v2f o;
                o.normal = DecodeNormal(vert.zw);
                o.worldPos = float4(DecodePosition(vert.xy) + o.normal * 0.1, 1);
                o.pos = TransformWorldToHClip(o.worldPos);
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float3 posWS = i.worldPos;
                float3 V = normalize(posWS - _WorldSpaceCameraPos);

                // primary visibility is the rasterized fluid surface
                float3 n0 = SurfaceNormal(posWS);
                if (dot(n0, n0) < 1e-6) n0 = normalize(i.normal);

                float2 seed = float2(posWS.x, posWS.y);
                return float4(RenderFluid(posWS, V, n0, seed), 1.0);
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

            ZWrite On
            ZTest LEqual
            ColorMask 0
            Cull Back

            HLSLPROGRAM

            #pragma vertex Vertex
            #pragma fragment ShadowPassFragment

            #pragma shader_feature_local _ALPHATEST_ON
            #pragma shader_feature_local_fragment _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A
            #pragma shader_feature_local_vertex _ENABLE_WIND
            #pragma shader_feature_local_vertex _WINDCOLORCHANNELCONTROL_ON

            #pragma multi_compile_instancing
            #include_with_pragmas "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DOTS.hlsl"

            #pragma multi_compile_fragment _ LOD_FADE_CROSSFADE
            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW

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



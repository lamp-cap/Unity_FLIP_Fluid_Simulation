Shader "ParticleRendering/ParticleInstance"
{
    Properties
    {
        _Radius ("Radius", Float) = 0.1
        _VelocityRange ("Range", Vector) = (2,8,0,0)
        _SlowColor ("_SlowColor", Color) = (1, 1, 1, 1)
        _FastColor ("_FastColor", Color) = (1, 1, 1, 1)
        _FresnelPower ("Fresnel Power", Float) = 1
    }

    SubShader
    {
        Tags {
            "RenderType" = "Opaque" 
            "RenderPipeline" = "UniversalPipeline" 
            "IgnoreProjector" = "True" 
        }
        LOD 300
        
        Cull Off
        ZClip Off
        ZWrite On
        ZTest LEqual
        Blend One Zero
        BlendOp Add

        Pass
        {
            Tags { "LightMode" = "UniversalForward" }
            HLSLPROGRAM
            // #pragma enable_d3d11_debug_symbols
            #pragma target   5.0
            #pragma vertex   Vertex
            #pragma geometry Geometry
            #pragma fragment Fragment
            #pragma multi_compile_fragment _ _SCREEN_SPACE_OCCLUSION
            #include "ParticleUtils.hlsl"
            ENDHLSL
        }

        Pass
        {
            Tags { "LightMode" = "DepthOnly" }

            HLSLPROGRAM
            
            #pragma target   5.0
            #pragma vertex   Vertex
            #pragma geometry Geometry
            #pragma fragment FragDepth
            #include "ParticleUtils.hlsl"
            ENDHLSL
        }
    }
}

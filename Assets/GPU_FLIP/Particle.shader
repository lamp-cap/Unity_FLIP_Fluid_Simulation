Shader "ParticleRendering/ParticleInstance"
{
    Properties
    {
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

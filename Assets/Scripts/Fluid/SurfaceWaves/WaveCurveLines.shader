Shader "SurfaceWaves/WaveCurveLines"
{
    Properties
    {
        _Width ("Ribbon half-width (m)", Float) = 0.02
        _AmpGain ("Steepness brightness gain", Float) = 1
        _MaxAge ("Age fade (s)", Float) = 1
    }

    SubShader
    {
        Tags {
            "RenderType" = "Transparent"
            "RenderPipeline" = "UniversalPipeline"
            "IgnoreProjector" = "True"
            "Queue" = "Transparent"
        }
        LOD 100

        Cull Off      // ribbons are one-sided but camera-facing
        ZClip Off
        ZWrite On
        ZTest LEqual

        Pass
        {
            Tags { "LightMode" = "UniversalForward" }

            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma target 5.0
            #pragma vertex Vertex
            #pragma fragment Fragment
            #include "WaveCurveLines.hlsl"
            ENDHLSL
        }

        // Pass
        // {
        //     Tags { "LightMode" = "DepthOnly" }

        //     HLSLPROGRAM
        //     #pragma target 5.0
        //     #pragma vertex Vertex
        //     #pragma fragment FragDepth
        //     #include "WaveCurveLines.hlsl"
        //     ENDHLSL
        // }
    }
}

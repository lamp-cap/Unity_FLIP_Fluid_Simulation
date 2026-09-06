Shader "Custom/DrawVolumeBox"
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
        _BoxMargin ("Box Margin (world units)", Range(0,2)) = 0.4
    }
    SubShader
    {
        Pass
        {
            Tags {
                "RenderPipeline" = "UniversalPipeline"
                "IgnoreProjector" = "True"
                "RenderType" = "Transparent"
                "Queue"="Transparent"
            }
            // Render the box back faces so the pass survives the camera
            // entering the volume; the fluid surface is found by ray-marching.
            Cull Front
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag
            #pragma shader_feature_local ENABLE_LIGHTS

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #include "FluidPathTracer.hlsl"

            float _BoxMargin; // grow the cube past the field so rays enter through empty space

            // Unit-cube corners (two triangles per face, CCW outward) scaled to [0,_Size].
            static const float3 BOX_VERTS[36] = {
                // -Z
                float3(0,0,0), float3(1,1,0), float3(1,0,0),
                float3(0,0,0), float3(0,1,0), float3(1,1,0),
                // +Z
                float3(0,0,1), float3(1,0,1), float3(1,1,1),
                float3(0,0,1), float3(1,1,1), float3(0,1,1),
                // -X
                float3(0,0,0), float3(0,0,1), float3(0,1,1),
                float3(0,0,0), float3(0,1,1), float3(0,1,0),
                // +X
                float3(1,0,0), float3(1,1,1), float3(1,0,1),
                float3(1,0,0), float3(1,1,0), float3(1,1,1),
                // -Y
                float3(0,0,0), float3(1,0,0), float3(1,0,1),
                float3(0,0,0), float3(1,0,1), float3(0,0,1),
                // +Y
                float3(0,1,0), float3(0,1,1), float3(1,1,1),
                float3(0,1,0), float3(1,1,1), float3(1,1,0)
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
            };

            v2f vert(uint id : SV_VertexID)
            {
                v2f o;
                // Expand the cube by _BoxMargin on every side so rays approach the
                // density-field boundary through empty space instead of starting on it.
                o.worldPos = lerp(-_BoxMargin, _Size + _BoxMargin, BOX_VERTS[id]);
                o.pos = TransformWorldToHClip(o.worldPos);
                return o;
            }

            struct FragOut
            {
                float4 color : SV_Target;
                float  depth : SV_Depth;
            };

            FragOut frag(v2f i)
            {
                FragOut o;
                float3 ro = _WorldSpaceCameraPos;
                float3 rd = normalize(i.worldPos - ro);

                // Clip the camera ray to the margin-expanded box, then march. Samples
                // in the margin read as empty (-1), so a wall-touching surface closes
                // cleanly at the field boundary instead of being clipped by the face.
                float2 tb = insect(ro, rd, -_BoxMargin, _Size + _BoxMargin);
                float tN = max(tb.x, 0.0);
                bool inside = SampleDistance(ro) > _Threshold; // camera submerged?
                float td = (tb.y > tN) ? RaycastFluid(ro, rd, tN, tb.y, inside) : -1.0;
                if (td <= 0.0) discard;

                float3 hitWS = ro + rd * td;
                float3 n0 = SurfaceNormal(hitWS);

                float2 seed = float2(hitWS.x, hitWS.y);
                o.color = float4(RenderFluid(hitWS, rd, n0, seed), 1.0);

                // Write true surface depth so the fluid composites with the scene.
                float4 clip = TransformWorldToHClip(hitWS);
                o.depth = clip.z / clip.w;
                return o;
            }

            ENDHLSL
        }
    }
}

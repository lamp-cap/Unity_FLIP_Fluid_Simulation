Shader "Unlit/GPUDebugNumer"
{
    Properties
    {
        _DigitAtlas ("Texture", 2D) = "white" {}
        _DigitWidth ("width", Float) = 1
        _DigitHeight ("height", Float) = 1
        _DisplayPos ("Pos", Vector) = (0,0,0,0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            Cull Off
            HLSLPROGRAM
            #pragma target 5.0
            #pragma vertex vert
            #pragma fragment frag

            struct Varyings
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            StructuredBuffer<uint> _Counter;
            float _DigitWidth;
            float _DigitHeight;
            float4 _DisplayPos;

            Varyings vert(uint vertexID : SV_VertexID)
            {
                Varyings o;
                
                uint count = _Counter[0];
                int digitCount =  count > 0 ? floor(log10(count)) + 1 : 1;
                float totalWidth = digitCount * _DigitWidth * 0.1;
                const float2 corners[4] = {
                    float2(0, 0),
                    float2(0, 1),
                    float2(1, 1),
                    float2(1, 0) 
                };
                
                float2 pos = float2(
                    corners[vertexID].x * totalWidth,
                    corners[vertexID].y * _DigitHeight * 0.1
                );
                
                #if UNITY_UV_STARTS_AT_TOP
                pos.y = -pos.y + _DigitHeight * 0.1;
                #endif
                pos.x += (8 - digitCount) * _DigitWidth * 0.1;
                o.vertex = float4(pos + _DisplayPos.xy, 0, 1);
                o.uv = corners[vertexID] * float2(digitCount, 1);
                o.uv.x = digitCount - o.uv.x;
                
                return o;
            }

            sampler2D _DigitAtlas;
            float _AtlasColumns; // 图集有多少列（通常是10）

            half4 frag(Varyings i) : SV_Target
            {
                uint count = _Counter[0];
                int columnIndex = (int)floor(i.uv.x);
                float digit = (count / (int)pow(10, columnIndex)) % 10 ;
                float2 uv = (1 - frac(i.uv)) * float2(0.1, 1) + float2(digit * 0.1, 0);
                half4 color =  tex2D(_DigitAtlas, uv);
                
                return color;
            }
            ENDHLSL
        }
    }
}

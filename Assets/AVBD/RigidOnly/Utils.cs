using Unity.Burst;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD.RigidOnly
{
    public struct Equation6
    {
        public float3x3 lhsLin;
        public float3x3 lhsAng;
        public float3x3 lhsCross;
        public float3 rhsLin;
        public float3 rhsAng;

        public readonly void Solve(out float3 xLin, out float3 xAng)
        {
            xLin = xAng = float3(0);
            // Extract elements from lower triangle storage
            float A11 = lhsLin[0][0];
            float A21 = lhsLin[0][1], A22 = lhsLin[1][1];
            float A31 = lhsLin[0][2], A32 = lhsLin[1][2], A33 = lhsLin[2][2];
            float A41 = lhsCross[0][0], A42 = lhsCross[1][0], A43 = lhsCross[2][0], A44 = lhsAng[0][0];
            float A51 = lhsCross[0][1], A52 = lhsCross[1][1], A53 = lhsCross[2][1], A54 = lhsAng[0][1], A55 = lhsAng[1][1];
            float A61 = lhsCross[0][2], A62 = lhsCross[1][2], A63 = lhsCross[2][2], A64 = lhsAng[0][2], A65 = lhsAng[1][2], A66 = lhsAng[2][2];

            // Step 1: LDL^T decomposition
            float L21 = A21 / A11;
            float L31 = A31 / A11;
            float L41 = A41 / A11;
            float L51 = A51 / A11;
            float L61 = A61 / A11;

            float D1 = A11;

            float D2 = A22 - L21 * L21 * D1;

            float L32 = (A32 - L21 * L31 * D1) / D2;
            float L42 = (A42 - L21 * L41 * D1) / D2;
            float L52 = (A52 - L21 * L51 * D1) / D2;
            float L62 = (A62 - L21 * L61 * D1) / D2;

            float D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2);

            float L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3;
            float L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3;
            float L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3;

            float D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3);

            float L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
            float L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4;

            float D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);

            float L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;

            float D6 = A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);

            // Step 2: Forward substitution: Solve Ly = b
            float y1 = -rhsLin[0];
            float y2 = -rhsLin[1] - L21 * y1;
            float y3 = -rhsLin[2] - L31 * y1 - L32 * y2;
            float y4 = -rhsAng[0] - L41 * y1 - L42 * y2 - L43 * y3;
            float y5 = -rhsAng[1] - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
            float y6 = -rhsAng[2] - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

            // Step 3: Diagonal solve: Solve Dz = y
            float z1 = y1 / D1;
            float z2 = y2 / D2;
            float z3 = y3 / D3;
            float z4 = y4 / D4;
            float z5 = y5 / D5;
            float z6 = y6 / D6;

            // Step 4: Backward substitution: Solve L^T x = z
            xAng[2] = z6;
            xAng[1] = z5 - L65 * xAng[2];
            xAng[0] = z4 - L54 * xAng[1] - L64 * xAng[2];
            xLin[2] = z3 - L43 * xAng[0] - L53 * xAng[1] - L63 * xAng[2];
            xLin[1] = z2 - L32 * xLin[2] - L42 * xAng[0] - L52 * xAng[1] - L62 * xAng[2];
            xLin[0] = z1 - L21 * xLin[1] - L31 * xLin[2] - L41 * xAng[0] - L51 * xAng[1] - L61 * xAng[2];
        }
    }

    public struct OBB
    {
        public float3 center;
        public float3 half;
        public float3x3 axis;
        public quaternion rotation;

        public static OBB makeOBB(Rigid body)
        {
            OBB box = new OBB();
            box.center = body.positionLin;
            box.rotation = body.positionAng;
            box.half = body.size * 0.5f;
            box.axis[0] = rotate(body.positionAng, float3(1.0f, 0.0f, 0.0f));
            box.axis[1] = rotate(body.positionAng, float3(0.0f, 1.0f, 0.0f));
            box.axis[2] = rotate(body.positionAng, float3(0.0f, 0.0f, 1.0f));
            return box;
        }
        public UnityEngine.Bounds ToAABB_Fast()
        {
            // 获取旋转矩阵
            float3x3 rotMatrix = new float3x3(rotation);
    
            // 计算局部半长在旋转后的世界轴向上的投影长度
            // 公式：extent_i = |R| * half，其中|R|是旋转矩阵的绝对值
            float3 extent = new float3(
                abs(rotMatrix.c0.x) * half.x + abs(rotMatrix.c1.x) * half.y + abs(rotMatrix.c2.x) * half.z,
                abs(rotMatrix.c0.y) * half.x + abs(rotMatrix.c1.y) * half.y + abs(rotMatrix.c2.y) * half.z,
                abs(rotMatrix.c0.z) * half.x + abs(rotMatrix.c1.z) * half.y + abs(rotMatrix.c2.z) * half.z
            );
    
            // AABB的中心就是OBB的中心
            // AABB的半长是extent
            return new UnityEngine.Bounds
            {
                min = center - extent,
                max = center + extent
            };
        }
    };
    
    [BurstCompile]
    public static class Utils
    {
        public const float PENALTY_MIN =1.0f;           // Minimum penalty parameter
        public const float PENALTY_MAX =10000000000.0f; // Maximum penalty parameter
        public const float COLLISION_MARGIN =0.01f;     // Margin for collision detection to avoid flickering contacts
        public const float STICK_THRESH =0.00001f;      // Position threshold for sticking contacts (ie static friction)
        
        public static float3x3 skew(in float3 r)
        {
            return float3x3(
                0, -r.z, r.y,
                r.z, 0, -r.x,
                -r.y, r.x, 0);
        }
        
        public static float4 mul(in this float4 a, in float4 b)
        {
            return new (
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
        }

        public static float3 sub(in this quaternion a, in quaternion b)
        {
            return a.value.mul(inverse(b).value).xyz * 2.0f;
        }

        public static quaternion add(in this quaternion a, in float3 b)
        {
            return normalize(a.value + new float4(b.x, b.y, b.z, 0).mul(a.value) * 0.5f);
        }

        public static float3x3 outer(in this float3 a, in float3 b)
        {
            return transpose(float3x3(b * a.x, b * a.y, b * a.z));
        }
        
        public static float3x3 diagonal(float m00, float m11, float m22)
        {
            return float3x3(m00, 0, 0, 0, m11, 0, 0, 0, m22);
        }

        public static float3 transform(in float3 qLin, in quaternion qAng, in float3 v)
        {
            return rotate(qAng, v) + qLin;
        }

        public static float3x3 orthonormal(in float3 normal)
        {
            float3 t1 = abs(normal.x) > abs(normal.z)
                ? float3(-normal.y, normal.x, 0)
                : float3(0, -normal.z, normal.y);
            t1 = normalize(t1);
            float3 t2 = cross(normal, t1);
            return transpose(float3x3(normal, t1, t2));
        }

        public static float3x3 diagonalize(in float3x3 m)
        {
            return diagonal(length(m.c0), length(m.c1), length(m.c2));
        }

        [BurstCompile]
        public static void SolveEq6(in this Equation6 eq, out float3 a, out float3 b)
        {
            eq.Solve(out a, out b);
        }
    }
}

using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD.Cloth
{
    /// <summary>
    /// VBD 布料求解的数学核心。
    /// 对应参考实现:
    ///   VBDTriMeshStVK.cpp :: greenStrain / accumlateStVKForceAndHessian
    ///   VBDClothPhysics.h   :: assembleMembraneHessian / assembleMembraneForceAndHessian
    ///   CuMatrix::solve3x3_psd_stable
    /// 注意:参考实现里 6x6 D2W_DFDF -> 3x3 的优化已经做好(switch(faceVertexOrder) + assembleMembraneHessian),
    /// 这里移植的就是该已优化路径,不是被注释掉的 6x3 DF_DX 旧写法。
    /// </summary>
    public static class ClothMath
    {
        // ---------------------------------------------------------------
        // 变形梯度 F (3x2):F = Ds * DmInv
        // Ds = [x1-x0, x2-x0] (3x2)
        // 对应 TriMeshFEM::calculateDeformationGradient
        // ---------------------------------------------------------------
        public static float3x2 DeformationGradient(float3 x0, float3 x1, float3 x2, float2x2 DmInv)
        {
            // Ds 列向量
            float3 ds0 = x1 - x0;
            float3 ds1 = x2 - x0;
            // F = Ds * DmInv  -> 3x2 = (3x2)*(2x2)
            float3 f0 = ds0 * DmInv.c0.x + ds1 * DmInv.c0.y;
            float3 f1 = ds0 * DmInv.c1.x + ds1 * DmInv.c1.y;
            return new float3x2(f0, f1);
        }

        // ---------------------------------------------------------------
        // Green 应变 G = 0.5 * (F^T F - I)  (2x2 对称)
        // 对应 VBDTriMeshStVK.cpp :: greenStrain
        // ---------------------------------------------------------------
        public static float2x2 GreenStrain(float3x2 F)
        {
            float3 f0 = F.c0;
            float3 f1 = F.c1;
            float g00 = 0.5f * (dot(f0, f0) - 1.0f);
            float g11 = 0.5f * (dot(f1, f1) - 1.0f);
            float g01 = 0.5f * dot(f0, f1);
            return new float2x2(g00, g01,
                                g01, g11);
        }

        /// <summary>
        /// StVK 膜能量:对单个相邻面,累加该顶点(faceVertexOrder ∈ {0,1,2})的力与 3x3 Hessian。
        /// 完整对应 accumlateStVKForceAndHessian 的内层循环(单个 neighbour face)。
        /// faceArea 为 restpose 面积,DmInv 为该面的逆材料矩阵。
        /// </summary>
        public static void AccumulateStVKFace(
            float3 x0, float3 x1, float3 x2,
            float2x2 DmInv, float faceArea,
            float lambda, float miu,
            int faceVertexOrder,
            ref float3 force, ref float3x3 hessian)
        {
            float3x2 F = DeformationGradient(x0, x1, x2, DmInv);
            float2x2 G = GreenStrain(F);

            // 第二 Piola-Kirchhoff 应力 S = 2*miu*G + lambda*tr(G)*I
            float trG = G.c0.x + G.c1.y;
            float2x2 S = new float2x2(
                2f * miu * G.c0.x + lambda * trG, 2f * miu * G.c1.x,
                2f * miu * G.c0.y, 2f * miu * G.c1.y + lambda * trG);

            // F12 = -faceArea * F * S * DmInv^T   (3x2)
            // 先 FS = F * S (3x2)
            float3 fs0 = F.c0 * S.c0.x + F.c1 * S.c0.y;
            float3 fs1 = F.c0 * S.c1.x + F.c1 * S.c1.y;
            // FS * DmInv^T : DmInv^T 行 = DmInv 列。result col j = FS * (DmInv row j)
            // DmInv^T.c0 = (DmInv(0,0), DmInv(0,1)) = (c0.x, c1.x)
            // DmInv^T.c1 = (DmInv(1,0), DmInv(1,1)) = (c0.y, c1.y)
            float3 f12_0 = -faceArea * (fs0 * DmInv.c0.x + fs1 * DmInv.c1.x);
            float3 f12_1 = -faceArea * (fs0 * DmInv.c0.y + fs1 * DmInv.c1.y);

            float Dm11 = DmInv.c0.x, Dm21 = DmInv.c0.y, Dm12 = DmInv.c1.x, Dm22 = DmInv.c1.y;

            // ---- 取 m1,m2 并累加 force,对应 switch(faceVertexOrder) ----
            float m1, m2;
            switch (faceVertexOrder)
            {
                case 0:
                    m1 = -Dm11 - Dm21;
                    m2 = -Dm12 - Dm22;
                    force += -f12_0 - f12_1;
                    break;
                case 1:
                    m1 = Dm11;
                    m2 = Dm12;
                    force += f12_0;
                    break;
                default: // 2
                    m1 = Dm21;
                    m2 = Dm22;
                    force += f12_1;
                    break;
            }

            // ---- 纯 3x3 小矩阵 Hessian(不再组装 6x6) ----
            // 推导:6x6 D2W_DFDF 按 F 的两列分块为 [[A, B],[B^T, C]];
            // 该顶点的 DF_DX = [m1*I3; m2*I3](6x3),故
            //   H = DF_DX^T * D2W * DF_DX = m1^2*A + m1*m2*(B+B^T) + m2^2*C
            // 其中(a=F.c0, b=F.c1, e_uu/e_vv/e_uv 为 Green 应变分量,e_sum=e_uu+e_vv):
            //   A      = (λ+2μ) a⊗a + μ b⊗b + (2μ e_uu + λ e_sum) I
            //   C      = (λ+2μ) b⊗b + μ a⊗a + (2μ e_vv + λ e_sum) I
            //   B+B^T  = (λ+μ)(a⊗b + b⊗a) + 4μ e_uv I
            // 合并同类项后,H 只需 a⊗a / b⊗b / (a⊗b+b⊗a) / I 四个块的标量加权:
            float3 a = F.c0, b = F.c1;
            float e_uu = G.c0.x, e_vv = G.c1.y, e_uv = G.c1.x;
            float e_sum = e_uu + e_vv;

            float cA = 2f * miu * e_uu + lambda * e_sum;   // A 的 I 系数
            float cC = 2f * miu * e_vv + lambda * e_sum;   // C 的 I 系数
            float lam2mu = lambda + 2f * miu;

            float m1m1 = m1 * m1, m2m2 = m2 * m2, m1m2 = m1 * m2;

            float kaa = m1m1 * lam2mu + m2m2 * miu;          // a⊗a 系数
            float kbb = m1m1 * miu + m2m2 * lam2mu;          // b⊗b 系数
            float kab = m1m2 * (lambda + miu);               // (a⊗b + b⊗a) 系数
            float kI = m1m1 * cA + m2m2 * cC + 4f * miu * e_uv * m1m2; // I 系数

            // 全部乘以 faceArea(对应原先 D.Scale(faceArea))
            kaa *= faceArea; kbb *= faceArea; kab *= faceArea; kI *= faceArea;

            float3x3 zero3 = new float3x3(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f);
            float3x3 h = AddOuter(zero3, a, a, kaa);         // += kaa * a⊗a
            h = AddOuter(h, b, b, kbb);                      // += kbb * b⊗b
            h = AddOuter(h, a, b, kab);                      // += kab * a⊗b
            h = AddOuter(h, b, a, kab);                      // += kab * b⊗a
            h.c0.x += kI; h.c1.y += kI; h.c2.z += kI;        // += kI * I

            hessian += h;
        }

        /// <summary>h += s * (u ⊗ w)。列主序:列 j = u * w[j]。</summary>
        static float3x3 AddOuter(float3x3 h, float3 u, float3 w, float s)
        {
            h.c0 += (s * w.x) * u;
            h.c1 += (s * w.y) * u;
            h.c2 += (s * w.z) * u;
            return h;
        }

        // ---------------------------------------------------------------
        // 3x3 PSD 稳定求解 H dx = f  ( dx = H^-1 f )
        // 对应 CuMatrix::solve3x3_psd_stable:对角线加正则,失败回退。
        // 返回是否成功。
        // ---------------------------------------------------------------
        public static bool Solve3x3PSD(float3x3 H, float3 f, out float3 dx)
        {
            // 对称化
            float3x3 A = H;
            float det = determinant(A);
            // 退化检测,加微小正则保证正定
            if (abs(det) < 1e-12f)
            {
                float eps = 1e-7f * (abs(A.c0.x) + abs(A.c1.y) + abs(A.c2.z) + 1f);
                A.c0.x += eps; A.c1.y += eps; A.c2.z += eps;
                det = determinant(A);
                if (abs(det) < 1e-20f)
                {
                    dx = float3(0);
                    return false;
                }
            }
            dx = mul(inverse(A), f);
            return all(isfinite(dx));
        }
    }
}

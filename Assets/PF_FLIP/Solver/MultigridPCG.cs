using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public class MultigridPCG : MonoBehaviour
{
    public struct MultiGridSolver
    {
        public NativeArray<float> Lhs;
        public NativeArray<float> Rhs;
        public int GridRes;
        public float H;

        public void Test()
        {
            Smooth(Lhs, Rhs, GridRes, H, 16);
        }
        
        public void SolveMultiGrid(out int iter, out float rs)
        {
            NativeArray<float> v_old = new NativeArray<float>(Lhs.Length, Allocator.Temp);
            float norm = 0;
            for (iter = 0; iter < 10; iter++)
            {
                Lhs.CopyTo(v_old);
                MultiGridVCycle(Lhs, Rhs, GridRes, H);
                
                norm = math.sqrt(Res(Lhs, v_old));
                if (norm < H * H * 0.5f)
                    break;
                
                // Debug.Log("MG iter " + iter + " res: " + norm);
            }

            Residual(Lhs, Rhs, v_old, GridRes, H);
            rs = math.sqrt(Dot(v_old, v_old));
            // Debug.Log("MG converged in iter " + iter + " res: " + norm);

            v_old.Dispose();
        }
        
        public void Solve_GS()
        {
            NativeArray<float> v_old = new NativeArray<float>(Lhs.Length, Allocator.Temp);
            float norm = 0;
            for (int iter = 0; iter < 3; iter++)
            {
                Smooth(Lhs, Rhs, GridRes, H, 16);

                Residual(Lhs, Rhs, v_old, GridRes, H);
                norm = math.sqrt(Dot(v_old, v_old));
                Debug.Log("GS iter " + iter + "*16 res: " + norm);
                if (norm < H * H)
                    break;
            }

            v_old.Dispose();
        }
        
        public void Solve_GSRB()
        {
            NativeArray<float> v_old = new NativeArray<float>(Lhs.Length, Allocator.Temp);
            float norm = 0;
            for (int iter = 0; iter < 3; iter++)
            {
                SmoothRB(Lhs, Rhs, GridRes, H, 16);
                
                Residual(Lhs, Rhs, v_old, GridRes, H);
                norm = math.sqrt(Dot(v_old, v_old));
                Debug.Log("GS RB iter " + iter + "*16 res: " + norm);
                if (norm < H * H)
                    break;
            }

            v_old.Dispose();
        }
        
        public void Solve_ConjugateGradient(out int iter, out float rs)
        {
            NativeArray<float> r = new NativeArray<float>(Rhs, Allocator.Temp);
            NativeArray<float> p = new NativeArray<float>(Rhs, Allocator.Temp);
            NativeArray<float> Ap = new NativeArray<float>(Rhs.Length, Allocator.Temp);
            float rs_old = Dot(r, r);
        
            for (iter = 0; iter < Rhs.Length; iter++)
            {
                Laplacian(p, Ap);
                float alpha = rs_old / Dot(p, Ap);
                for (int i = 0; i < Lhs.Length; i++)
                    Lhs[i] += alpha * p[i];
                for (int i = 0; i < Lhs.Length; i++)
                    r[i] -= alpha * Ap[i];
                float rs_new = Dot(r, r);
                if (math.sqrt(rs_new) < H * H * 0.5f)
                {
                    rs_old = rs_new;
                    break;
                }

                for (int i = 1; i < Lhs.Length; i++)
                    p[i] = r[i] + (rs_new / rs_old) * p[i];
                rs_old = rs_new;
                
            }
            
            Residual(Lhs, Rhs, r, GridRes, H);
            rs = math.sqrt(Dot(r, r));
            // Debug.Log($"ConjugateGradient converged in {iter}/{Rhs.Length} iterations. rs:{math.sqrt(rs_old)}");
            
            r.Dispose();
            p.Dispose();
            Ap.Dispose();
        }

        public void SolveMGPCG(out int iter, out float rs)
        {
            NativeArray<float> r = new NativeArray<float>(Rhs.Length, Allocator.Temp);
            Residual(Lhs, Rhs, r, GridRes, H);
            NativeArray<float> z = new NativeArray<float>(r, Allocator.Temp);
            MultiGridVCycle(z, r, GridRes, H);
            NativeArray<float> p = new NativeArray<float>(z, Allocator.Temp);
            NativeArray<float> Ap = new NativeArray<float>(p.Length, Allocator.Temp);

            float rz_old = Dot(r, z);

            for (iter = 0; iter < 20; iter++)
            {
                Laplacian(p, Ap);
                float alpha = rz_old / Dot(p, Ap);
                for (int i = 0; i < r.Length; i++)
                    Lhs[i] += alpha * p[i];

                for (int i = 0; i < r.Length; i++)
                    r[i] -= alpha * Ap[i];
                
                if (math.sqrt(Dot(r, r)) < H * H * 0.1f)
                    break;
                
                for (int i = 0; i < z.Length; i++)
                    z[i] = 0;
                
                MultiGridVCycle(z, r, GridRes, H);
                float rz_new = Dot(r, z);
                float beta = rz_new / rz_old;
                for (int i = 0; i < p.Length; i++)
                    p[i] = z[i] + beta * p[i];
                rz_old = rz_new;
            }

            Residual(Lhs, Rhs, r, GridRes, H);
            rs = math.sqrt(Dot(r, r));
            // Debug.Log($"MGPCG converged in {iter}/{Rhs.Length} iterations. rs:{rs}");

            r.Dispose();
            z.Dispose();
            p.Dispose();
            Ap.Dispose();
        }

        private void Laplacian(NativeArray<float> v, NativeArray<float> result)
        {
            int gridRes = GridRes;
            float h = H;
            float ih2 = 1f / (h * h);
            for (int y = 1; y < gridRes - 1; y++)
            for (int x = 1; x < gridRes - 1; x++)
            {
                int i = Coord2Index(x, y, gridRes);
                result[i] = (4 * v[i] - NeighborSum(v, x, y, gridRes)) * ih2;
            }
            
        }

        private float Dot(NativeArray<float> lhs, NativeArray<float> rhs)
        {
            float sum = 0;
            for (int i = 0; i < lhs.Length; i++)
                sum += lhs[i] * rhs[i];
            return sum;
        }

        private float Res(NativeArray<float> x, NativeArray<float> x_old)
        {
            float norm = 0;
            for (int i = 0; i < x.Length; i++)
            {
                float temp = x[i] - x_old[i];
                norm += temp * temp;
            }
            return norm;
        }

        public void MultiGridVCycle(NativeArray<float> v, NativeArray<float> b, int res, float h)
        {
            if (res < 4)
            {
                v[4] = b[4] * H * H * 0.25f;
                return;
            }

            Smooth(v, b, res, h, 3);

            var r = new NativeArray<float>(b.Length, Allocator.Temp);
            Residual(v, b, r, res, h);

            int resC = res / 2 + 1;
            var rc = new NativeArray<float>(resC * resC, Allocator.Temp);
            Restrict(r, rc, res);
            
            var ec = new NativeArray<float>(rc.Length, Allocator.Temp);
            MultiGridVCycle(ec, rc, resC, h * 2);
            
            Prolongate(ec, v, res);
            
            BackSmooth(v, b, res, h, 3);
            
            r.Dispose();
            rc.Dispose();
            ec.Dispose();
        }

        private void Smooth(NativeArray<float> v, NativeArray<float> f, int res, float h, int count)
        {
            float h2 = h * h;
            
            for (int iter = 0; iter < count; iter++)
            {
                for (int y = 1; y < res - 1; y++)
                for (int x = 1; x < res - 1; x++)
                {
                    int i = Coord2Index(x, y, res);
                    v[i] = 0.25f * (h2 * f[i] + NeighborSum(v, x, y, res));
                }
            }
        }
        private void SmoothRB(NativeArray<float> v, NativeArray<float> f, int res, float h, int count)
        {
            float h2 = h * h;
            
            for (int iter = 0; iter < count; iter++)
            {
                for (int y = 1; y < res - 1; y++)
                for (int x = 1; x < res - 1; x++)
                {
                    if (((x + y) & 1) != 0) continue;
                    int i = Coord2Index(x, y, res);
                    v[i] = 0.25f * (h2 * f[i] + NeighborSum(v, x, y, res));
                }
            
                for (int y = 1; y < res - 1; y++)
                for (int x = 1; x < res - 1; x++)
                {
                    if (((x + y) & 1) != 1) continue;
                    int i = Coord2Index(x, y, res);
                    v[i] = 0.25f * (h2 * f[i] + NeighborSum(v, x, y, res));
                }
            }
        }

        private void BackSmooth(NativeArray<float> v, NativeArray<float> f, int res, float h, int count)
        {
            float h2 = h * h;

            for (int iter = 0; iter < count; iter++)
            {
                for (int y = res - 2; y > 0; y--)
                for (int x = res - 2; x > 0; x--)
                {
                    int i = Coord2Index(x, y, res);
                    v[i] = 0.25f * (h2 * f[i] + NeighborSum(v, x, y, res));
                }

                for (int y = 1; y < res - 1; y++)
                for (int x = 1; x < res - 1; x++)
                {
                    int i = Coord2Index(x, y, res);
                    v[i] = 0.25f * (h2 * f[i] + NeighborSum(v, x, y, res));
                }
            }
        }

        private void Residual(NativeArray<float> v, NativeArray<float> b, NativeArray<float> r, int res, float h)
        {
            float ih2 = 1f / (h * h);
            for (int y = 1; y < res - 1; y++)
            for (int x = 1; x < res - 1; x++)
            {
                int i = Coord2Index(x, y, res);
                float ax = (4 * v[i] - NeighborSum(v, x, y, res)) * ih2;
                r[i] = b[i] - ax;
            }
        }

        private void Restrict(NativeArray<float> rf, NativeArray<float> rc, int res)
        {
            int gridResC = res / 2 + 1;
            int gridResF = res;
            float3 weight = new float3(0.25f, 0.125f, 0.0625f);
            
            for (int y = 1; y < gridResC - 1; y++)
            for (int x = 1; x < gridResC - 1; x++)
            {
                float sum = 0;
                for (int ly = -1; ly <= 1; ly++)
                for (int lx = -1; lx <= 1; lx++)
                {
                    int offset = math.abs(ly) + math.abs(lx);
                    sum += rf[Coord2Index(x * 2 + lx, y * 2 + ly, gridResF)] * weight[offset];
                }
                
                rc[Coord2Index(x, y, gridResC)] = sum;
            }
        }

        private void Prolongate(NativeArray<float> ec, NativeArray<float> ef, int res)
        {
            int gridResF = res;
            int gridResC = res / 2 + 1;
            
            for (int y = 0; y < gridResC - 1; y++)
            for (int x = 0; x < gridResC - 1; x++)
            {
                float cur = ec[Coord2Index(x, y, gridResC)];
                float right = ec[Coord2Index(x + 1, y, gridResC)];
                float up = ec[Coord2Index(x, y + 1, gridResC)];
                float upright = ec[Coord2Index(x + 1, y + 1, gridResC)];

                if (x > 0 && y > 0)
                    ef[Coord2Index(x * 2, y * 2, gridResF)] += cur;
                if (x * 2 + 1 < res - 1 && y > 0)
                    ef[Coord2Index(x * 2 + 1, y * 2, gridResF)] += 0.5f * (cur + right);
                if (y * 2 + 1 < res - 1 && x > 0)
                    ef[Coord2Index(x * 2, y * 2 + 1, gridResF)] += 0.5f * (cur + up);
                if (x * 2 + 1 < res - 1 && y * 2 + 1 < res - 1)
                    ef[Coord2Index(x * 2 + 1, y * 2 + 1, gridResF)] += 0.25f * (cur + right + up + upright);
            }
        }

        private float NeighborSum(NativeArray<float> v, int x, int y, int gridRes)
        {
            return v[Coord2Index(x - 1, y, gridRes)] + v[Coord2Index(x + 1, y, gridRes)]
                   + v[Coord2Index(x, y - 1, gridRes)] + v[Coord2Index(x, y + 1, gridRes)];
        }

        private int Coord2Index(int x, int y, int gridRes) => y * gridRes + x;
    }

    public struct MultiGrid1DSolver
    {
        public NativeArray<float> Lhs;
        public NativeArray<float> Rhs;
        public int GridRes;
        public float H;

        public void Test()
        {
            Smooth(Lhs, Rhs, GridRes, H, 16);
        }
        
        public void SolveMG(out int iter, out float rs)
        {
            NativeArray<float> v_old = new NativeArray<float>(Lhs.Length, Allocator.Temp);
            float norm = 0;
            for (iter = 0; iter < 30; iter++)
            {
                Lhs.CopyTo(v_old);
                MultiGridVCycle(Lhs, Rhs, GridRes, H);

                norm = math.sqrt(Res(Lhs, v_old));
                if (norm < H * H)
                    break;
                // Debug.Log("MG iter " + iter + " res: " + norm);
            }
            
            Residual(Lhs, Rhs, v_old, GridRes, H);
            rs = math.sqrt(Dot(v_old, v_old));

            v_old.Dispose();
        }
        
        public void Solve_GS()
        {
            NativeArray<float> v_old = new NativeArray<float>(Lhs.Length, Allocator.Temp);
            float norm = 0;
            for (int iter = 0; iter < 5; iter++)
            {
                Smooth(Lhs, Rhs, GridRes, H, 16);

                Residual(Lhs, Rhs, v_old, GridRes, H);
                norm = math.sqrt(Dot(v_old, v_old));
                if (norm < H * H)
                    break;
                Debug.Log("GS iter " + iter + "*16 res: " + norm);
            }

            v_old.Dispose();
        }
        
        public void Solve_GSRB()
        {
            NativeArray<float> v_old = new NativeArray<float>(Lhs.Length, Allocator.Temp);
            float norm = 0;
            for (int iter = 0; iter < 5; iter++)
            {
                SmoothRB(Lhs, Rhs, GridRes, H, 16);

                Residual(Lhs, Rhs, v_old, GridRes, H);
                norm = math.sqrt(Dot(v_old, v_old));
                if (norm < H * H)
                    break;
                Debug.Log("GS RB iter " + iter + "*16 res: " + norm);
            }

            v_old.Dispose();
        }
        
        public void Solve_ConjugateGradient(out int iter, out float rs)
        {
            var u = Lhs;
            var b = Rhs;
            NativeArray<float> r = new NativeArray<float>(b.Length, Allocator.Temp);
            Residual(u, b, r, GridRes, H);
            NativeArray<float> p = new NativeArray<float>(r, Allocator.Temp);
            NativeArray<float> Ap = new NativeArray<float>(b.Length, Allocator.Temp);
            float rs_old = Dot(r, r);

            for (iter = 0; iter < Rhs.Length; iter++)
            {
                Laplacian(p, Ap);
                float alpha = rs_old / Dot(p, Ap);
                for (int i = 1; i < u.Length - 1; i++)
                    u[i] += alpha * p[i];
                for (int i = 1; i < r.Length - 1; i++)
                    r[i] -= alpha * Ap[i];
                float rs_new = Dot(r, r);
                // if (math.sqrt(rs_new) < H*H*0.1f)
                //     break;

                float beta = rs_new / rs_old;
                for (int i = 1; i < p.Length - 1; i++)
                    p[i] = r[i] + beta * p[i];
                rs_old = rs_new;
            }
            
            Residual(u, b, r, GridRes, H);
            rs = math.sqrt(Dot(r, r));
            // Debug.Log($"ConjugateGradient converged in {iter}/{Rhs.Length} iterations. rs:{math.sqrt(rs_old)}");

            r.Dispose();
            p.Dispose();
            Ap.Dispose();
        }

        public void SolveMGPCG(out int iter, out float rs)
        {
            NativeArray<float> r = new NativeArray<float>(Rhs, Allocator.Temp);
            NativeArray<float> z = new NativeArray<float>(r.Length, Allocator.Temp);
            MultiGridVCycle(z, r, GridRes, H);
            NativeArray<float> p = new NativeArray<float>(z, Allocator.Temp);
            NativeArray<float> Ap = new NativeArray<float>(z.Length, Allocator.Temp);
            float rz_old = Dot(r, z);

            for (iter = 0; iter < Rhs.Length / 2; iter++)
            {
                Laplacian(p, Ap);
                float alpha = rz_old / Dot(p, Ap);
                for (int i = 0; i < r.Length; i++)
                    Lhs[i] += alpha * p[i];

                for (int i = 0; i < r.Length; i++)
                    r[i] -= alpha * Ap[i];

                if (math.sqrt(Dot(r, r)) < H * H * 0.005f)
                    break;

                for (int i = 0; i < z.Length; i++)
                    z[i] = 0;
                
                MultiGridVCycle(z, r, GridRes, H);
                float rz_new = Dot(r, z);
                float beta = rz_new / rz_old;
                for (int i = 0; i < r.Length; i++)
                    p[i] = z[i] + beta * p[i];
                rz_old = rz_new;
                // Debug.Log($"MGPCG {iter} iterations. rs:{math.sqrt(rz_old)} {residual}");
            }

            Residual(Lhs, Rhs, r, GridRes, H);
            rs = math.sqrt(Dot(r, r));
            // Debug.Log($"MGPCG converged in {iter}/{Rhs.Length} iterations. final residual: {rs}");

            r.Dispose();
            z.Dispose();
            p.Dispose();
            Ap.Dispose();
        }

        private void Laplacian(NativeArray<float> v, NativeArray<float> result)
        {
            int gridRes = GridRes;
            float h = H;
            float ih2 = 1f / (h * h);
            for (int i = 1; i < gridRes - 1; i++)
                result[i] = (2 * v[i] - NeighborSum(v, i)) * ih2;
        }

        private float Dot(NativeArray<float> lhs, NativeArray<float> rhs)
        {
            float sum = 0;
            for (int i = 1; i < lhs.Length - 1; i++)
                sum += lhs[i] * rhs[i];
            return sum;
        }

        private float Res(NativeArray<float> x, NativeArray<float> x_old)
        {
            float norm = 0;
            for (int i = 1; i < x.Length - 1; i++)
            {
                float temp = x[i] - x_old[i];
                norm += temp * temp;
            }

            return norm;
        }

        private void MultiGridVCycle(NativeArray<float> v, NativeArray<float> b, int res, float h)
        {
            if (res < 4)
            {
                v[1] = b[1] * H * H * 0.5f;
                return;
            }

            Smooth(v, b, res, h, 3);

            var r = new NativeArray<float>(b.Length, Allocator.Temp);
            Residual(v, b, r, res, h);

            int resC = res / 2 + 1;
            var rc = new NativeArray<float>(resC, Allocator.Temp);
            Restrict(r, rc, res);

            var ec = new NativeArray<float>(resC, Allocator.Temp);
            MultiGridVCycle(ec, rc, resC, h * 2);

            Prolongate(ec, v, res);

            Smooth(v, b, res, h, 3);

            // if (res == GridRes)
            // {
            //     Residual(v, b, r, res, h);
            //     float rs = 0;
            //     for (int i = 0; i < r.Length; i++)
            //         rs += r[i] * r[i];
            //     
            //     Debug.Log("////////////////////// residual: " + math.sqrt(rs));
            // }
            r.Dispose();
            rc.Dispose();
            ec.Dispose();
        }

        private void Smooth(NativeArray<float> v, NativeArray<float> f, int res, float h, int smoothCount)
        {
            float h2 = h * h;
            float omega = 1;
            for (int iter = 0; iter < smoothCount; iter++)
            {
                for (int i = 1; i < res - 1; i++)
                    v[i] = (1 - omega) * v[i] + omega * 0.5f * (h2 * f[i] + NeighborSum(v, i));
                
                // for (int i = res - 2; i > 0; i--)
                //     v[i] = (1 - omega) * v[i] + omega * 0.5f * (h2 * f[i] + NeighborSum(v, i));
            }
        }

        private void SmoothRB(NativeArray<float> v, NativeArray<float> f, int res, float h, int smoothCount)
        {
            float h2 = h * h;
            float omega = 1;
            for (int iter = 0; iter < smoothCount; iter++)
            {
                for (int i = 1; i < res - 1; i+=2)
                    v[i] = (1 - omega) * v[i] + omega * 0.5f * (h2 * f[i] + NeighborSum(v, i));
                
                for (int i = 2; i < res - 1; i+=2)
                    v[i] = (1 - omega) * v[i] + omega * 0.5f * (h2 * f[i] + NeighborSum(v, i));
                
                // for (int i = 2; i < res - 1; i+=2)
                //     v[i] = (1 - omega) * v[i] + omega * 0.5f * (h2 * f[i] + NeighborSum(v, i));
                //
                // for (int i = 1; i < res - 1; i+=2)
                //     v[i] = (1 - omega) * v[i] + omega * 0.5f * (h2 * f[i] + NeighborSum(v, i));
            }
        }

        private float NeighborSum(NativeArray<float> v, int i)
        {
            return v[i - 1] + v[i + 1];
        }

        private void Residual(NativeArray<float> v, NativeArray<float> b, NativeArray<float> r, int res, float h)
        {
            float ih2 = 1f / (h * h);
            for (int i = 1; i < res - 1; i++)
            {
                float ax = (2 * v[i] - NeighborSum(v, i)) * ih2;
                r[i] = b[i] - ax;
            }
        }

        private void Restrict(NativeArray<float> rf, NativeArray<float> rc, int res)
        {
            int gridResC = res / 2 + 1;

            for (int i = 1; i < gridResC - 1; i++)
                rc[i] = 0.25f * NeighborSum(rf, i * 2) + 0.5f * rf[i * 2];
        }

        private void Prolongate(NativeArray<float> ec, NativeArray<float> ef, int res)
        {
            int gridResC = res / 2 + 1;

            for (int i = 0; i < gridResC - 1; i++)
            {
                float cur = ec[i];
                float right = ec[i + 1];
                ef[i * 2] += cur;
                if (i * 2 + 1 < res - 1)
                    ef[i * 2 + 1] += 0.5f * (cur + right);
            }
        }
    }
}

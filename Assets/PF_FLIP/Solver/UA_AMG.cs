using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace PF_FLIP
{
    public class UnSmoothedAggregationMultiGridSolver
    {
        public NativeArray<float> Lhs;
        public NativeArray<float> Rhs;
        public NativeArray<float3> A; // x: center, y: left, z: down
        public int GridRes;
        public float H;

        private NativeArray<T> AllocateTemp<T>(int length, bool clear = false) where T : unmanaged
        {
            return new NativeArray<T>(length, Allocator.Temp);
        }
        
        public void Solve_MG(out int iter, out float rs)
        {
            NativeArray<float> v_old = AllocateTemp<float>(Lhs.Length);
            float norm = 0;
            for (iter = 0; iter < 10; iter++)
            {
                Lhs.CopyTo(v_old);
                MultiGridVCycle(A, Lhs, Rhs, GridRes, H);

                norm = math.sqrt(Res(Lhs, v_old));
                if (norm < H * H * 2)
                    break;

                // Debug.Log("MG iter " + iter + " res: " + norm);
            }

            Residual(A, Lhs, Rhs, v_old, GridRes, H);
            rs = math.sqrt(Dot(v_old, v_old));
            // Debug.Log("MG converged in iter " + iter + " res: " + norm);

            v_old.Dispose();
        }
        
        public void Solve_MGF(out int iter, out float rs)
        {
            NativeArray<float> v_old = AllocateTemp<float>(Lhs.Length);
            float norm = 0;
            // NativeArray<float> res = new NativeArray<float>(Lhs.Length, Allocator.Temp);
            for (iter = 0; iter < 10; iter++)
            {
                Lhs.CopyTo(v_old);
                
                MultiGridFCycle(A, Lhs, Rhs, GridRes, H);
                // if (iter < 2) MultiGridFCycle(A, Lhs, Rhs, GridRes, H);
                // else MultiGridVCycle(A, Lhs, Rhs, GridRes, H);
                
                norm = math.sqrt(Res(Lhs, v_old));
                if (norm < H * H * 2)
                    break;

                // Debug.Log("MG iter " + iter + " res: " + norm);
            }

            Residual(A, Lhs, Rhs, v_old, GridRes, H);
            rs = math.sqrt(Dot(v_old, v_old));
            // Debug.Log("MG converged in iter " + iter + " res: " + norm);

            v_old.Dispose();
        }

        public void Solve_GS(out int iter, out float rs)
        {
            NativeArray<float> v_old = AllocateTemp<float>(Lhs.Length);
            float norm = 0;

            Residual(A, Lhs, Rhs, v_old, GridRes, H);
            norm = math.sqrt(Dot(v_old, v_old));
            Debug.Log("GS iter 0 rs: " + norm);
            for (iter = 0; iter < 32; iter++)
            {
                Lhs.CopyTo(v_old);
                Smooth(A, Lhs, Rhs, GridRes, H, 32);

                Residual(A, Lhs, Rhs, v_old, GridRes, H);
                norm = math.sqrt(Dot(v_old, v_old));
                Debug.Log("GS iter " + iter*32 + " rs: " + norm);
                if (norm < H * H * 0.5f)
                    break;
            }

            iter *= 32;
            rs = norm;

            v_old.Dispose();
        }

        public void Solve_SSOR(float omega, out int iter, out float rs)
        {
            NativeArray<float> v_old = AllocateTemp<float>(Lhs.Length);

            Residual(A, Lhs, Rhs, v_old, GridRes, H);
            var norm = math.sqrt(Dot(v_old, v_old));
            // Debug.Log("SSOR iter 0 rs: " + norm);
            for (iter = 0; iter < 32; iter++)
            {
                Lhs.CopyTo(v_old);
                SSOR(A, Lhs, Rhs, GridRes, H, 32, omega);

                Residual(A, Lhs, Rhs, v_old, GridRes, H);
                norm = math.sqrt(Dot(v_old, v_old));
                // Debug.Log("SSOR iter " + iter*32 + " rs: " + norm);
                if (norm < H * H)
                    break;
            }

            iter *= 32;
            rs = norm;

            v_old.Dispose();
        }

        public void Solve_ConjugateGradient(out int iter, out float rs)
        {
            NativeArray<float> r = new NativeArray<float>(Rhs, Allocator.Temp);
            NativeArray<float> p = new NativeArray<float>(Rhs, Allocator.Temp);
            NativeArray<float> Ap = AllocateTemp<float>(Rhs.Length);
            float rs_old = Dot(r, r);

            for (iter = 0; iter < math.min(512, Rhs.Length); iter++)
            {
                Laplacian(A, p, Ap, GridRes, H);
                float alpha = rs_old / Dot(p, Ap);
                for (int i = 0; i < Lhs.Length; i++)
                    Lhs[i] += alpha * p[i];
                for (int i = 0; i < Lhs.Length; i++)
                    r[i] -= alpha * Ap[i];
                float rs_new = Dot(r, r);
                if (math.sqrt(rs_new) < H * H * 0.1f)
                {
                    rs_old = rs_new;
                    break;
                }

                float beta = rs_new / rs_old;
                for (int i = 0; i < Lhs.Length; i++)
                    p[i] = r[i] + beta * p[i];
                rs_old = rs_new;
            }
            
            rs = math.sqrt(rs_old);
            // Debug.Log($"ConjugateGradient converged in {iter}/{Rhs.Length} iterations. rs:{math.sqrt(rs_old)}");

            r.Dispose();
            p.Dispose();
            Ap.Dispose();
        }

        public void Solve_FMGPCG(out int iter, out float rs)
        {
            NativeArray<float> r = new NativeArray<float>(Rhs, Allocator.Temp);
            NativeArray<float> z = AllocateTemp<float>(Rhs.Length, true);
            MultiGridFCycle(A, z, r, GridRes, H);
            NativeArray<float> p = new NativeArray<float>(z, Allocator.Temp);
            NativeArray<float> Ap = AllocateTemp<float>(p.Length);

            float rz_old = Dot(r, z);

            for (iter = 0; iter < 10 && math.abs(rz_old) > 1e-6f; iter++)
            {
                Laplacian(A, p, Ap, GridRes, H);
                float alpha = rz_old / Dot(p, Ap);
                for (int i = 0; i < r.Length; i++)
                    Lhs[i] += alpha * p[i];

                for (int i = 0; i < r.Length; i++)
                    r[i] -= alpha * Ap[i];

                if (math.sqrt(Dot(r, r)) < H * H * 0.1f)
                    break;

                for (int i = 0; i < z.Length; i++)
                    z[i] = 0;
                
                MultiGridFCycle(A ,z, r, GridRes, H);
                float rz_new = Dot(r, z);
                float beta = rz_new / rz_old;
                for (int i = 0; i < p.Length; i++)
                    p[i] = z[i] + beta * p[i];
                rz_old = rz_new;
            }

            Residual(A, Lhs, Rhs, r, GridRes, H);
            rs = math.sqrt(Dot(r, r));
            // Debug.Log($"MGPCG converged in {iter}/{Rhs.Length} iterations. rs:{}");

            r.Dispose();
            z.Dispose();
            p.Dispose();
            Ap.Dispose();
        }

        public void Solve_MGPCG(out int iter, out float rs)
        {
            NativeArray<float> r = new NativeArray<float>(Rhs, Allocator.Temp);
            NativeArray<float> z = AllocateTemp<float>(Rhs.Length, true);
            MultiGridVCycle(A, z, r, GridRes, H);
            NativeArray<float> p = new NativeArray<float>(z, Allocator.Temp);
            NativeArray<float> Ap = AllocateTemp<float>(p.Length);

            float rz_old = Dot(r, z);

            for (iter = 0; iter < 10 && math.abs(rz_old) > 1e-6f; iter++)
            {
                Laplacian(A, p, Ap, GridRes, H);
                float alpha = rz_old / Dot(p, Ap);
                for (int i = 0; i < r.Length; i++)
                    Lhs[i] += alpha * p[i];

                for (int i = 0; i < r.Length; i++)
                    r[i] -= alpha * Ap[i];

                if (math.sqrt(Dot(r, r)) < H * H * 0.1f)
                    break;

                for (int i = 0; i < z.Length; i++)
                    z[i] = 0;

                MultiGridVCycle(A, z, r, GridRes, H);
                float rz_new = Dot(r, z);
                float beta = rz_new / rz_old;
                for (int i = 0; i < p.Length; i++)
                    p[i] = z[i] + beta * p[i];
                rz_old = rz_new;
            }

            Residual(A, Lhs, Rhs, r, GridRes, H);
            rs = math.sqrt(Dot(r, r));
            // Debug.Log($"MGPCG converged in {iter}/{Rhs.Length} iterations. rs:{}");

            r.Dispose();
            z.Dispose();
            p.Dispose();
            Ap.Dispose();
        }

        private void Laplacian(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> result, int res, float h)
        {
            float ih2 = 1f / (h * h);
            for (int y = 0; y < res; y++)
            for (int x = 0; x < res; x++)
            {
                int i = Coord2Index(x, y, res);
                if (Invalid(a[i].x)) continue;
                result[i] = (a[i].x * v[i] + NeighborSum(a, v, x, y, res)) * ih2;
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
        
        private void FullMultiGridInitialize(
            NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, int res, float h)
        {
            if (res <= 4)
            {
                Smooth(a, v, b, res, h, 4);
                return;
            }

            PreSmooth(a, v, b, res, h, 2);
            
            var r = AllocateTemp<float>(b.Length);
            Residual(a, v, b, r, res, h);

            int resC = res >> 1;
            var rc = AllocateTemp<float>(resC * resC);
            var ac = AllocateTemp<float3>(resC * resC);
            Restriction(r, a, rc, ac, res);

            var ec = AllocateTemp<float>(rc.Length);
            
            FullMultiGridInitialize(ac, ec, rc, resC, h);

            Prolongation(ec, a, v, res);

            PostSmooth(a, v, b, res, h, 2);
            
            r.Dispose();
            rc.Dispose();
            ec.Dispose();
            ac.Dispose();
        }

        private void MultiGridFCycle(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, int res, float h)
        {
            if (res <= 4)
            {
                Smooth(a, v, b, res, h, 4);
                return;
            }
            
            FullMultiGridInitialize(a, v, b, res, h);

            PreSmooth(a, v, b, res, h, 3);

            var r = AllocateTemp<float>(b.Length);
            Residual(a, v, b, r, res, h);

            int resC = res >> 1;
            var rc = AllocateTemp<float>(resC * resC);
            var ac = AllocateTemp<float3>(resC * resC);
            Restriction(r, a, rc, ac, res);

            var ec = AllocateTemp<float>(rc.Length);
            // for (int i = 0; i < 2; i++)
            MultiGridFCycle(ac, ec, rc, resC, h);

            Prolongation(ec, a, v, res);

            PostSmooth(a, v, b, res, h, 3);
            
            r.Dispose();
            rc.Dispose();
            ec.Dispose();
            ac.Dispose();
        }

        private void MultiGridVCycle(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, int res, float h)
        {
            if (res <= 4)
            {
                Smooth(a, v, b, res, h, 4);
                return;
            }

            PreSmooth(a, v, b, res, h, 3);

            var r = AllocateTemp<float>(b.Length);
            Residual(a, v, b, r, res, h);

            int resC = res >> 1;
            var rc = AllocateTemp<float>(resC * resC);
            var ac = AllocateTemp<float3>(resC * resC);
            Restriction(r, a, rc, ac, res);

            var ec = AllocateTemp<float>(rc.Length);
            // for (int i = 0; i < 2; i++)
                MultiGridVCycle(ac, ec, rc, resC, h);

            Prolongation(ec, a, v, res);

            PostSmooth(a, v, b, res, h, 3);
            
            r.Dispose();
            rc.Dispose();
            ec.Dispose();
            ac.Dispose();
        }
        
        private void Smooth(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> temp, 
            NativeArray<float> f, int res, float h, int count)
        {
            float h2 = h * h;
            float omega = 0.8f;

            for (int iter = 0; iter < count; iter++)
            {
                for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++)
                {
                    int i = Coord2Index(x, y, res);
                    temp[i] = math.lerp(v[i], (h2 * f[i] - NeighborSum(a, v, x, y, res)) / a[i].x, omega);
                }

                for (int y = res - 1; y >= 0; y--)
                for (int x = res - 1; x >= 0; x--)
                {
                    int i = Coord2Index(x, y, res);
                    v[i] = math.lerp(temp[i], (h2 * f[i] - NeighborSum(a, temp, x, y, res)) / a[i].x, omega);
                }
            }
        }
        
        private void PreSmooth(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f, int res, float h, int count)
        {
            float h2 = h * h;

            for (int iter = 0; iter < count; iter++)
            {
                for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++)
                {
                    int i = Coord2Index(x, y, res);
                    v[i] = Invalid(a[i].x) ? 0 : (h2 * f[i] - NeighborSum(a, v, x, y, res)) / a[i].x;
                }
            }
        }
        
        private void PostSmooth(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f, int res, float h, int count)
        {
            float h2 = h * h;

            for (int iter = 0; iter < count; iter++)
            {
                for (int y = res - 1; y >= 0; y--)
                for (int x = res - 1; x >= 0; x--)
                {
                    int i = Coord2Index(x, y, res);
                    v[i] = Invalid(a[i].x) ? 0 : (h2 * f[i] - NeighborSum(a, v, x, y, res)) / a[i].x;
                }
            }
        }

        private void Smooth(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f, int res, float h, int count)
        {
            float h2 = h * h;

            for (int iter = 0; iter < count; iter++)
            {
                for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++)
                {
                    int i = Coord2Index(x, y, res);
                    v[i] = Invalid(a[i].x) ? 0 : (h2 * f[i] - NeighborSum(a, v, x, y, res)) / a[i].x;
                }

                for (int y = res - 1; y >= 0; y--)
                for (int x = res - 1; x >= 0; x--)
                {
                    int i = Coord2Index(x, y, res);
                    v[i] = Invalid(a[i].x) ? 0 : (h2 * f[i] - NeighborSum(a, v, x, y, res)) / a[i].x;
                }
            }
        }

        private void SSOR(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f, int res, float h, int count, float omega)
        {
            float h2 = h * h;

            for (int iter = 0; iter < count; iter++)
            {
                for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++)
                {
                    int i = Coord2Index(x, y, res);
                    v[i] = Invalid(a[i].x) ? 0 : 
                        math.lerp(v[i], (h2 * f[i] - NeighborSum(a, v, x, y, res)) / a[i].x, omega);
                }

                for (int y = res - 1; y >= 0; y--)
                for (int x = res - 1; x >= 0; x--)
                {
                    int i = Coord2Index(x, y, res);
                    v[i] = Invalid(a[i].x) ? 0 : 
                        math.lerp(v[i], (h2 * f[i] - NeighborSum(a, v, x, y, res)) / a[i].x, omega);
                }
            }
        }

        private void Residual(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, NativeArray<float> r, int res, float h)
        {
            float ih2 = 1f / (h * h);
            for (int y = 0; y < res; y++)
            for (int x = 0; x < res; x++)
            {
                int i = Coord2Index(x, y, res);
                r[i] = b[i] - ih2 * (a[i].x * v[i] + NeighborSum(a, v, x, y, res));
            }
        }

        private void Restriction(NativeArray<float> rf, NativeArray<float3> af, NativeArray<float> rc, NativeArray<float3> ac, int res)
        {
            int gridResC = res >> 1;
            int gridResF = res;

            for (int y = 0; y < gridResC; y++)
            for (int x = 0; x < gridResC; x++)
            {
                int ci = Coord2Index(x, y, gridResC);
                float r_coarse = 0;
                float3 A_coarse = float3.zero;
                for (int yy = 0; yy < 2; yy++)
                for (int xx = 0; xx < 2; xx++)
                {
                    int fi = Coord2Index(x * 2 + xx, y * 2 + yy, gridResF);
                    float3 A_fine = af[fi];
                    if (Invalid(A_fine.x))
                        continue;
                    
                    A_coarse.x += A_fine.x;
                    r_coarse += rf[fi];
                    
                    if (xx == 0) A_coarse.y += A_fine.y;
                    else A_coarse.x += A_fine.y * 2;
                    
                    if (yy == 0) A_coarse.z += A_fine.z;
                    else A_coarse.x += A_fine.z * 2;
                }

                rc[ci] = r_coarse * 0.25f;
                ac[ci] = A_coarse * 0.25f;
            }
        }

        private void Prolongation(NativeArray<float> ec, NativeArray<float3> af, NativeArray<float> ef, int res)
        {
            int gridResF = res;
            int gridResC = res >> 1;

            for (int y = 0; y < gridResC; y++)
            for (int x = 0; x < gridResC; x++)
            {
                float e = ec[Coord2Index(x, y, gridResC)];
                for (int yy = 0; yy < 2; yy++)
                for (int xx = 0; xx < 2; xx++)
                {
                    int fi = Coord2Index(x * 2 + xx, y * 2 + yy, gridResF);
                    if (!Invalid(af[fi].x)) ef[fi] += e * 2;
                }
            }
        }

        private float NeighborSum(NativeArray<float3> a, NativeArray<float> v, int x, int y, int gridRes)
        {
            float3 ac = a[Coord2Index(x, y, gridRes)];
            float3 ar = x < gridRes - 1 ? a[Coord2Index(x + 1, y, gridRes)] : float3.zero;
            float3 at = y < gridRes - 1 ? a[Coord2Index(x, y + 1, gridRes)] : float3.zero;
            float sum = 0;
            if (!Invalid(ac.y)) sum += ac.y * v[Coord2Index(x - 1, y, gridRes)];
            if (!Invalid(ac.z)) sum += ac.z * v[Coord2Index(x, y - 1, gridRes)];
            if (!Invalid(ar.y)) sum += ar.y * v[Coord2Index(x + 1, y, gridRes)];
            if (!Invalid(at.z)) sum += at.z * v[Coord2Index(x, y + 1, gridRes)];

            return sum;
        }
        
        private int Coord2Index(int x, int y, int gridRes) => y * gridRes + x;
        
        private bool Invalid(float x) => math.abs(x) < 1e-5f;
    }

    public struct UAAMG1DSolver
    {
        public NativeArray<float> Lhs;
        public NativeArray<float> Rhs;
        public NativeArray<float3> A;
        public int GridRes;
        public float H;

        public void Test()
        {
            Smooth(A, Lhs, Rhs, GridRes, H, 1);
            // MultiGridVCycle(Lhs, Rhs, A, GridRes, H);
        }

        public void Solve_MG(out int iter, out float rs)
        {
            NativeArray<float> v_old = new NativeArray<float>(Lhs.Length, Allocator.Temp);
            float norm = 0;
            for (iter = 0; iter < 10; iter++)
            {
                Lhs.CopyTo(v_old);
                MultiGridVCycle(Lhs, Rhs, A, GridRes, H);

                norm = math.sqrt(Res(Lhs, v_old));
                if (norm < H * H)
                    break;
                // Debug.Log("MG iter " + iter + " res: " + norm);
            }

            Residual(A, Lhs, Rhs, v_old, GridRes, H);
            rs = math.sqrt(Dot(v_old, v_old));

            v_old.Dispose();
        }
        public void Solve_GS(out int iter, out float rs)
        {
            NativeArray<float> v_old = new NativeArray<float>(Lhs.Length, Allocator.Temp);
            float norm = 0;
            for (iter = 0; iter < 500; iter++)
            {
                Lhs.CopyTo(v_old);
                Smooth(A, Lhs, Rhs, GridRes, H, 16);

                norm = math.sqrt(Res(Lhs, v_old));
                if (norm < H * H)
                    break;
                Debug.Log("GS iter " + iter + "*16 res: " + norm);
            }

            rs = norm;
            v_old.Dispose();
        }

        public void Solve_ConjugateGradient(out int iter, out float rs)
        {
            var u = Lhs;
            var b = Rhs;
            NativeArray<float> r = new NativeArray<float>(b.Length, Allocator.Temp);
            Residual(A, u, b, r, GridRes, H);
            NativeArray<float> p = new NativeArray<float>(r, Allocator.Temp);
            NativeArray<float> Ap = new NativeArray<float>(b.Length, Allocator.Temp);
            float rs_old = Dot(r, r);

            for (iter = 0; iter < b.Length; iter++)
            {
                Laplacian(A, p, Ap, GridRes, H);
                float alpha = rs_old / Dot(p, Ap);
                for (int i = 0; i < u.Length; i++)
                    u[i] += alpha * p[i];
                for (int i = 0; i < r.Length; i++)
                    r[i] -= alpha * Ap[i];
                float rs_new = Dot(r, r);
                // if (math.sqrt(rs_new) < H * H * 0.1f)
                //     break;

                float beta = rs_new / rs_old;
                for (int i = 0; i < p.Length; i++)
                    p[i] = r[i] + beta * p[i];
                rs_old = rs_new;
            }

            Residual(A, u, b, r, GridRes, H);
            rs = math.sqrt(Dot(r, r));
            // Debug.Log($"ConjugateGradient converged in {iter}/{Rhs.Length} iterations. rs:{math.sqrt(rs_old)}");

            r.Dispose();
            p.Dispose();
            Ap.Dispose();
        }

        public void Solve_MGPCG(out int iter, out float rs)
        {
            NativeArray<float> r = new NativeArray<float>(Rhs, Allocator.Temp);
            NativeArray<float> z = new NativeArray<float>(r, Allocator.Temp);
            MultiGridVCycle(z, r, A, GridRes, H);
            NativeArray<float> p = new NativeArray<float>(z, Allocator.Temp);
            NativeArray<float> Ap = new NativeArray<float>(z.Length, Allocator.Temp);
            float rz_old = Dot(r, z);

            for (iter = 0; iter < Rhs.Length / 2; iter++)
            {
                Laplacian(A, p, Ap, GridRes, H);
                float alpha = rz_old / Dot(p, Ap);
                for (int i = 0; i < r.Length; i++)
                    Lhs[i] += alpha * p[i];

                for (int i = 0; i < r.Length; i++)
                    r[i] -= alpha * Ap[i];

                if (math.sqrt(Dot(r, r)) < H * H * 0.005f)
                    break;

                for (int i = 0; i < z.Length; i++)
                    z[i] = 0;
                MultiGridVCycle(z, r, A, GridRes, H);
                float rz_new = Dot(r, z);
                float beta = rz_new / rz_old;
                for (int i = 0; i < r.Length; i++)
                    p[i] = z[i] + beta * p[i];
                rz_old = rz_new;
                // Debug.Log($"MGPCG {iter} iterations. rs:{math.sqrt(rz_old)} {residual}");
            }

            Residual(A, Lhs, Rhs, r, GridRes, H);
            rs = math.sqrt(Dot(r, r));
            // Debug.Log($"MGPCG converged in {iter}/{Rhs.Length} iterations. final residual: {rs}");

            r.Dispose();
            z.Dispose();
            p.Dispose();
            Ap.Dispose();
        }

        private void Laplacian(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> result, int res, float h)
        {
            float ih2 = 1f / (h * h);
            for (int i = 0; i < res; i++)
                result[i] = (a[i].y * v[i] + NeighborSum(a[i], v, i)) * ih2;
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

        private void MultiGridVCycle(NativeArray<float> v, NativeArray<float> b, NativeArray<float3> a, int res, float h)
        {
            if (res <= 4)
            {
                Smooth(a, v, b, res, h, 2);
                return;
            }

            Smooth(a, v, b, res, h, 2);

            var r = new NativeArray<float>(b.Length, Allocator.Temp);
            Residual(a, v, b, r, res, h);

            int resC = res / 2;
            var rc = new NativeArray<float>(resC, Allocator.Temp);
            var ac = new NativeArray<float3>(resC, Allocator.Temp);
            Restrict(r, a, rc, ac, res);

            var ec = new NativeArray<float>(resC, Allocator.Temp);
            // for (int i = 0; i < 2; i++) // W-cycle
                MultiGridVCycle(ec, rc, ac, resC, h);

            Prolongate(ec, a, v, res);

            Smooth(a, v, b, res, h, 2);

            // if (res == GridRes)
            // {
            //     Residual(a, v, b, r, res, h);
            //     float rs = 0;
            //     for (int i = 0; i < r.Length; i++)
            //         rs += r[i] * r[i];
            //     
            //     Debug.Log("////////////////////// residual: " + math.sqrt(rs));
            // }
            r.Dispose();
            rc.Dispose();
            ec.Dispose();
            ac.Dispose();
        }

        private void Smooth(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f, int res, float h, int smoothCount)
        {
            float h2 = h * h;
            for (int iter = 0; iter < smoothCount; iter++)
            {
                for (int i = 0; i < res; i++)
                    v[i] = (h2 * f[i] - NeighborSum(a[i], v, i)) / a[i].y;
                
                for (int i = res - 1; i >= 0; i--)
                    v[i] = (h2 * f[i] - NeighborSum(a[i], v, i)) / a[i].y;
            }
        }
        
        private void Residual(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, NativeArray<float> r, int res, float h)
        {
            float ih2 = 1f / (h * h);
            for (int i = 0; i < res; i++)
                r[i] = b[i] - ih2 * (a[i].y * v[i] + NeighborSum(a[i], v, i));
        }

        private float NeighborSum(float3 a, NativeArray<float> v, int i)
        {
            return ((a.x != 0) ? (a.x * v[i - 1]) : 0) + ((a.z != 0) ? (a.z * v[i + 1]) : 0);
        }

        private void Restrict(NativeArray<float> rf, NativeArray<float3> af, NativeArray<float> rc,
            NativeArray<float3> ac, int res)
        {
            int gridResC = res / 2;
            for (int i = 0; i < gridResC; i++)
            {
                int idx0 = i * 2;
                int idx1 = i * 2 + 1;
        
                float3 af0 = af[idx0];
                float3 af1 = af[idx1];
        
                float3 A_coarse = float3.zero;
        
                A_coarse.x = af0.x;
        
                A_coarse.y = af0.y + af0.z + af1.x + af1.y;
        
                A_coarse.z = af1.z;
        
                rc[i] = (rf[idx0] + rf[idx1]) * 0.5f;
                ac[i] = A_coarse * 0.5f;
            }
        }

        private void Prolongate(NativeArray<float> ec, NativeArray<float3> af, NativeArray<float> ef, int res)
        {
            int gridResC = res / 2;

            for (int i = 0; i < gridResC; i++)
            {
                float cur = ec[i];
                if (af[i * 2].y > 0) ef[i * 2] += cur * 2;
                if (af[i * 2 + 1].y > 0) ef[i * 2 + 1] += cur * 2;
            }
        }
    }
}

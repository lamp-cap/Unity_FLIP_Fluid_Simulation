using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace PF_FLIP
{
    public class Neumann_UAAMGSolver : System.IDisposable
    {
        public NativeArray<float3>[] As; // x: center, y: left, z: down
        private NativeArray<float>[] Rs;
        public NativeArray<float> F;
        
        public NativeArray<float> V;
        
        public NativeArray<float>[] Zs;
        public NativeArray<float3> A => As[0]; // x: center, y: left, z: down
        
        private NativeArray<float> R => Rs[0];
        private NativeArray<float> Z => Zs[0];
        private NativeArray<float> P;
        private NativeArray<float> Ap;
        private readonly int GridRes;
        private readonly float H;
        private readonly int L;
        private const int BatchSize = 64;
        private NativeReference<float> rs_old;
        private NativeReference<float> pAp;
        private NativeReference<float> rs_new;

        public Neumann_UAAMGSolver(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, int gridRes, float h)
        {
            L = (int)(math.log2(gridRes) - 1);
            Debug.Log("Neumann UAAMG Solver levels: " + L);
            As = new NativeArray<float3>[L];
            Zs = new NativeArray<float>[L];
            Rs = new NativeArray<float>[L];
            V = v;
            F = b;
            As[0] = a;
            Zs[0] = new NativeArray<float>(gridRes * gridRes, Allocator.Persistent);
            Rs[0] = new NativeArray<float>(gridRes * gridRes, Allocator.Persistent);
            int res = gridRes;
            for (int i = 1; i < L; i++)
            {
                res >>= 1;
                As[i] = new NativeArray<float3>(res * res, Allocator.Persistent);
                Zs[i] = new NativeArray<float>(res * res, Allocator.Persistent);
                Rs[i] = new NativeArray<float>(res * res, Allocator.Persistent);
            }
            
            GridRes = gridRes;
            H = h;
            P = new NativeArray<float>(gridRes * gridRes, Allocator.Persistent);
            Ap = new NativeArray<float>(gridRes * gridRes, Allocator.Persistent);
            
            rs_old = new NativeReference<float>(0, Allocator.Persistent);
            rs_new = new NativeReference<float>(0, Allocator.Persistent);
            pAp = new NativeReference<float>(0, Allocator.Persistent);
        }

        #region Grid Type Utils

        private const uint SOLID = 2;
        private const uint AIR = 1;
        private const uint FLUID = 0;
        private static bool IsSolidCell(uint gridTypes)
        {
            return (gridTypes & 3u) == SOLID;
        }
        private static bool IsFluidCell(uint gridTypes)
        {
            return (gridTypes & 3u) == FLUID;
        }
        private static bool IsAirCell(uint gridTypes)
        {
            return (gridTypes & 3u) == AIR;
        }
        private static uint2 NeighborGridTypeLB(uint gridTypes)
        {
            return new uint2((gridTypes >> 2) & 3u, (gridTypes >> 6) & 3u);
        }
        private static uint2 NeighborGridTypeAxis(int axis, uint gridTypes)
        {
            return new uint2((gridTypes >> (axis * 4 + 2)) & 3u,
                (gridTypes >> (axis * 4 + 4)) & 3u);
        }
        private static bool2 IsNeighborSolidCell(int2 dir, uint gridTypes)
        {
            return new bool2(((gridTypes >> ((dir.x << 1) + 2)) & 3u) == SOLID,
                ((gridTypes >> ((dir.y << 1) + 6)) & 3u) == SOLID);
        }
        #endregion

        public void Solve_Jacobi(int maxIter, out float rs)
        {
            // new ClearJob(V).Schedule().Complete();
            // new ClearJob(Z).Schedule().Complete();
            JobHandle handle = default; 
            for (int iter = 0; iter < maxIter; iter++)
            {
                handle = new SmoothJacobi(F, A, V, Z,  H, GridRes).Schedule(V.Length, BatchSize, handle);
                handle = new SmoothJacobi(F, A, Z, V,  H, GridRes).Schedule(V.Length, BatchSize, handle);
            }
            handle.Complete();

            new Residual(A, V, F, R,  GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_new).Schedule().Complete();
            rs = math.sqrt(rs_new.Value);
        }

        private JobHandle SmoothJob_Jacobi(int maxIter, JobHandle handle = default)
        {
            for (int iter = 0; iter < maxIter; iter++)
            {
                handle = new SmoothJacobi(F, A, Z, V,  H, GridRes).Schedule(V.Length, BatchSize, handle);
                handle = new SmoothJacobi(F, A, V, Z,  H, GridRes).Schedule(V.Length, BatchSize, handle);
            }

            return handle;
        }
        
        public void Solve_MG(int maxIter, out float rs)
        {
            R.CopyFrom(F);
            new ClearJob(Z).Schedule().Complete();
            for (int iter = 0; iter < maxIter; iter++)
            {
                MultiGridVCycle().Complete();
                // MultiGridVCycle(A, Z, R, 0, GridRes, H);
            }

            V.CopyFrom(Z);
            new Residual(A, V, F, R,  GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_new).Schedule().Complete();
            rs = math.sqrt(rs_new.Value);
        }
        
        public void Solve_MGFCycle(int maxIter, out float rs)
        {
            R.CopyFrom(F);
            new ClearJob(Z).Schedule().Complete();
            for (int iter = 0; iter < maxIter; iter++)
            {
                MultiGridFCycle().Complete();
                // MultiGridVCycle(A, Z, R, 0, GridRes, H);
            }

            V.CopyFrom(Z);
            new Residual(A, V, F, R,  GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_new).Schedule().Complete();
            rs = math.sqrt(rs_new.Value);
        }
        
        public void Solve_GSRB(int maxIter, out float rs)
        {
            new ClearJob(V).Schedule().Complete();
            PreSmoothJob(A, V, F, GridRes, H, maxIter).Complete();
            // new SymmetricGaussSeidel(F, A, V, H, GridRes, maxIter).Schedule().Complete();
            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_new).Schedule().Complete();
            rs = math.sqrt(rs_new.Value);
        }
        
        public void Solve_GS(int maxIter, out float rs)
        {
            new ClearJob(V).Schedule().Complete();
            // SmoothJob(A, V, F, GridRes, H, maxIter).Complete();
            new GaussSeidel(F, A, V, H, GridRes, maxIter).Schedule().Complete();
            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_new).Schedule().Complete();
            rs = math.sqrt(rs_new.Value);
        }
        
        public void Solve_SSOR(float omega, int iter, out float rs)
        {
            new ClearJob(V).Schedule().Complete();
            JobHandle handle = default;
            // for (int i = 0; i < iter; i++)
            // {
                // handle = new SSOR_Phase(F, A, V, H, GridRes, 0, omega).Schedule(V.Length, BatchSize, handle);
                // handle = new SSOR_Phase(F, A, V, H, GridRes, 1, omega).Schedule(V.Length, BatchSize, handle);
                // handle = new SSOR_Phase(F, A, V, H, GridRes, 1, omega).Schedule(V.Length, BatchSize, handle);
                // handle = new SSOR_Phase(F, A, V, H, GridRes, 0, omega).Schedule(V.Length, BatchSize, handle);
            // }
            handle = new SymmetricSuccessiveOverRelaxation(F, A, V, H, GridRes, iter, omega).Schedule(handle);
            handle.Complete();
        
            new Residual(A, V, F, R,  GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_new).Schedule().Complete();
            rs = math.sqrt(rs_new.Value);
        }
        
        public void Solve_CG(int maxIter, out int iter, out float rs)
        {
            // new ClearJob(V).Schedule().Complete();
            new Residual(A, V, F, R,  GridRes, H).Schedule(R.Length, BatchSize).Complete();
            // R.CopyFrom(F);
            P.CopyFrom(R);
            
            new Dot(R, R, rs_old).Schedule().Complete();

            if (math.abs(rs_old.Value) > 1e-3f)
            {
                for (iter = 0; iter < maxIter; iter++)
                {
                    var handle = new Laplace(A, P, Ap, GridRes, H).Schedule(R.Length, BatchSize);

                    handle = new Dot(P, Ap, pAp).Schedule(handle);

                    handle = new UpdateVR(P, Ap, V, R, rs_old, pAp).Schedule(V.Length, BatchSize, handle);

                    new Dot(R, R, rs_new).Schedule(handle).Complete();

                    if (math.sqrt(rs_new.Value) < H * H)
                        break;

                    new UpdateP(R, P, rs_new, rs_old).Schedule(P.Length, BatchSize).Complete();
                    (rs_old, rs_new) = (rs_new, rs_old);
                }
            }
            else iter = 0;

            new Residual(A, V, F, R,  GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_old).Schedule().Complete();
            rs = math.sqrt(rs_old.Value);
        }

        public void Solve_MGPCG(int maxIter, out float rs)
        {
            new ClearJob(V).Schedule().Complete();
            Z.CopyFrom(V);
            R.CopyFrom(F);
            MultiGridVCycle().Complete();
            P.CopyFrom(Z);

            new Dot(R, Z, rs_old).Schedule().Complete();
            if (math.abs(rs_old.Value) > 1e-3f)
            {
                for (int iter = 0; iter < maxIter; iter++)
                {
                    var handle = new Laplace(A, P,  Ap, GridRes, H).Schedule(R.Length, BatchSize);
                    handle = new Dot(P, Ap, pAp).Schedule(handle);
                    new UpdateVR(P, Ap, V, R, rs_old, pAp).Schedule(V.Length, BatchSize, handle).Complete();

                    if (iter == maxIter - 1) break;

                    handle = new ClearJob(Z).Schedule();
                    handle = MultiGridVCycle(handle);
                    handle = new Dot(R, Z, rs_new).Schedule(handle);
                    new UpdateP(Z, P, rs_new, rs_old).Schedule(P.Length, BatchSize, handle).Complete();

                    // new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
                    // new Dot(R, R, rs_old).Schedule().Complete();
                    // Debug.Log($"iter: {iter}, rs: {rs_old.Value}");
                    rs_old.Value = rs_new.Value;
                }
            }

            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_old).Schedule().Complete();
            rs = math.sqrt(rs_old.Value);
        }

        public void Solve_SMGPCG(int maxIter, out float rs)
        {
            new ClearJob(V).Schedule().Complete();
            Z.CopyFrom(V);
            R.CopyFrom(F);
            SmoothJob_Jacobi(8).Complete();
            P.CopyFrom(Z);

            new Dot(R, Z, rs_old).Schedule().Complete();
            if (math.abs(rs_old.Value) > 1e-3f)
            {
                for (int iter = 0; iter < maxIter; iter++)
                {
                    var handle = new Laplace(A, P,  Ap, GridRes, H).Schedule(R.Length, BatchSize);
                    handle = new Dot(P, Ap, pAp).Schedule(handle);
                    new UpdateVR(P, Ap, V, R, rs_old, pAp).Schedule(V.Length, BatchSize, handle).Complete();

                    if (iter == maxIter - 1) break;

                    handle = new ClearJob(Z).Schedule();
                    handle = SmoothJob_Jacobi(8, handle);
                    handle = new Dot(R, Z, rs_new).Schedule(handle);
                    new UpdateP(Z, P, rs_new, rs_old).Schedule(P.Length, BatchSize, handle).Complete();

                    // new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
                    // new Dot(R, R, rs_old).Schedule().Complete();
                    // Debug.Log($"iter: {iter}, rs: {rs_old.Value}");
                    rs_old.Value = rs_new.Value;
                }
            }

            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_old).Schedule().Complete();
            rs = math.sqrt(rs_old.Value);
        }
        
        public void Solve_MGFPCG(int maxIter, out float rs)
        {
            new ClearJob(V).Schedule().Complete();
            Z.CopyFrom(V);
            R.CopyFrom(F);
            MultiGridFCycle().Complete();
            P.CopyFrom(Z);

            new Dot(R, Z, rs_old).Schedule().Complete();
            if (math.abs(rs_old.Value) > 1e-3f)
            {
                for (int iter = 0; iter < maxIter; iter++)
                {
                    var handle = new Laplace(A, P,  Ap, GridRes, H).Schedule(R.Length, BatchSize);
                    handle = new Dot(P, Ap, pAp).Schedule(handle);
                    new UpdateVR(P, Ap, V, R, rs_old, pAp).Schedule(V.Length, BatchSize, handle).Complete();

                    if (iter == maxIter - 1) break;

                    handle = new ClearJob(Z).Schedule();
                    handle = MultiGridFCycle(handle);
                    handle = new Dot(R, Z, rs_new).Schedule(handle);
                    new UpdateP(Z, P, rs_new, rs_old).Schedule(P.Length, BatchSize, handle).Complete();

                    // new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
                    // new Dot(R, R, rs_old).Schedule().Complete();
                    // Debug.Log($"iter: {iter}, rs: {rs_old.Value}");
                    rs_old.Value = rs_new.Value;
                }
            }

            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_old).Schedule().Complete();
            rs = math.sqrt(rs_old.Value);
        }

        [BurstCompile]
        private struct Laplace : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float> _v;
            [ReadOnly] private NativeArray<float3> _a;
            [WriteOnly] private NativeArray<float> _result;
            private readonly float _ih2;
            private readonly int _res;
            
            public Laplace(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> result, 
                int res, float h)
            {
                _ih2 = 1f / (h * h);
                _res = res;
                _v = v;
                _a = a;
                _result = result;
            }
            
            public void Execute(int i)
            {
                float3 a = _a[i];
                if (a.x != 0)
                {
                    int x = i % _res;
                    int y = i / _res;
                    float sum = NeighborSum(_a, _v, a, x, y, _res, out float ac);
                    _result[i] = _ih2 * (ac + sum);
                }
                else
                    _result[i] = 0;
            }
        }
        
        [BurstCompile]
        private struct Dot : IJob
        {
            [ReadOnly] private NativeArray<float> _lhs;
            [ReadOnly] private NativeArray<float> _rhs;
            [WriteOnly] private NativeReference<float> _result;
            
            public Dot(NativeArray<float> lhs, NativeArray<float> rhs, NativeReference<float> result)
            {
                _lhs = lhs;
                _rhs = rhs;
                _result = result;
            }
            
            public void Execute()
            {
                float sum = 0;
                for (int i = 0; i < _lhs.Length; i++)
                    sum += _lhs[i] * _rhs[i];
                _result.Value = sum;
            }
        }
        
        [BurstCompile]
        private struct UpdateVR : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float> _p;
            [ReadOnly] private NativeArray<float> _ap;
            private NativeArray<float> _v;
            private NativeArray<float> _r;
            [ReadOnly] private NativeReference<float> _rsOld;
            [ReadOnly] private NativeReference<float> _pAp;
            

            public UpdateVR(NativeArray<float> p, NativeArray<float> ap, NativeArray<float> v, 
                NativeArray<float> r, NativeReference<float> rsOld, NativeReference<float> pAp)
            {
                _p = p;
                _ap = ap;
                _v = v;
                _r = r;
                _rsOld = rsOld;
                _pAp = pAp;
            }
            
            public void Execute(int i)
            {
                float alpha = _rsOld.Value / _pAp.Value;
                _v[i] += alpha * _p[i];
                _r[i] -= alpha * _ap[i];
            }
        }
        
        [BurstCompile]
        private struct UpdateP : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float> _z;
            private NativeArray<float> _p;
            [ReadOnly] private NativeReference<float> _rsNew;
            [ReadOnly] private NativeReference<float> _rsOld;
            
            public UpdateP(NativeArray<float> z, NativeArray<float> p, NativeReference<float> rsNew, NativeReference<float> rsOld)
            {
                _z = z;
                _p = p;
                _rsNew = rsNew;
                _rsOld = rsOld;
            }
            
            public void Execute(int i)
            {
                _p[i] = _z[i] + _p[i] * _rsNew.Value / _rsOld.Value;
            }
        }
        
        private JobHandle MultiGridFCycle(JobHandle handle = default)
        {
            float h = H;
            int top = L - 1;
            for (int i = 0; i < top; i++)
            {
                int res = GridRes >> i;
                handle = FullMultiGridInitialize(i, handle);
                handle = PreSmoothJob(As[i], Zs[i], Rs[i], res, h, 3, handle);
                handle = new Restriction(Rs[i], As[i], Zs[i], Rs[i + 1], As[i + 1], Zs[i + 1], res, h)
                    .Schedule(Rs[i + 1].Length, BatchSize, handle);
            }

            handle = new SymmetricGaussSeidel(Rs[top], As[top], Zs[top], h, GridRes >> top, 4).Schedule(handle);
            // handle = SmoothJob(As[top], Zs[top], Rs[top], GridRes >> top, h, 8, handle);

            for (int i = top - 1; i >= 0; i--)
            {
                int res = GridRes >> i;
                handle = new Prolongation(Zs[i+1], As[i], Zs[i], res).Schedule(Zs[i].Length, BatchSize, handle);
                handle = PostSmoothJob(As[i], Zs[i], Rs[i],  res, h, 3, handle);
            }

            return handle;
        }

        private JobHandle FullMultiGridInitialize(int level, JobHandle handle = default)
        {
            float h = H;
            int top = L - 1;
            for (int i = level; i < top; i++)
            {
                int res = GridRes >> i;
                handle = PreSmoothJob(As[i], Zs[i], Rs[i], res, h, 2, handle);
                handle = new Restriction(Rs[i], As[i], Zs[i], Rs[i + 1], As[i + 1], Zs[i + 1], res, h)
                    .Schedule(Rs[i + 1].Length, BatchSize, handle);
            }

            handle = new SymmetricGaussSeidel(Rs[top], As[top], Zs[top], h, GridRes >> top, 4).Schedule(handle);

            for (int i = top - 1; i >= level; i--)
            {
                int res = GridRes >> i;
                handle = new Prolongation(Zs[i+1], As[i], Zs[i], res).Schedule(Zs[i].Length, BatchSize, handle);
                // handle = SmoothJob(As[i], Zs[i], Rs[i],  res, h, 2, handle);
                handle = PostSmoothJob(As[i], Zs[i], Rs[i],  res, h, 2, handle);
            }

            return handle;
        }

        private JobHandle MultiGridVCycle(JobHandle handle = default)
        {
            float h = H;
            int top = L - 1;
            for (int i = 0; i < top; i++)
            {
                int res = GridRes >> i;
                handle = PreSmoothJob(As[i], Zs[i], Rs[i], res, h, 3, handle);
                handle = new Restriction(Rs[i], As[i], Zs[i], Rs[i + 1], As[i + 1], Zs[i + 1], res, h)
                    .Schedule(Rs[i + 1].Length, BatchSize, handle);
            }

            handle = new SymmetricGaussSeidel(Rs[top], As[top], Zs[top], h, GridRes >> top, 4).Schedule(handle);
            // handle = SmoothJob(As[top], Zs[top], Rs[top], GridRes >> top, h, 8, handle);

            for (int i = top - 1; i >= 0; i--)
            {
                int res = GridRes >> i;
                handle = new Prolongation(Zs[i+1], As[i], Zs[i], res).Schedule(Zs[i].Length, BatchSize, handle);
                handle = PostSmoothJob(As[i], Zs[i], Rs[i],  res, h, 3, handle);
            }

            return handle;
        }
        
        private void MultiGridVCycle(NativeArray<float3> af, NativeArray<float> ef, NativeArray<float> rf, 
            int level, int res, float h)
        {
            if (res <= 4)
            {
                new SymmetricGaussSeidel(rf, af, ef, h, res, 4).Schedule().Complete();
                // SmoothJob(af, ef, rf, res, h, 8).Complete();
                return;
            }

            SmoothJob(af, ef, rf, res, h, 4).Complete();
            
            int resC = res / 2;
            var rc = Rs[level + 1];
            var ac = As[level + 1];
            var ec = Zs[level + 1];
            new Restriction(rf, af, ef, rc, ac, ec, res, h).Schedule(rc.Length, BatchSize).Complete();

            // for (int i = 0; i < 2; i++) // W-cycle
            MultiGridVCycle(ac, ec, rc, level + 1, resC, h);

            new Prolongation(ec, af, ef, res).Schedule(ef.Length, BatchSize).Complete();

            SmoothJob(af, ef, rf, res, h, 4).Complete();
        }

        private JobHandle PreSmoothJob(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f,
            int res, float h, int count, JobHandle handle = default)
        {
            for (int iter = 0; iter < count; iter++)
            {
                handle = new GaussSeidelPhase(f, a, v, h, res, 0).Schedule(v.Length, BatchSize, handle);
                handle = new GaussSeidelPhase(f, a, v, h, res, 1).Schedule(v.Length, BatchSize, handle);
            }
            // handle = new SymmetricGaussSeidel(f, a, v, h, res, count).Schedule(handle);
            
            return handle;
        }
        private JobHandle PostSmoothJob(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f,
            int res, float h, int count, JobHandle handle = default)
        {
            for (int iter = 0; iter < count; iter++)
            {
                handle = new GaussSeidelPhase(f, a, v, h, res, 1).Schedule(v.Length, BatchSize, handle);
                handle = new GaussSeidelPhase(f, a, v, h, res, 0).Schedule(v.Length, BatchSize, handle);
            }
            // handle = new SymmetricGaussSeidel(f, a, v, h, res, count).Schedule(handle);
            
            return handle;
        }
        private JobHandle SmoothJob(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f,
            int res, float h, int count, JobHandle handle = default)
        {
            for (int iter = 0; iter < count; iter++)
            {
                handle = new GaussSeidelPhase(f, a, v, h, res, 0).Schedule(v.Length, BatchSize, handle);
                handle = new GaussSeidelPhase(f, a, v, h, res, 1).Schedule(v.Length, BatchSize, handle);
                handle = new GaussSeidelPhase(f, a, v, h, res, 1).Schedule(v.Length, BatchSize, handle);
                handle = new GaussSeidelPhase(f, a, v, h, res, 0).Schedule(v.Length, BatchSize, handle);
            }
            // handle = new SymmetricGaussSeidel(f, a, v, h, res, count).Schedule(handle);
            
            return handle;
        }

        [BurstCompile]
        private struct SmoothJacobi : IJobParallelFor
        {
            [ReadOnly] private readonly NativeArray<float> _f;
            [ReadOnly] private readonly NativeArray<float3> _a;
            [ReadOnly] private readonly NativeArray<float> _v;
            [WriteOnly] private NativeArray<float> _result;
            private readonly float _h2;
            private readonly int _res;

            public SmoothJacobi(NativeArray<float> f, NativeArray<float3> a, NativeArray<float> v, 
                NativeArray<float> r, float h, int res)
            {
                _f = f;
                _a = a;
                _v = v;
                _h2 = h * h;
                _res = res;
                _result = r;
            }

            public void Execute(int i)
            {
                float res = 0;
                float3 a = _a[i];
                if (!InActive(a.x))
                {
                    int x = i % _res;
                    int y = i / _res;
                    res = (_h2 * _f[i] - NeighborSum(_a, _v, a, x, y, _res)) / a.x;
                }
                _result[i] = res;
            }
        }

        [BurstCompile]
        private struct GaussSeidel : IJob
        {
            [ReadOnly] private NativeArray<float> _f;
            [ReadOnly] private NativeArray<float3> _a;
            private NativeArray<float> _v;
            private readonly float _h2;
            private readonly int _res;
            private readonly int _count;

            public GaussSeidel(NativeArray<float> f, NativeArray<float3> a, NativeArray<float> v, 
                float h, int res, int count)
            {
                _f = f;
                _a = a;
                _v = v;
                _h2 = h * h;
                _res = res;
                _count = count;
            }

            public void Execute()
            {
                for (int iter = 0; iter < _count; iter++)
                {
                    for (int y = 0; y < _res; y++)
                    for (int x = 0; x < _res; x++)
                    {
                        int i = Coord2Index(x, y, _res);
                        float3 a = _a[i];
                        _v[i] = InActive(a.x) ? 0 : (_h2 * _f[i] - NeighborSum(_a, _v, a, x, y, _res)) / a.x;
                    }
                }
            }
        }

        [BurstCompile]
        private struct SymmetricGaussSeidel : IJob
        {
            [ReadOnly] private NativeArray<float> _f;
            [ReadOnly] private NativeArray<float3> _a;
            private NativeArray<float> _v;
            private readonly float _h2;
            private readonly int _res;
            private readonly int _count;

            public SymmetricGaussSeidel(NativeArray<float> f, NativeArray<float3> a, NativeArray<float> v, 
                float h, int res, int count)
            {
                _f = f;
                _a = a;
                _v = v;
                _h2 = h * h;
                _res = res;
                _count = count;
            }

            public void Execute()
            {
                for (int iter = 0; iter < _count; iter++)
                {
                    for (int y = 0; y < _res; y++)
                    for (int x = 0; x < _res; x++)
                    {
                        int i = Coord2Index(x, y, _res);
                        float3 a = _a[i];
                        _v[i] = InActive(a.x) ? 0 : (_h2 * _f[i] - NeighborSum(_a, _v, a, x, y, _res)) / a.x;
                    }

                    for (int y = _res - 1; y >= 0; y--)
                    for (int x = _res - 1; x >= 0; x--)
                    {
                        int i = Coord2Index(x, y, _res);
                        float3 a = _a[i];
                        _v[i] = InActive(a.x) ? 0 : (_h2 * _f[i] - NeighborSum(_a, _v, a, x, y, _res)) / a.x;
                    }
                }
            }
        }

        [BurstCompile]
        private struct SymmetricSuccessiveOverRelaxation : IJob
        {
            [ReadOnly] private NativeArray<float> _f;
            [ReadOnly] private NativeArray<float3> _a;
            private NativeArray<float> _v;
            private readonly float _h2;
            private readonly int _res;
            private readonly int _count;
            private readonly float _omega;

            public SymmetricSuccessiveOverRelaxation(NativeArray<float> f, NativeArray<float3> a, NativeArray<float> v, 
                float h, int res, int count, float omega)
            {
                _f = f;
                _a = a;
                _v = v;
                _h2 = h * h;
                _res = res;
                _count = count;
                _omega = omega;
            }

            public void Execute()
            {
                for (int iter = 0; iter < _count; iter++)
                {
                    for (int y = 0; y < _res; y++)
                    for (int x = 0; x < _res; x++)
                    {
                        int i = Coord2Index(x, y, _res);
                        float3 a = _a[i];
                        if (InActive(a.x)) continue;
                        _v[i] = math.lerp(_v[i], (_h2 * _f[i] - NeighborSum(_a, _v, a, x, y, _res)) / a.x, _omega);
                    }

                    for (int y = _res - 1; y >= 0; y--)
                    for (int x = _res - 1; x >= 0; x--)
                    {
                        int i = Coord2Index(x, y, _res);
                        float3 a = _a[i];
                        if (InActive(a.x)) continue;
                        _v[i] = math.lerp(_v[i], (_h2 * _f[i] - NeighborSum(_a, _v, a, x, y, _res)) / a.x, _omega);
                    }
                }
            }
        }

        [BurstCompile]
        private struct GaussSeidelPhase : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float> _f;
            [ReadOnly] private NativeArray<float3> _a;
            [NativeDisableParallelForRestriction] private NativeArray<float> _v;
            private readonly float _h2;
            private readonly int _res;
            private readonly int _phase;
            
            public GaussSeidelPhase(NativeArray<float> f, NativeArray<float3> a, NativeArray<float> v,
                float h, int res, int phase)
            {
                _f = f;
                _a = a;
                _v = v;
                _h2 = h * h;
                _res = res;
                _phase = phase;
            }
            
            public void Execute(int i)
            {
                int y = i / _res;
                int x = i % _res;
                
                if (((x + y) & 1) == _phase)
                {
                    float3 a = _a[i];
                    if (InActive(a.x)) _v[i] = 0;
                    else
                    {
                        _v[i] = math.lerp(_v[i], 
                            (_h2 * _f[i] - NeighborSum(_a, _v, a, x, y, _res)) / a.x, 1.2f);
                    }
                }
            }
        }

        [BurstCompile]
        private struct SSOR_Phase : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float> _f;
            [ReadOnly] private NativeArray<float3> _a;
            [NativeDisableParallelForRestriction] private NativeArray<float> _v;
            private readonly float _h2;
            private readonly int _res;
            private readonly int _phase;
            private readonly float _omega;
            
            public SSOR_Phase(NativeArray<float> f, NativeArray<float3> a, NativeArray<float> v,
                float h, int res, int phase, float omega)
            {
                _f = f;
                _a = a;
                _v = v;
                _h2 = h * h;
                _res = res;
                _phase = phase;
                _omega = omega;
            }
            
            public void Execute(int i)
            {
                int y = i / _res;
                int x = i % _res;
                
                if (((x + y) & 1) == _phase)
                {
                    float3 a = _a[i];
                    float newValue = InActive(a.x) ? 0 : (_h2 * _f[i] - NeighborSum(_a, _v, a, x, y, _res)) / a.x;
                    _v[i] = InActive(a.x) ? 0 : (1 - _omega) * _v[i] + _omega * newValue;
                }
            }
        }
        
        [BurstCompile]
        private struct Residual : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float> _b;
            [ReadOnly] private NativeArray<float> _v;
            [ReadOnly] private NativeArray<float3> _a;
            [WriteOnly] private NativeArray<float> _r;
            
            private readonly float _ih2;
            private readonly int _res;

            public Residual(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f, NativeArray<float> r, 
                int res, float h)
            {
                _ih2 = 1f / (h * h);
                _res = res;
                _b = f;
                _v = v;
                _a = a;
                _r = r;
            }
            
            public void Execute(int i)
            {
                float3 a = _a[i];
                if (InActive(a.x)) _r[i] = 0;
                else
                {
                    int x = i % _res;
                    int y = i / _res;
                    float sum = NeighborSum(_a, _v, a, x, y, _res, out float ac);
                    _r[i] = _b[i] - _ih2 * (ac + sum);
                }
            }
        }
        
        [BurstCompile]
        private struct Restriction : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float3> _aFine;
            [ReadOnly] private NativeArray<float> _fFine;
            [ReadOnly] private NativeArray<float> _vFine;
            [WriteOnly] private NativeArray<float3> _aCoarse;
            [WriteOnly] private NativeArray<float> _rCoarse;
            [WriteOnly] private NativeArray<float> _eCoarse;

            private readonly int _res;
            private readonly float _ih2;
            
            public Restriction(NativeArray<float> ff, NativeArray<float3> af, NativeArray<float> vf,
                NativeArray<float> rc, NativeArray<float3> ac, NativeArray<float> ec, int res, float h)
            {
                _aFine = af;
                _fFine = ff;
                _aCoarse = ac;
                _vFine = vf;
                _rCoarse = rc;
                _eCoarse = ec;
                _res = res;
                _ih2 = 1f / (h * h);
            }
            
            public void Execute(int ci)
            {
                int gridResC = _res >> 1;
                int gridResF = _res;
                int x = ci % gridResC;
                int y = ci / gridResC;
                
                float rCoarse = 0;
                float3 aCoarse = float3.zero;
                for (int yy = 0; yy < 2; yy++)
                for (int xx = 0; xx < 2; xx++)
                {
                    int fx = x * 2 + xx;
                    int fy = y * 2 + yy;
                    int fi = Coord2Index(fx, fy, gridResF);
                    float3 aFine = _aFine[fi];
                    if (InActive(aFine.x)) 
                        continue;
                    aCoarse.x += aFine.x;
                    float sum = NeighborSum(_aFine, _vFine, aFine, fx, fy, gridResF, out float ac);
                    float rFine = _fFine[fi] - _ih2 * (ac + sum);
                    rCoarse += rFine;
                    
                    if (xx == 0) aCoarse.y += aFine.y;
                    else aCoarse.x += aFine.y * 2;
                    if (yy == 0) aCoarse.z += aFine.z;
                    else aCoarse.x += aFine.z * 2;
                }

                _rCoarse[ci] = rCoarse * 0.25f;
                _aCoarse[ci] = aCoarse * 0.25f;
                _eCoarse[ci] = 0;
            }
        }

        [BurstCompile]
        private struct Prolongation : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float3> _a;
            [ReadOnly] private NativeArray<float> _eCoarse;
            private NativeArray<float> _eFine;
            private readonly int _res;
            
            public Prolongation(NativeArray<float> ec, NativeArray<float3> af, NativeArray<float> ef, int res)
            {
                _a = af;
                _eCoarse = ec;
                _eFine = ef;
                _res = res;
            }
            
            public void Execute(int fi)
            {
                float3 a = _a[fi];
                if (InActive(a.x)) return;
                int fx = fi % _res;
                int fy = fi / _res;
                _eFine[fi] += _eCoarse[Coord2Index(fx >> 1, fy >> 1, _res >> 1)] * 2;
            }
        }

        [BurstCompile]
        private struct ClearJob : IJob
        {
            [WriteOnly] private NativeArray<float> _array;
            
            public ClearJob(NativeArray<float> array)
            {
                _array = array;
            }
            
            public void Execute()
            {
                for (int i = 0; i < _array.Length; i++)
                    _array[i] = 0;
            }
        }

        private static float NeighborSum(NativeArray<float3> a, NativeArray<float> v, float3 ac,
            int x, int y, int gridRes, out float Ac)
        {
            float sum = 0;
            float3 ar = x < gridRes - 1 ? a[Coord2Index(x + 1, y, gridRes)] : float3.zero;
            float3 au = y < gridRes - 1 ? a[Coord2Index(x, y + 1, gridRes)] : float3.zero;
            sum += InActive(ac.y) ? 0 : (ac.y * v[Coord2Index(x - 1, y, gridRes)]);
            sum += InActive(ac.z) ? 0 : (ac.z * v[Coord2Index(x, y - 1, gridRes)]);
            sum += InActive(ar.y) ? 0 : (ar.y * v[Coord2Index(x + 1, y, gridRes)]);
            sum += InActive(au.z) ? 0 : (au.z * v[Coord2Index(x, y + 1, gridRes)]);

            Ac = ac.x * v[Coord2Index(x, y, gridRes)];
            
            return sum;
        }

        private static float NeighborSum(NativeArray<float3> a, NativeArray<float> v, float3 ac, int x, int y, int gridRes)
        {
            float sum = 0;
            float3 ar = x < gridRes - 1 ? a[Coord2Index(x + 1, y, gridRes)] : float3.zero;
            float3 au = y < gridRes - 1 ? a[Coord2Index(x, y + 1, gridRes)] : float3.zero;
            sum += InActive(ac.y) ? 0 : ac.y * v[Coord2Index(x - 1, y, gridRes)];
            sum += InActive(ac.z) ? 0 : ac.z * v[Coord2Index(x, y - 1, gridRes)];
            sum += InActive(ar.y) ? 0 : ar.y * v[Coord2Index(x + 1, y, gridRes)];
            sum += InActive(au.z) ? 0 : au.z * v[Coord2Index(x, y + 1, gridRes)];
            
            return sum;
        }

        private static int Coord2Index(int x, int y, int gridRes) => y * gridRes + x;

        private static bool InActive(float x) => math.abs(x) < 1e-6f;
        
        public void Dispose()
        {
            P.Dispose();
            Ap.Dispose();

            Rs[0].Dispose();
            Zs[0].Dispose();
            for (int i = 1; i < L; i++)
            {
                As[i].Dispose();
                Zs[i].Dispose();
                Rs[i].Dispose();
            }

            rs_old.Dispose();
            rs_new.Dispose();
            pAp.Dispose();
        }
    }
}

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace PF_FLIP
{
    public class Jobs_UAAMGSolver : System.IDisposable
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

        public Jobs_UAAMGSolver(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, int gridRes, float h)
        {
            L = (int)(math.log2(gridRes) - 1);
            Debug.Log("UAAMG Solver levels: " + L);
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
        }

        #region Grid Type Utils

        private static bool IsFluidCell(float3 a)
        {
            return math.abs(a.x) > 1e-5f;
        }
        #endregion

        public void Solve_Jacobi(int maxIter, out float rs)
        {
            new ClearJob(V).Schedule().Complete();
            new ClearJob(Z).Schedule().Complete();
            JobHandle handle = default; 
            for (int iter = 0; iter < maxIter; iter++)
            {
                handle = new SmoothJacobi(F, A, V, Z,  H, GridRes).Schedule(V.Length, BatchSize, handle);
                handle = new SmoothJacobi(F, A, Z, V,  H, GridRes).Schedule(V.Length, BatchSize, handle);
            }
            handle.Complete();

            var rs_ref = new NativeReference<float>(0, Allocator.TempJob);
            new Residual(A, V, F, R,  GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_ref).Schedule().Complete();
            rs = math.sqrt(rs_ref.Value);
            rs_ref.Dispose();
        }
        
        public void Solve_MG(int maxIter, out float rs)
        {
            var rs_ref = new NativeReference<float>(0, Allocator.TempJob);
            R.CopyFrom(F);
            new ClearJob(Z).Schedule().Complete();
            for (int iter = 0; iter < maxIter; iter++)
            {
                MultiGridVCycle().Complete();
                // MultiGridVCycle(Z, F, A, 0, GridRes, H);
            }

            V.CopyFrom(Z);
            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_ref).Schedule().Complete();
            rs = math.sqrt(rs_ref.Value);
            rs_ref.Dispose();
        }
        
        public void Solve_GS(int maxIter, out float rs)
        {
            new ClearJob(V).Schedule().Complete();
            SmoothJob(A, V, F, GridRes, H, maxIter).Complete();
            var rs_ref = new NativeReference<float>(0, Allocator.TempJob);
            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_ref).Schedule().Complete();
            rs = math.sqrt(rs_ref.Value);
            rs_ref.Dispose();
        }
        
        public void Solve_CG(int maxIter, out int iter, out float rs)
        {
            var rs_old = new NativeReference<float>(0, Allocator.TempJob);
            var pAp = new NativeReference<float>(0, Allocator.TempJob);
            var rs_new = new NativeReference<float>(0, Allocator.TempJob);
            
            new ClearJob(V).Schedule().Complete();
            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            P.CopyFrom(R);
            
            new Dot(R, R, rs_old).Schedule().Complete();

            if (rs_old.Value > H * H * 1e-3f)
            {
                for (iter = 0; iter < maxIter; iter++)
                {
                    var handle = new Laplace(A, P, Ap, GridRes, H).Schedule(R.Length, BatchSize);

                    handle = new Dot(P, Ap, pAp).Schedule(handle);

                    handle = new UpdateVR(P, Ap, V, R, rs_old, pAp).Schedule(V.Length, BatchSize, handle);

                    new Dot(R, R, rs_new).Schedule(handle).Complete();

                    if (rs_new.Value < H * H * 0.01f)
                        break;

                    new UpdateP(R, P, rs_old, rs_new).Schedule(P.Length, BatchSize).Complete();
                    (rs_old, rs_new) = (rs_new, rs_old);
                }
            }
            else
                iter = 0;

            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_old).Schedule().Complete();
            rs = math.sqrt(rs_old.Value);

            rs_old.Dispose();
            rs_new.Dispose();
            pAp.Dispose();
        }

        public void Solve_MGPCG(int maxIter, out float rs)
        {
            var rz_old = new NativeReference<float>(0, Allocator.TempJob);
            var pAp = new NativeReference<float>(0, Allocator.TempJob);
            var rz_new = new NativeReference<float>(0, Allocator.TempJob);
            
            new ClearJob(V).Schedule().Complete();
            Z.CopyFrom(V);
            R.CopyFrom(F);
            MultiGridVCycle().Complete();
            P.CopyFrom(Z);

            new Dot(R, Z, rz_old).Schedule().Complete();
            if (math.abs(rz_old.Value) > 1e-5f)
            {
                for (int iter = 0; iter < maxIter; iter++)
                {
                    var handle = new Laplace(A, P, Ap, GridRes, H).Schedule(R.Length, BatchSize);
                    handle = new Dot(P, Ap, pAp).Schedule(handle);
                    new UpdateVR(P, Ap, V, R, rz_old, pAp).Schedule(V.Length, BatchSize, handle).Complete();

                    if (iter == maxIter - 1) break;
                    
                    handle = new ClearJob(Z).Schedule();
                    handle = MultiGridVCycle(handle);
                    handle = new Dot(R, Z, rz_new).Schedule(handle);
                    new UpdateP(Z, P, rz_old, rz_new).Schedule(P.Length, BatchSize, handle).Complete();

                    (rz_old, rz_new) = (rz_new, rz_old);
                }
            }

            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rz_old).Schedule().Complete();

            rs = math.sqrt(rz_old.Value);

            rz_old.Dispose();
            rz_new.Dispose();
            pAp.Dispose();
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
                int x = i % _res;
                int y = i / _res;
                float3 ac = _a[Coord2Index(x, y, _res)];
                if (IsFluidCell(ac))
                    _result[i] = _ih2 * (_a[i].x * _v[i] + NeighborSum(_a, _v, ac, x, y, _res));
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
            [ReadOnly] private NativeReference<float> _rzOld;
            [ReadOnly] private NativeReference<float> _rzNew;
            
            public UpdateP(NativeArray<float> z, NativeArray<float> p, NativeReference<float> rsOld, NativeReference<float> rsNew)
            {
                _z = z;
                _p = p;
                _rzOld = rsOld;
                _rzNew = rsNew;
            }
            
            public void Execute(int i)
            {
                float beta = _rzNew.Value / _rzOld.Value;
                _p[i] = _z[i] + beta * _p[i];
            }
        }

        private JobHandle MultiGridVCycle(JobHandle handle = default)
        {
            float h = H;
            int top = L - 1;
            for (int i = 0; i < top; i++)
            {
                int res = GridRes >> i;
                handle = PreSmoothJob(As[i], Zs[i], Rs[i], res, h, 3, handle);
                handle = new Restriction(Rs[i], As[i], Zs[i], Rs[i + 1], As[i + 1], Zs[i+1], res, h)
                    .Schedule(As[i + 1].Length, BatchSize, handle);
            }

            handle = new SymmetricGaussSeidel(Rs[top], As[top], Zs[top], h, GridRes >> top, 4).Schedule(handle);

            for (int i = top - 1; i >= 0; i--)
            {
                int res = GridRes >> i;
                handle = new Prolongation(Zs[i+1], Zs[i], As[i], res).Schedule(Zs[i].Length, BatchSize, handle);
                handle = PostSmoothJob(As[i], Zs[i], Rs[i], res, h, 3, handle);
            }

            return handle;
        }
        
        private void MultiGridVCycle(NativeArray<float> vf, NativeArray<float> rf, NativeArray<float3> af, 
            int level, int res, float h)
        {
            if (res <= 4)
            {
                SmoothJob(af, vf, rf, res, h, 2).Complete();
                return;
            }

            SmoothJob(af, vf, rf, res, h, 2).Complete();
            
            int resC = res / 2;
            var rc = Rs[level + 1];
            var ac = As[level + 1];
            var vc = Zs[level + 1];
            new Restriction(rf, af, vf, rc, ac, vc, res, h)
                .Schedule(vc.Length, BatchSize).Complete();
            // new UpdateNeighborType(tcTemp, tc, res).Schedule(tc.Length, BatchSize).Complete();
            // for (int i = 0; i < 2; i++) // W-cycle
            MultiGridVCycle(vc, rc, ac, level + 1, resC, h);

            new Prolongation(vc, vf, af, res).Schedule(vf.Length, BatchSize).Complete();

            SmoothJob(af, vf, rf, res, h, 2).Complete();
        }

        private JobHandle PreSmoothJob(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f,
            int res, float h, int count, JobHandle handle = default)
        {
            for (int iter = 0; iter < count; iter++)
            {
                handle = new GaussSeidelPhase(f, a, v, h, res, 0).Schedule(v.Length, BatchSize, handle);
                handle = new GaussSeidelPhase(f, a, v, h, res, 1).Schedule(v.Length, BatchSize, handle);
            }
            return handle;

            // return new SymmetricGaussSeidel(f, a, v, h, res, count).Schedule(handle);
        }
        private JobHandle PostSmoothJob(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f,
            int res, float h, int count, JobHandle handle = default)
        {
            for (int iter = 0; iter < count; iter++)
            {
                handle = new GaussSeidelPhase(f, a, v, h, res, 1).Schedule(v.Length, BatchSize, handle);
                handle = new GaussSeidelPhase(f, a, v, h, res, 0).Schedule(v.Length, BatchSize, handle);
            }
            return handle;

            // return new SymmetricGaussSeidel(f, a, v, h, res, count).Schedule(handle);
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
            return handle;

            // return new SymmetricGaussSeidel(f, a, v, h, res, count).Schedule(handle);
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
                        float3 ac = _a[i];
                        if (!IsFluidCell(ac)) continue;
                        _v[i] = (_h2 * _f[i] - NeighborSum(_a, _v, ac, x, y, _res)) / _a[i].x;
                    }

                    for (int y = _res - 1; y >= 0; y--)
                    for (int x = _res - 1; x >= 0; x--)
                    {
                        int i = Coord2Index(x, y, _res);
                        float3 ac = _a[i];
                        if (!IsFluidCell(ac)) continue;
                        _v[i] = (_h2 * _f[i] - NeighborSum(_a, _v, ac, x, y, _res)) / _a[i].x;
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
                    float v = 0;
                    float3 ac = _a[i];
                    if (!InActive(ac.x))
                        v = (_h2 * _f[i] - NeighborSum(_a, _v, ac, x, y, _res)) / _a[i].x;
                    _v[i] = v;
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
                    float sum = NeighborSum(_a, _v, a, x, y, _res);
                    _r[i] = _b[i] - _ih2 * (a.x * _v[i] + sum);
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
                int gridResF = _res;
                int gridResC = gridResF >> 1;
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
                    if (!IsFluidCell(aFine))
                        continue;
                    aCoarse.x += aFine.x;
                    float sum = NeighborSum(_aFine, _vFine, aFine, fx, fy, gridResF);
                    float rFine = _fFine[fi] - _ih2 * (aFine.x * _vFine[fi] + sum);
                    rCoarse += rFine;
                    
                    if (xx == 0) aCoarse.y += aFine.y;
                    else aCoarse.x += aFine.y * 2;
                    
                    if (yy == 0) aCoarse.z += aFine.z;
                    else aCoarse.x += aFine.z * 2;
                }

                _rCoarse[ci] = rCoarse * (0.25f);
                _aCoarse[ci] = aCoarse * (0.25f);
                _eCoarse[ci] = 0;
            }
        }
        
        // [BurstCompile]
        // private struct UpdateNeighborType : IJobParallelFor
        // {
        //     [ReadOnly] private NativeArray<uint> _typesR;
        //     [WriteOnly] private NativeArray<uint> _typesW;
        //
        //     private readonly int _res;
        //     
        //     public UpdateNeighborType(NativeArray<uint> tr,NativeArray<uint> tw, int res)
        //     {
        //         _typesR = tr;
        //         _typesW = tw;
        //         _res = res;
        //     }
        //     
        //     public void Execute(int ci)
        //     {
        //         int gridResC = _res >> 1;
        //         int x = ci % gridResC;
        //         int y = ci / gridResC;
        //         
        //         uint left = (x == 0) ? SOLID : _typesR[Coord2Index(x - 1, y, gridResC)];
        //         uint right = (x == gridResC - 1) ? SOLID : _typesR[Coord2Index(x + 1, y, gridResC)];
        //         uint down = (y == 0) ? SOLID : _typesR[Coord2Index(x, y - 1, gridResC)];
        //         uint up = (y == gridResC - 1) ? SOLID : _typesR[Coord2Index(x, y + 1, gridResC)];
        //
        //         uint center = _typesR[ci];
        //         
        //         _typesW[ci] = center | (left << 2) | (right << 4) | (down << 6) | (up << 8);
        //     }
        // }
        
        [BurstCompile]
        private struct Prolongation : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float3> _aFine;
            [ReadOnly] private NativeArray<float> _eCoarse;
            private NativeArray<float> _eFine;

            private readonly int _res;
            
            public Prolongation(NativeArray<float> ec, NativeArray<float> ef, NativeArray<float3> af, int res)
            {
                _aFine = af;
                _eCoarse = ec;
                _eFine = ef;
                _res = res;
            }
            
            public void Execute(int fi)
            {
                float3 ac = _aFine[fi];
                if (!IsFluidCell(ac)) return;
                int x = fi % _res;
                int y = fi / _res;
                int cx = x >> 1;
                int cy = y >> 1;
                int gridResC = _res >> 1;
                _eFine[fi] += _eCoarse[Coord2Index(cx, cy, gridResC)] * 2;
            }
        }
        
        private static float NeighborSum(NativeArray<float3> a, NativeArray<float> v, float3 ac, int x, int y, int gridRes)
        {
            float sum = 0;
            float3 ar = x < gridRes - 1 ? a[Coord2Index(x + 1, y, gridRes)] : float3.zero;
            float3 at = y < gridRes - 1 ? a[Coord2Index(x, y + 1, gridRes)] : float3.zero;
            sum += InActive(ac.y) ? 0 : (ac.y * v[Coord2Index(x - 1, y, gridRes)]);
            sum += InActive(ac.z) ? 0 : (ac.z * v[Coord2Index(x, y - 1, gridRes)]);
            sum += InActive(ar.y) ? 0 : (ar.y * v[Coord2Index(x + 1, y, gridRes)]);
            sum += InActive(at.z) ? 0 : (at.z * v[Coord2Index(x, y + 1, gridRes)]);
            
            return sum;
        }

        private static int Coord2Index(int x, int y, int gridRes) => y * gridRes + x;
        
        private static bool InActive(float x) => math.abs(x) < 1e-6f;
        public void Dispose()
        {
            P.Dispose();
            Ap.Dispose();

            Zs[0].Dispose();
            Rs[0].Dispose();
            for (int i = 1; i < L; i++)
            {
                As[i].Dispose();
                Zs[i].Dispose();
                Rs[i].Dispose();
            }
        }
    }

    public struct Jobs_UAAMG1DSolver
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
            Residual(A, Lhs, Rhs, r, GridRes, H);
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

                if (math.sqrt(Dot(r, r)) < H * H * 0.1f)
                    break;

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
                result[i] = (a[i].y * v[i] + NeighborSum(a, v, i)) * ih2;
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
                    v[i] = (h2 * f[i] - NeighborSum(a, v, i)) / a[i].y;
                
                for (int i = res - 1; i >= 0; i--)
                    v[i] = (h2 * f[i] - NeighborSum(a, v, i)) / a[i].y;
            }
        }
        
        private void Residual(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, NativeArray<float> r, int res, float h)
        {
            float ih2 = 1f / (h * h);
            for (int i = 0; i < res; i++)
                r[i] = b[i] - ih2 * (a[i].y * v[i] + NeighborSum(a, v, i));
        }

        private float NeighborSum(NativeArray<float3> a, NativeArray<float> v, int i)
        {
            return ((a[i].x != 0) ? (a[i].x * v[i - 1]) : 0) + ((a[i].z != 0) ? (a[i].z * v[i + 1]) : 0);
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
                if (af[i * 2].y > 0) ef[i * 2] += cur;
                if (af[i * 2 + 1].y > 0) ef[i * 2 + 1] += cur;
            }
        }
    }
}

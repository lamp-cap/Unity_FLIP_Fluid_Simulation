using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace NarrowBand
{
    // Dedicated copy of PF_FLIP's Neumann_UAAMGSolver for the two-phase sim.
    // Only the MG-preconditioned CG solve is kept (warm-started, relative
    // residual early exit) — the Jacobi/GS/SOR/plain-CG/F-cycle variants and
    // their helpers were deleted, so this solver can be retuned for the
    // 1:1000 phase mobility contrast without touching the single-phase one.
    public class TwoPhaseMGSolver : System.IDisposable
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
        private bool _warmStarted;
        // final relative residual of the previous solve — a badly unconverged
        // frame leaves a pressure field that is a worse initial guess than
        // zero, so the warm start is dropped until the solve recovers
        private float _lastRelResidual;
        // diagnostics: how many solves fell back to a cold start
        public int ColdStarts;
        // last solve's relative residual (rs/bNorm, the convergence quality)
        // and the problem scale |b|, for per-frame logging
        public float LastRelResidual => _lastRelResidual;
        public float LastBNorm { get; private set; }

        public TwoPhaseMGSolver(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, int gridRes, float h)
        {
            L = (int)(math.log2(gridRes) - 1);
            Debug.Log("TwoPhaseMGSolver levels: " + L);
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

        public void Solve_MGPCG(int maxIter, out float rs)
        {
            // warm start: keep the previous frame's pressure as the initial
            // guess — pressure is temporally coherent, so the first V-cycle
            // starts from a far better point than zero. Cleared on the first
            // call (fresh NativeArray holds garbage) and after any frame that
            // failed to converge (its leftover V would compound the error)
            if (!_warmStarted || _lastRelResidual > 1e-2f)
            {
                new ClearJob(V).Schedule().Complete();
                _warmStarted = true;
                ColdStarts++;
            }
            var mgHandle = new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize);
            // the fine matrix is fixed for the whole solve, so the coarse
            // hierarchy is built once per frame here instead of inside every
            // V-cycle (the residual half of Restriction stays per-cycle)
            for (int i = 0; i < L - 1; i++)
                mgHandle = new RestrictMatrix(As[i], As[i + 1], GridRes >> i)
                    .Schedule(As[i + 1].Length, BatchSize, mgHandle);
            new Dot(R, R, rs_new).Schedule(mgHandle).Complete();
            float bNorm = math.sqrt(rs_new.Value); // for the relative early-exit test
            new ClearJob(Z).Schedule().Complete();
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

                    // early exit on relative residual: iterations are only spent
                    // when the phase mobility contrast actually demands them
                    new Dot(R, R, rs_new).Schedule().Complete();
                    if (math.sqrt(rs_new.Value) < 1e-4f * bNorm) break;

                    if (iter == maxIter - 1) break;

                    handle = new ClearJob(Z).Schedule();
                    handle = MultiGridVCycle(handle);
                    handle = new Dot(R, Z, rs_new).Schedule(handle);
                    new UpdateP(Z, P, rs_new, rs_old).Schedule(P.Length, BatchSize, handle).Complete();

                    rs_old.Value = rs_new.Value;
                }
            }

            new Residual(A, V, F, R, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_old).Schedule().Complete();
            rs = math.sqrt(rs_old.Value);
            // bNorm is the initial residual norm: if it was already negligible
            // the warm start nailed this frame for free, otherwise the ratio
            // measures how well the solve actually converged
            _lastRelResidual = bNorm > 1e-3f ? rs / bNorm : 0f;
            LastBNorm = bNorm;
        }

        private JobHandle MultiGridVCycle(JobHandle handle = default)
        {
            float h = H;
            int top = L - 1;
            for (int i = 0; i < top; i++)
            {
                int res = GridRes >> i;
                // 2 pre-smoothing sweeps (was 3): the 600-frame harness
                // showed typical frames exit PCG at the tolerance anyway,
                // so the V-cycle is mostly paying for smoothing time
                handle = PreSmoothJob(As[i], Zs[i], Rs[i], res, h, 2, handle);
                handle = new Restriction(Rs[i], As[i], Zs[i], Rs[i + 1], Zs[i + 1], res, h)
                    .Schedule(Rs[i + 1].Length, BatchSize, handle);
            }

            handle = new SymmetricGaussSeidel(Rs[top], As[top], Zs[top], h, GridRes >> top, 4).Schedule(handle);

            for (int i = top - 1; i >= 0; i--)
            {
                int res = GridRes >> i;
                handle = new Prolongation(Zs[i+1], As[i], Zs[i], res).Schedule(Zs[i].Length, BatchSize, handle);
                handle = PostSmoothJob(As[i], Zs[i], Rs[i],  res, h, 2, handle);
            }

            return handle;
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
                            (_h2 * _f[i] - NeighborSum(_a, _v, a, x, y, _res)) / a.x, 1.3f);
                    }
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

        // matrix half of the old Restriction: As[0] is fixed for the whole
        // solve, so the coarse hierarchy only needs to be built once per
        // frame, not once per V-cycle
        [BurstCompile]
        private struct RestrictMatrix : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float3> _aFine;
            [WriteOnly] private NativeArray<float3> _aCoarse;
            private readonly int _res;

            public RestrictMatrix(NativeArray<float3> af, NativeArray<float3> ac, int res)
            {
                _aFine = af;
                _aCoarse = ac;
                _res = res;
            }

            public void Execute(int ci)
            {
                int gridResC = _res >> 1;
                int x = ci % gridResC;
                int y = ci / gridResC;

                float3 aCoarse = float3.zero;
                for (int yy = 0; yy < 2; yy++)
                for (int xx = 0; xx < 2; xx++)
                {
                    float3 aFine = _aFine[Coord2Index(x * 2 + xx, y * 2 + yy, _res)];
                    if (InActive(aFine.x))
                        continue;
                    aCoarse.x += aFine.x;
                    if (xx == 0) aCoarse.y += aFine.y;
                    else aCoarse.x += aFine.y * 2;
                    if (yy == 0) aCoarse.z += aFine.z;
                    else aCoarse.x += aFine.z * 2;
                }

                _aCoarse[ci] = aCoarse * 0.25f;
            }
        }

        // residual half of the old Restriction: the coarse RHS depends on the
        // current correction, so it must run inside every V-cycle
        [BurstCompile]
        private struct Restriction : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float3> _aFine;
            [ReadOnly] private NativeArray<float> _fFine;
            [ReadOnly] private NativeArray<float> _vFine;
            [WriteOnly] private NativeArray<float> _rCoarse;
            [WriteOnly] private NativeArray<float> _eCoarse;

            private readonly int _res;
            private readonly float _ih2;

            public Restriction(NativeArray<float> ff, NativeArray<float3> af, NativeArray<float> vf,
                NativeArray<float> rc, NativeArray<float> ec, int res, float h)
            {
                _aFine = af;
                _fFine = ff;
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
                for (int yy = 0; yy < 2; yy++)
                for (int xx = 0; xx < 2; xx++)
                {
                    int fx = x * 2 + xx;
                    int fy = y * 2 + yy;
                    int fi = Coord2Index(fx, fy, gridResF);
                    float3 aFine = _aFine[fi];
                    if (InActive(aFine.x))
                        continue;
                    float sum = NeighborSum(_aFine, _vFine, aFine, fx, fy, gridResF, out float ac);
                    float rFine = _fFine[fi] - _ih2 * (ac + sum);
                    rCoarse += rFine;
                }

                _rCoarse[ci] = rCoarse * 0.25f;
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

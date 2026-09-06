using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace PF_FLIP
{
    public class Jobs_UAAMGPCGSolver : System.IDisposable
    {
        public NativeArray<float>[] Vs;
        public NativeArray<float3>[] As; // x: center, y: left, z: down
        public NativeArray<uint>[] Ts;
        private NativeArray<uint>[] TsTemp;
        private NativeArray<float>[] Rs;
        public NativeArray<float> F;
        public NativeArray<float2> B;
        
        private NativeArray<float> V => Vs[0];
        private NativeArray<float3> A => As[0]; // x: center, y: left, z: down
        private NativeArray<uint> T => Ts[0];
        
        private NativeArray<float> R => Rs[0];
        private NativeArray<float> Z;
        private NativeArray<float> P;
        private NativeArray<float> Ap;
        private readonly int GridRes;
        private readonly float H;
        private readonly int L;
        private const int BatchSize = 64;

        public Jobs_UAAMGPCGSolver(NativeArray<float2> beta, NativeArray<float> v, NativeArray<float> f, 
            NativeArray<uint> type, int gridRes, float h)
        {
            L = (int)(math.log2(gridRes) - 1);
            Debug.Log("UAAMG Solver levels: " + L);
            As = new NativeArray<float3>[L];
            Vs = new NativeArray<float>[L];
            Rs = new NativeArray<float>[L];
            Ts = new NativeArray<uint>[L];
            TsTemp = new NativeArray<uint>[L];
            Vs[0] = v;
            F = f;
            B = beta;
            Ts[0] = type;
            Rs[0] = new NativeArray<float>(gridRes * gridRes, Allocator.Persistent);
            int res = gridRes;
            for (int i = 1; i < L; i++)
            {
                res >>= 1;
                Vs[i] = new NativeArray<float>(res * res, Allocator.Persistent);
                Rs[i] = new NativeArray<float>(res * res, Allocator.Persistent);
                Ts[i] = new NativeArray<uint>(res * res, Allocator.Persistent);
                TsTemp[i] = new NativeArray<uint>(res * res, Allocator.Persistent);
            }
            
            GridRes = gridRes;
            H = h;
            Z = new NativeArray<float>(gridRes * gridRes, Allocator.Persistent);
            P = new NativeArray<float>(gridRes * gridRes, Allocator.Persistent);
            Ap = new NativeArray<float>(gridRes * gridRes, Allocator.Persistent);
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
            var rs_ref = new NativeReference<float>(0, Allocator.TempJob);
            JobHandle handle = default;
            // for (int i = 0; i < maxIter; i += 32)
            // {
                for (int iter = 0; iter < maxIter; iter++)
                {
                    handle = new SmoothJacobi(F, V, Z, T, B, H, GridRes).Schedule(V.Length, BatchSize, handle);
                    handle = new SmoothJacobi(F, Z, V, T, B, H, GridRes).Schedule(V.Length, BatchSize, handle);
                }
                handle.Complete();

            //     new Residual(B, V, F, R, T, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            //     new Dot(R, R, rs_ref).Schedule().Complete();
            //     rs = math.sqrt(rs_ref.Value);
            //     Debug.Log("Jacobi iter " + (i + 32) + " res: " + rs);
            // }
            new Residual(B, V, F, R, T, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_ref).Schedule().Complete();
            rs = math.sqrt(rs_ref.Value);
            rs_ref.Dispose();
        }
        
        public void Solve_MG(int maxIter, out float rs)
        {
            var rs_ref = new NativeReference<float>(0, Allocator.TempJob);
            Rs[0].CopyFrom(F);
            for (int iter = 0; iter < maxIter; iter++)
            {
                // MultiGridVCycle().Complete();
                MultiGridVCycle(V, F, A, T, 0, GridRes, H);
            }

            new Residual(B, V, F, R, T, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_ref).Schedule().Complete();
            rs = math.sqrt(rs_ref.Value);
            rs_ref.Dispose();
        }
        
        [BurstCompile]
        private struct CalcAverage : IJob
        {
            [ReadOnly] public NativeArray<float> Lhs;
            [WriteOnly] public NativeReference<float> Rhs;

            public void Execute()
            {
                float sum = 0;
                for (int i = 0; i < Lhs.Length; i++)
                    sum += Lhs[i];
                
                Rhs.Value = sum / Lhs.Length;
            }
        }
        
        [BurstCompile]
        private struct SubAverage : IJobParallelFor
        {
            public NativeArray<float> Lhs;
            public float Rhs;

            public void Execute(int i)
            {
                Lhs[i] -= Rhs;
            }
        }
        
        public void Solve_GS(int maxIter, out float rs)
        {
            SmoothJob(A, V, F, T, GridRes, H, maxIter).Complete();
            var rs_ref = new NativeReference<float>(0, Allocator.TempJob);
            new Residual(B, V, F, R, T, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_ref).Schedule().Complete();
            rs = math.sqrt(rs_ref.Value);
            rs_ref.Dispose();
        }
        
        public void Solve_GS(out int iter, out float rs)
        {
            rs = 10000;
            var rs_ref = new NativeReference<float>(0, Allocator.TempJob);
            for (iter = 0; iter < 32; iter++)
            {
                SmoothJob(A, V, F, T, GridRes, H, 32).Complete();

                new Residual(B, V, F, R, T, GridRes, H).Schedule(R.Length, BatchSize).Complete();
                new Dot(R, R, rs_ref).Schedule().Complete();
                rs = math.sqrt(rs_ref.Value);
                Debug.Log("GS iter " + (iter + 1) * 32 + " res: " + rs);
                if (rs < H * H)
                    break;
            }

            rs_ref.Dispose();
            iter *= 32;
        }
        
        public void Solve_CG(int maxIter, out int iter, out float rs)
        {
            var rs_old = new NativeReference<float>(0, Allocator.TempJob);
            var pAp = new NativeReference<float>(0, Allocator.TempJob);
            var rs_new = new NativeReference<float>(0, Allocator.TempJob);
            
            new Residual(B, V, F, R, T, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            P.CopyFrom(R);
            
            new Dot(R, R, rs_old).Schedule().Complete();
            if (rs_old.Value > 1e-5f)
            {

                for (iter = 0; iter < maxIter; iter++)
                {
                    var handle = new Laplace(B, P, T, Ap, GridRes, H).Schedule(R.Length, BatchSize);

                    handle = new Dot(P, Ap, pAp).Schedule(handle);

                    handle = new UpdateVR(P, Ap, V, R, rs_old, pAp).Schedule(V.Length, BatchSize, handle);

                    new Dot(R, R, rs_new).Schedule(handle).Complete();

                    if (math.sqrt(rs_new.Value) < H * H)
                        break;

                    float beta = rs_new.Value / rs_old.Value;
                    new UpdateP(R, P, beta).Schedule(P.Length, BatchSize).Complete();
                    (rs_old, rs_new) = (rs_new, rs_old);
                }
            }
            else iter = 0;
            
            new Residual(B, V, F, R, T, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            new Dot(R, R, rs_old).Schedule().Complete();
            rs = math.sqrt(rs_old.Value);

            rs_old.Dispose();
            rs_new.Dispose();
            pAp.Dispose();
        }

        public void Solve_MGPCG(int maxIter, out float rs)
        {
            new Residual(B, V, F, R, T, GridRes, H).Schedule(R.Length, BatchSize).Complete();
            MultiGridVCycle(V, R, A, T, 0, GridRes, H);
            P.CopyFrom(Z);

            var rz_old = new NativeReference<float>(0, Allocator.TempJob);
            var pAp = new NativeReference<float>(0, Allocator.TempJob);
            var rz_new = new NativeReference<float>(0, Allocator.TempJob);

            new Dot(R, Z, rz_old).Schedule().Complete();

            for (int iter = 0; iter < maxIter; iter++)
            {
                new Laplace(B, P, T, Ap, GridRes, H).Schedule(R.Length, BatchSize).Complete();
                new Dot(P, Ap, pAp).Schedule().Complete();
                new UpdateVR(P, Ap, V, R, rz_old, pAp).Schedule(V.Length, BatchSize).Complete();

                if (iter == maxIter - 1) break;
                MultiGridVCycle(V, R, A, T, 0, GridRes, H);

                new Dot(R, Z, rz_new).Schedule().Complete();

                float beta = rz_new.Value / rz_old.Value;

                new UpdateP(Z, P, beta).Schedule(P.Length, BatchSize).Complete();

                rz_old.Value = rz_new.Value;
            }

            new Residual(B, V, F, R, T, GridRes, H).Schedule(R.Length, BatchSize).Complete();

            new Dot(R, R, rz_old).Schedule().Complete();

            rs = math.sqrt(rz_old.Value);

            rz_old.Dispose();
            rz_new.Dispose();
            pAp.Dispose();
        }

        [BurstCompile]
        private struct Laplace : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float> _v;
            [ReadOnly] private NativeArray<float2> _b;
            [ReadOnly] private NativeArray<uint> _t;
            [WriteOnly] private NativeArray<float> _result;
            private readonly float _ih2;
            private readonly int _res;
            
            public Laplace(NativeArray<float2> b, NativeArray<float> v, NativeArray<uint> t, NativeArray<float> result, 
                int res, float h)
            {
                _ih2 = 1f / (h * h);
                _res = res;
                _v = v;
                _b = b;
                _t = t;
                _result = result;
            }
            
            public void Execute(int i)
            {
                uint gridType = _t[i];
                if (IsFluidCell(gridType))
                {
                    int x = i % _res;
                    int y = i / _res;
                    float2 betaLB = _b[i];
                    float betaL = betaLB.x;
                    float betaD = betaLB.y;
                    float betaR = _b[Coord2Index(x + 1, y, _res)].x;
                    float betaT = _b[Coord2Index(x, y + 1, _res)].y;
                    
                    bool4 isSolid = IsNeighborSolid(gridType);
                    float vC = _v[i];
                    float vL = isSolid.x ? vC : _v[Coord2Index(x - 1, y, _res)];
                    float vR = isSolid.y ? vC : _v[Coord2Index(x + 1, y, _res)];
                    float vD = isSolid.z ? vC : _v[Coord2Index(x, y - 1, _res)];
                    float vT = isSolid.w ? vC : _v[Coord2Index(x, y + 1, _res)];
                    float laplace = betaL * vL + betaR * vR + betaD * vD + betaT * vT 
                                    - (betaL + betaR + betaD + betaT) * vC;
                    _result[i] = _ih2 * laplace;
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
            private readonly float _beta;
            
            public UpdateP(NativeArray<float> z, NativeArray<float> p, float beta)
            {
                _z = z;
                _p = p;
                _beta = beta;
            }
            
            public void Execute(int i)
            {
                _p[i] = _z[i] + _beta * _p[i];
            }
        }

        private JobHandle MultiGridVCycle(JobHandle handle = default)
        {
            float h = H;
            int top = L - 1;
            for (int i = 0; i < top; i++)
            {
                int res = GridRes >> i;
                handle = SmoothJob(As[i], Vs[i], Rs[i], Ts[i], res, h, 2, handle);
                handle = new Restriction(Rs[i], As[i], Ts[i], Vs[i], Rs[i + 1], As[i + 1], TsTemp[i + 1], res, h)
                    .Schedule(Ts[i + 1].Length, BatchSize, handle);
                handle = new UpdateNeighborType(TsTemp[i+1], Ts[i+1], res).Schedule(Ts[i+1].Length, BatchSize, handle);
            }

            handle = new SymmetricGaussSeidel(Rs[top], As[top], Vs[top], Ts[top], h, GridRes >> top, 2).Schedule(handle);

            for (int i = top - 1; i >= 0; i--)
            {
                int res = GridRes >> i;
                handle = new Prolongation(Vs[i+1], Ts[i], Vs[i], res).Schedule(Vs[i].Length, BatchSize, handle);
                handle = SmoothJob(As[i], Vs[i], Rs[i], Ts[i], res, h, 2, handle);
            }

            return handle;
        }
        
        private void MultiGridVCycle(NativeArray<float> vf, NativeArray<float> rf, NativeArray<float3> af, 
            NativeArray<uint> tf, int level, int res, float h)
        {
            if (res <= 4)
            {
                SmoothJob(af, vf, rf, tf, res, h, 2).Complete();
                return;
            }

            SmoothJob(af, vf, rf, tf, res, h, 2).Complete();
            
            int resC = res / 2;
            var rc = Rs[level + 1];
            var ac = As[level + 1];
            var tcTemp = TsTemp[level + 1];
            var tc = Ts[level + 1];
            new Restriction(rf, af, tf, vf, rc, ac, tcTemp, res, h)
                .Schedule(tc.Length, BatchSize).Complete();
            new UpdateNeighborType(tcTemp, tc, res).Schedule(tc.Length, BatchSize).Complete();

            var vc = Vs[level + 1];
            // for (int i = 0; i < 2; i++) // W-cycle
            MultiGridVCycle(vc, rc, ac, tc, level + 1, resC, h);

            new Prolongation(vc, tf, vf, res).Schedule(vf.Length, BatchSize).Complete();

            SmoothJob(af, vf, rf, tf, res, h, 2).Complete();
        }

        private JobHandle SmoothJob(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f, NativeArray<uint> t, 
            int res, float h, int count, JobHandle handle = default)
        {
            for (int iter = 0; iter < count; iter++)
            {
                handle = new GaussSeidelPhase(f, a, v, t, h, res, 0).Schedule(v.Length, BatchSize, handle);
                handle = new GaussSeidelPhase(f, a, v, t, h, res, 1).Schedule(v.Length, BatchSize, handle);
                handle = new GaussSeidelPhase(f, a, v, t, h, res, 1).Schedule(v.Length, BatchSize, handle);
                handle = new GaussSeidelPhase(f, a, v, t, h, res, 0).Schedule(v.Length, BatchSize, handle);
            }
            
            return handle;

            // return new SymmetricGaussSeidel(f, a, v, t, h, res, count).Schedule(handle);
        }

        [BurstCompile]
        private struct SmoothJacobi : IJobParallelFor
        {
            [ReadOnly] private readonly NativeArray<float> _f;
            [ReadOnly] private readonly NativeArray<float> _v;
            [ReadOnly] private readonly NativeArray<uint> _t;
            [ReadOnly] private readonly NativeArray<float2> _b;
            [WriteOnly] private NativeArray<float> _result;
            private readonly float _h2;
            private readonly int _res;

            public SmoothJacobi(NativeArray<float> f, NativeArray<float> v, NativeArray<float> r,
                NativeArray<uint> type, NativeArray<float2> b, float h, int res)
            {
                _f = f;
                _v = v;
                _t = type;
                _b = b;
                _h2 = h * h;
                _res = res;
                _result = r;
            }

            public void Execute(int i)
            {
                uint gridType = _t[i];
                float res = 0;
                if (IsFluidCell(gridType))
                {
                    int x = i % _res;
                    int y = i / _res;
                    float2 betaLB = _b[i];
                    float betaL = betaLB.x;
                    float betaD = betaLB.y;
                    float betaR = _b[Coord2Index(x + 1, y, _res)].x;
                    float betaT = _b[Coord2Index(x, y + 1, _res)].y;
                    float betaSum = betaL + betaR + betaD + betaT;
                    if (betaSum > 1e-4f)
                    {
                        bool4 isSolid = IsNeighborSolid(gridType);
                        float vC = _v[i];
                        float vL = isSolid.x ? vC : _v[Coord2Index(x - 1, y, _res)];
                        float vR = isSolid.y ? vC : _v[Coord2Index(x + 1, y, _res)];
                        float vD = isSolid.z ? vC : _v[Coord2Index(x, y - 1, _res)];
                        float vT = isSolid.w ? vC : _v[Coord2Index(x, y + 1, _res)];
                        float sum = betaL * vL + betaR * vR + betaD * vD + betaT * vT;

                        res = (sum - _h2 * _f[i]) / betaSum;
                    }
                }
                _result[i] = res;
            }
        }

        [BurstCompile]
        private struct SymmetricGaussSeidel : IJob
        {
            [ReadOnly] private NativeArray<float> _f;
            [ReadOnly] private NativeArray<uint> _gridTypes;
            [ReadOnly] private NativeArray<float3> _a;
            private NativeArray<float> _v;
            private readonly float _h2;
            private readonly int _res;
            private readonly int _count;

            public SymmetricGaussSeidel(NativeArray<float> f, NativeArray<float3> a, NativeArray<float> v,
                NativeArray<uint> gridTypes, float h, int res, int count)
            {
                _f = f;
                _a = a;
                _v = v;
                _gridTypes = gridTypes;
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
                        uint gridType = _gridTypes[i];
                        if (!IsFluidCell(gridType)) continue;
                        _v[i] = (_h2 * _f[i] - NeighborSum(_a, _v, gridType, x, y, _res)) / _a[i].x;
                    }

                    for (int y = _res - 1; y >= 0; y--)
                    for (int x = _res - 1; x >= 0; x--)
                    {
                        int i = Coord2Index(x, y, _res);
                        uint gridType = _gridTypes[i];
                        if (!IsFluidCell(gridType)) continue;
                        _v[i] = (_h2 * _f[i] - NeighborSum(_a, _v, gridType, x, y, _res)) / _a[i].x;
                    }
                }
            }
        }

        [BurstCompile]
        private struct GaussSeidelPhase : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float> _f;
            [ReadOnly] private NativeArray<float3> _a;
            [ReadOnly] private NativeArray<uint> _gridTypes;
            [NativeDisableParallelForRestriction] private NativeArray<float> _v;
            private readonly float _h2;
            private readonly int _res;
            private readonly int _phase;
            
            public GaussSeidelPhase(NativeArray<float> f, NativeArray<float3> a, NativeArray<float> v, NativeArray<uint> gridTypes,
                float h, int res, int phase)
            {
                _f = f;
                _a = a;
                _v = v;
                _gridTypes = gridTypes;
                _h2 = h * h;
                _res = res;
                _phase = phase;
            }
            
            public void Execute(int i)
            {
                int y = i / _res;
                int x = i % _res;
                uint gridType = _gridTypes[i];
                if (!IsFluidCell(gridType)) return;
                
                if (((x + y) & 1) == _phase)
                {
                    _v[i] = (_h2 * _f[i] - NeighborSum(_a, _v, gridType, x, y, _res)) / _a[i].x;
                }
            }
        }
        
        [BurstCompile]
        private struct Residual : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float> _f;
            [ReadOnly] private NativeArray<float> _v;
            [ReadOnly] private NativeArray<float2> _b;
            [ReadOnly] private NativeArray<uint> _gridTypes;
            [WriteOnly] private NativeArray<float> _r;
            
            private readonly float _ih2;
            private readonly int _res;

            public Residual(NativeArray<float2> b, NativeArray<float> v, NativeArray<float> f, NativeArray<float> r, 
                NativeArray<uint> gridTypes, int res, float h)
            {
                _ih2 = 1f / (h * h);
                _res = res;
                _f = f;
                _v = v;
                _b = b;
                _r = r;
                _gridTypes = gridTypes;
            }
            
            public void Execute(int i)
            {
                uint gridType = _gridTypes[i];
                float residual = 0;
                if (IsFluidCell(gridType))
                {
                    int x = i % _res;
                    int y = i / _res;

                    float2 betaLB = _b[i];
                    float betaL = betaLB.x;
                    float betaD = betaLB.y;
                    float betaR = _b[Coord2Index(x + 1, y, _res)].x;
                    float betaT = _b[Coord2Index(x, y + 1, _res)].y;

                    bool4 isSolid = IsNeighborSolid(gridType);
                    float vC = _v[i];
                    float vL = isSolid.x ? vC : _v[Coord2Index(x - 1, y, _res)];
                    float vR = isSolid.y ? vC : _v[Coord2Index(x + 1, y, _res)];
                    float vD = isSolid.z ? vC : _v[Coord2Index(x, y - 1, _res)];
                    float vT = isSolid.w ? vC : _v[Coord2Index(x, y + 1, _res)];
                    float laplace = betaL * vL + betaR * vR + betaD * vD + betaT * vT 
                                    - (betaL + betaR + betaD + betaT) * vC;

                    residual = _f[i] - _ih2 * laplace;
                }
                _r[i] = residual;
            }
        }
        
        [BurstCompile]
        private struct Restriction : IJobParallelFor
        {
            [ReadOnly] private NativeArray<float3> _aFine;
            [ReadOnly] private NativeArray<float> _fFine;
            [ReadOnly] private NativeArray<float> _vFine;
            [ReadOnly] private NativeArray<uint> _typesFine;
            [WriteOnly] private NativeArray<float3> _aCoarse;
            [WriteOnly] private NativeArray<float> _rCoarse;
            [WriteOnly] private NativeArray<uint> _typesCoarse;

            private readonly int _res;
            private readonly float _ih2;
            
            public Restriction(NativeArray<float> ff, NativeArray<float3> af, NativeArray<uint> tf, NativeArray<float> vf,
                NativeArray<float> rc, NativeArray<float3> ac,NativeArray<uint> tc, int res, float h)
            {
                _aFine = af;
                _fFine = ff;
                _aCoarse = ac;
                _vFine = vf;
                _rCoarse = rc;
                _typesFine = tf;
                _typesCoarse = tc;
                _res = res;
                _ih2 = 1f / (h * h);
            }
            
            public void Execute(int ci)
            {
                int gridResC = _res >> 1;
                int x = ci % gridResC;
                int y = ci / gridResC;
                
                float rCoarse = 0;
                float3 aCoarse = float3.zero;
                uint4 types = uint4.zero;
                for (int yy = 0; yy < 2; yy++)
                for (int xx = 0; xx < 2; xx++)
                {
                    int fi = Coord2Index(x * 2 + xx, y * 2 + yy, _res);
                    uint typeFine = _typesFine[fi];
                    types[yy * 2 + xx] = typeFine;
                    if (IsSolidCell(typeFine)) continue;
                    float3 aFine = _aFine[fi];
                    
                    aCoarse.x += aFine.x;
                    float rFine = !IsFluidCell(typeFine)
                        ? 0
                        : _fFine[fi] - _ih2 * (_aFine[fi].x * _vFine[fi]
                              + NeighborSum(_aFine, _vFine, typeFine, x * 2 + xx, y * 2 + yy, _res));
                    rCoarse += rFine;
                    
                    if (xx == 0) aCoarse.y += aFine.y;
                    else aCoarse.x += aFine.y * 2;
                    
                    if (yy == 0) aCoarse.z += aFine.z;
                    else aCoarse.x += aFine.z * 2;
                }

                _rCoarse[ci] = rCoarse * 0.25f;
                _aCoarse[ci] = aCoarse * 0.25f;
                _typesCoarse[ci] = math.any(types == SOLID) ? SOLID : (math.any(types == FLUID) ? FLUID : AIR);
            }
        }
        
        [BurstCompile]
        private struct UpdateNeighborType : IJobParallelFor
        {
            [ReadOnly] private NativeArray<uint> _typesR;
            [WriteOnly] private NativeArray<uint> _typesW;

            private readonly int _res;
            
            public UpdateNeighborType(NativeArray<uint> tr,NativeArray<uint> tw, int res)
            {
                _typesR = tr;
                _typesW = tw;
                _res = res;
            }
            
            public void Execute(int ci)
            {
                int gridResC = _res >> 1;
                int x = ci % gridResC;
                int y = ci / gridResC;
                
                uint left = (x == 0) ? SOLID : _typesR[Coord2Index(x - 1, y, gridResC)];
                uint right = (x == gridResC - 1) ? SOLID : _typesR[Coord2Index(x + 1, y, gridResC)];
                uint down = (y == 0) ? SOLID : _typesR[Coord2Index(x, y - 1, gridResC)];
                uint up = (y == gridResC - 1) ? SOLID : _typesR[Coord2Index(x, y + 1, gridResC)];

                uint center = _typesR[ci];
                
                _typesW[ci] = center | (left << 2) | (right << 4) | (down << 6) | (up << 8);
            }
        }
        
        [BurstCompile]
        private struct Prolongation : IJobParallelFor
        {
            [ReadOnly] private NativeArray<uint> _tFine;
            [ReadOnly] private NativeArray<float> _eCoarse;
            private NativeArray<float> _eFine;

            private readonly int _res;
            
            public Prolongation(NativeArray<float> ec, NativeArray<uint> tf, NativeArray<float> ef, int res)
            {
                _tFine = tf;
                _eCoarse = ec;
                _eFine = ef;
                _res = res;
            }
            
            public void Execute(int fi)
            {
                uint gridType = _tFine[fi];
                if (!IsFluidCell(gridType)) return;
                int x = fi % _res;
                int y = fi / _res;
                int cx = x >> 1;
                int cy = y >> 1;
                int gridResC = _res >> 1;
                _eFine[fi] += _eCoarse[Coord2Index(cx, cy, gridResC)] * 2;
            }
        }

        private static float NeighborSum(NativeArray<float3> a, NativeArray<float> v, 
            uint gridType, int x, int y, int gridRes)
        {
            float sum = 0;
            uint2 xType = NeighborGridTypeAxis(0, gridType);
            uint2 yType = NeighborGridTypeAxis(1, gridType);
            float3 ac = a[Coord2Index(x, y, gridRes)];
            sum += ac.y * v[Coord2Index(IsSolidCell(xType.x) ? x : x - 1, y, gridRes)];
            sum += ac.z * v[Coord2Index(x, IsSolidCell(yType.x) ? y : y - 1, gridRes)];
            sum += a[Coord2Index(x + 1, y, gridRes)].y * v[Coord2Index(IsSolidCell(xType.y) ? x : x + 1, y, gridRes)];
            sum += a[Coord2Index(x, y + 1, gridRes)].z * v[Coord2Index(x, IsSolidCell(yType.y) ? y : y + 1, gridRes)];
            
            return sum;
        }

        private static bool4 IsNeighborSolid(uint gridType)
        {
            uint2 xType = NeighborGridTypeAxis(0, gridType);
            uint2 yType = NeighborGridTypeAxis(1, gridType);
            return new bool4(IsSolidCell(xType.x), IsSolidCell(xType.y),
                IsSolidCell(yType.x), IsSolidCell(yType.y));
        }

        private static int Coord2Index(int x, int y, int gridRes) => y * gridRes + x;
        
        public void Dispose()
        {
            Z.Dispose();
            P.Dispose();
            Ap.Dispose();

            Rs[0].Dispose();
            for (int i = 1; i < L; i++)
            {
                Vs[i].Dispose();
                Rs[i].Dispose();
                Ts[i].Dispose();
                TsTemp[i].Dispose();
            }
        }
    }
}

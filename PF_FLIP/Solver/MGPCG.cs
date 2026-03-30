using System.Diagnostics;
using Sirenix.OdinInspector;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Unity.Jobs;
using Random = UnityEngine.Random;

namespace PF_FLIP
{
    public class MGPCG : MonoBehaviour
    {
        private const int N = 256;
        
        private const byte AIR = 0;
        private const byte SOLID = 1;
        private const byte FLUID = 2;
    
        private static bool IsSolid(byte type)
        {
            return type == SOLID;
        }
        
        private static int NotSolid(byte type)
        {
            return type != SOLID ? 1 : 0;
        }

        private static void InitializeData(int i, NativeArray<float> v, NativeArray<float> b, NativeArray<float3> a)
        {
            new InitJob((uint)i, a, v, b).Schedule().Complete();
        }

        [BurstCompile]
        private struct InitJob : IJob
        {
            private NativeArray<float> _v;
            private NativeArray<float> _b;
            private NativeArray<float3> _a;
            private Unity.Mathematics.Random _rnd;
            public InitJob(uint i, NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b)
            {
                _v = v;
                _b = b;
                _a = a;
                _rnd = new Unity.Mathematics.Random(i + 1);
            }

            public void Execute()
            {
                var types = new NativeArray<byte>(_a.Length, Allocator.Temp);
                
                for (int y = 0; y < N; y++)
                for (int x = 0; x < N; x++)
                {
                    float rs = math.sin(x * ( 1f + _rnd.NextFloat()) + _rnd.NextFloat()) +
                               math.sin(x * ( 3f + _rnd.NextFloat()) + _rnd.NextFloat()) + 
                               math.sin(x * ( 7f + _rnd.NextFloat()) + _rnd.NextFloat()) + 
                               math.sin(x * (13f + _rnd.NextFloat()) + _rnd.NextFloat()) + 
                               math.sin(x * (17f + _rnd.NextFloat()) + _rnd.NextFloat()) * 0.5f + 0.5f;
                    byte type = FLUID; // 2 Fluid
            
                    // 0 Air, Dirichlet boundary condition
                    if (y > (math.sin(x * 0.3f + _rnd.NextFloat())
                             + math.sin(x * 0.07f + _rnd.NextFloat())
                             + math.sin(x * 0.013f) + _rnd.NextFloat()) * 5 
                        + N * (_rnd.NextFloat() * 0.5f + 0.5f))
                    {
                        type = AIR;
                        rs = 0;
                    }
            
                    types[y * N + x] = type;
                    _b[y * N + x] = rs;
                    _v[y * N + x] = 0;
                }
        
                for (int y = 0; y < N; y++)
                for (int x = 0; x < N; x++)
                {
                    byte cType = types[y * N + x];
                    byte lType = x > 0 ? types[y * N + x - 1] : SOLID;
                    byte rType = x < N - 1 ? types[y * N + x + 1] : SOLID; // 1 Solid
                    byte dType = y > 0 ? types[(y - 1) * N + x] : SOLID;
                    byte tType = y < N - 1 ? types[(y + 1) * N + x] : SOLID;
            
                    float3 A = float3.zero;

                    if (cType == FLUID)
                    {
                        A.x = NotSolid(lType) + NotSolid(rType) + NotSolid(dType) + NotSolid(tType);
                        A.y = lType == FLUID ? -1 : 0;
                        A.z = dType == FLUID ? -1 : 0;
                    }

                    _a[y * N + x] = A;
                }

                types.Dispose();
            }
        }
        private AMG_Solver Init2D()
        {
            var b = new NativeArray<float>(N * N, Allocator.Temp);
            var v = new NativeArray<float>(N * N, Allocator.Temp);
            var a = new NativeArray<float3>(N * N, Allocator.Temp);

            InitializeData(Random.Range(0, 1000000), v, b, a);

            return new AMG_Solver()
            {
                Lhs = v,
                Rhs = b,
                A = a,
                GridRes = N,
                H = 1,
            };
        }

        private class TestSw
        {
            private Stopwatch _sw;
            public delegate void Init(int i);
            public delegate void Action(out float rs);

            public TestSw()
            {
                _sw = Stopwatch.StartNew();
                _sw.Stop();
            }
            
            public string Run(Init init, Action action, int runs)
            {
                float sum = 0;
                _sw.Restart();
                for (int i = 0; i < runs; i++)
                {
                    init(i);
                    action(out var rs);
                    sum += rs;
                }
                _sw.Stop();
                return ($"  \trs {sum / runs}, \ttime { _sw.ElapsedMilliseconds} ms\n");
            }
        }
    
        private void Init2DJobs()
        {
            var b = new NativeArray<float>(N * N, Allocator.TempJob);
            var v = new NativeArray<float>(N * N, Allocator.TempJob);
            var a = new NativeArray<float3>(N * N, Allocator.TempJob);

            int runs = 200;


            var sw = new TestSw();
            var solver = new Neumann_UAAMGSolver(a, v, b, N, 1);

            var msg = $"Neumann_UAAMGSolver({runs} runs)\n";

            msg += "Jacobi:    \titer 64, " + sw.Run(
                i => { InitializeData(i, solver.V, solver.F, solver.A); },
                (out float rs) => { solver.Solve_Jacobi(64, out rs); },runs);
            
            // msg += "GS:          \titer 64,  " + sw.Run(
            //     i => { InitializeData(i, solver.V, solver.F, solver.A); },
            //     (out float rs) => { solver.Solve_GS(64, out rs); }, runs);
            //
            // msg += "SSOR 1.5:    \titer 64,  " + sw.Run(
            //     i => { InitializeData(i, solver.V, solver.F, solver.A); },
            //     (out float rs) => { solver.Solve_SSOR(1.5f,64, out rs); }, runs);
            
            // msg += "SSOR 1.9:    \titer 64,  " + sw.Run(
            //     i => { InitializeData(i, solver.V, solver.F, solver.A); },
            //     (out float rs) => { solver.Solve_SSOR(1.8f,64, out rs); }, runs);
            
            msg += "MG:          \titer 8,    " + sw.Run(
                i => { InitializeData(i, solver.V, solver.F, solver.A); },
                (out float rs) => { solver.Solve_MG(8, out rs); }, runs);

            // msg += "CG:          \titer 64, " + sw.Run(
            //     i => { InitializeData(i, solver.V, solver.F, solver.A); },
            //     (out float rs) => { solver.Solve_CG(64, out _, out rs); }, runs);
            
            msg += "MGF:        \titer 6,    " + sw.Run(
                i => { InitializeData(i, solver.V, solver.F, solver.A); },
                (out float rs) => { solver.Solve_MGFCycle(6, out rs); }, runs);
            
            msg += "MGPCG: \titer 6,    " + sw.Run(
                i => { InitializeData(i, solver.V, solver.F, solver.A); },
                (out float rs) => { solver.Solve_MGPCG(6, out rs); }, runs);
            
            msg += "SMGPCG: \titer 5,    " + sw.Run(
                i => { InitializeData(i, solver.V, solver.F, solver.A); },
                (out float rs) => { solver.Solve_SMGPCG(5, out rs); }, runs);
            
            msg += "MGFPCG:\titer 4,    " + sw.Run(
                i => { InitializeData(i, solver.V, solver.F, solver.A); },
                (out float rs) => { solver.Solve_MGFPCG(4, out rs); }, runs);
            
            solver.Dispose();
            // var solver2 = new Jobs_UAAMGSolver(a, v, b, N, 1);
            //
            // msg += "Jobs_UAAMGSolver\n";
            //
            // // msg += "Jacobi:    \titer 128," + sw.Run(
            // //     i => { InitializeData(i, solver2.V, solver2.F, solver2.A); },
            // //     (out float rs) => { solver2.Solve_Jacobi(128, out rs); }, runs);
            //
            // msg += "GS:          \titer 64,  " + sw.Run(
            //     i => { InitializeData(i, solver2.V, solver2.F, solver2.A); },
            //     (out float rs) => { solver2.Solve_GS(64, out rs); }, runs);
            //
            // msg += "MG:          \titer 8,    " + sw.Run(
            //     i => { InitializeData(i, solver2.V, solver2.F, solver2.A); },
            //     (out float rs) => { solver2.Solve_MG(8, out rs); }, runs);
            //
            // // msg += "CG:          \titer 64,  " + sw.Run(
            // //     i => { InitializeData(i, solver2.V, solver2.F, solver2.A); },
            // //     (out float rs) => { solver2.Solve_CG(64, out _, out rs); }, runs);
            //
            // msg += "MGPCG: \titer 3,    " + sw.Run(
            //     i => { InitializeData(i, solver2.V, solver2.F, solver2.A); },
            //     (out float rs) => { solver2.Solve_MGPCG(3, out rs); }, runs);
            
            // solver2.Dispose();
            
            Debug.Log(msg);
            
            b.Dispose();
            v.Dispose();
            a.Dispose();
        }

        [TitleGroup("Common"),Button]
        public void Solve_MG()
        {
            Init2D().Solve_MG(out var iter, out var rs);
            Debug.Log($"Solve_MG converged in iter {iter} res {rs}");
        }

        [TitleGroup("Common"),Button]
        public void Solve_GS()
        {
            Init2D().Solve_GS(out var iter, out var rs);
            Debug.Log($"Solve_GS converged in iter {iter} res {rs}");
        }

        [TitleGroup("Common"),Button]
        public void Solve_CG()
        {
            Init2D().Solve_ConjugateGradient(out var iter, out var rs);
            Debug.Log($"CG converged in iter {iter} res {rs}");
        }

        [TitleGroup("Common"),Button]
        public void Solve_MGPCG()
        {
            Init2D().Solve_MGPCG(out var iter, out var rs);
            Debug.Log(rs < 1 ? $"MGPCG converged in iter {iter} res {rs}" 
                : $"MGPCG not converged in iter {iter} res {rs}");
        }

        [TitleGroup("Jobs"), Button]
        public void Solve_Jobs()
        {
            Init2DJobs();
        }
        [TitleGroup("Jobs")]
        public GridShortCut shortcut;

        private void InitializeData(NativeArray<float> v, NativeArray<float> b, NativeArray<float3> a)
        {
            if (shortcut == null)
                InitializeData(2345, v, b, a);
            else
            {
                for (int i = 0; i < a.Length; i++)
                {
                    a[i] = shortcut.laplacian[i];
                    b[i] = shortcut.divergence[i];
                    v[i] = 0;
                }
            }
        }
        
        [TitleGroup("Jobs"), Button]
        public void Jobs_Solve_MGPCG()
        {
            var b = new NativeArray<float>(N * N, Allocator.TempJob);
            var v = new NativeArray<float>(N * N, Allocator.TempJob);
            var a = new NativeArray<float3>(N * N, Allocator.TempJob);
            
            InitializeData(v, b, a);
            
            new UnSmoothedAggregationMultiGridSolver()
            {
                Lhs = v,
                Rhs = b,
                A = a,
                GridRes = N,
                H = 1,
            }.Solve_MGPCG(out int iter, out float rs1);

            var solver = new Neumann_UAAMGSolver(a, v, b, N, 1);
            solver.Solve_MGPCG(iter, out float rs2);
            solver.Dispose();
            
            Debug.Log($"MGPCG converged in iter {iter} res {rs1} (UnSmoothedAggregationMultiGridSolver) vs res {rs2} (Neumann_UAAMGSolver)");
            b.Dispose();
            v.Dispose();
            a.Dispose();
        }
        [TitleGroup("Jobs"), Button]
        public void Jobs_Solve_FMGPCG()
        {
            var b = new NativeArray<float>(N * N, Allocator.TempJob);
            var v = new NativeArray<float>(N * N, Allocator.TempJob);
            var a = new NativeArray<float3>(N * N, Allocator.TempJob);
            
            InitializeData(v, b, a);
            // var shortCut = ScriptableObject.CreateInstance<GridShortCut>();
            // shortCut.laplacian = a.ToArray();
            // shortCut.divergence = b.ToArray();
            // UnityEditor.AssetDatabase.CreateAsset(shortCut,
            //     $"Assets/PF_FLIP/ShortCut_Random.asset");
            // UnityEditor.AssetDatabase.Refresh();
            
            new UnSmoothedAggregationMultiGridSolver()
            {
                Lhs = v,
                Rhs = b,
                A = a,
                GridRes = N,
                H = 1,
            }.Solve_FMGPCG(out int iter, out float rs1);

            var solver = new Neumann_UAAMGSolver(a, v, b, N, 1);
            solver.Solve_MGFPCG(iter, out float rs2);
            solver.Dispose();
            
            Debug.Log($"FMGPCG converged in iter {iter} res {rs1} (UnSmoothedAggregationMultiGridSolver) vs res {rs2} (Neumann_UAAMGSolver)");
            b.Dispose();
            v.Dispose();
            a.Dispose();
        }

        [TitleGroup("Jobs"), Button]
        public void Jobs_Solve_MG()
        {
            var b = new NativeArray<float>(N * N, Allocator.TempJob);
            var v = new NativeArray<float>(N * N, Allocator.TempJob);
            var a = new NativeArray<float3>(N * N, Allocator.TempJob);
            
            InitializeData(v, b, a);
            
            new UnSmoothedAggregationMultiGridSolver()
            {
                Lhs = v,
                Rhs = b,
                A = a,
                GridRes = N,
                H = 1,
            }.Solve_MG(out int iter, out float rs1);

            var solver = new Neumann_UAAMGSolver(a, v, b, N, 1);
            solver.Solve_MG(iter, out float rs2);
            solver.Dispose();
            
            Debug.Log($"MG converged in iter {iter} res {rs1} (UnSmoothedAggregationMultiGridSolver) vs res {rs2} (Neumann_UAAMGSolver)");
            b.Dispose();
            v.Dispose();
            a.Dispose();
        }

        [TitleGroup("Jobs"), Button]
        public void Jobs_Solve_FMG()
        {
            var b = new NativeArray<float>(N * N, Allocator.TempJob);
            var v = new NativeArray<float>(N * N, Allocator.TempJob);
            var a = new NativeArray<float3>(N * N, Allocator.TempJob);
            
            InitializeData(v, b, a);
            
            new UnSmoothedAggregationMultiGridSolver()
            {
                Lhs = v,
                Rhs = b,
                A = a,
                GridRes = N,
                H = 1,
            }.Solve_MGF(out int iter, out float rs1);

            var solver = new Neumann_UAAMGSolver(a, v, b, N, 1);
            solver.Solve_MGFCycle(iter, out float rs2);
            solver.Dispose();
            
            Debug.Log($"MGF converged in iter {iter} res {rs1} (UnSmoothedAggregationMultiGridSolver) vs res {rs2} (Neumann_UAAMGSolver)");
            b.Dispose();
            v.Dispose();
            a.Dispose();
        }

        [TitleGroup("Jobs"), Button]
        public void Jobs_Solve_SSOR()
        {
            var b = new NativeArray<float>(N * N, Allocator.TempJob);
            var v = new NativeArray<float>(N * N, Allocator.TempJob);
            var a = new NativeArray<float3>(N * N, Allocator.TempJob);
            
            InitializeData(v, b, a);
            
            new UnSmoothedAggregationMultiGridSolver()
            {
                Lhs = v,
                Rhs = b,
                A = a,
                GridRes = N,
                H = 1,
            }.Solve_SSOR(1.5f, out int iter, out float rs1);

            var solver = new Neumann_UAAMGSolver(a, v, b, N, 1);
            solver.Solve_SSOR(1.5f, iter, out float rs2);
            solver.Dispose();
            
            Debug.Log($"SSOR converged in iter {iter} res {rs1} (UnSmoothedAggregationMultiGridSolver) vs res {rs2} (Neumann_UAAMGSolver)");
            b.Dispose();
            v.Dispose();
            a.Dispose();
        }

        [TitleGroup("Jobs"), Button]
        public void Jobs_Solve_GS()
        {
            var b = new NativeArray<float>(N * N, Allocator.TempJob);
            var v = new NativeArray<float>(N * N, Allocator.TempJob);
            var a = new NativeArray<float3>(N * N, Allocator.TempJob);
            
            InitializeData(v, b, a);

            var solver = new Neumann_UAAMGSolver(a, v, b, N, 1);
            solver.Solve_GS(32, out float rs2);
            solver.Solve_GSRB(32, out float rs1);
            solver.Dispose();
            
            Debug.Log($"GS converged in iter 128 res {rs1} (Red-black) vs res {rs2} (normal)");
            b.Dispose();
            v.Dispose();
            a.Dispose();
        }
        
        private class AMG_Solver
        {
            public NativeArray<float> Lhs;
            public NativeArray<float> Rhs;
            public NativeArray<float3> A; // x: center, y: left, z: down
            public int GridRes;
            public float H;

            public void Solve_MG(out int iter, out float rs)
            {
                NativeArray<float> v_old = new NativeArray<float>(Lhs.Length, Allocator.Temp);
                float norm = 0;
                for (iter = 0; iter < 16; iter++)
                {
                    Lhs.CopyTo(v_old);
                    MultiGridVCycle(A, Lhs, Rhs, GridRes, H);

                    norm = math.sqrt(Diff(Lhs, v_old));
                    if (norm < H * H)
                        break;

                    // Debug.Log("MG iter " + iter + " diff: " + norm);
                }

                Residual(A, Lhs, Rhs, v_old, GridRes, H);
                rs = math.sqrt(Dot(v_old, v_old));

                v_old.Dispose();
            }

            public void Solve_GS(out int iter, out float rs)
            {
                NativeArray<float> v_old = new NativeArray<float>(Lhs.Length, Allocator.Temp);
                float norm = 0;
                for (iter = 0; iter < 100; iter++)
                {
                    Lhs.CopyTo(v_old);
                    Smooth(A, Lhs, Rhs, GridRes, H, 32);

                    norm = math.sqrt(Diff(Lhs, v_old));
                    Debug.Log("SGS iter " + (iter * 32) + " diff: " + norm);
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
                NativeArray<float> Ap = new NativeArray<float>(Rhs.Length, Allocator.Temp);
                float rs_old = Dot(r, r);

                for (iter = 0; iter < Rhs.Length; iter++)
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

                r.Dispose();
                p.Dispose();
                Ap.Dispose();
            }

            public void Solve_MGPCG(out int iter, out float rs)
            {
                NativeArray<float> r = new NativeArray<float>(Rhs.Length, Allocator.Temp);
                // NativeArray<float> rs0 = new NativeArray<float>(Rhs.Length, Allocator.Temp);
                Residual(A, Lhs, Rhs, r, GridRes, H);
                NativeArray<float> z = new NativeArray<float>(Rhs.Length, Allocator.Temp);
                MultiGridVCycle(A, z, r, GridRes, H);
                // Residual(A, z, Rhs, rs0, GridRes, H);
                // rs = math.sqrt(Dot(rs0, rs0));
                NativeArray<float> p = new NativeArray<float>(z, Allocator.Temp);
                NativeArray<float> Ap = new NativeArray<float>(p.Length, Allocator.Temp);

                float rz_old = Dot(r, z);
                // Debug.Log("MGPCG initial rs: " + rs + "___rz_old: " + rz_old);

                for (iter = 0; iter < 16; iter++)
                {
                    Laplacian(A, p, Ap, GridRes, H);
                    float alpha = rz_old / Dot(p, Ap);
                    for (int i = 0; i < r.Length; i++)
                        Lhs[i] += alpha * p[i];

                    for (int i = 0; i < r.Length; i++)
                        r[i] -= alpha * Ap[i];

                    if (math.sqrt(Dot(r, r)) < H * H * 0.02f)
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

            private float Diff(NativeArray<float> x, NativeArray<float> x_old)
            {
                float norm = 0;
                for (int i = 0; i < x.Length; i++)
                {
                    float temp = x[i] - x_old[i];
                    norm += temp * temp;
                }

                return norm;
            }

            private void MultiGridVCycle(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b,
                int res, float h)
            {
                if (res <= 4)
                {
                    Smooth(a, v, b, res, h, 4);
                    return;
                }

                Smooth(a, v, b, res, h, 2);

                var r = new NativeArray<float>(b.Length, Allocator.Temp);
                Residual(a, v, b, r, res, h);

                int resC = res >> 1;
                var rc = new NativeArray<float>(resC * resC, Allocator.Temp);
                var ac = new NativeArray<float3>(resC * resC, Allocator.Temp);
                Restrict(r, a, rc, ac, res);

                var ec = new NativeArray<float>(rc.Length, Allocator.Temp);
                // for (int i = 0; i < 2; i++)
                MultiGridVCycle(ac, ec, rc, resC, h);

                Prolongate(ec, a, v, res);

                Smooth(a, v, b, res, h, 2);

                r.Dispose();
                rc.Dispose();
                ec.Dispose();
                ac.Dispose();
            }

            private void Smooth(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f,
                int res, float h, int count)
            {
                float h2 = h * h;

                for (int iter = 0; iter < count; iter++)
                {
                    for (int y = 0; y < res; y++)
                    for (int x = 0; x < res; x++)
                    {
                        int i = Coord2Index(x, y, res);
                        v[i] = (h2 * f[i] - NeighborSum(a, v, x, y, res)) / a[i].x;
                    }

                    for (int y = res - 1; y >= 0; y--)
                    for (int x = res - 1; x >= 0; x--)
                    {
                        int i = Coord2Index(x, y, res);
                        v[i] = (h2 * f[i] - NeighborSum(a, v, x, y, res)) / a[i].x;
                    }
                }
            }

            private void Residual(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, NativeArray<float> r,
                int res, float h)
            {
                float ih2 = 1f / (h * h);
                for (int y = 0; y < res; y++)
                for (int x = 0; x < res; x++)
                {
                    int i = Coord2Index(x, y, res);
                    if (IsActive(a[i].x))
                        r[i] = b[i] - ih2 * (a[i].x * v[i] + NeighborSum(a, v, x, y, res));
                }
            }

            private void Restrict(NativeArray<float> rf, NativeArray<float3> af, NativeArray<float> rc, NativeArray<float3> ac,
                int res)
            {
                int gridResC = res >> 1;
                int gridResF = res;

                for (int y = 0; y < gridResC; y++)
                for (int x = 0; x < gridResC; x++)
                {
                    int ci = Coord2Index(x, y, gridResC);
                    float r_coarse = 0;
                    float3 A_coarse = float3.zero;
                    for (int cy = 0; cy < 2; cy++)
                    for (int cx = 0; cx < 2; cx++)
                    {
                        int fi = Coord2Index(x * 2 + cx, y * 2 + cy, gridResF);
                        float3 A_fine = af[fi];
                        if (!IsActive(A_fine.x))
                            continue;
                        A_coarse.x += A_fine.x;
                        r_coarse += rf[fi];

                        if (cx == 0) A_coarse.y += A_fine.y;
                        else A_coarse.x += A_fine.y * 2;

                        if (cy == 0) A_coarse.z += A_fine.z;
                        else A_coarse.x += A_fine.z * 2;
                    }

                    rc[ci] = r_coarse * 0.25f;
                    ac[ci] = A_coarse * 0.25f;
                }
            }

            private void Prolongate(NativeArray<float> ec, NativeArray<float3> af, NativeArray<float> ef, int res)
            {
                int gridResF = res;
                int gridResC = res >> 1;

                for (int y = 0; y < gridResC; y++)
                for (int x = 0; x < gridResC; x++)
                {
                    float cur = ec[Coord2Index(x, y, gridResC)];
                    for (int cy = 0; cy < 2; cy++)
                    for (int cx = 0; cx < 2; cx++)
                    {
                        int i = Coord2Index(x * 2 + cx, y * 2 + cy, gridResF);
                        if (IsActive(af[i].x)) ef[i] += cur * 2;
                    }
                }
            }

            private float NeighborSum(NativeArray<float3> a, NativeArray<float> v, int x, int y, int gridRes)
            {
                float3 ac = a[Coord2Index(x, y, gridRes)];
                float3 ar = x < gridRes - 1 ? a[Coord2Index(x + 1, y, gridRes)] : float3.zero;
                float3 au = y < gridRes - 1 ? a[Coord2Index(x, y + 1, gridRes)] : float3.zero;
                return (IsActive(ac.y) ? ac.y * v[Coord2Index(x - 1, y, gridRes)] : 0)
                       + (IsActive(ac.z) ? ac.z * v[Coord2Index(x, y - 1, gridRes)] : 0)
                       + (IsActive(ar.y) ? ar.y * v[Coord2Index(x + 1, y, gridRes)] : 0)
                       + (IsActive(au.z) ? au.z * v[Coord2Index(x, y + 1, gridRes)] : 0);
            }

            private bool IsActive(float x) => math.abs(x) > 1e-4f;

            private int Coord2Index(int x, int y, int gridRes) => y * gridRes + x;
        }
    }
}

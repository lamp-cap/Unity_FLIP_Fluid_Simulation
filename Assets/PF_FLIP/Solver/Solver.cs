using System.Diagnostics;
using PF_FLIP;
using Unity.Collections;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Random = UnityEngine.Random;

public class Solver : MonoBehaviour
{
    public class Measure
    {
        public Stopwatch sw;
        public float ElapsedMilliseconds => (float)sw.Elapsed.TotalMilliseconds;

        public delegate void Func(out int iter, out float rs);
        
        public Measure()
        {
            sw = Stopwatch.StartNew();
        }
        
        public void MeasureAction(Func action, int runs, out float iterAvg, out float rsMax, out float rsMin, out float rsAvg)
        {
            float sum = 0;
            float rsSum = 0;
            float rs_min = float.MaxValue;
            float rs_max = float.MinValue;
            sw.Restart();
            for (int i = 0; i < runs; i++)
            {
                action(out var iter, out var rs);
                sum += iter;
                rsSum += rs;
                if (rs < rs_min) rs_min = rs;
                if (rs > rs_max) rs_max = rs;
            }
            sw.Stop();
            iterAvg = sum / runs;
            rsMax = rs_max;
            rsMin = rs_min;
            rsAvg = rsSum / runs;
        }
    }
    private struct Grid1D
    {
        private readonly double[] _values;
        public readonly int Len;
        public readonly double H;
        public readonly double H2;
        public readonly double Ih2;
        
        public Grid1D(int len)
        {
            _values = new double[len];
            H = 60f / (len + 1);
            H2 = H * H;
            Ih2 = 1 / H2;
            Len = len;
        }
        
        public double this[int i]
        {
            get
            {
                if (i < 0 || i >= Len) return 0; // Dirichlet boundary condition
                return _values[i];
            }
            set
            {
                if (i < 0 || i >= Len) return;
                _values[i] = value;
            }
        }

        public Grid1D Copy()
        {
            Grid1D copy = new Grid1D(Len);
            for (int i = 0; i < Len; i++)
                copy[i] = _values[i];
            
            return copy;
        }
        
        public void CopyTo(Grid1D target)
        {
            for (int i = 0; i < Len; i++)
                target[i] = _values[i];
        }

        public static Grid1D operator +(Grid1D a, Grid1D b)
        {
            Grid1D result = new Grid1D(a.Len);
            for (int i = 0; i < a.Len; i++)
                result[i] = a[i] + b[i];
            
            return result;
        }
        
        public static Grid1D operator -(Grid1D a, Grid1D b)
        {
            Grid1D result = new Grid1D(a.Len);
            for (int i = 0; i < a.Len; i++)
                result[i] = a[i] - b[i];
            
            return result;
        }
        
        public static Grid1D operator -(Grid1D a)
        {
            Grid1D result = new Grid1D(a.Len);
            for (int i = 0; i < a.Len; i++)
                result[i] = -a[i];
            
            return result;
        }

        public static Grid1D operator *(double scalar, Grid1D a)
        {
            Grid1D result = new Grid1D(a.Len);
            for (int i = 0; i < a.Len; i++)
                result[i] = a[i] * scalar;

            return result;
        }

        public static Grid1D operator *(Grid1D a, double scalar)
        {
            return scalar * a;
        }
        
        public static Grid1D operator *(Grid1D a, Grid1D b)
        {
            Grid1D result = new Grid1D(a.Len);
            for (int i = 0; i < a.Len; i++)
                result[i] = a[i] * b[i];
            
            return result;
        }
        
        public static double Dot(Grid1D a, Grid1D b)
        {
            double result = 0;
            for (int i = 0; i < a.Len; i++)
                result += a[i] * b[i];
            
            return result;
        }

        public void Clear()
        {
            for (int i = 0; i < Len; i++)
                _values[i] = 0;
        }
    }

    // 参数设置
    public int N = 64;

    private static Grid1D Laplacian(Grid1D u)
    {
        Grid1D v = new Grid1D(u.Len);
        for (int i = 0; i < u.Len; i++)
            v[i] = (u[i] * 2 - u[i - 1] - u[i + 1]) * u.Ih2;
        
        return v;
    }
    
    private const string LogTemplate = "\n{0}\t:  useIter: {1,-6}  \t{2} runs: {3,-6} ms  \tresidual:  {4,-10:F6}";

    #region UAAMG1D
    private UAAMG1DSolver Init1D()
    {
        var b = new NativeArray<float>(N, Allocator.Temp);
        var v = new NativeArray<float>(N, Allocator.Temp);
        var a = new NativeArray<float3>(N, Allocator.Temp);
        
        for (int x = 0; x < N; x++)
        {
            a[x] = new float3(-1, 2, -1);
            v[x] = 0;
            b[x] = Random.value * 2 - 1;
        }
        a[0] = new float3(0, 2, -1);
        a[N - 1] = new float3(-1, 2, 0);

        return new UAAMG1DSolver()
        {
            Lhs = v,
            Rhs = b,
            A = a,
            GridRes = N,
            H = 1,
        };
    }
    
    private UAAMG1DSolver Init1D(NativeArray<float>v, NativeArray<float>b, NativeArray<float3> a)
    {
        for (int x = 0; x < N; x++)
        {
            a[x] = new float3(-1, 2, -1);
            b[x] = Random.value * 2 - 1;
            v[x] = 0;
        }
        a[0] = new float3(0, 2, -1);
        a[N - 1] = new float3(-1, 2, 0);

        return new UAAMG1DSolver()
        {
            Lhs = v,
            Rhs = b,
            A = a,
            GridRes = N,
            H = 1,
        };
    }

    public void UAAMG1D_MG()
    {
        Init1D().Solve_MG(out var iter, out var rs);
        Debug.Log($"UAAMG1D_MG iter {iter} res {rs}");
    }
    
    public void UAAMG1D_GS()
    {
        Init1D().Solve_GS(out var iter, out var rs);
        Debug.Log($"UAAMG1D_GS iter {iter} res {rs}");
    }
    
    public void UAAMG1D_CG()
    {
        Init1D().Solve_ConjugateGradient(out var iter, out var rs);
        Debug.Log($"UAAMG1D_CG iter {iter} res {rs}");
    }
    
    public void UAAMG1D_MGPCG()
    {
        Init1D().Solve_MGPCG(out var iter, out var rs);
        Debug.Log($"UAAMG1D_CG iter {iter} res {rs}");
    }

    public void UAAMG1D_BenchMark()
    {
        int runs = 200;
        var b = new NativeArray<float>(N, Allocator.Temp);
        var v = new NativeArray<float>(N, Allocator.Temp);
        var a = new NativeArray<float3>(N, Allocator.Temp);

        var sw = new Measure();
        float iterAvg, rsMax, rsMin, rsAvg;
        sw.MeasureAction((out int iter, out float rs) =>
        {
            Init1D(v, b, a).Solve_MGPCG(out iter, out rs);
        }, runs, out iterAvg, out rsMax, out rsMin, out rsAvg);

        Debug.LogFormat(D1LogTemplate, "MGPCG", iterAvg, runs, sw.ElapsedMilliseconds, rsMin, rsMax,
            rsAvg);
        
        sw.MeasureAction((out int iter, out float rs) =>
        {
            Init1D(v, b, a).Solve_MG(out iter, out rs);
        }, runs, out iterAvg, out rsMax, out rsMin, out rsAvg);

        Debug.LogFormat(D1LogTemplate, "    MG", iterAvg, runs, sw.ElapsedMilliseconds, rsMin, rsMax,
            rsAvg);

        sw.MeasureAction((out int iter, out float rs) =>
        {
            Init1D(v, b, a).Solve_ConjugateGradient(out iter, out rs);
        }, runs, out iterAvg, out rsMax, out rsMin, out rsAvg);

        Debug.LogFormat(D1LogTemplate, "    CG", iterAvg, runs, sw.ElapsedMilliseconds, rsMin, rsMax,
            rsAvg);

        b.Dispose();
        v.Dispose();
    }

    #endregion

    #region UAAMG2D

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

    private UnSmoothedAggregationMultiGridSolver Init2D(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b)
    {
        var types = new byte[a.Length];
        for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
        {
            float rs = Random.value * 2 - 1;
            byte type = FLUID; // 2 Fluid
            
            // 0 Air, Dirichlet boundary condition
            if (y > (math.sin(x * 0.1f) + math.sin(x * 0.07f) + math.sin(x * 0.013f)) * 5 + N * 0.7f)
            {
                type = AIR;
                rs = 0;
            }
            
            types[y * N + x] = type;
            b[y * N + x] = rs;
            v[y * N + x] = 0;
        }
        
        for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
        {
            byte cType = types[y * N + x];
            byte lType = x > 1 ? types[y * N + x - 1] : SOLID;
            byte rType = x < N - 1 ? types[y * N + x + 1] : SOLID; // 1 Solid
            byte dType = y > 1 ? types[(y - 1) * N + x] : SOLID;
            byte tType = y < N - 1 ? types[(y + 1) * N + x] : SOLID;
            
            float3 A = float3.zero;

            if (cType == FLUID)
            {
                A.x = NotSolid(lType) + NotSolid(rType) + NotSolid(dType) + NotSolid(tType);
                A.y = lType == FLUID ? -1 : 0;
                A.z = dType == FLUID ? -1 : 0;
            }

            a[y * N + x] = A;
        }

        return new UnSmoothedAggregationMultiGridSolver()
        {
            Lhs = v,
            Rhs = b,
            A = a,
            GridRes = N,
            H = 1,
        };
    }
    
    public void UAAMG2D_MGPCG()
    {
        var b = new NativeArray<float>(N * N, Allocator.Temp);
        var v = new NativeArray<float>(N * N, Allocator.Temp);
        var a = new NativeArray<float3>(N * N, Allocator.Temp);
        
        Init2D(a, v, b).Solve_MGPCG(out var iter, out var rs);
        Debug.LogFormat($"MGPCG iter {iter} res {rs}");
        
        a.Dispose();
        b.Dispose();
        v.Dispose();
    }

    public void UAAMG2D_BenchMark()
    {
        int runs = 100;
        var b = new NativeArray<float>(N * N, Allocator.Temp);
        var v = new NativeArray<float>(N * N, Allocator.Temp);
        var a = new NativeArray<float3>(N * N, Allocator.Temp);

        var sw = new Measure();
        float iterAvg, rsMax, rsMin, rsAvg;
        
        sw.MeasureAction((out int iter, out float rs) =>
        {
            Init2D(a, v, b).Solve_MGPCG(out iter, out rs);
        }, runs, out iterAvg, out rsMax, out rsMin, out rsAvg);
        Debug.LogFormat(D2LogTemplate, "MGPCG", iterAvg, runs, sw.ElapsedMilliseconds, rsMin, rsMax, rsAvg);
        
        sw.MeasureAction((out int iter, out float rs) =>
        {
            Init2D(a, v, b).Solve_FMGPCG(out iter, out rs);
        }, runs, out iterAvg, out rsMax, out rsMin, out rsAvg);
        Debug.LogFormat(D2LogTemplate, "FMGPCG", iterAvg, runs, sw.ElapsedMilliseconds, rsMin, rsMax, rsAvg);
        
        sw.MeasureAction((out int iter, out float rs) =>
        {
            Init2D(a, v, b).Solve_MG(out iter, out rs);
        }, runs, out iterAvg, out rsMax, out rsMin, out rsAvg);
        Debug.LogFormat(D2LogTemplate, "     MG", iterAvg, runs, sw.ElapsedMilliseconds, rsMin, rsMax, rsAvg);
        
        sw.MeasureAction((out int iter, out float rs) =>
        {
            Init2D(a, v, b).Solve_MGF(out iter, out rs);
        }, runs, out iterAvg, out rsMax, out rsMin, out rsAvg);
        Debug.LogFormat(D2LogTemplate, "   MGF", iterAvg, runs, sw.ElapsedMilliseconds, rsMin, rsMax, rsAvg);
        
        sw.MeasureAction((out int iter, out float rs) =>
        {
            Init2D(a, v, b).Solve_ConjugateGradient(out iter, out rs);
        }, runs, out iterAvg, out rsMax, out rsMin, out rsAvg);
        Debug.LogFormat(D2LogTemplate, "     CG", iterAvg, runs, sw.ElapsedMilliseconds, rsMin, rsMax, rsAvg);

        a.Dispose();
        b.Dispose();
        v.Dispose();
    }
    
    #endregion

    #region GMG1D

    private MultigridPCG.MultiGrid1DSolver Init1D(NativeArray<float> v, NativeArray<float> b)
    {
        for (int x = 1; x < N - 1; x++)
        {
            b[x] = Random.value * 2 - 1;
            v[x] = 0;
        }
        
        return new MultigridPCG.MultiGrid1DSolver()
        {
            Lhs = v,
            Rhs = b,
            GridRes = N,
            H = 1,
        };
    }
    
    public void Test1D_MG()
    {
        var b = new NativeArray<float>(N, Allocator.Temp);
        var v = new NativeArray<float>(N, Allocator.Temp);
        Init1D(v, b).SolveMG(out _, out _);
        b.Dispose();
        v.Dispose();
    }
    
    public void Test1D_GS()
    {
        var b = new NativeArray<float>(N, Allocator.Temp);
        var v = new NativeArray<float>(N, Allocator.Temp);
        var solver = Init1D(v, b);
        solver.Solve_GS();
        for (int i = 0; i < N; i++)
        {
            v[i] = 0;
        }
        solver.Solve_GSRB();
        b.Dispose();
        v.Dispose();
    }
    
    public void Test1D_CG()
    {
        var b = new NativeArray<float>(N, Allocator.Temp);
        var v = new NativeArray<float>(N, Allocator.Temp);
        Init1D(v, b).Solve_ConjugateGradient(out _, out _);
        b.Dispose();
        v.Dispose();
    }
    
    public void Test1D_MGPCG()
    {
        var b = new NativeArray<float>(N, Allocator.Temp);
        var v = new NativeArray<float>(N, Allocator.Temp);
        Init1D(v, b).SolveMGPCG(out var iter, out  var rs);
        Debug.Log("Test1D_MGPCG iter " + iter + " res " + rs);
        b.Dispose();
        v.Dispose();
    }
    
    private const string D1LogTemplate = "\n1D {0}\t: average Iter: {1,-6} \t{2} runs: {3,-6} ms" +
                                         "\t rsMin: {4} \trsMax: {5} \taverage residual: {6}";
    private const string D2LogTemplate = "\n2D {0}\t: average Iter: {1,-6} \t{2} runs: {3,-6} ms" +
                                         "\t rsMin: {4} \trsMax: {5} \taverage residual: {6}";

    public void Grid1DBenchMark()
    {
        int runs = 500;
        var b = new NativeArray<float>(N, Allocator.Temp);
        var v = new NativeArray<float>(N, Allocator.Temp);
        var sw = Stopwatch.StartNew();
        int iter = 0;
        float rs = 0;
        float rs_sum = 0;
        int sum = 0;
        float rs_min = float.MaxValue;
        float rs_max = float.MinValue;
        for (int i = 0; i < runs; i++)
        {
            Init1D(v, b).SolveMGPCG(out iter, out rs);
            sum += iter;
            rs_sum += rs;
            if (rs < rs_min) rs_min = rs;
            if (rs > rs_max) rs_max = rs;
        }
        
        sw.Stop();
        Debug.LogFormat(D1LogTemplate, "MGPCG", (float)sum/runs, runs, sw.ElapsedMilliseconds, rs_min, rs_max, rs_sum/runs);

        sum = 0;
        rs_sum = 0;
        rs_min = float.MaxValue;
        rs_max = float.MinValue;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init1D(v, b).SolveMG(out iter, out rs);
            sum += iter;
            rs_sum += rs;
            if (rs < rs_min) rs_min = rs;
            if (rs > rs_max) rs_max = rs;
        }
        
        sw.Stop();
        Debug.LogFormat(D1LogTemplate, "    MG", (float)sum/runs, runs, sw.ElapsedMilliseconds, rs_min, rs_max, rs_sum/runs);
        
        sum = 0;
        rs_sum = 0;
        rs_min = float.MaxValue;
        rs_max = float.MinValue;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init1D(v, b).Solve_ConjugateGradient(out iter, out rs);
            sum += iter;
            rs_sum += rs;
            if (rs < rs_min) rs_min = rs;
            if (rs > rs_max) rs_max = rs;
        }
        
        sw.Stop();
        Debug.LogFormat(D1LogTemplate, "    CG", (float)sum/runs, runs, sw.ElapsedMilliseconds, rs_min, rs_max, rs_sum/runs);
        
        b.Dispose();
        v.Dispose();
    }

    #endregion

    #region GMG2D

    private MultigridPCG.MultiGridSolver Init2D(NativeArray<float> v, NativeArray<float> b)
    {
        float[] rndPhase = new []
        {
            Random.value * 3 + 0.1f,
            Random.value * 3 + 0.1f,
            Random.value * 3 + 0.1f,
            Random.value * 3 + 0.1f,
            Random.value * 3 + 0.1f,
            Random.value * 3 + 0.1f,
            Random.value * 3 + 0.1f,
            Random.value * 3 + 0.1f,
            Random.value * 3 + 0.1f,
            Random.value * 3 + 0.1f,
        };
        
        for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
        {
            int i = y * N + x;
            if (y == 0 || y == N - 1 || x == 0 || x == N - 1)
                b[i] = 0;
            else
                for (int j = 0; j < rndPhase.Length / 2; j++)
                    b[i] += math.PI * math.sin(x * rndPhase[j] + rndPhase[j * 2])
                                    * math.sin(y * rndPhase[j] + rndPhase[j * 2]);

            v[i] = 0;
        }

        return new MultigridPCG.MultiGridSolver()
        {
            Lhs = v,
            Rhs = b,
            GridRes = N,
            H = 1,
        };
    }
    
    public void Test2D_CG()
    {
        var b = new NativeArray<float>(N * N, Allocator.Temp);
        var v = new NativeArray<float>(N * N, Allocator.Temp);
        Init2D(v, b).Solve_ConjugateGradient(out _, out _);
        b.Dispose();
        v.Dispose();
    }
    
    public void Test2D_GS()
    {
        var b = new NativeArray<float>(N * N, Allocator.Temp);
        var v = new NativeArray<float>(N * N, Allocator.Temp);
        Init2D(v, b).Solve_GS();
        for (int i = 0; i < v.Length; i++)
            v[i] = 0;
        Init2D(v, b).Solve_GSRB();
        b.Dispose();
        v.Dispose();
    }

    public void Test2D_MG()
    {
        var b = new NativeArray<float>(N * N, Allocator.Temp);
        var v = new NativeArray<float>(N * N, Allocator.Temp);
        Init2D(v, b).SolveMultiGrid(out _, out _);
        b.Dispose();
        v.Dispose();
    }
    
    public void Test2D_MGPCG()
    {
        var b = new NativeArray<float>(N * N, Allocator.Temp);
        var v = new NativeArray<float>(N * N, Allocator.Temp);
        Init2D(v, b).SolveMGPCG(out _, out _);
        b.Dispose();
        v.Dispose();
    }

    public void Grid2DBenchMark()
    {
        int runs = 20;
        var b = new NativeArray<float>(N * N, Allocator.Temp);
        var v = new NativeArray<float>(N * N, Allocator.Temp);
        var sw = Stopwatch.StartNew();
        int iter = 0;
        float rs = 0;
        float rs_sum = 0;
        int sum = 0;
        float rs_min = float.MaxValue;
        float rs_max = float.MinValue;
        for (int i = 0; i < runs; i++)
        {
            Init2D(v, b).SolveMGPCG(out iter, out rs);
            sum += iter;
            rs_sum += rs;
            if (rs < rs_min) rs_min = rs;
            if (rs > rs_max) rs_max = rs;
        }
        
        sw.Stop();
        Debug.LogFormat(D2LogTemplate, "MGPCG", (float)sum/runs, runs, sw.ElapsedMilliseconds, rs_min, rs_max, rs_sum/runs);
        
        rs_sum = 0;
        sum = 0;
        rs_min = float.MaxValue;
        rs_max = float.MinValue;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init2D(v, b).SolveMultiGrid(out iter, out rs);
            sum += iter;
            rs_sum += rs;
            if (rs < rs_min) rs_min = rs;
            if (rs > rs_max) rs_max = rs;
        }
        
        sw.Stop();
        Debug.LogFormat(D2LogTemplate, "   MG ", (float)sum/runs, runs, sw.ElapsedMilliseconds, rs_min, rs_max, rs_sum/runs);
        
        // rs_sum = 0;
        // sum = 0;
        // rs_min = float.MaxValue;
        // rs_max = float.MinValue;
        // sw.Restart();
        // for (int i = 0; i < runs; i++)
        // {
        //     Init2D(v, b).Solve_ConjugateGradient(out iter, out rs);
        //     sum += iter;
        //     rs_sum += rs;
        //     if (rs < rs_min) rs_min = rs;
        //     if (rs > rs_max) rs_max = rs;
        // }
        //
        // sw.Stop();
        // Debug.LogFormat(D2LogTemplate, "   CG ", (float)sum/runs, runs, sw.ElapsedMilliseconds, rs_min, rs_max, rs_sum/runs);

        b.Dispose();
        v.Dispose();
    }
    
    #endregion
    
    #region Grid1D Solvers

    public void Solve_CG()
    {
        int runs = 100;
        int iter = 0;
        Grid1D b = new Grid1D(N);
        void Init()
        {
            for (int i = 0; i < N; i++)
            {
                // float x = (i + 1) * b.H;
                // b[i] = math.PI * math.PI * math.sin(math.PI * x);
                // b[i] = math.PI * math.PI * math.sin(math.PI * x)
                //        + 9 * math.PI * math.PI * math.sin(3 * math.PI * x)
                //        + 25 * math.PI * math.PI * math.sin(5 * math.PI * x);
                b[i] = Random.value;
            }
        }
        
        var sw = Stopwatch.StartNew();
        Grid1D u = new Grid1D(b.Len);
        
        var msg = "";
        double rs_sum = 0;
        
        for (int i = 0; i < runs; i++)
        {
            Init();
            rs_sum += ValuateResult(ConjugateGradient(b, out iter), b);
        }
        sw.Stop();
        msg += string.Format(LogTemplate, $"             {"CG",20}", iter, runs, sw.ElapsedMilliseconds, rs_sum/runs);
        
        rs_sum = 0;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init();
            rs_sum += ValuateResult(Jacobi(b, out iter), b);
        }
        
        sw.Stop();
        msg += string.Format(LogTemplate, $"         {"Jacobi",20}", iter, runs, sw.ElapsedMilliseconds, rs_sum/runs);

        rs_sum = 0;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init();
            u = Chebyshev_Jacobi(b, out iter);
            rs_sum += ValuateResult(u, b);
        }
        sw.Stop();
        msg += string.Format(LogTemplate, $"{"ChebyshevJacobi",20}", iter, runs, sw.ElapsedMilliseconds, rs_sum/runs);
        
        rs_sum = 0;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init();
            u = GaussSeidel(b, out iter);
            rs_sum += ValuateResult(u, b);
        }
        sw.Stop();
        msg += string.Format(LogTemplate, $"    {"GaussSeidel",20}", iter, runs, sw.ElapsedMilliseconds, rs_sum/runs);
        
        rs_sum = 0;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init();
            u = SSOR(new Grid1D(b.Len), b, 1, out iter);
            rs_sum += ValuateResult(u, b);
        }
        sw.Stop();
        msg += string.Format(LogTemplate, $"       {"SSOR 1.0",20}", iter, runs, sw.ElapsedMilliseconds, rs_sum/runs);
        
        rs_sum = 0;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init();
            u = SSOR(new Grid1D(b.Len), b, 1.5f, out iter);
            rs_sum += ValuateResult(u, b);
        }
        sw.Stop();
        msg += string.Format(LogTemplate, $"       {"SSOR 1.5",20}", iter, runs, sw.ElapsedMilliseconds, rs_sum/runs);
        
        rs_sum = 0;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init();
            u = SSOR(new Grid1D(b.Len), b, 1.9f, out iter);
            rs_sum += ValuateResult(u, b);
        }
        sw.Stop();
        msg += string.Format(LogTemplate, $"       {"SSOR 1.9",20}", iter, runs, sw.ElapsedMilliseconds, rs_sum/runs);
        
        // msg += "\nChebyshev_SSOR: ";
        // u = Chebyshev_SSOR(b);
        // ValuateResult(u,b, ref msg);
        
        rs_sum = 0;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init();
            u = MultiGrid_VCycle(b, out iter);
            rs_sum += ValuateResult(u, b);
        }
        sw.Stop();
        msg += string.Format(LogTemplate, $"      {"MG_VCycle",20}", iter, runs, sw.ElapsedMilliseconds, rs_sum/runs);
        
        rs_sum = 0;
        sw.Restart();
        for (int i = 0; i < runs; i++)
        {
            Init();
            u = MGPCG(b, out iter);
            rs_sum += ValuateResult(u, b);
        }
        sw.Stop();
        msg += string.Format(LogTemplate, $"          {"MGPCG",20}", iter, runs, sw.ElapsedMilliseconds, rs_sum/runs);

        // msg+= "\ne: ";
        // for (int i = 0; i < u.Len; i++)
        // {
        //     msg+= $"{math.sin(math.PI * (i + 1) * h):F3} ";
        // }
        // msg+= "\ns: ";
        // for (int i = 0; i < u.Len; i++)
        // {
        //     msg+= $"{u[i]:F3} ";
        // }
        Debug.Log(msg);
    }

    public void Test()
    {
        var v1 = new Grid1D(N);
        var b = new Grid1D(N);
        var v2 = new Grid1D(N);
        for (int i = 0; i < N; i++)
        {
            b[i] = Random.value * 2 - 1;
            v1[i] = Random.value * 2 - 1;
            v2[i] = Random.value * 2 - 1;
        }

        var u1 = v1.Copy();
        var u2 = v2.Copy();
        MGTools.Smooth(u1, b, 16);
        MGTools.Smooth(u2, b, 16);
        double e1 = 0, e2 = 0;
        for (int i = 0; i < N; i++)
        {
            e1 += u1[i] * v2[i];
            e2 += v1[i] * u2[i];
        }
        Debug.Log($"/////Grid1D///// uTv {e1} vTu {e2}, {math.abs(e1-e2)}");
    }
    
    private double ValuateResult(Grid1D x, Grid1D b)
    {
        Grid1D r = b - Laplacian(x);
        double rs = Grid1D.Dot(r, r);
        return math.sqrt(rs);
        // var err = new float[N];
        // float sum = 0;
        // for (int i = 0; i < N; i++)
        // {
        //     float expected = math.sin(math.PI * (i + 1) * h);
        //     err[i] = x[i] - expected;
        //     sum += err[i] * (N + 1) * (N + 1) * err[i] * (N + 1) * (N + 1);
        //     // msg += $"{err[i]*(N+1)*(N+1):F2} ";
        // }
        // msg += $"    residual:  {math.sqrt(rs)}";
    }

    private Grid1D Jacobi(Grid1D b, out int iter, int maxIter = 5000)
    {
        Grid1D u = new Grid1D(N);
        Grid1D u_new = new Grid1D(N);
        double norm = 0;
        for (iter = 0; iter < maxIter; iter++)
        {
            for (int i = 0; i < N; i++)
                u_new[i] = 0.5f * (u.H2 * b[i] + u[i - 1] + u[i + 1]);
            
            (u, u_new) = (u_new, u);
            
            // if ((iter & 31) != 0) continue;
            Grid1D rs = u_new - u;
            norm = math.sqrt(Grid1D.Dot(rs, rs));
            if (norm < u.H2)
                break;
        }
        
        // Debug.Log($"Jacobi converged in {iter} iterations. rs:{norm}");
        return u;
    }

    private Grid1D Chebyshev_Jacobi(Grid1D b, out int iter, int maxIter = 500)
    {
        Grid1D x = new Grid1D(b.Len);
        Grid1D x_old = new Grid1D(b.Len);
        Grid1D r = new Grid1D(b.Len);
        double rho = math.cos(math.PI * b.H);
        double beta = 1;
        double err = 1;
        for (iter = 0; iter < maxIter; iter++)
        {
            for (int j = 0; j < b.Len; j++)
                r[j] = 0.5f * (b[j] * b.H2 + (x[j - 1] + x[j + 1]));
            
            if (iter == 0)
                beta = 1;
            else
                beta = iter == 1 ? 2 / (2 - rho * rho) : 1 / (1 - (rho * rho / 4) * beta);
            
            for (int j = 0; j < b.Len; j++)
            {
                double temp = beta * r[j] + (1 - beta) * x_old[j];
                x_old[j] = x[j];
                x[j] = temp;
            }

            // if ((iter & 15) != 0) continue;

            var rs = x - x_old;
            err = math.sqrt(Grid1D.Dot(rs, rs));
            if (err > b.H2) continue;
            break;
        }
        
        // Debug.Log($"Jacobi with chebyshev acceleration converged in {iter} iterations. rs:{err}");

        return x;
    }

    private Grid1D GaussSeidel(Grid1D b, out int iter, int maxIter = 5000)
    {
        Grid1D u = new Grid1D(N);
        Grid1D u_old = new Grid1D(b.Len);
        double h2 = u.H2;
        double norm = 0;
        for (iter = 0; iter < maxIter; iter++)
        {
            // if ((iter & 31) == 0)
                u.CopyTo(u_old);
            
            for (int i = 0; i < N; i++)
                u[i] = 0.5f * (h2 * b[i] + u[i - 1] + u[i + 1]);
            
            // if ((iter & 31) != 0) continue;
            u_old -= u;
            norm = math.sqrt(Grid1D.Dot(u_old, u_old));
            if (norm < h2)
                break;
        }
        
        // Debug.Log($"Gauss-Seidel converged in {iter} iterations. rs:{norm}");
        return u;
    }
    
    private Grid1D SSOR(Grid1D u, Grid1D b, float omega, out int iter, int maxIter = 5000)
    {
        Grid1D u_old = new Grid1D(N);
        double norm = 0;
        double h2 = u.H2;
        for (iter = 0; iter < maxIter; iter++)
        {
            if ((iter & 15) == 0)
                u.CopyTo(u_old);
            for (int i = 0; i < N; i++)
                u[i] = (1 - omega) * u[i] + omega * 0.5f * (h2 * b[i] + u[i - 1] + u[i + 1]);

            for (int i = N - 1; i >= 0; i--)
                u[i] = (1 - omega) * u[i] + omega * 0.5f * (h2 * b[i] + u[i - 1] + u[i + 1]);
            
            // if ((iter & 15) != 0) continue;
            var rs = u - u_old;
            norm = math.sqrt(Grid1D.Dot(rs, rs));
            if (norm < h2)
                break;
        }
        
        // Debug.Log($"SSOR {omega} converged in {iter} iterations. rs:{norm}");
        return u;
    }

    private Grid1D Chebyshev_SSOR(Grid1D b, int maxIter = 500)
    {
        Grid1D u = new Grid1D(N);
        Grid1D u_new = new Grid1D(N);
        Grid1D u_temp = new Grid1D(N);
        
        double h2 = u.H2;
        double rho_jacobi = math.cos(math.PI * b.H2); // 雅可比迭代的谱半径
    
        double rho_ssor = (rho_jacobi);
    
        // 切比雪夫多项式参数
        double[] chebyshevParams = new double[maxIter];
    
        for (int k = 0; k < maxIter; k++)
        {
            if (k == 0)
                chebyshevParams[k] = 1.0f;
            else
            {
                // 切比雪夫多项式的最优参数
                double theta_k = math.PI * (2 * k + 1) / (2 * maxIter);
                chebyshevParams[k] = 2.0 / (1.0 + rho_ssor * math.cos(theta_k));
            }
        }
    
        double norm = 0;
        
        int iter;
        for (iter = 0; iter < maxIter; iter++)
        {
            // 保存前一次迭代结果用于收敛性检查
            u.CopyTo(u_temp);
        
            for (int i = 0; i < N; i++)
                u[i] = (h2 * b[i] + u[i - 1] + u[i + 1]) * 0.5f;

            for (int i = N - 1; i >= 0; i--)
                u[i] = (h2 * b[i] + u[i - 1] + u[i + 1]) * 0.5f;

            // 应用切比雪夫加速
            double alpha = chebyshevParams[iter];
            for (int i = 0; i < N; i++)
            {
                u[i] = alpha * u[i] + (1 - alpha) * u_temp[i];
            }
        
            // 检查收敛性
            for (int i = 0; i < N; i++)
                norm += (u[i] - u_temp[i]) * (u[i] - u_temp[i]);
            norm = math.sqrt(norm);
        
            if (norm < h2)
                break;
        }
        Debug.Log($"SSOR_Chebyshev converged in {iter} iterations. rs:{norm}");
        return u;
    }
    
    private Grid1D ConjugateGradient(Grid1D b, out int iter)
    {
        Grid1D u = new Grid1D(b.Len);
        Grid1D r = b - Laplacian(u);
        Grid1D p = r.Copy();
        double rs_old = Grid1D.Dot(r, r);

        for (iter = 0; iter < b.Len; iter++)
        {
            Grid1D Ap = Laplacian(p);
            double alpha = rs_old / Grid1D.Dot(p, Ap);
            u += alpha * p;
            r -= alpha * Ap;
            double rs_new = Grid1D.Dot(r, r);
            if (math.sqrt(rs_new) < b.H2)
            {
                rs_old = rs_new;
                break;
            }
            p = r + (rs_new / rs_old) * p;
            rs_old = rs_new;
        }

        // Debug.Log($"ConjugateGradient converged in {i} iterations. rs:{math.sqrt(rs_old)}");
        
        return u;
    }
    
    private Grid1D MGPCG(Grid1D b, out int iter, int maxIter = 500)
    {
        Grid1D x = new Grid1D(b.Len);
        Grid1D r = b;
        Grid1D z = new Grid1D(b.Len);
        MultiGridVCycle(z, r, 3);
        Grid1D p = z.Copy();
        double rz_old = Grid1D.Dot(r, z);

        for (iter = 0; iter < maxIter; iter++)
        {
            Grid1D Ap = Laplacian(p);
            double alpha = rz_old / Grid1D.Dot(p, Ap);
            x += alpha * p;
            r -= alpha * Ap;

            if (math.sqrt(Grid1D.Dot(r, r)) < b.H2 * 0.01f)
                break;

            z.Clear();
            MultiGridVCycle(z, r, 3);
            double rz_new = Grid1D.Dot(r, z);
            p = z + (rz_new / rz_old) * p;
            rz_old = rz_new;
            // Debug.Log($"MGPCG {iter} iterations. rs:{math.sqrt(rz_old)}");
        }

        // Debug.Log($"MGPCG converged in {iter} iterations. rs:{math.sqrt(rz_old)}");
        return x;
    }

    private Grid1D MultiGrid_VCycle(Grid1D b, out int iter, int maxIter = 50, int smooth = 5)
    {
        Grid1D x = new Grid1D(b.Len);
        Grid1D x_old = x.Copy();
        double norm = 0;
        for (iter = 0; iter < maxIter; iter++)
        {
            x.CopyTo(x_old);
            MultiGridVCycle(x, b, smooth);
            
            Grid1D r = x - x_old;
            norm = math.sqrt(Grid1D.Dot(r, r));
            if (norm < b.H2)
                break;
            // Debug.Log($"Multigrid V-Cycle {iter} iterations. rs:{norm}");
        }
        
        // Debug.Log($"Multigrid V-Cycle converged in {iter} iterations. rs:{norm}");
        return x;
    }
    
    private void MultiGridVCycle(Grid1D v, Grid1D b, int nu1 = 2)
    {
        if (v.Len < 2)                 // 最粗网格
        {
            v[0] = b[0] * b.H2 * 0.5f;
            return;
        }

        // 1. 预平滑
        // MGTools.DampedSSOR(v, b, 2f/3f, 2);
        MGTools.Smooth(v, b, nu1);

        // 2. 计算残差
        var r = b - Laplacian(v);

        // 3. 限制到粗网格
        var rc = new Grid1D(v.Len / 2);
        MGTools.Restrict(r, rc);

        // 4. 递归求解残差方程 A e = rc
        var ec = new Grid1D(rc.Len);
        MultiGridVCycle(ec, rc, nu1);

        // 5. 误差校正
        MGTools.Prolongate(ec, v);

        // 6. 后平滑
        MGTools.Smooth(v, b, nu1);
        // MGTools.BackDampedSSOR(v, b, 2f/3f, 2);
    }
    
    private class MGTools
    {
        // 加权 Jacobi 平滑 (ω=2/3)
        // public static void Smooth(Grid1D v, Grid1D f, int nu)
        // {
        //     double omega = 2f / 3;
        //     var next = new Grid1D(v.Len);
        //     for (int iter = 0; iter < nu; iter++)
        //     {
        //         for (int i = 0; i < v.Len; i++)
        //             next[i] = (1 - omega) * v[i] + omega * 0.5f * (v[i - 1] + v[i + 1] + f.H2 * f[i]);
        //         
        //         next.CopyTo(v);
        //     }
        // }
        
        public static void Smooth(Grid1D v, Grid1D f, int nu)
        {
            // for (int i = 0; i < v.Len; i+=2)
            //     v[i] = (1 - omega) * v[i] + omega * 0.5f * (v[i - 1] + v[i + 1] + f.H2 * f[i]);
            // for (int i = 1; i < v.Len; i+=2)
            //     v[i] = (1 - omega) * v[i] + omega * 0.5f * (v[i - 1] + v[i + 1] + f.H2 * f[i]);
            //
            // for (int i = 1; i < v.Len; i+=2)
            //     v[i] = (1 - omega) * v[i] + omega * 0.5f * (v[i - 1] + v[i + 1] + f.H2 * f[i]);
            // for (int i = 0; i < v.Len; i+=2)
            //     v[i] = (1 - omega) * v[i] + omega * 0.5f * (v[i - 1] + v[i + 1] + f.H2 * f[i]);
            for (int iter = 0; iter < nu; iter++)
            {
                for (int i = 0; i < v.Len; i++)
                    v[i] = 0.5 * (v[i - 1] + v[i + 1] + f.H2 * f[i]);
                for (int i = v.Len - 1; i >= 0; i--)
                    v[i] = 0.5 * (v[i - 1] + v[i + 1] + f.H2 * f[i]);
            }
        }

        public static void Restrict(Grid1D rf, Grid1D rc)
        {
            for (int i = 0; i < rc.Len; i++)
                rc[i] = 0.25f * rf[2 * i] + 0.5f * rf[2 * i + 1] + 0.25f * rf[2 * i + 2];
        }

        public static void Prolongate(Grid1D ec, Grid1D ef)
        {
            for (int i = 0; i < ef.Len; i++)
            {
                if ((i & 1) == 0) ef[i] += 0.5f * (ec[(i >> 1) - 1] + ec[i >> 1]);
                else ef[i] += ec[i >> 1];
            }
        }
    }
    
    #endregion
}

[CustomEditor(typeof(Solver))]
public class SolverEditor : Editor
{
    
}
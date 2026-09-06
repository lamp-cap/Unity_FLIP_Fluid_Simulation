using System.Diagnostics;
using PF_FLIP;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Random = UnityEngine.Random;

public class TestMSBG : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private int[] _levels;
    private Texture2D _tex;
    public SolverType solverType;
    [Range(0, 4)] public int sampleLevel;
    [Range(1, 10)] public int iter;

    public void TestMSBG2D()
    {
        var sbg = new MultiResSparseBlockGrids();
        _levels = new int[MSBGConstants.GridCount];
        const int w = 1024;
        _tex = new Texture2D(w, w, TextureFormat.ARGB32, false, true);
        _tex.filterMode = FilterMode.Point;
        var colors = _tex.GetPixels();
        var field = new float[w * w];
        var sw = Stopwatch.StartNew();
        sbg.RandomInit(iter, solverType, out float rs);
        Debug.Log($"MG: iter {iter}, rs {rs}, time {sw.ElapsedMilliseconds} ms");
        sbg.GetLevels(_levels);
        sbg.SampleField(field, w, sampleLevel);
        for (int i = 0; i < colors.Length; i++)
        {
            float f = field[i];
            colors[i] = new Color(math.saturate(f), math.saturate(-f)*0.5f,math.saturate(-f),  1);
        }
        _tex.SetPixels(colors);
        _tex.Apply();
        GetComponent<MeshRenderer>().sharedMaterial.mainTexture = _tex;
        sbg.Dispose();
    }
    
    private const int N = MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth;
    private void InitializeData( NativeArray<float2> vel, NativeArray<float> v, NativeArray<float> b, NativeArray<float3> a)
    {
        for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
        {
            float fx = (x + 0.5f)/N;
            float fy = (y + 0.5f)/N;
            if (math.length(new float2(fx, fy)) > 0.75f) continue;
            float3 A = new float3(4, -1, -1);
            if (x == 0)
            {
                A.y = 0;
                A.x -= 1;
            }
            if (y == 0)
            {
                A.z = 0;
                A.x -= 1;
            }
            a[y * N + x] = A;
            v[y * N + x] = 0;
            vel[y * N + x] = new float2(Random.value * 2 - 1, Random.value * 2 - 1);
        }

        float d = 0;
        for (int y = 0; y < N - 1; y++)
        for (int x = 0; x < N - 1; x++)
        {
            float2 c = vel[y * N + x];
            float2 right = vel[y * N + x + 1];
            float2 up = vel[(y + 1) * N + x];
            b[y * N + x] = right.x - c.x + up.y - c.y;
            d += b[y * N + x] * b[y * N + x];
        }
        
        Debug.Log($"Initialize: sum d {d}");
    }
    
    public void TestUAAMG2D()
    {
        var vel = new NativeArray<float2>(N * N, Allocator.TempJob);
        var b = new NativeArray<float>(N * N, Allocator.TempJob);
        var v = new NativeArray<float>(N * N, Allocator.TempJob);
        var a = new NativeArray<float3>(N * N, Allocator.TempJob);

        InitializeData(vel, v, b, a);

        var solver = new Neumann_UAAMGSolver(a, v, b, N, 1);
        float rs = 0;
            
        for (int i = 0; i < N * N; i++) v[i] = 0;
        switch (solverType)
        {
            case SolverType.CG:
                solver.Solve_CG(iter * 8, out _, out rs);
                break;
            case SolverType.GS:
                solver.Solve_GS(iter * 8, out rs);
                break;
            case SolverType.MG:
                solver.Solve_MG(iter, out rs);
                break;
            case SolverType.MGPCG:
                solver.Solve_MGPCG(iter, out rs);
                break;
            case SolverType.None: break;
        }
        Debug.Log($"{solverType.ToString()}: iter {iter}, rs {rs}");
        
        for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
        {
            float c = v[y * N + x];
            float left = x > 0 ? v[y * N + x - 1] : c;
            float down = y > 0 ? v[(y - 1) * N + x] : c;
            if (a[y * N + x].x > 0)
                vel[y * N + x] += new float2(c - left, c - down);
        }

        float d = 0;
        for (int y = 0; y < N - 1; y++)
        for (int x = 0; x < N - 1; x++)
        {
            float2 c = vel[y * N + x];
            float2 right = vel[y * N + x + 1];
            float2 up = vel[(y + 1) * N + x];
            b[y * N + x] = right.x - c.x + up.y - c.y;
            d += b[y * N + x] * b[y * N + x];
        }
        Debug.Log($"After solve: d {d}");
        
        const int w = N;
        _tex = new Texture2D(w, w, TextureFormat.ARGB32, false, true);
        _tex.filterMode = FilterMode.Point;
        var colors = _tex.GetPixels();
        for (int i = 0; i < colors.Length; i++)
        {
            float f = b[i] * 10;
            colors[i] = new Color(math.saturate(f), 0, math.saturate(-f), 1);
        }
        _tex.SetPixels(colors);
        _tex.Apply();
        GetComponent<MeshRenderer>().sharedMaterial.mainTexture = _tex;
            
        solver.Dispose();
        vel.Dispose();
        b.Dispose();
        v.Dispose();
        a.Dispose();
    }
    
    private readonly Color[] _colors =
    {
        Color.red, new Color(1, 0.5f, 0), Color.grey, 
    };

    private void OnDrawGizmos()
    {
        if (_levels == null || _levels.Length != MSBGConstants.GridCount) return;
        Vector3 size = new Vector3(0.5f, 0.5f, 0);
        int w = MSBGConstants.GridWidth;
        for (int y = 0; y < w; y++)
        for (int x = 0; x < w; x++)
        {
            int level = _levels[x + y * w];
            if (level < 0) continue;
            Gizmos.color = _colors[level % _colors.Length];
            Gizmos.DrawWireCube(new Vector3(x, y, 0)*0.5f, size);
        }
    }
}

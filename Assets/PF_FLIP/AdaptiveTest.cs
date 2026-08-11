using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using Random = Unity.Mathematics.Random;

[CustomEditor(typeof(AdaptiveTest))]
public class AdaptiveTestEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        var adaptiveTest = target as AdaptiveTest;
        if (GUILayout.Button("Solve"))
        {
            adaptiveTest.Solve();
        }
        if (GUILayout.Button("Test"))
        {
            adaptiveTest.Test();
        }
    }
}
public class AdaptiveTest : MonoBehaviour
{
    public enum SolverType
    {
        CG,
        MG,
        MGPCG
    }

    public SolverType solverType;
    [Range(1, 100)]
    public int maxIterations = 10;
    private int[] _levels;
    private int[] _startIndices;
    private float2[] _velocity;
    private float3[] _params;
    private float2[] _params2;
    private float[] _pressure;
    private float[] _flux;
    
    private const int GridWidth = 6;

    private void OnDrawGizmos()
    {
        if (_levels == null || _startIndices == null || _velocity == null || _flux == null || _params == null)
            return;
        
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = _levels[i];
            int ptr = _startIndices[i];
            int width = BlockWidth(level);
            float h = GetH(level);
            float half = h * 0.5f;
            float2 posBase = new float2(x * 8, y * 8);
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                var center = new Vector3(posBase.x + xx * h + half, posBase.y + yy * h + half, 0f);
                int idx = ptr + BlockCoord2Idx(xx, yy, width);
                
                float2 vel = _velocity[idx];
                float div = _flux[idx] * 0.2f;
                // float div = _pressure[idx];
                Gizmos.color = new Color(math.max(0, div), math.max(0, -div), 0);
                // Gizmos.DrawLine(center, center + new Vector3(vel.x, vel.y, 0f));
                // Gizmos.DrawCube(center, new Vector3(h, h, 0f));
                
                float3 ps = _params[idx];
                float2 ps2 = _params2[idx];
                Handles.Label(center, $"{ps.x:F2}");
                float t = half * 0.8f;
                Handles.Label(center + new Vector3(-t,0,0.01f), $"{ps.y:F2}");
                Handles.Label(center + new Vector3(0,-t,0.01f), $"{ps.z:F2}");
                Handles.Label(center + new Vector3(t,0,0.01f), $"{ps2.x:F2}");
                Handles.Label(center + new Vector3(0,t,0.01f), $"{ps2.y:F2}");
                Gizmos.color = Color.white;
                Gizmos.DrawWireCube(center, new Vector3(h, h, 0f));
            }
        }
    }

    public void Test()
    {
        _levels = new int[GridWidth * GridWidth];
        _startIndices = new int[GridWidth * GridWidth];
        // init
        int ptr = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = 2;
            if (x > 0 && x < GridWidth - 1 && y > 0 && y < GridWidth - 1)
            {
                if (x > 1 && x < GridWidth - 2 && y > 1 && y < GridWidth - 2)
                    level = 0;
                else level = 1;
            }
            _levels[i] = level;
            _startIndices[i] = ptr;
            ptr += BlockWidth(level) * BlockWidth(level);
        }
        
        var rnd = new Random(12345);
        
        FillMatrix(_params, _levels, _startIndices);
        _params[0].x += 1f;
        
        float[] v1 = new float[ptr];
        float[] v2 = new float[ptr];
        float[] b1 = new float[ptr];
        float[] b2 = new float[ptr];
        float sum1 = 0, sum2 = 0;
        for (int i = 0; i < ptr; i++)
        {
            b1[i] = rnd.NextFloat(-1f, 1f);
            b2[i] = rnd.NextFloat(-1f, 1f);
            sum1 += b1[i];
            sum2 += b2[i];
        }

        for (int i = 0; i < ptr; i++)
        {
            b1[i] -= sum1 / ptr;
            b2[i] -= sum2 / ptr;
        }

        sum1 = 0;
        sum2 = 0;
        for (int i = 0; i < ptr; i++)
        {
            sum1 += b1[i];
            sum2 += b2[i];
        }
        
        Debug.Log("b1 sum: " + sum1 + ", b2 sum: " + sum2);
        switch (solverType)
        {
            case SolverType.CG:
                SolveCG(maxIterations * 8, _levels, _startIndices, _params, v1, b1);
                SolveCG(maxIterations * 8, _levels, _startIndices, _params, v2, b2);
                break;
            case SolverType.MG:
                SolveMG(maxIterations, _levels, _startIndices, _params, v1, b1);
                SolveMG(maxIterations, _levels, _startIndices, _params, v2, b2);
                break;
            case SolverType.MGPCG:
                SolveMGPCG(maxIterations, _levels, _startIndices, _params, v1, b1);
                SolveMGPCG(maxIterations, _levels, _startIndices, _params, v2, b2);
                break;
        }

        float dot1 = Dot(b1, v2);
        float dot2 = Dot(b2, v1);
        Debug.Log($"Symmetry check: (v1, Av2)={dot1}, (Av1, v2)={dot2}, diff={Math.Abs(dot1-dot2)}");
    }

    public void Solve()
    {
        _levels = new int[GridWidth * GridWidth];
        _startIndices = new int[GridWidth * GridWidth];
        // init
        int ptr = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = 2;
            if (x > 0 && x < GridWidth - 1 && y > 0 && y < GridWidth - 1)
            {
                if (x > 1 && x < GridWidth - 2 && y > 1 && y < GridWidth - 2)
                    level = 0;
                else level = 1;
            }
            _levels[i] = level;
            _startIndices[i] = ptr;
            ptr += BlockWidth(level) * BlockWidth(level);
        }
        
        _velocity   = new float2[ptr];
        _flux = new float[ptr];
        _params = new float3[ptr];
        _pressure = new float[ptr];
        var rnd = new Random(1234);
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = _levels[i];
            int offset = _startIndices[i];
            int width = BlockWidth(level);
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                var vel = rnd.NextFloat2(-1f, 1f);
                if (x == 0 && xx == 0) vel.x = 0;
                if (y == 0 && yy == 0) vel.y = 0;
                _velocity[offset + BlockCoord2Idx(xx, yy, width)] = vel;
            }
        }

        FillMatrix(_params, _levels, _startIndices);
        _params[0].x += 1f;

        float fluxSum = 0;
        CalcFlux(_levels, _startIndices, _velocity, _flux);
        for (int i = 0; i < ptr; i++)
            fluxSum += (_flux[i]);
        Debug.Log("Init with " + fluxSum);
        
        switch (solverType)
        {
            case SolverType.CG:
                SolveCG(64, _levels, _startIndices, _params, _pressure, _flux);
                break;
            case SolverType.MG:
                SolveMG(8, _levels, _startIndices, _params, _pressure, _flux);
                break;
            case SolverType.MGPCG:
                SolveMGPCG(8, _levels, _startIndices, _params, _pressure, _flux);
                break;
        }
        Debug.Log("Residual: "+ Residual(_levels, _startIndices, _params, _pressure, _flux));

        ApplyPressure(_levels, _startIndices, _params, _pressure, _velocity);
        fluxSum = 0;
        CalcFlux(_levels, _startIndices, _velocity, _flux);
        for (int i = 0; i < ptr; i++)
            fluxSum += math.abs(_flux[i]);
        Debug.Log("End with flux: " + fluxSum);
    }

    private void FillMatrix(float3[] matrix, int[] levels, int[] indices)
    {
        _params2 = new float2[matrix.Length];
        int GetLevel(int2 block, int2 local)
        {
            int level = levels[Coord2Idx(block)];
            int width = BlockWidth(level);
            if (local.x >= 0 && local.x < width && local.y >= 0 && local.y < width) return level;
            if (local.x < 0) block.x -= 1;
            else if (local.x >= width) block.x += 1;
            
            if (local.y < 0) block.y -= 1;
            else if (local.y >= width) block.y += 1;
            
            if (block.x < 0 || block.y < 0 || block.x >= GridWidth || block.y >= GridWidth)
                return -1;
            return levels[Coord2Idx(block)];
        }
        
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = levels[i];
            int ptr = indices[i];
            int width = BlockWidth(level);
            float h = GetH(level);
            int2 coord = new int2(x, y);
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int4 ox = new int4(-1, 0, 1, 0);
                int4 oy = new int4(0, -1, 0, 1);
                float4 ps = float4.zero;
                float sum = 0;
                for (int n = 0; n < 4; n++)
                {
                    int levelN = GetLevel(coord, new int2(xx + ox[n], yy + oy[n]));
                    if (levelN >= 0)
                    {
                        float hn = GetH(levelN);
                        if (levelN == level)
                        {
                            // ps[n] = h / h;
                            // sum += h / h;
                            ps[n] = 1f;
                            sum += 1f;
                        }
                        else
                        {
                            // ps[n] = math.min(h, hn) / (0.5f *(h + hn));
                            ps[n] = 2f / 3f;
                            if (levelN < level) sum += 2 * ps[n]; // two neighbors
                            else sum += ps[n];
                        }
                    }
                }
                matrix[ptr + BlockCoord2Idx(xx, yy, width)] = new float3(sum, -ps.x, -ps.y); // center, left, down
                _params2[ptr + BlockCoord2Idx(xx, yy, width)] = new float2(-ps.z, -ps.w); // right, up
            }
        }
    }

    private void CalcFlux(int[] levels, int[] indices, float2[] velocity, float[] flux)
    {
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = levels[i];
            int ptr = indices[i];
            int width = BlockWidth(level);
            int haloBlockWidth = width + 2;
            float2[] haloBlock = new float2[haloBlockWidth * haloBlockWidth];
            FillHaloBlock(velocity, levels, indices, haloBlock, new int2(x, y));
            float h = GetH(level);
            for (int yy = 1; yy <= width; yy++)
            for (int xx = 1; xx <= width; xx++)
            {
                float2 vel = haloBlock[BlockCoord2Idx(xx, yy, haloBlockWidth)];
                float un = haloBlock[BlockCoord2Idx(xx + 1, yy, haloBlockWidth)].x;
                float vn = haloBlock[BlockCoord2Idx(xx, yy + 1, haloBlockWidth)].y;
                
                flux[ptr + BlockCoord2Idx(xx-1, yy-1, width)] = (un - vel.x) * h + (vn - vel.y) * h;
            }
        }
    }
    
    private void SolveCG(int maxIter, int[] levels, int[] indices, float3[] matrix, float[] v, float[] flux)
    {
        // Implementation for solving the conjugate gradient method
        int numCells = flux.Length;
        var r = new float[numCells];
        var p = new float[numCells];
        flux.CopyTo(r, 0);
        r.CopyTo(p, 0);
        var Ap = new float[numCells];
        float pAp, rsNew, rsOld = 0;

        rsOld = Dot(r, r);

        var msg = "CG init with rs" + rsOld + "\n";

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Apply Laplace
            ApplyLaplace(levels, indices, matrix, p, Ap);

            pAp = Dot(p, Ap);

            float alpha = rsOld / pAp;
            for (var i = 0; i < numCells; i++)
            {
                v[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            if (iter % 20 == 19)
            {
                // 保持残差和常数向量正交（只对 INTERIOR 做）
                float mean = 0;
                for (int i = 0; i < numCells; i++) mean += r[i];
                mean /= numCells;
                for (int i = 0; i < numCells; i++) r[i] -= mean;

                // 如果 r 均值归零，p 也应该和常数正交
                // 因为 p 是 r 的线性组合
                float meanP = 0;
                for (int i = 0; i < numCells; i++) meanP += p[i];
                meanP /= numCells;
                for (int i = 0; i < numCells; i++) p[i] -= meanP;
            }

            rsNew = Dot(r, r);

            msg += $"iter{iter + 1} \trsNew:{rsNew}\n";
            if (rsNew < 1e-6f) break;

            float beta = rsNew / rsOld;
            for (int i = 0; i < numCells; i++)
                p[i] = r[i] + beta * p[i];

            rsOld = rsNew;
        }
        
        msg += "\nresidual: " + Residual(levels, indices, matrix, v, flux);
        Debug.Log(msg);
    }

    private float Dot(float[] a, float[] b)
    {
        float dot = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
        }

        return dot;
    }

    private void ApplyLaplace(int[] levels, int[] indices, float3[] matrix, float[] p, float[] Ap)
    {
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = levels[i], offset = indices[i], width = BlockWidth(level);
                
            int haloWidth = width + 2;
            var temp = new float[haloWidth * haloWidth];
            var param = new float3[haloWidth * haloWidth];

            // fill halo block
            FillHaloBlock(p, matrix, levels, indices, temp, param, new int2(x, y));
                
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int ii = offset + yy * width + xx;

                float xc = temp[BlockCoord2Idx(xx + 1, yy + 1, haloWidth)];
                float xl = temp[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                float xr = temp[BlockCoord2Idx(xx + 2, yy + 1, haloWidth)];
                float xb = temp[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                float xt = temp[BlockCoord2Idx(xx + 1, yy + 2, haloWidth)];
                float3 ac = param[BlockCoord2Idx(xx + 1, yy + 1, haloWidth)];
                float al = ac.y;
                float ar = param[BlockCoord2Idx(xx + 2, yy + 1, haloWidth)].y;
                float ab = ac.z;
                float at = param[BlockCoord2Idx(xx + 1, yy + 2, haloWidth)].z;
                Ap[ii] = xl * al + xr * ar + xb * ab + xt * at + ac.x * xc;
            }
        }
    }

    private void SolveMG(int maxIter, int[] levels, int[] indices, float3[] matrix, float[] v, float[] flux)
    {
        var ls = new[]
            { levels, new int[GridWidth * GridWidth], new int[GridWidth * GridWidth], new int[GridWidth * GridWidth] };
        var ids = new []
            { indices, new int[GridWidth * GridWidth], new int[GridWidth * GridWidth], new int[GridWidth * GridWidth] };
        int ptr1 = 0, ptr2 = 0, ptr3 = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int l0 = levels[i];
            int l1 = math.max(1, l0);
            int l2 = math.max(2, l1);
            int l3 = math.max(3, l2);
            ls[1][i] = l1;
            ls[2][i] = l2;
            ls[3][i] = l3;
            ids[1][i] = ptr1;
            ids[2][i] = ptr2;
            ids[3][i] = ptr3;
            ptr1 += BlockWidth(l1) * BlockWidth(l1);
            ptr2 += BlockWidth(l2) * BlockWidth(l2);
            ptr3 += BlockWidth(l3) * BlockWidth(l3);
        }

        var As = new[] { matrix, new float3[ptr1], new float3[ptr2], new float3[ptr3] };
        var xs = new[] { v, new float[ptr1], new float[ptr2], new float[ptr3] };
        var bs = new[] { flux, new float[ptr1], new float[ptr2], new float[ptr3] };

        var msg = ("init with residual: " + Residual(ls[0], ids[0], As[0], xs[0], bs[0]));
        for (int i = 0; i < maxIter; i++)
        {
            MultiGridVCycle(ls, ids, As, xs, bs);
            msg += "\niter " + (i + 1) + ", residual: " + Residual(ls[0], ids[0], As[0], xs[0], bs[0]);
        }
        msg += "\nresidual: " + Residual(ls[0], ids[0], As[0], xs[0], bs[0]);
        Debug.Log(msg);
    }
    private void SolveMGPCG(int maxIter, int[] levels, int[] indices, float3[] matrix, float[] v, float[] flux)
    {
        var ls = new[]
            { levels, new int[GridWidth * GridWidth], new int[GridWidth * GridWidth], new int[GridWidth * GridWidth] };
        var ids = new []
            { indices, new int[GridWidth * GridWidth], new int[GridWidth * GridWidth], new int[GridWidth * GridWidth] };
        int ptr1 = 0, ptr2 = 0, ptr3 = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int l0 = levels[i];
            int l1 = math.max(1, l0);
            int l2 = math.max(2, l1);
            int l3 = math.max(3, l2);
            ls[1][i] = l1;
            ls[2][i] = l2;
            ls[3][i] = l3;
            ids[1][i] = ptr1;
            ids[2][i] = ptr2;
            ids[3][i] = ptr3;
            ptr1 += BlockWidth(l1) * BlockWidth(l1);
            ptr2 += BlockWidth(l2) * BlockWidth(l2);
            ptr3 += BlockWidth(l3) * BlockWidth(l3);
        }
        int numCells = flux.Length;
        var r = new float[numCells];
        var z = new float[numCells];
        var p = new float[numCells];
        var Ap = new float[numCells];
        flux.CopyTo(r, 0);

        var As = new[] { matrix, new float3[ptr1], new float3[ptr2], new float3[ptr3] };
        var xs = new[] { z, new float[ptr1], new float[ptr2], new float[ptr3] };
        var bs = new[] { r, new float[ptr1], new float[ptr2], new float[ptr3] };
        
        MultiGridVCycle(ls, ids, As, xs, bs);
        xs[0].CopyTo(p, 0);
        
        float pAp, rzNew, rzOld = 0;
        for (var i = 0; i < numCells; i++)
            rzOld += r[i] * z[i];

        var msg = ("begin with residual: " + Residual(ls[0], ids[0], As[0], xs[0], bs[0]));
        
        for (int iter = 0; iter < maxIter; iter++)
        {
            ApplyLaplace(levels, indices, matrix, p, Ap);

            pAp = 0;
            for (var i = 0; i < numCells; i++)
                pAp += p[i] * Ap[i];

            float alpha = rzOld / pAp;
            for (var i = 0; i < numCells; i++)
            {
                v[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            if (iter % 2 == 1)
            {
                // 保持残差和常数向量正交（只对 INTERIOR 做）
                float mean = 0;
                for (int i = 0; i < numCells; i++) mean += r[i];
                mean /= numCells;
                for (int i = 0; i < numCells; i++) r[i] -= mean;

                // 如果 r 均值归零，p 也应该和常数正交
                // 因为 p 是 r 的线性组合
                float meanP = 0;
                for (int i = 0; i < numCells; i++) meanP += p[i];
                meanP /= numCells;
                for (int i = 0; i < numCells; i++) p[i] -= meanP;
            }

            float rs = 0;
            for (var i = 0; i < numCells; i++)
                rs += r[i] * r[i];
            msg += $"\niter{iter + 1}: \trsNew:{rs}";
            if (rs < 1e-6f || iter >= maxIter) break;
            
            for (var i = 0; i < numCells; i++)
                z[i] = 0;
            MultiGridVCycle(ls, ids, As, xs, bs);

            rzNew = 0;
            for (var i = 0; i < numCells; i++)
                rzNew += r[i] * z[i];
            msg += $"\t rzNew:{rzNew}";

            float beta = rzNew / rzOld;
            for (int i = 0; i < numCells; i++)
                p[i] = z[i] + beta * p[i];

            rzOld = rzNew;
        }
        
        msg += "\nresidual: " + Residual(levels, indices, matrix, v, flux);
        Debug.Log(msg);
    }

    private void MultiGridVCycle(int[][] lvs, int[][] ids, float3[][] As, float[][] xs, float[][] bs)
    {
        int smoothIter = 3;
        
        for (int i = 0; i < 3; i++)
        {
            for (int iter = 0; iter < smoothIter; iter++)
                GaussSeidelPhase(i, lvs[i], ids[i], As[i], xs[i], bs[i], true);
            
            Restriction(bs[i], As[i], xs[i], lvs[i], ids[i], bs[i + 1], As[i + 1], xs[i + 1], lvs[i + 1], ids[i + 1]);
        }
        
        // for (int iter = 0; iter < smoothIter; iter++)
        // {
        //     GaussSeidelPhase(3, lvs[3], ids[3], As[3], xs[3], bs[3], true);
        //     GaussSeidelPhase(3, lvs[3], ids[3], As[3], xs[3], bs[3], false);
        // }
        for (int iter = 0; iter < 4; iter++)
        {
            for (int y = 0; y < GridWidth; y++)
            for (int x = 0; x < GridWidth; x++)
            {
                int i = Coord2Idx(x, y);
                float xl = x > 0 ? xs[3][Coord2Idx(x - 1, y)] : 0;
                float xr = x < GridWidth - 1 ? xs[3][Coord2Idx(x + 1, y)] : 0;
                float xb = y > 0 ? xs[3][Coord2Idx(x, y - 1)] : 0;
                float xt = y < GridWidth - 1 ? xs[3][Coord2Idx(x, y + 1)] : 0;
                float3 ac = As[3][i];
                float ar = x < GridWidth - 1 ? As[3][Coord2Idx(x + 1, y)].y : 0;
                float at = y < GridWidth - 1 ? As[3][Coord2Idx(x, y + 1)].z : 0;
                xs[3][i] = (bs[3][i] - (xl * ac.y + xr * ar + xb * ac.z + xt * at)) / ac.x;
            }

            for (int y = GridWidth - 1; y >= 0; y--)
            for (int x = GridWidth - 1; x >= 0; x--)
            {
                int i = Coord2Idx(x, y);
                float xl = x > 0 ? xs[3][Coord2Idx(x - 1, y)] : 0;
                float xr = x < GridWidth - 1 ? xs[3][Coord2Idx(x + 1, y)] : 0;
                float xb = y > 0 ? xs[3][Coord2Idx(x, y - 1)] : 0;
                float xt = y < GridWidth - 1 ? xs[3][Coord2Idx(x, y + 1)] : 0;
                float3 ac = As[3][i];
                float ar = x < GridWidth - 1 ? As[3][Coord2Idx(x + 1, y)].y : 0;
                float at = y < GridWidth - 1 ? As[3][Coord2Idx(x, y + 1)].z : 0;
                xs[3][i] = (bs[3][i] - (xl * ac.y + xr * ar + xb * ac.z + xt * at)) / ac.x;
            }
        }

        for (int i = 2; i >= 0; i--)
        {
            Prolongation(xs[i+1], lvs[i+1], ids[i+1], xs[i], lvs[i], ids[i]);
            for (int iter = 0; iter < smoothIter; iter++)
                GaussSeidelPhase(i, lvs[i], ids[i], As[i], xs[i], bs[i], false);
        }
    }

    private void GaussSeidelPhase(int targetLevel, int[] l, int[] id, float3[] a, float[] v, float[] b, bool red_black)
    {
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = l[i];
            if (level != targetLevel) continue;
            int offset = id[i], width = BlockWidth(level);
                
            int haloWidth = width + 2;
            var blockB = new float[width * width];
            var blockV = new float[haloWidth * haloWidth];
            var blockA = new float3[haloWidth * haloWidth];

            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int localId = BlockCoord2Idx(xx, yy, width);
                blockB[localId] = b[offset + localId];
            }
            
            // fill halo block
            FillHaloBlock(v, a, l, id, blockV, blockA, new int2(x, y));
            
            for (int yy = 1; yy <= width; yy++)
            for (int xx = 1; xx <= width; xx++)
            {
                int phase = red_black ? 0 : 1;
                if (((xx + yy) & 1) != phase) continue;
                
                int localId = BlockCoord2Idx(xx - 1, yy - 1, width);

                float xl = blockV[BlockCoord2Idx(xx - 1, yy, haloWidth)];
                float xr = blockV[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                float xb = blockV[BlockCoord2Idx(xx, yy - 1, haloWidth)];
                float xt = blockV[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                float3 ac = blockA[BlockCoord2Idx(xx, yy, haloWidth)];
                float al = ac.y;
                float ar = blockA[BlockCoord2Idx(xx + 1, yy, haloWidth)].y;
                float ab = ac.z;
                float at = blockA[BlockCoord2Idx(xx, yy + 1, haloWidth)].z;
                blockV[BlockCoord2Idx(xx, yy, haloWidth)] = (blockB[localId] - (xl * al + xr * ar + xb * ab + xt * at)) / ac.x;
            }
            
            // copy to buffer
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                v[offset + BlockCoord2Idx(xx, yy, width)] = blockV[BlockCoord2Idx(xx + 1, yy + 1, haloWidth)];
            }
        }
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = l[i];
            if (level != targetLevel) continue;
            int offset = id[i], width = BlockWidth(level);
                
            int haloWidth = width + 2;
            var blockB = new float[width * width];
            var blockV = new float[haloWidth * haloWidth];
            var blockA = new float3[haloWidth * haloWidth];

            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int localId = BlockCoord2Idx(xx, yy, width);
                blockB[localId] = b[offset + localId];
            }
            
            // fill halo block
            FillHaloBlock(v, a, l, id, blockV, blockA, new int2(x, y));
            
            for (int yy = 1; yy <= width; yy++)
            for (int xx = 1; xx <= width; xx++)
            {
                int phase = red_black ? 1 : 0;
                if (((xx + yy) & 1) != phase) continue;
                
                int localId = BlockCoord2Idx(xx - 1, yy - 1, width);

                float xl = blockV[BlockCoord2Idx(xx - 1, yy, haloWidth)];
                float xr = blockV[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                float xb = blockV[BlockCoord2Idx(xx, yy - 1, haloWidth)];
                float xt = blockV[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                float3 ac = blockA[BlockCoord2Idx(xx, yy, haloWidth)];
                float al = ac.y;
                float ar = blockA[BlockCoord2Idx(xx + 1, yy, haloWidth)].y;
                float ab = ac.z;
                float at = blockA[BlockCoord2Idx(xx, yy + 1, haloWidth)].z;
                blockV[BlockCoord2Idx(xx, yy, haloWidth)] = (blockB[localId] - (xl * al + xr * ar + xb * ab + xt * at)) / ac.x;
            }
            
            // copy to buffer
            
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                v[offset + BlockCoord2Idx(xx, yy, width)] = blockV[BlockCoord2Idx(xx + 1, yy + 1, haloWidth)];
            }
        }
    }

    private void Restriction(float[] bf, float3[] af, float[] vf, int[] lf, int[] idf,
        float[] rc, float3[] ac, float[] vc, int[] lc, int[] idc)
    {
        for (int gy = 0; gy < GridWidth; gy++)
        for (int gx = 0; gx < GridWidth; gx++)
        {
            int gid = Coord2Idx(gx, gy);
            int levelF = lf[gid];
            int phf = idf[gid];
            int levelC = lc[gid];
            int phc = idc[gid];
            
            int blockWidthF = BlockWidth(levelF);
            int haloWidth = blockWidthF + 2;
            var blockV = new float[haloWidth * haloWidth];
            var blockA = new float3[haloWidth * haloWidth];
            var blockR = new float[blockWidthF * blockWidthF];
            FillHaloBlock(vf, af, lf, idf, blockV, blockA, new int2(gx, gy));
            for (int fy = 0; fy < blockWidthF; fy++)
            for (int fx = 0; fx < blockWidthF; fx++)
            {
                int localId = BlockCoord2Idx(fx, fy, blockWidthF);

                float xc = blockV[BlockCoord2Idx(fx + 1, fy + 1, haloWidth)];
                float xl = blockV[BlockCoord2Idx(fx, fy + 1, haloWidth)];
                float xr = blockV[BlockCoord2Idx(fx + 2, fy + 1, haloWidth)];
                float xb = blockV[BlockCoord2Idx(fx + 1, fy, haloWidth)];
                float xt = blockV[BlockCoord2Idx(fx + 1, fy + 2, haloWidth)];
                float3 a = blockA[BlockCoord2Idx(fx + 1, fy + 1, haloWidth)];
                float al = a.y;
                float ar = blockA[BlockCoord2Idx(fx + 2, fy + 1, haloWidth)].y;
                float ab = a.z;
                float at = blockA[BlockCoord2Idx(fx + 1, fy + 2, haloWidth)].z;
                if (a.x == 0) continue;
                blockR[localId] = bf[phf + localId] - (xl * al + xr * ar + xb * ab + xt * at + a.x * xc);
            }

            if (levelF < levelC)
            {
                int blockWidthC = BlockWidth(levelC);
                for (int cy = 0; cy < blockWidthC; cy++)
                for (int cx = 0; cx < blockWidthC; cx++)
                {
                    float rCoarse = 0;
                    float3 aCoarse = float3.zero;
                    for (int yy = 0; yy < 2; yy++)
                    for (int xx = 0; xx < 2; xx++)
                    {
                        int fx = cx * 2 + xx;
                        int fy = cy * 2 + yy;
                        float rFine = blockR[BlockCoord2Idx(fx, fy, blockWidthF)];
                        rCoarse += rFine;
                        float3 aFine = blockA[BlockCoord2Idx(fx + 1, fy + 1, haloWidth)];
                        aCoarse.x += aFine.x;
                        if (xx == 0) aCoarse.y += aFine.y;
                        else aCoarse.x += aFine.y * 2;
                        if (yy == 0) aCoarse.z += aFine.z;
                        else aCoarse.x += aFine.z * 2;
                    }

                    int ci = phc + BlockCoord2Idx(cx, cy, blockWidthC);
                    rc[ci] = rCoarse;
                    ac[ci] = aCoarse;
                    vc[ci] = 0;
                }
            }
            else // copy
            {
                for (int y = 0; y < blockWidthF; y++)
                for (int x = 0; x < blockWidthF; x++)
                {
                    int ii = BlockCoord2Idx(x, y, blockWidthF);
                    int ci = phc + ii;
                    float3 aFine = blockA[BlockCoord2Idx(x + 1, y + 1, haloWidth)];
                    if (aFine.x == 0) continue;
                    if (x == 0 && gx > 0 && lf[Coord2Idx(gx - 1, gy)] < lc[Coord2Idx(gx - 1, gy)]) aFine.y *= 2;
                    if (y == 0 && gy > 0 && lf[Coord2Idx(gx, gy - 1)] < lc[Coord2Idx(gx, gy - 1)]) aFine.z *= 2;
                    float rFine = blockR[BlockCoord2Idx(x, y, blockWidthF)];
                    rc[ci] = rFine;
                    ac[ci] = aFine;
                    vc[ci] = 0;
                }
            }
        }
    }
    
    private void Prolongation(float[] vc, int[] lc, int[] idc, float[] vf, int[] lf, int[] idf)
    {
        for (int gy = 0; gy < GridWidth; gy++)
        for (int gx = 0; gx < GridWidth; gx++)
        {
            int gid = Coord2Idx(gx, gy);
            int levelF = lf[gid];
            int phf = idf[gid];
            int levelC = lc[gid];
            int phc = idc[gid];
                
            int blockWidthC = BlockWidth(levelC);
            int blockWidthF = BlockWidth(levelF);
            if (levelF < levelC)
            {
                for (int y = 0; y < blockWidthC; y++)
                for (int x = 0; x < blockWidthC; x++)
                {
                    float eCoarse = vc[phc + BlockCoord2Idx(x, y, blockWidthC)];
                    for (int yy = 0; yy < 2; yy++)
                    for (int xx = 0; xx < 2; xx++)
                    {
                        int fx = x * 2 + xx;
                        int fy = y * 2 + yy;
                        int fi = phf + BlockCoord2Idx(fx, fy, blockWidthF);

                        vf[fi] += eCoarse;
                    }
                }
            }
            else
            {
                int blockSize = blockWidthC * blockWidthC;
                for (int ii = 0; ii < blockSize; ii++)
                    vf[phf + ii] += vc[phc + ii];
            }
        }
    }
    
    private float Residual(int[] l, int[] id, float3[] a, float[] v, float[] b)
    {
        float r = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = l[i], offset = id[i], width = BlockWidth(level);
                
            int haloWidth = width + 2;
            var blockV = new float[haloWidth * haloWidth];
            var blockA = new float3[haloWidth * haloWidth];
            
            // fill halo block
            FillHaloBlock(v, a, l, id, blockV, blockA, new int2(x, y));
            
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int localId = BlockCoord2Idx(xx, yy, width);

                float xc = blockV[BlockCoord2Idx(xx + 1, yy + 1, haloWidth)];
                float xl = blockV[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                float xr = blockV[BlockCoord2Idx(xx + 2, yy + 1, haloWidth)];
                float xb = blockV[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                float xt = blockV[BlockCoord2Idx(xx + 1, yy + 2, haloWidth)];
                float3 ac = blockA[BlockCoord2Idx(xx + 1, yy + 1, haloWidth)];
                float al = ac.y;
                float ar = blockA[BlockCoord2Idx(xx + 2, yy + 1, haloWidth)].y;
                float ab = ac.z;
                float at = blockA[BlockCoord2Idx(xx + 1, yy + 2, haloWidth)].z;
                if (ac.x == 0) continue;
                float residual = b[offset + localId] - (xl * al + xr * ar + xb * ab + xt * at + ac.x * xc);
                r += residual * residual;
            }
        }

        return r;
    }
    
    private void ApplyPressure(int[] levels, int[] indices, float3[] matrix, float[] pressure, float2[] velocity)
    {
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y), level = levels[i], offset = indices[i], width = BlockWidth(level);
            
            int haloWidth = width + 2;
            var temp = new float[haloWidth * haloWidth];
            var param = new float[haloWidth * haloWidth];

            // fill halo block
            FillHaloBlock(pressure, levels, indices, temp, param, new int2(x, y));
            
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int ii = offset + yy * width + xx;

                float p = temp[BlockCoord2Idx(xx + 1, yy + 1, haloWidth)];
                float up = temp[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                float ua = param[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                float vp = temp[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                float va = param[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                
                float2 delta = new float2((p - up) * ua, (p - vp) * va);
                velocity[ii] += delta;
            }
        }

    }

    private static void FillHaloBlock(float2[] v, int[] levels, int[] startIndices, float2[] block, int2 coord)
    {
        int level = levels[Coord2Idx(coord)];
        int blockWidth = BlockWidth(level);
        int haloBlockWidth = blockWidth + 2;
        for (int by = 0; by < blockWidth; by++)
        for (int bx = 0; bx < blockWidth; bx++)
        {
            int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
            int physicsIdx = startIndices[Coord2Idx(coord)] + BlockCoord2Idx(bx, by, blockWidth);
                
            block[localIdx] = v[physicsIdx];
        }
        int4 ox = new int4(-1, 0, 1, 0);
        int4 oy = new int4(0, -1, 0, 1);
        
        for (int n = 0; n < 4; n++)
        {
            int2 dir = new int2(ox[n], oy[n]);
            int2 curr = coord + dir;
            if (curr.x < 0 || curr.y < 0 || curr.x >= GridWidth || curr.y >= GridWidth)
                continue;
            
            int nLevel = levels[Coord2Idx(curr)];
            if (nLevel < 0)
                continue;
            
            int phn = startIndices[Coord2Idx(curr)];
            if (nLevel == level)
            {
                for (int c = 0; c < blockWidth; c++)
                {
                    int2 nCoord = math.select(math.select(c, 0, dir > 0), blockWidth - 1, dir < 0);
                    int nLocalIdx = BlockCoord2Idx(nCoord, blockWidth);
                    int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                    block[paddingIdx] = v[phn + nLocalIdx];
                }
            }
            else if (nLevel > level)
            {
                int nBlockWidth = BlockWidth(nLevel);
                for (int c = 0; c < blockWidth; c++)
                {
                    int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                    int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                    int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                    block[paddingIdx] = v[phn + nLocalIdx];
                }
            }
            else // n_level < level
            {
                int nBlockWidth = BlockWidth(nLevel);
                for (int c = 0; c < blockWidth; c++)
                {
                    int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                    int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                    int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                    int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                    int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                    block[paddingIdx] = (v[phn + nLocalIdx0] + v[phn + nLocalIdx1]) * 0.5f;
                }
            }
        }
    }
    private static void FillHaloBlock(float[] v, int[] levels, int[] indices, float[] blockV, float[] blockP, int2 coord)
    {
        int level = levels[Coord2Idx(coord)];
        int width = BlockWidth(level);
        int haloWidth = width + 2;
        int offset = indices[Coord2Idx(coord)];
        // center
        for (int by = 0; by < width; by++)
        for (int bx = 0; bx < width; bx++)
        {
            int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloWidth);
            int physicsIdx = offset + BlockCoord2Idx(bx, by, width);

            blockV[localIdx] = v[physicsIdx];
            blockP[localIdx] = 1f / GetH(level);
        }

        // halo
        int4 ox = new int4(-1, 0, 1, 0);
        int4 oy = new int4(0, -1, 0, 1);
        for (int n = 0; n < 4; n++)
        {
            int2 dir = new int2(ox[n], oy[n]);
            int2 curr = coord + dir;
            if (curr.x < 0 || curr.y < 0 || curr.x >= GridWidth || curr.y >= GridWidth)
                continue;

            int nLevel = levels[BlockCoord2Idx(curr, GridWidth)];
            int phn = indices[BlockCoord2Idx(curr, GridWidth)];
            int nBlockWidth = BlockWidth(nLevel);
            
            if (nLevel == level)
            {
                for (int c = 0; c < width; c++)
                {
                    int2 nCoord = math.select(math.select(c, 0, dir > 0), width - 1, dir < 0);
                    int nLocalIdx = BlockCoord2Idx(nCoord, width);
                    int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                    blockV[paddingIdx] = v[phn + nLocalIdx];
                    blockP[paddingIdx] = 1f / GetH(nLevel);
                }
            }
            else if (nLevel > level)
            {
                for (int c = 0; c < width; c++)
                {
                    int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                    int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                    int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                    blockV[paddingIdx] = v[phn + nLocalIdx];
                    blockP[paddingIdx] = 1f / (0.5f * (GetH(level) + GetH(nLevel)));
                }
            }
            else // n_level < level
            {
                for (int c = 0; c < width; c++)
                {
                    int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                    int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                    int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                    int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                    int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                    blockV[paddingIdx] = (v[phn + nLocalIdx0] + v[phn + nLocalIdx1]) * 0.5f;
                    blockP[paddingIdx] = 1f / (0.5f * (GetH(level) + GetH(nLevel)));
                }
            }
        }
    }
    
    private static void FillHaloBlock(float[] v, float3[] p, int[] levels, int[] indices, float[] blockV, float3[] blockP, int2 coord)
    {
        int level = levels[Coord2Idx(coord)];
        int width = BlockWidth(level);
        int haloWidth = width + 2;
        int offset = indices[Coord2Idx(coord)];
        // center
        for (int by = 0; by < width; by++)
        for (int bx = 0; bx < width; bx++)
        {
            int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloWidth);
            int physicsIdx = offset + BlockCoord2Idx(bx, by, width);

            blockV[localIdx] = v[physicsIdx];
            blockP[localIdx] = p[physicsIdx];
        }

        // halo
        int4 ox = new int4(-1, 0, 1, 0);
        int4 oy = new int4(0, -1, 0, 1);
        for (int n = 0; n < 4; n++)
        {
            int2 dir = new int2(ox[n], oy[n]);
            int2 curr = coord + dir;
            if (curr.x < 0 || curr.y < 0 || curr.x >= GridWidth || curr.y >= GridWidth)
                continue;

            int nLevel = levels[BlockCoord2Idx(curr, GridWidth)];
            int phn = indices[BlockCoord2Idx(curr, GridWidth)];
            int nBlockWidth = BlockWidth(nLevel);
            
            if (nLevel == level)
            {
                for (int c = 0; c < width; c++)
                {
                    int2 nCoord = math.select(math.select(c, 0, dir > 0), width - 1, dir < 0);
                    int nLocalIdx = BlockCoord2Idx(nCoord, width);
                    int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                    blockV[paddingIdx] = v[phn + nLocalIdx];
                    blockP[paddingIdx] = p[phn + nLocalIdx];
                }
            }
            else if (nLevel > level)
            {
                for (int c = 0; c < width; c++)
                {
                    int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                    int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                    int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                    blockV[paddingIdx] = v[phn + nLocalIdx];
                    blockP[paddingIdx] = p[phn + nLocalIdx];
                }
            }
            else // n_level < level
            {
                for (int c = 0; c < width; c++)
                {
                    int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                    int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                    int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                    int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                    int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                    blockV[paddingIdx] = v[phn + nLocalIdx0] + v[phn + nLocalIdx1];
                    blockP[paddingIdx] = p[phn + nLocalIdx0]; // the two fine cells' params are the same
                }
            }
        }
    }

    private static int BlockWidth(int level) => 1 << (3 - level);
    private static int BlockCoord2Idx(int2 coord, int res) => coord.x + coord.y * res;
    private static int BlockCoord2Idx(int x, int y, int res) => x + y * res;
    private static int Coord2Idx(int x, int y) => x + y * GridWidth;
    private static int Coord2Idx(int2 coord) => coord.x + coord.y * GridWidth;
        
    private static float GetH(int level) => 1 << level;
}

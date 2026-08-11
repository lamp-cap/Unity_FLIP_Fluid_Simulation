using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using Random = Unity.Mathematics.Random;

public class AdaptiveSolver : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    
    public enum SolverType
    {
        CG,
        MG,
        MGPCG
    }

    public SolverType solverType;
    [Range(1, 100)]
    public int maxIterations = 10;

    private NativeArray<int2> _gridInfos;
    private NativeArray<float2> _gridVelocity;
    private NativeArray<float3> _gridLaplacian;
    private NativeArray<float> _pressure;
    private NativeArray<float> _flux;
    
    private const int GridWidth = 16;

    // private void OnDrawGizmos()
    // {
    //     if (_levels == null || _startIndices == null || _velocity == null || _flux == null || _gridLaplacian == null)
    //         return;
    //     
    //     for (int y = 0; y < GridWidth; y++)
    //     for (int x = 0; x < GridWidth; x++)
    //     {
    //         int i = Coord2Idx(x, y);
    //         int level = _levels[i];
    //         int ptr = _startIndices[i];
    //         int width = BlockWidth(level);
    //         float h = GetH(level);
    //         float half = h * 0.5f;
    //         float2 posBase = new float2(x * 8, y * 8);
    //         for (int yy = 0; yy < width; yy++)
    //         for (int xx = 0; xx < width; xx++)
    //         {
    //             var center = new Vector3(posBase.x + xx * h + half, posBase.y + yy * h + half, 0f);
    //             int idx = ptr + BlockCoord2Idx(xx, yy, width);
    //             
    //             float2 vel = _velocity[idx];
    //             float div = _flux[idx] * 0.2f;
    //             // float div = _pressure[idx];
    //             Gizmos.color = new Color(math.max(0, div), math.max(0, -div), 0);
    //             // Gizmos.DrawLine(center, center + new Vector3(vel.x, vel.y, 0f));
    //             // Gizmos.DrawCube(center, new Vector3(h, h, 0f));
    //             
    //             float3 ps = _gridLaplacian[idx];
    //             float2 ps2 = _gridLaplacian2[idx];
    //             Handles.Label(center, $"{ps.x:F2}");
    //             float t = half * 0.8f;
    //             Handles.Label(center + new Vector3(-t,0,0.01f), $"{ps.y:F2}");
    //             Handles.Label(center + new Vector3(0,-t,0.01f), $"{ps.z:F2}");
    //             Handles.Label(center + new Vector3(t,0,0.01f), $"{ps2.x:F2}");
    //             Handles.Label(center + new Vector3(0,t,0.01f), $"{ps2.y:F2}");
    //             Gizmos.color = Color.white;
    //             Gizmos.DrawWireCube(center, new Vector3(h, h, 0f));
    //         }
    //     }
    // }

    public void Initialize()
    {
    }

    public void Solve()
    {
        _gridInfos = new NativeArray<int2>(GridWidth * GridWidth, Allocator.Persistent);
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
            _gridInfos[i] = new int2(level, ptr);
            ptr += BlockWidth(level) * BlockWidth(level);
        }
        
        _gridVelocity = new NativeArray<float2>(ptr, Allocator.Persistent);
        _flux = new NativeArray<float>(ptr, Allocator.Persistent);
        _gridLaplacian = new NativeArray<float3>(ptr, Allocator.Persistent);
        _pressure = new NativeArray<float>(ptr, Allocator.Persistent);
        var rnd = new Random(1234);
    }

    public void Step()
    {
        CalcFlux(_gridInfos, _gridVelocity, _flux);
        
        switch (solverType)
        {
            case SolverType.CG:
                SolveCG(64, _gridInfos, _gridLaplacian, _pressure, _flux);
                break;
            case SolverType.MG:
                SolveMG(8, _gridInfos, _gridLaplacian, _pressure, _flux);
                break;
            case SolverType.MGPCG:
                SolveMGPCG(8, _gridInfos, _gridLaplacian, _pressure, _flux);
                break;
        }

        ApplyPressure(_gridInfos, _pressure, _gridVelocity);
        
    }

    private void FillMatrix(NativeArray<float3> matrix, NativeArray<int> levels, NativeArray<int> indices)
    {
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
                // _gridLaplacian2[ptr + BlockCoord2Idx(xx, yy, width)] = new float2(-ps.z, -ps.w); // right, up
            }
        }
    }

    private void CalcFlux(NativeArray<int2> infos, NativeArray<float2> velocity, NativeArray<float> flux)
    {
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int2 info = infos[i];
            int level = info.x;
            int ptr = info.y;
            int width = BlockWidth(level);
            int haloBlockWidth = width + 2;
            NativeArray<float2> haloBlock = new NativeArray<float2>(haloBlockWidth * haloBlockWidth, Allocator.Temp);
            FillHaloBlock(velocity, infos, haloBlock, new int2(x, y));
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
    
    private void SolveCG(int maxIter, NativeArray<int2> infos, NativeArray<float3> matrix, NativeArray<float> v, NativeArray<float> flux)
    {
        // Implementation for solving the conjugate gradient method
        int numCells = flux.Length;
        var r = new NativeArray<float>(numCells, Allocator.Temp);
        var p = new NativeArray<float>(numCells, Allocator.Temp);
        flux.CopyTo(r);
        r.CopyTo(p);
        var Ap = new NativeArray<float>(numCells, Allocator.Temp);
        float pAp, rsNew, rsOld = 0;

        rsOld = Dot(r, r);

        var msg = "CG init with rs" + rsOld + "\n";

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Apply Laplace
            ApplyLaplace(infos, matrix, p, Ap);

            pAp = Dot(p, Ap);

            float alpha = rsOld / pAp;
            for (var i = 0; i < numCells; i++)
            {
                v[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            rsNew = Dot(r, r);

            msg += $"iter{iter + 1} \trsNew:{rsNew}\n";
            if (rsNew < 1e-6f) break;

            float beta = rsNew / rsOld;
            for (int i = 0; i < numCells; i++)
                p[i] = r[i] + beta * p[i];

            rsOld = rsNew;
        }
        
        Debug.Log(msg);
    }

    private float Dot(NativeArray<float> a, NativeArray<float> b)
    {
        float dot = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
        }

        return dot;
    }

    private void ApplyLaplace(NativeArray<int2> infos, NativeArray<float3> matrix, NativeArray<float> p, NativeArray<float> Ap)
    {
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int2 info = infos[i];
            int level = info.x, offset = info.y, width = BlockWidth(level);

            int haloWidth = width + 2;
            var temp =  new NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);
            var param = new NativeArray<float3>(haloWidth * haloWidth, Allocator.Temp);

            // fill halo block
            FillHaloBlock(p, matrix, infos, temp, param, new int2(x, y));
                
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

    private void SolveMG(int maxIter, NativeArray<int2> infos, NativeArray<float3> matrix, NativeArray<float> v, NativeArray<float> flux)
    {
        var ifs = new[]
            { infos, new NativeArray<int2>(GridWidth * GridWidth, Allocator.Persistent), new NativeArray<int2>(GridWidth * GridWidth, Allocator.Persistent), new NativeArray<int2>(GridWidth * GridWidth, Allocator.Persistent) };
        int ptr1 = 0, ptr2 = 0, ptr3 = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int l0 = infos[i].x;
            int l1 = math.max(1, l0);
            int l2 = math.max(2, l1);
            int l3 = math.max(3, l2);
            ifs[1][i] =new int2(l1, ptr1);
            ifs[2][i] = new int2(l2, ptr2);
            ifs[3][i] = new int2(l3, ptr3);
            ptr1 += BlockWidth(l1) * BlockWidth(l1);
            ptr2 += BlockWidth(l2) * BlockWidth(l2);
            ptr3 += BlockWidth(l3) * BlockWidth(l3);
        }

        var As = new[] { matrix, new NativeArray<float3>(ptr1, Allocator.TempJob), new NativeArray<float3>(ptr2, Allocator.TempJob), new NativeArray<float3>(ptr3, Allocator.TempJob) };
        var xs = new[] { v, new NativeArray<float>(ptr1, Allocator.TempJob), new NativeArray<float>(ptr2, Allocator.TempJob), new NativeArray<float>(ptr3, Allocator.TempJob) };
        var bs = new[] { flux, new NativeArray<float>(ptr1, Allocator.TempJob), new NativeArray<float>(ptr2, Allocator.TempJob), new NativeArray<float>(ptr3, Allocator.TempJob) };

        var msg = ("init with residual: " + Residual(ifs[0], As[0], xs[0], bs[0]));
        for (int i = 0; i < maxIter; i++)
        {
            MultiGridVCycle(ifs, As, xs, bs);
            msg += "\niter " + (i + 1) + ", residual: " + Residual(ifs[0], As[0], xs[0], bs[0]);
        }
        Debug.Log(msg);
    }
    private void SolveMGPCG(int maxIter, NativeArray<int2> infos, NativeArray<float3> matrix, NativeArray<float> v, NativeArray<float> flux)
    {
        var ifs = new[]
            { infos, new NativeArray<int2>(GridWidth * GridWidth, Allocator.Persistent), new NativeArray<int2>(GridWidth * GridWidth, Allocator.Persistent), new NativeArray<int2>(GridWidth * GridWidth, Allocator.Persistent) };
        int ptr1 = 0, ptr2 = 0, ptr3 = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int l0 = infos[i].x;
            int l1 = math.max(1, l0);
            int l2 = math.max(2, l1);
            int l3 = math.max(3, l2);
            ifs[1][i] =new int2(l1, ptr1);
            ifs[2][i] = new int2(l2, ptr2);
            ifs[3][i] = new int2(l3, ptr3);
            ptr1 += BlockWidth(l1) * BlockWidth(l1);
            ptr2 += BlockWidth(l2) * BlockWidth(l2);
            ptr3 += BlockWidth(l3) * BlockWidth(l3);
        }
        int numCells = flux.Length;
        var r =  new NativeArray<float>(numCells, Allocator.TempJob);
        var z =  new NativeArray<float>(numCells, Allocator.TempJob);
        var p =  new NativeArray<float>(numCells, Allocator.TempJob);
        var Ap = new NativeArray<float>(numCells, Allocator.TempJob);
        flux.CopyTo(r);

        var As = new[] { matrix, new NativeArray<float3>(ptr1, Allocator.TempJob), new NativeArray<float3>(ptr2, Allocator.TempJob), new NativeArray<float3>(ptr3, Allocator.TempJob) };
        var xs = new[] { z, new NativeArray<float>(ptr1, Allocator.TempJob), new NativeArray<float>(ptr2, Allocator.TempJob), new NativeArray<float>(ptr3, Allocator.TempJob) };
        var bs = new[] { r, new NativeArray<float>(ptr1, Allocator.TempJob), new NativeArray<float>(ptr2, Allocator.TempJob), new NativeArray<float>(ptr3, Allocator.TempJob) };

        MultiGridVCycle(ifs, As, xs, bs);
        xs[0].CopyTo(p);
        
        float pAp, rzNew, rzOld = 0;
        for (var i = 0; i < numCells; i++)
            rzOld += r[i] * z[i];

        var msg = ("begin with residual: " + Residual(ifs[0],  As[0], xs[0], bs[0]));
        
        for (int iter = 0; iter < maxIter; iter++)
        {
            ApplyLaplace(infos, matrix, p, Ap);

            pAp = 0;
            for (var i = 0; i < numCells; i++)
                pAp += p[i] * Ap[i];

            float alpha = rzOld / pAp;
            for (var i = 0; i < numCells; i++)
            {
                v[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }
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

            float rs = 0;
            for (var i = 0; i < numCells; i++)
                rs += r[i] * r[i];
            msg += $"\niter{iter + 1}: \trsNew:{rs}";
            if (rs < 1e-6f || iter >= maxIter) break;
            
            for (var i = 0; i < numCells; i++)
                z[i] = 0;
            MultiGridVCycle(ifs, As, xs, bs);

            rzNew = 0;
            for (var i = 0; i < numCells; i++)
                rzNew += r[i] * z[i];
            msg += $"\t rzNew:{rzNew}";

            float beta = rzNew / rzOld;
            for (int i = 0; i < numCells; i++)
                p[i] = z[i] + beta * p[i];

            rzOld = rzNew;
        }
        
        Debug.Log(msg);
    }

    private void MultiGridVCycle(NativeArray<int2>[] infos, NativeArray<float3>[] As, NativeArray<float>[] xs, NativeArray<float>[] bs)
    {
        int smoothIter = 3;
        
        for (int i = 0; i < 3; i++)
        {
            for (int iter = 0; iter < smoothIter; iter++)
                GaussSeidelPhase(i, infos[i], As[i], xs[i], bs[i], true);
            
            Restriction(bs[i], As[i], xs[i], infos[i], bs[i + 1], As[i + 1], xs[i + 1], infos[i + 1]);
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
            Prolongation(xs[i+1], infos[i+1], xs[i], infos[i]);
            for (int iter = 0; iter < smoothIter; iter++)
                GaussSeidelPhase(i, infos[i], As[i], xs[i], bs[i], false);
        }
    }

    private void GaussSeidelPhase(int targetLevel, NativeArray<int2> infos, NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, bool red_black)
    {
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = infos[i].x;
            if (level != targetLevel) continue;
            int offset = infos[i].y, width = BlockWidth(level);
                
            int haloWidth = width + 2;
            var blockB = new NativeArray<float> (width * width, Allocator.Temp);
            var blockV = new NativeArray<float> (haloWidth * haloWidth, Allocator.Temp);
            var blockA = new NativeArray<float3>(haloWidth * haloWidth, Allocator.Temp);

            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int localId = BlockCoord2Idx(xx, yy, width);
                blockB[localId] = b[offset + localId];
            }
            
            // fill halo block
            FillHaloBlock(v, a, infos, blockV, blockA, new int2(x, y));
            
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
            int level = infos[i].x;
            if (level != targetLevel) continue;
            int offset = infos[i].y, width = BlockWidth(level);
                
            int haloWidth = width + 2;
            var blockB = new NativeArray<float>(width * width, Allocator.Temp);
            var blockV = new NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);
            var blockA = new NativeArray<float3>(haloWidth * haloWidth, Allocator.Temp);

            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int localId = BlockCoord2Idx(xx, yy, width);
                blockB[localId] = b[offset + localId];
            }
            
            // fill halo block
            FillHaloBlock(v, a, infos, blockV, blockA, new int2(x, y));
            
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

    private void Restriction(NativeArray<float> bf, NativeArray<float3> af, NativeArray<float> vf, NativeArray<int2> infof,
        NativeArray<float> rc, NativeArray<float3> ac, NativeArray<float> vc, NativeArray<int2> infoc)
    {
        for (int gy = 0; gy < GridWidth; gy++)
        for (int gx = 0; gx < GridWidth; gx++)
        {
            int gid = Coord2Idx(gx, gy);
            int levelF = infof[gid].x;
            int phf = infof[gid].y;
            int levelC = infoc[gid].x;
            int phc = infoc[gid].y;
            
            int blockWidthF = BlockWidth(levelF);
            int haloWidth = blockWidthF + 2;
            var blockV = new NativeArray<float >(haloWidth * haloWidth, Allocator.Temp);
            var blockA = new NativeArray<float3>(haloWidth * haloWidth, Allocator.Temp);
            var blockR = new NativeArray<float >(blockWidthF * blockWidthF, Allocator.Temp);
            FillHaloBlock(vf, af, infof, blockV, blockA, new int2(gx, gy));
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
                    if (x == 0 && gx > 0 && infof[Coord2Idx(gx - 1, gy)].x < infoc[Coord2Idx(gx - 1, gy)].x) aFine.y *= 2;
                    if (y == 0 && gy > 0 && infof[Coord2Idx(gx, gy - 1)].x < infoc[Coord2Idx(gx, gy - 1)].x) aFine.z *= 2;
                    float rFine = blockR[BlockCoord2Idx(x, y, blockWidthF)];
                    rc[ci] = rFine;
                    ac[ci] = aFine;
                    vc[ci] = 0;
                }
            }
        }
    }
    
    private void Prolongation(NativeArray<float> vc, NativeArray<int2> infoc, NativeArray<float> vf, NativeArray<int2> infof)
    {
        for (int gy = 0; gy < GridWidth; gy++)
        for (int gx = 0; gx < GridWidth; gx++)
        {
            int gid = Coord2Idx(gx, gy);
            int levelF = infof[gid].x;
            int phf = infof[gid].y;
            int levelC = infoc[gid].x;
            int phc = infoc[gid].y;

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
    
    private float Residual(NativeArray<int2> infos, NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b)
    {
        float r = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int level = infos[i].x, offset = infos[i].y, width = BlockWidth(level);
                
            int haloWidth = width + 2;
            var blockV = new NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);
            var blockA = new NativeArray<float3>(haloWidth * haloWidth, Allocator.Temp);

            // fill halo block
            FillHaloBlock(v, a, infos, blockV, blockA, new int2(x, y));
            
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
    
    private void ApplyPressure(NativeArray<int2> infos, NativeArray<float> pressure, NativeArray<float2> velocity)
    {
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y), level = infos[i].x, offset = infos[i].y, width = BlockWidth(level);
            
            int haloWidth = width + 2;
            var temp = new  NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);
            var param = new NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);

            // fill halo block
            FillHaloBlock(pressure, infos, temp, param, new int2(x, y));
            
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

    private static void FillHaloBlock(NativeArray<float2> v, NativeArray<int2> infos, NativeArray<float2> block, int2 coord)
    {
        int2 info = infos[Coord2Idx(coord)];
        int level = info.x;
        int ptr = info.y;
        int blockWidth = BlockWidth(level);
        int haloBlockWidth = blockWidth + 2;
        for (int by = 0; by < blockWidth; by++)
        for (int bx = 0; bx < blockWidth; bx++)
        {
            int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
            int physicsIdx = infos[Coord2Idx(coord)].y + BlockCoord2Idx(bx, by, blockWidth);
                
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
            
            int2 nInfo = infos[Coord2Idx(curr)];
            int nLevel = nInfo.x;
            if (nLevel < 0)
                continue;
            
            int phn = nInfo.y;
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
    private static void FillHaloBlock(NativeArray<float> v, NativeArray<int2> infos, NativeArray<float> blockV, NativeArray<float> blockP, int2 coord)
    {
        int level = infos[Coord2Idx(coord)].x;
        int width = BlockWidth(level);
        int haloWidth = width + 2;
        int offset = infos[Coord2Idx(coord)].y;
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

            int2 nInfo = infos[BlockCoord2Idx(curr, GridWidth)];
            int nLevel = nInfo.x;
            int phn = nInfo.y;
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
    
    private static void FillHaloBlock(NativeArray<float> v, NativeArray<float3> p, NativeArray<int2> infos, NativeArray<float> blockV, NativeArray<float3> blockP, int2 coord)
    {
        int2 info = infos[Coord2Idx(coord)];
        int level = info.x;
        int width = BlockWidth(level);
        int haloWidth = width + 2;
        int offset = info.y;
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

            int2 nInfo = infos[BlockCoord2Idx(curr, GridWidth)];
            int nLevel = nInfo.x;
            int phn = nInfo.y;
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

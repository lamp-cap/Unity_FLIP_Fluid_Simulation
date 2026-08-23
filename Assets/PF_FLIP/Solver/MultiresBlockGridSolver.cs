using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class MultiresBlockGridSolver : System.IDisposable
{
    private const uint SOLID = 2;
    private const uint AIR = 1;
    private const uint FLUID = 0;
    
    public NativeArray<float2> GridVelocityAlt;
    public NativeArray<float2> GridVelocityAltDS;
    public NativeArray<float> _pressure;

    private NativeArray<int2>[] _infoPymaid;
    private NativeArray<float3>[] _coefPymaid;
    private NativeArray<float>[] _xPymaid;
    private NativeArray<float>[] _bPymaid;
    
    private NativeArray<float> _r;
    private NativeArray<float> _z;
    private NativeArray<float> _p;
    private NativeArray<float> _Ap;
    
    private NativeReference<float> _pAp;
    private NativeReference<float> _rzOld;
    private NativeReference<float> _rzNew;
    private NativeReference<float> _temp;
    
    private NativeReference<int> _cellCount;
    
    public NativeArray<int2> GridInfos;
    public NativeArray<int2> GridInfosOld;
    public NativeArray<float3> GridLaplacian;
    public NativeArray<float2> GridVelocity;
    public NativeArray<float2> GridVelocityDS;
    public NativeArray<float> Flux;
    public NativeArray<float> SDF;
    public NativeArray<int> BlockLevel;
    public NativeArray<uint> GridTypes;
    
    public const int GridWidth = 16;
    public const int BlockCount = GridWidth * GridWidth;
    public const int BaseBlockSize = 8;
    public const int BaseLevelWidth = GridWidth * BaseBlockSize;
    public const int BaseCellSize = 1;

    private const int LevelCount = 5;

    public int CellCount => _cellCount.Value;
    
    public MultiresBlockGridSolver()
    {
        int[] poolSize = new int[LevelCount] { BlockCount * 64, BlockCount * 16, BlockCount * 4, BlockCount, BlockCount / 4 };

        _infoPymaid = new NativeArray<int2>[LevelCount];
        _coefPymaid = new NativeArray<float3>[LevelCount];
        _xPymaid = new NativeArray<float>[LevelCount];
        _bPymaid = new NativeArray<float>[LevelCount];
        
        for (int i = 0; i < LevelCount; i++)
        {
            int count = poolSize[i];
            _coefPymaid[i] = new NativeArray<float3>(count, Allocator.Persistent);
            _xPymaid[i] = new NativeArray<float>(count, Allocator.Persistent);
            _bPymaid[i] = new NativeArray<float>(count, Allocator.Persistent);
            _infoPymaid[i] = new NativeArray<int2>(GridWidth * GridWidth, Allocator.Persistent);
        }

        GridInfos = _infoPymaid[0];
        GridLaplacian = _coefPymaid[0];
        GridInfosOld = new NativeArray<int2>(GridWidth * GridWidth, Allocator.Persistent);
        
        int numCells = poolSize[0];
        _r = _bPymaid[0];
        _z = _xPymaid[0];
        _p = new NativeArray<float>(numCells, Allocator.Persistent);
        _Ap = new NativeArray<float>(numCells, Allocator.Persistent);
        GridVelocity = new NativeArray<float2>(numCells, Allocator.Persistent);
        GridVelocityAlt = new NativeArray<float2>(numCells, Allocator.Persistent);
        GridVelocityDS = new NativeArray<float2>(numCells, Allocator.Persistent);
        GridVelocityAltDS = new NativeArray<float2>(numCells, Allocator.Persistent);
        _pressure = new NativeArray<float>(numCells, Allocator.Persistent);
        Flux = new NativeArray<float>(numCells, Allocator.Persistent);
        SDF = new NativeArray<float>(numCells, Allocator.Persistent);
        GridTypes = new NativeArray<uint>(numCells, Allocator.Persistent);
        BlockLevel = new NativeArray<int>(BlockCount, Allocator.Persistent);

        _pAp = new NativeReference<float>(Allocator.Persistent);
        _rzOld = new NativeReference<float>(Allocator.Persistent);
        _rzNew = new NativeReference<float>(Allocator.Persistent);
        _temp = new NativeReference<float>(Allocator.Persistent);
        _cellCount = new NativeReference<int>(Allocator.Persistent);
    }

    public void Dispose()
    {
        for (int i = 0; i < LevelCount; i++)
        {
            _coefPymaid[i].Dispose();
            _xPymaid[i]   .Dispose();
            _bPymaid[i]   .Dispose();
            _infoPymaid[i].Dispose();
        }
        _p .Dispose();
        _Ap.Dispose();
        GridVelocityAlt.Dispose();
        GridVelocity.Dispose();
        GridVelocityDS.Dispose();
        GridVelocityAltDS.Dispose();
        _pressure.Dispose();
        Flux.Dispose();
        SDF.Dispose();
        GridTypes.Dispose();
        BlockLevel.Dispose();
        GridInfosOld.Dispose();
        
        _pAp .Dispose();
        _rzOld.Dispose();
        _rzNew.Dispose();
        _temp.Dispose();
        _cellCount.Dispose();
    }
    
    public void Solve()
    {
        CalcFlux();
        
        SolveMGPCG(4, GridInfos, GridLaplacian, _pressure, Flux);

        ApplyPressure(GridInfos, _pressure, GridVelocity);
    }

    public void SwapVelocity()
    {
        (GridVelocity, GridVelocityAlt) = (GridVelocityAlt, GridVelocity);
    }

    public void SetCellCount(int cellCount)
    {
        _cellCount.Value = cellCount;
    }

    public void SolveCG(int maxIter)
    {
        SolveCG(maxIter, GridInfos, GridLaplacian, _pressure, Flux);
    }

    private void SolveCG(int maxIter, NativeArray<int2> infos, NativeArray<float3> matrix, NativeArray<float> v, NativeArray<float> b)
    {
        var r = _r;
        var p = _p;
        b.CopyTo(r);
        r.CopyTo(p);
        int numCells = _cellCount.Value;
        var Ap = _Ap;
        float rs = Dot(r, r, _rzOld);

        var msg = "CG init with rs: " + rs + "\n";

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Apply Laplace
            ApplyLaplace(infos, matrix, p, Ap);

            Dot(p, Ap, _pAp);

            UpdateVR(p, Ap, v, r, _rzOld, _pAp);
            // if (iter % 20 == 19)
            // {
            //     // 保持残差和常数向量正交（只对 INTERIOR 做）
            //     float mean = 0;
            //     for (int i = 0; i < numCells; i++) mean += r[i];
            //     mean /= numCells;
            //     for (int i = 0; i < numCells; i++) r[i] -= mean;
            //     float meanP = 0;
            //     for (int i = 0; i < numCells; i++) meanP += p[i];
            //     meanP /= numCells;
            //     for (int i = 0; i < numCells; i++) p[i] -= meanP;
            // }

            rs = Dot(r, r, _rzNew);

            msg += $"iter {iter + 1} \trs: {rs}\n";
            if (rs < 1e-6f) break;
            
            UpdateP(r, p, _rzNew, _rzOld);

            (_rzOld, _rzNew) = (_rzNew, _rzOld);
        }
        
        Debug.Log(msg + "Residual:" + Residual(infos, matrix, v, b));
    }


    public void SolveMG(int maxIter)
    {
        SolveMG(maxIter, GridInfos, GridLaplacian, _pressure, Flux);
    }
    
    private void SolveMG(int maxIter, NativeArray<int2> infos, NativeArray<float3> matrix, NativeArray<float> v, NativeArray<float> flux)
    {
        var ifs = _infoPymaid;
        int ptr1 = 0, ptr2 = 0, ptr3 = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int l0 = infos[i].x;
            int l1 = l0 < 0 ? -1 : math.max(1, l0);
            int l2 = l0 < 0 ? -1 : math.max(2, l1);
            int l3 = l0 < 0 ? -1 : math.max(3, l2);
            ifs[1][i] = new int2(l1, ptr1);
            ifs[2][i] = new int2(l2, ptr2);
            ifs[3][i] = new int2(l3, ptr3);
            ptr1 += l1 < 0 ? 0 : BlockWidth(l1) * BlockWidth(l1);
            ptr2 += l2 < 0 ? 0 : BlockWidth(l2) * BlockWidth(l2);
            ptr3 += l3 < 0 ? 0 : BlockWidth(l3) * BlockWidth(l3);
        }

        var As = _coefPymaid;
        var xs = _xPymaid;
        var bs = _bPymaid;
        
        flux.CopyTo(bs[0]);

        var msg = ("init with residual: " + Residual(ifs[0], As[0], xs[0], bs[0]));

        for (int i = 0; i < 3; i++)
            RestrictionCoefficients(As[i], ifs[i], As[i + 1], ifs[i + 1]);
        for (int i = 0; i < maxIter; i++)
        {
            MultiGridVCycle(ifs, As, xs, bs);
            msg += "\niter " + (i + 1) + ", residual: " + Residual(ifs[0], As[0], xs[0], bs[0]);
        }
        _pressure.CopyFrom(xs[0]);
        Debug.Log(msg + "\nResidual:" + Residual(infos, matrix, v, flux));
    }
    public void SolveMGPCG(int maxIter)
    {
        SolveMGPCG(maxIter, GridInfos, GridLaplacian, _pressure, Flux);
    }
    private void SolveMGPCG(int maxIter, NativeArray<int2> infos, NativeArray<float3> matrix, NativeArray<float> v, NativeArray<float> flux)
    {
        var ifs = _infoPymaid;
        int ptr1 = 0, ptr2 = 0, ptr3 = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int l0 = infos[i].x;
            int l1 = l0 < 0 ? -1 : math.max(1, l0);
            int l2 = l0 < 0 ? -1 : math.max(2, l1);
            int l3 = l0 < 0 ? -1 : math.max(3, l2);
            ifs[1][i] = new int2(l1, ptr1);
            ifs[2][i] = new int2(l2, ptr2);
            ifs[3][i] = new int2(l3, ptr3);
            ptr1 += l1 < 0 ? 0 : BlockWidth(l1) * BlockWidth(l1);
            ptr2 += l2 < 0 ? 0 : BlockWidth(l2) * BlockWidth(l2);
            ptr3 += l3 < 0 ? 0 : BlockWidth(l3) * BlockWidth(l3);
        }

        var As = _coefPymaid;
        var xs = _xPymaid;
        var bs = _bPymaid;
        
        flux.CopyTo(_r);

        var msg = ("init with residual: " + Residual(ifs[0], As[0], xs[0], bs[0]));
        int numCells = _cellCount.Value; // active cells only — the pool tail must stay 0
        var r =  _r;
        var z =  _z;
        var p =  _p;
        var Ap = _Ap;
        
        for (var i = 0; i < numCells; i++)
        {
            v[i] = 0;
            z[i] = 0;
        }

        for (int i = 0; i < 3; i++)
            RestrictionCoefficients(As[i], ifs[i], As[i + 1], ifs[i + 1]);
        
        MultiGridVCycle(ifs, As, xs, bs);
        z.CopyTo(p);
        
        float rzOld = Dot(r, z, _rzOld);

        msg += ("\nbegin with residual: " + Residual(ifs[0],  As[0], xs[0], bs[0]) + " rs: " + rzOld);
        
        for (int iter = 0; iter < maxIter; iter++)
        {
            ApplyLaplace(infos, matrix, p, Ap);
            
            Dot(p, Ap, _pAp);

            UpdateVR(p, Ap, v, r, _rzOld, _pAp);

            // if ((iter & 1) == 1)
            // {
            //     // 保持残差和常数向量正交（只对 INTERIOR 做）
            //     float mean = 0;
            //     for (int i = 0; i < numCells; i++) mean += r[i];
            //     mean /= numCells;
            //     for (int i = 0; i < numCells; i++) r[i] -= mean;
            //
            //     // 如果 r 均值归零，p 也应该和常数正交
            //     // 因为 p 是 r 的线性组合
            //     float meanP = 0;
            //     for (int i = 0; i < numCells; i++) meanP += p[i];
            //     meanP /= numCells;
            //     for (int i = 0; i < numCells; i++) p[i] -= meanP;
            // }

            float rs = 0;
            for (var i = 0; i < numCells; i++)
                rs += r[i] * r[i];
            msg += $"\niter{iter + 1}: \trs:{rs}";
            if (rs < 1e-6f || iter >= maxIter) break;
            
            for (var i = 0; i < numCells; i++)
                z[i] = 0;
            MultiGridVCycle(ifs, As, xs, bs);

            rzOld = Dot(r, z, _rzNew);

            msg += $" \trsNew: {rs}\n";
            if (rzOld < 1e-6f) break;
            
            UpdateP(z, p, _rzNew, _rzOld);

            (_rzOld, _rzNew) = (_rzNew, _rzOld);
        }
        
        Debug.Log(msg + "\nResidual:" + Residual(infos, matrix, v, flux));
    }

    // Applies ONE multigrid V-cycle as a linear preconditioner: result = M^{-1} * rhs,
    // starting from a zero initial guess so the mapping is strictly linear.
    // Use this to check preconditioner symmetry — Krylov solvers (CG/MGPCG) are NOT
    // linear in rhs, so testing symmetry through SolveMGPCG is meaningless.
    public void ApplyVCycle(NativeArray<float> rhs, NativeArray<float> result)
    {
        var ifs = _infoPymaid;
        int ptr1 = 0, ptr2 = 0, ptr3 = 0;
        for (int y = 0; y < GridWidth; y++)
        for (int x = 0; x < GridWidth; x++)
        {
            int i = Coord2Idx(x, y);
            int l0 = ifs[0][i].x;
            int l1 = l0 < 0 ? -1 : math.max(1, l0);
            int l2 = l0 < 0 ? -1 : math.max(2, l1);
            int l3 = l0 < 0 ? -1 : math.max(3, l2);
            ifs[1][i] = new int2(l1, ptr1);
            ifs[2][i] = new int2(l2, ptr2);
            ifs[3][i] = new int2(l3, ptr3);
            ptr1 += l1 < 0 ? 0 : BlockWidth(l1) * BlockWidth(l1);
            ptr2 += l2 < 0 ? 0 : BlockWidth(l2) * BlockWidth(l2);
            ptr3 += l3 < 0 ? 0 : BlockWidth(l3) * BlockWidth(l3);
        }

        var As = _coefPymaid;
        var xs = _xPymaid;
        var bs = _bPymaid;

        for (int i = 0; i < 3; i++)
            RestrictionCoefficients(As[i], ifs[i], As[i + 1], ifs[i + 1]);

        // rhs -> bs[0] (= _r); zero the solution (= _z) so M is applied from x = 0.
        rhs.CopyTo(bs[0]);
        var z = xs[0];
        for (int i = 0; i < z.Length; i++) z[i] = 0;

        MultiGridVCycle(ifs, As, xs, bs);

        xs[0].CopyTo(result);
    }

    private void MultiGridVCycle(NativeArray<int2>[] infos, NativeArray<float3>[] As, NativeArray<float>[] xs, NativeArray<float>[] bs)
    {
        int smoothIter = 4;
        
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
        for (int iter = 0; iter < 32; iter++)
        {
            for (int y = 0; y < GridWidth; y++)
            for (int x = 0; x < GridWidth; x++)
            {
                int i = Coord2Idx(x, y);
                int level = infos[3][i].x;
                if (level < 0) continue;
                float3 ac = As[3][i];
                if (ac.x < 1e-5f) continue;
                float ar = x < GridWidth - 1 ? As[3][Coord2Idx(x + 1, y)].y : 0;
                float at = y < GridWidth - 1 ? As[3][Coord2Idx(x, y + 1)].z : 0;
                float xl = x > 0 ? xs[3][Coord2Idx(x - 1, y)] : 0;
                float xr = x < GridWidth - 1 ? xs[3][Coord2Idx(x + 1, y)] : 0;
                float xb = y > 0 ? xs[3][Coord2Idx(x, y - 1)] : 0;
                float xt = y < GridWidth - 1 ? xs[3][Coord2Idx(x, y + 1)] : 0;
                float xc = xs[3][Coord2Idx(x, y)];
                xs[3][i] = math.lerp(xc, (bs[3][i] - (xl * ac.y + xr * ar + xb * ac.z + xt * at)) / ac.x, 1.3f);
            }

            for (int y = GridWidth - 1; y >= 0; y--)
            for (int x = GridWidth - 1; x >= 0; x--)
            {
                int i = Coord2Idx(x, y);
                int level = infos[3][i].x;
                if (level < 0) continue;
                float3 ac = As[3][i];
                if (ac.x < 1e-5f) continue;
                float ar = x < GridWidth - 1 ? As[3][Coord2Idx(x + 1, y)].y : 0;
                float at = y < GridWidth - 1 ? As[3][Coord2Idx(x, y + 1)].z : 0;
                float xl = x > 0 ? xs[3][Coord2Idx(x - 1, y)] : 0;
                float xr = x < GridWidth - 1 ? xs[3][Coord2Idx(x + 1, y)] : 0;
                float xb = y > 0 ? xs[3][Coord2Idx(x, y - 1)] : 0;
                float xt = y < GridWidth - 1 ? xs[3][Coord2Idx(x, y + 1)] : 0;
                float xc = xs[3][Coord2Idx(x, y)];
                xs[3][i] = math.lerp(xc, (bs[3][i] - (xl * ac.y + xr * ar + xb * ac.z + xt * at)) / ac.x, 1.3f);
            }
        }

        for (int i = 2; i >= 0; i--)
        {
            Prolongation(xs[i+1], infos[i+1], xs[i], infos[i]);
            for (int iter = 0; iter < smoothIter; iter++)
                GaussSeidelPhase(i, infos[i], As[i], xs[i], bs[i], false);
        }
    }
    
    public void FillMatrix(NativeArray<float3> matrix, NativeArray<int2> gridLut)
    {
        new BuildCoefficientJob()
        {
            GridLut = gridLut,
            Matrix = matrix,
            GridTypes = GridTypes,
        }.Schedule(BlockCount, 1).Complete();
    }
    
    private struct BuildCoefficientJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int2> GridLut;
        [ReadOnly] public NativeArray<uint> GridTypes;
        [NativeDisableParallelForRestriction, WriteOnly] public NativeArray<float3> Matrix;
        
        public void Execute(int i)
        {
            int2 coord = Idx2Coord(i);
            int2 info = GridLut[i];
            int level = info.x;
            if (level < 0) return;
            int ptr = info.y;
            int width = BlockWidth(level);
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int idx = ptr + BlockCoord2Idx(xx, yy, width);
                uint gridType = GridTypes[idx]; 
                float3 coefs = float3.zero;
                if (IsFluidCell(gridType))
                {
                    int4 ox = new int4(-1, 1, 0, 0);
                    int4 oy = new int4(0, 0, -1, 1);
                    float4 ps = float4.zero;
                    float sum = 0;
                    var neighborTypes = NeighborGridTypes(gridType);
                    for (int n = 0; n < 4; n++)
                    {
                        uint nType = neighborTypes[n];
                        if (IsSolidCell(nType)) continue;
                        bool fluidCell = IsFluidCell(nType);
                        int levelN = GetLevel(coord, new int2(xx + ox[n], yy + oy[n]));
                        if (levelN >= 0)
                        {
                            if (levelN == level)
                            {
                                ps[n] = fluidCell?1f:0f;
                                sum += 1f;
                            }
                            else
                            {
                                ps[n] = fluidCell ? 2f / 3f : 0f;
                                if (levelN < level) sum += 4f / 3f; // two neighbors
                                else sum += 2f / 3f;
                            }
                        }
                    }

                    coefs = new float3(sum, -ps.x, -ps.z);
                }
                Matrix[idx] = coefs; // center, left, down
            }
        }
        
        int GetLevel(int2 block, int2 local)
        {
            int2 info = GridLut[Coord2Idx(block)];
            int level = info.x;
            int width = BlockWidth(level);
            if (local.x >= 0 && local.x < width && local.y >= 0 && local.y < width) return level;
            if (local.x < 0) block.x -= 1;
            else if (local.x >= width) block.x += 1;
            
            if (local.y < 0) block.y -= 1;
            else if (local.y >= width) block.y += 1;
            
            if (block.x < 0 || block.y < 0 || block.x >= GridWidth || block.y >= GridWidth)
                return -2;
            return GridLut[Coord2Idx(block)].x;
        }
    }

    public int InitBaseCells(int4 bounds)
    {
        new ComputeBlockSDFJob(BlockLevel).Run();
        new AllocateCellsJob(BlockLevel, GridInfos, _cellCount).Run();
        GridInfos.CopyTo(GridInfosOld);
        new InitGridTypesJob(GridInfos, GridTypes, bounds).Schedule(BlockCount, 1).Complete();

        new ClearCellsJob()
        {
            f = _r,
            p = Flux,
        }.Schedule(_cellCount.Value, 64).Complete();
        return _cellCount.Value;
    }

    public void IterateCellSDF()
    {
        for (int i = 0; i < 4; i++)
        {
            new ComputeCellSDFJob(GridInfos, SDF).Schedule(BlockCount, 1).Complete();
        }
        
        new BuildCoefficientJob()
        {
            GridLut = GridInfos,
            Matrix = GridLaplacian,
            GridTypes = GridTypes,
        }.Schedule(BlockCount, 1).Complete();
    }

    public int AllocateBaseCells()
    {
        new ComputeBlockSDFJob(BlockLevel).Run();
        GridInfos.CopyTo(GridInfosOld);
        new AllocateCellsJob(BlockLevel, GridInfos, _cellCount).Run();

        return _cellCount.Value;
    }

    public void FillMatrix()
    {
        new BuildCoefficientJob()
        {
            GridLut = GridInfos,
            Matrix = GridLaplacian,
            GridTypes = GridTypes,
        }.Schedule(BlockCount, 1).Complete();
    }
    
    [BurstCompile]
    private struct ClearCellsJob : IJobParallelFor
    {
        [WriteOnly] public NativeArray<float> f;
        [WriteOnly] public NativeArray<float> p;
        
        public void Execute(int i)
        {
            f[i] = 0;
            p[i] = 0;
        }
        
        private static void FillHaloBlock(NativeArray<float> v, NativeArray<int2> infos, NativeArray<float> block, int2 coord)
        {
            int2 info = infos[Coord2Idx(coord)];
            int level = info.x; // must be 0
            int ptr = info.y;
            int blockWidth = BlockWidth(level);
            int haloBlockWidth = blockWidth + 2;
            for (int by = 0; by < blockWidth; by++)
            for (int bx = 0; bx < blockWidth; bx++)
            {
                int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
                int physicsIdx = ptr + BlockCoord2Idx(bx, by, blockWidth);
                    
                block[localIdx] = v[physicsIdx];
            }
            int4 ox = new int4(-1, 1, 0, 0);
            int4 oy = new int4(0, 0, -1, 1);
            
            for (int n = 0; n < 4; n++)
            {
                int2 dir = new int2(ox[n], oy[n]);
                int2 curr = coord + dir;
                if (curr.x < 0 || curr.y < 0 || curr.x >= GridWidth || curr.y >= GridWidth)
                    continue;
                
                int2 nInfo = infos[Coord2Idx(curr)];
                int nLevel = nInfo.x;
                if (nLevel != level)
                    continue;
                
                int phn = nInfo.y;
                for (int c = 0; c < blockWidth; c++)
                {
                    int2 nCoord = math.select(math.select(c, 0, dir > 0), blockWidth - 1, dir < 0);
                    int nLocalIdx = BlockCoord2Idx(nCoord, blockWidth);
                    int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                    block[paddingIdx] = v[phn + nLocalIdx];
                }
            }
        }
    }
    
    [BurstCompile]
    private struct ComputeCellSDFJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int2> GridLut;
        [NativeDisableParallelForRestriction] public NativeArray<float> SDF;

        public ComputeCellSDFJob(NativeArray<int2> lut, NativeArray<float> sdf)
        {
            GridLut = lut;
            SDF = sdf;
        }
        
        public void Execute(int i)
        {
            int2 coord = Idx2Coord(i);
            int2 info = GridLut[i];
            int level = info.x;
            if (level != 0) return;
            int ptr = info.y;
            int width = BlockWidth(level);
            int haloWidth = width + 2;
            NativeArray<float> haloBlock = new NativeArray<float>(haloWidth * haloWidth,
                Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int j = 0; j < haloWidth * haloWidth; j++)
                haloBlock[j] = -1;
            
            FillHaloBlock(SDF, GridLut, haloBlock, coord);
            for (int yy = 1; yy <= width; yy++)
            for (int xx = 1; xx <= width; xx++)
            {
                int idx = ptr + BlockCoord2Idx(xx - 1, yy - 1, width);
                float c = haloBlock[BlockCoord2Idx(xx, yy, haloWidth)];
                if (c < 0) continue;
                float l = haloBlock[BlockCoord2Idx(xx - 1, yy, haloWidth)];
                if (l >= 0) c = math.min(l + 1, c);
                float r = haloBlock[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                if (r >= 0) c = math.min(r + 1, c);
                float b = haloBlock[BlockCoord2Idx(xx, yy - 1, haloWidth)];
                if (b >= 0) c = math.min(b + 1, c);
                float t = haloBlock[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                if (t >= 0) c = math.min(t + 1, c);
                
                SDF[idx] = c;
            }

            haloBlock.Dispose();
        }
        
        private static void FillHaloBlock(NativeArray<float> v, NativeArray<int2> infos, NativeArray<float> block, int2 coord)
        {
            int2 info = infos[Coord2Idx(coord)];
            int level = info.x; // must be 0
            int ptr = info.y;
            int blockWidth = BlockWidth(level);
            int haloBlockWidth = blockWidth + 2;
            for (int by = 0; by < blockWidth; by++)
            for (int bx = 0; bx < blockWidth; bx++)
            {
                int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
                int physicsIdx = ptr + BlockCoord2Idx(bx, by, blockWidth);
                    
                block[localIdx] = v[physicsIdx];
            }
            int4 ox = new int4(-1, 1, 0, 0);
            int4 oy = new int4(0, 0, -1, 1);
            
            for (int n = 0; n < 4; n++)
            {
                int2 dir = new int2(ox[n], oy[n]);
                int2 curr = coord + dir;
                if (curr.x < 0 || curr.y < 0 || curr.x >= GridWidth || curr.y >= GridWidth)
                    continue;
                
                int2 nInfo = infos[Coord2Idx(curr)];
                int nLevel = nInfo.x;
                if (nLevel != level)
                    continue;
                
                int phn = nInfo.y;
                for (int c = 0; c < blockWidth; c++)
                {
                    int2 nCoord = math.select(math.select(c, 0, dir > 0), blockWidth - 1, dir < 0);
                    int nLocalIdx = BlockCoord2Idx(nCoord, blockWidth);
                    int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                    block[paddingIdx] = v[phn + nLocalIdx];
                }
            }
        }
    }
    
    [BurstCompile]
    private struct ComputeBlockSDFJob : IJob
    {
        private NativeArray<int> _blockLevel;
        
        public ComputeBlockSDFJob(NativeArray<int> level)
        {
            _blockLevel = level;
        }

        public void Execute()
        {
            int2 offset = new int2(1, 0);
            int rightBound = GridWidth - 1;
            for (int i = 0; i < BlockCount; i++)
            {
                var level = _blockLevel[i];
                if (level <= 0) continue;
                int2 coord = Idx2Coord(i);
                if (coord.x > 0)
                    level = math.min(level, 1 + _blockLevel[Coord2Idx(coord - offset.xy)]);
                if (coord.y > 0)
                    level = math.min(level, 1 + _blockLevel[Coord2Idx(coord - offset.yx)]);
                if (math.all(coord > 0))
                    level = math.min(level, 1 + _blockLevel[Coord2Idx(coord - offset.xx)]);
                if (coord.x < rightBound && coord.y > 0)
                    level = math.min(level, 1 + _blockLevel[Coord2Idx(coord - new int2(-1, 1))]);

                _blockLevel[i] = level;
            }
            
            for (int i = BlockCount - 1; i >= 0; i--)
            {
                var level = _blockLevel[i];
                if (level <= 0) continue;
                int2 coord = Idx2Coord(i);
                if (coord.x < rightBound)
                    level = math.min(level, 1 + _blockLevel[Coord2Idx(coord + offset.xy)]);
                if (coord.y < rightBound)
                    level = math.min(level, 1 + _blockLevel[Coord2Idx(coord + offset.yx)]);
                if (math.all(coord < rightBound))
                    level = math.min(level, 1 + _blockLevel[Coord2Idx(coord + offset.xx)]);
                if (coord.x > 0 && coord.y < rightBound)
                    level = math.min(level, 1 + _blockLevel[Coord2Idx(coord + new int2(-1, 1))]);

                _blockLevel[i] = level;
            }
        }
    }
    
    [BurstCompile]
    private struct AllocateCellsJob : IJob
    {
        private NativeArray<int> _blockLevel;
        private NativeArray<int2> _blockInfos;
        private NativeReference<int> _cellCount;
        
        public AllocateCellsJob(NativeArray<int> level, NativeArray<int2> infos, NativeReference<int> cellCount)
        {
            _blockLevel = level;
            _blockInfos = infos;
            _cellCount = cellCount;
        }

        public void Execute()
        {
            int counter = 0;
            for (int y = 0; y < GridWidth; y++)
            for (int x = 0; x < GridWidth; x++)
            {
                int i = Coord2Idx(x, y);
                int level = _blockLevel[i];

                _blockInfos[i] = new int2(level, counter);
                counter += level < 0 ? 0 : BlockWidth(level) * BlockWidth(level);
            }
            _cellCount.Value = counter;
        }
    }

    [BurstCompile]
    private struct InitGridTypesJob : IJobParallelFor
    {
        [ReadOnly] private NativeArray<int2> _gridLut;
        [NativeDisableParallelForRestriction, WriteOnly]
        private NativeArray<uint> _gridTypes;
        private int4 _bounds;
        
        public InitGridTypesJob(NativeArray<int2> lut, NativeArray<uint> types, int4 bounds)
        {
            _gridTypes = types;
            _gridLut = lut;
            _bounds = bounds;
        }

        public void Execute(int i)
        {
            int2 coord = Idx2Coord(i);
            int2 info = _gridLut[i];
            int level = info.x;
            if (level < 0) return;
            int ptr = info.y;
            int width = BlockWidth(level);
            int cellSize = 1 << level;
            for (int y = 0; y < width; y++)
            for (int x = 0; x < width; x++)
            {
                int2 cCoord = coord * 8 + new int2(x, y) * cellSize;
                int4 ox = new int4(-cellSize, cellSize, 0, 0);
                int4 oy = new int4(0, 0, -cellSize, cellSize);
                uint type = GetType(cCoord);
                for (int n = 0; n < 4; n++)
                    type |= (GetType(cCoord + new int2(ox[n], oy[n])) << ((n + 1) * 2));

                _gridTypes[ptr + BlockCoord2Idx(x, y, width)] = type;
            }
        }

        private uint GetType(int2 coord)
        {
            if (math.any(coord < 0) || math.any(coord >= BaseLevelWidth))
                return SOLID;
            if (math.all(coord >= _bounds.xy) && math.all(coord < _bounds.zw))
                return FLUID;
            return AIR;

        }
    }

    public void CalcFlux()
    {
        new CalcFluxJob()
        {
            GridLut = GridInfos, 
            GridVelocity = GridVelocity,
            GridFlux = Flux,
            GridTypes = GridTypes
        }.Schedule(BlockCount, 1).Complete();
    }

    [BurstCompile]
    private struct CalcFluxJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int2> GridLut;
        [ReadOnly] public NativeArray<float2> GridVelocity;
        [ReadOnly] public NativeArray<uint> GridTypes;
        // [ReadOnly] public NativeArray<float> GridDensity;

        [NativeDisableParallelForRestriction, WriteOnly]
        public NativeArray<float> GridFlux;

        public void Execute(int i)
        {
            int2 coord = Idx2Coord(i);
            int2 info = GridLut[i];
            int level = info.x;
            if (level < 0) return;
            int ptr = info.y;
            int width = BlockWidth(level);
            int haloWidth = width + 2;
            NativeArray<float2> haloBlock = new NativeArray<float2>(haloWidth * haloWidth, Allocator.Temp);
            FillHaloBlock(GridVelocity, GridLut, haloBlock, coord);
            float h = GetH(level);
            for (int yy = 1; yy <= width; yy++)
            for (int xx = 1; xx <= width; xx++)
            {
                int idx = ptr + BlockCoord2Idx(xx - 1, yy - 1, width);
                if (!IsFluidCell(GridTypes[idx])) continue;
                float2 vel = haloBlock[BlockCoord2Idx(xx, yy, haloWidth)];
                float un = haloBlock[BlockCoord2Idx(xx + 1, yy, haloWidth)].x;
                float vn = haloBlock[BlockCoord2Idx(xx, yy + 1, haloWidth)].y;
                
                GridFlux[idx] = (un - vel.x) * h + (vn - vel.y) * h;
            }

            haloBlock.Dispose();
        }
    }

    private float Dot(NativeArray<float> a, NativeArray<float> b, NativeReference<float> r)
    {
        new DotJob(a, b, r, _cellCount.Value).Schedule().Complete();

        return r.Value;
    }

    [BurstCompile]
    private struct DotJob : IJob
    {
        [ReadOnly] private NativeArray<float> _lhs;
        [ReadOnly] private NativeArray<float> _rhs;
        [WriteOnly] private NativeReference<float> _result;
        private readonly int _count;
            
        public DotJob(NativeArray<float> lhs, NativeArray<float> rhs, NativeReference<float> result, int count)
        {
            _lhs = lhs;
            _rhs = rhs;
            _result = result;
            _count = count;
        }
            
        public void Execute()
        {
            float sum = 0;
            for (int i = 0; i < _count; i++)
                sum += _lhs[i] * _rhs[i];
            _result.Value = sum;
        }
    }
    
    private void ApplyLaplace(NativeArray<int2> infos, NativeArray<float3> matrix, NativeArray<float> p, NativeArray<float> Ap)
    {
        new LaplaceJob(infos, matrix, p, Ap).Schedule(BlockCount, 1).Complete();
    }

    [BurstCompile]
    private struct LaplaceJob : IJobParallelFor
    {
        [ReadOnly] private NativeArray<int2> _lut;
        [ReadOnly] private NativeArray<float3> _a;
        [ReadOnly] private NativeArray<float> _v;
        [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float> _r;

        public LaplaceJob(NativeArray<int2> l, NativeArray<float3> a, NativeArray<float> v, NativeArray<float> r)
        {
            _a = a;
            _v = v;
            _lut = l;
            _r = r;
        }

        public void Execute(int i)
        {
            int2 info = _lut[i];
            int level = info.x;
            if (level < 0) return;
            int offset = info.y, width = BlockWidth(level);

            int haloWidth = width + 2;
            var blockV =  new NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);
            var blockA = new NativeArray<float3>(haloWidth * haloWidth, Allocator.Temp);

            // fill halo block
            FillHaloBlock(_v, _a, _lut, blockV, blockA, Idx2Coord(i));
                
            for (int y = 1; y <= width; y++)
            for (int x = 1; x <= width; x++)
            {
                float xc = blockV[BlockCoord2Idx(x, y, haloWidth)];
                float neighborSum = NeighborSum(blockV, blockA, out float ac, x, y, haloWidth);
                if (ac < 1e-5f) continue;
                int ii = offset + BlockCoord2Idx(x - 1, y - 1, width);
                _r[ii] = neighborSum + ac * xc;
            }
                
            blockA.Dispose();
            blockV.Dispose();
        }
    }

    private void UpdateVR(NativeArray<float> p, NativeArray<float> ap, NativeArray<float> v,
        NativeArray<float> r, NativeReference<float> rsOld, NativeReference<float> pAp)
    {
        new UpdateVRJob(p, ap, v, r, rsOld, pAp).Schedule(_cellCount.Value, 1).Complete();
    }
        
    [BurstCompile]
    private struct UpdateVRJob : IJobParallelFor
    {
        [ReadOnly] private NativeArray<float> _p;
        [ReadOnly] private NativeArray<float> _ap;
        [ReadOnly] private NativeReference<float> _rsOld;
        [ReadOnly] private NativeReference<float> _pAp;
        private NativeArray<float> _v;
        private NativeArray<float> _r;
        
        public UpdateVRJob(NativeArray<float> p, NativeArray<float> ap, NativeArray<float> v, 
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
        
    private void UpdateP(NativeArray<float> z, NativeArray<float> p, NativeReference<float> rsNew, NativeReference<float> rsOld)
    {
        new UpdatePJob(z, p, rsNew, rsOld).Schedule(_cellCount.Value, 1).Complete();
    }
    [BurstCompile]
    private struct UpdatePJob : IJobParallelFor
    {
        [ReadOnly] private NativeArray<float> _z;
        [ReadOnly] private NativeReference<float> _rsOld;
        [ReadOnly] private NativeReference<float> _rsNew;
        private NativeArray<float> _p;
            
        public UpdatePJob(NativeArray<float> z, NativeArray<float> p, NativeReference<float> rsNew, NativeReference<float> rsOld)
        {
            _z = z;
            _p = p;
            _rsOld = rsOld;
            _rsNew = rsNew;
        }
            
        public void Execute(int i)
        {
            float beta = _rsNew.Value / _rsOld.Value;
            _p[i] = _z[i] + beta * _p[i];
        }
    }
    
    private float Residual(NativeArray<int2> infos, NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b)
    {
        var r = new NativeArray<float>(b.Length, Allocator.TempJob);
        new ResidualJob(b, a, v, infos, r).Schedule(BlockCount, 1).Complete();

        new DotJob(r,r, _temp, _cellCount.Value).Run();
        r.Dispose();
        return _temp.Value;
    }
    
    [BurstCompile]
    private struct ResidualJob : IJobParallelFor
    {
        [ReadOnly] private NativeArray<int2> _lut;
        [ReadOnly] private NativeArray<float3> _a;
        [ReadOnly] private NativeArray<float> _f;
        [ReadOnly] private NativeArray<float> _v;
        [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float> _r;

        public ResidualJob(NativeArray<float> f, NativeArray<float3> a, NativeArray<float> v,
            NativeArray<int2> l, NativeArray<float> r)
        {
            _a = a;
            _f = f;
            _v = v;
            _lut = l;
            _r = r;
        }

        public void Execute(int i)
        {
            int2 info = _lut[i];
            int level = info.x;
            if (level < 0)
                return;
            
            int ph = info.y;
            
            int blockWidth = BlockWidth(level);
            int haloBlockWidth = blockWidth + 2;
            var haloBlockV = new NativeArray<float>(haloBlockWidth * haloBlockWidth, Allocator.Temp);
            var haloBlockA = new NativeArray<float3>(haloBlockWidth * haloBlockWidth, Allocator.Temp);
            
            FillHaloBlock(_v, _a, _lut, haloBlockV, haloBlockA, Idx2Coord(i));
            for (int y = 0; y < blockWidth; y++)
            for (int x = 0; x < blockWidth; x++)
            {
                int ii = ph + BlockCoord2Idx(x, y, blockWidth);

                float vFine = haloBlockV[BlockCoord2Idx(x + 1, y + 1, haloBlockWidth)];
                float neighborSum = NeighborSum(haloBlockV, haloBlockA, out float csum, x + 1, y + 1, haloBlockWidth);
                if (csum < 1e-5f) continue;
                float rFine = _f[ii] - (neighborSum + csum * vFine);
                _r[ii] = rFine;
            }
            haloBlockA.Dispose();
            haloBlockV.Dispose();
        }
    }
    
    private void GaussSeidelPhase(int targetLevel, NativeArray<int2> infos, NativeArray<float3> a, NativeArray<float> v, NativeArray<float> b, bool red_black)
    {
        new SmoothGaussSeidel(v, a, infos, b, red_black ? 0 : 1, targetLevel).Schedule(BlockCount, 1).Complete();
        new SmoothGaussSeidel(v, a, infos, b, red_black ? 1 : 0, targetLevel).Schedule(BlockCount, 1).Complete();
    }

    [BurstCompile]
    private struct SmoothGaussSeidel : IJobParallelFor
    {
        [ReadOnly] private NativeArray<int2> _lut;
        [ReadOnly] private NativeArray<float> _b;
        [ReadOnly] private NativeArray<float3> _a;
        [NativeDisableParallelForRestriction] private NativeArray<float> _v;
        private readonly int _phase;
        private readonly int _targetLevel;
        
        public SmoothGaussSeidel(NativeArray<float> v, NativeArray<float3> a, NativeArray<int2> lut, NativeArray<float> b, int phase, int targetLevel)
        {
            _v = v;
            _lut = lut; 
            _b = b;
            _a = a;
            _phase = phase;
            _targetLevel = targetLevel;
        }

        public void Execute(int i)
        {
            int2 info = _lut[i];
            int level = info.x;
            if (level != _targetLevel) return;
            int offset = info.y, width = BlockWidth(level);
                
            int haloWidth = width + 2;
            var blockV = new NativeArray<float> (haloWidth * haloWidth, Allocator.Temp);
            var blockA = new NativeArray<float3>(haloWidth * haloWidth, Allocator.Temp);
            
            // fill halo block
            FillHaloBlock(_v, _a, _lut, blockV, blockA, Idx2Coord(i));
            
            for (int yy = 1; yy <= width; yy++)
            for (int xx = 1; xx <= width; xx++)
            {
                if (((xx + yy) & 1) != _phase) continue;
                
                int localId = BlockCoord2Idx(xx - 1, yy - 1, width);
                
                float nsum = NeighborSum(blockV, blockA, out float diag, xx, yy,  haloWidth);
                var oldV = blockV[BlockCoord2Idx(xx, yy, haloWidth)];
                _v[offset + localId] = math.abs(diag) < 1e-5f ? 0 : math.lerp(oldV, (_b[offset + localId] - nsum) / diag, 1.3f);
            }
        }
    }
    private void Restriction(NativeArray<float> bf, NativeArray<float3> af, NativeArray<float> vf, NativeArray<int2> infof,
        NativeArray<float> rc, NativeArray<float3> ac, NativeArray<float> vc, NativeArray<int2> infoc)
    {
       new RestrictionJob(bf, af, vf, infof, rc, vc, infoc).Schedule(BlockCount, 1).Complete();
    }
    private void RestrictionCoefficients(NativeArray<float3> af, NativeArray<int2> infof,
        NativeArray<float3> ac, NativeArray<int2> infoc)
    {
        new RestrictCoefficientsJob(af, infof, ac, infoc).Schedule(BlockCount, 1).Complete();
    }

    [BurstCompile]
    private struct RestrictCoefficientsJob : IJobParallelFor
    {
        [ReadOnly] private NativeArray<int2> _lutFine;
        [ReadOnly] private NativeArray<float3> _aFine;
        [ReadOnly] private NativeArray<int2> _lutCoarse;
        [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float3> _aCoarse;
        
        public RestrictCoefficientsJob(NativeArray<float3> af, NativeArray<int2> lf,
            NativeArray<float3> ac, NativeArray<int2> lc)
        {
            _aFine = af;
            _lutFine = lf;
            _aCoarse = ac;
            _lutCoarse = lc;
        }

        public void Execute(int i)
        {
            int2 coord = Idx2Coord(i);
            int levelF = _lutFine[i].x;
            if (levelF < 0) return;
            int levelC = _lutCoarse[i].x;
            int phc = _lutCoarse[i].y;
            
            int blockWidthF = BlockWidth(levelF);
            int haloWidth = blockWidthF + 2;
            var blockA = new NativeArray<float3>(haloWidth * haloWidth, Allocator.Temp);
            FillHaloBlock( _aFine, _lutFine, blockA, Idx2Coord(i));

            if (levelF < levelC)
            {
                int blockWidthC = BlockWidth(levelC);
                for (int cy = 0; cy < blockWidthC; cy++)
                for (int cx = 0; cx < blockWidthC; cx++)
                {
                    float3 aCoarse = float3.zero;
                    for (int yy = 0; yy < 2; yy++)
                    for (int xx = 0; xx < 2; xx++)
                    {
                        int fx = cx * 2 + xx;
                        int fy = cy * 2 + yy;
                        float3 aFine = blockA[BlockCoord2Idx(fx + 1, fy + 1, haloWidth)];
                        aCoarse.x += aFine.x;
                        if (xx == 0) aCoarse.y += aFine.y;
                        else aCoarse.x += aFine.y * 2;
                        if (yy == 0) aCoarse.z += aFine.z;
                        else aCoarse.x += aFine.z * 2;
                    }

                    int ci = phc + BlockCoord2Idx(cx, cy, blockWidthC);
                    _aCoarse[ci] = aCoarse;
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
                    if (x == 0 && coord.x > 0 && _lutFine[Coord2Idx(coord.x - 1, coord.y)].x < _lutCoarse[Coord2Idx(coord.x - 1, coord.y)].x) aFine.y *= 2;
                    if (y == 0 && coord.y > 0 && _lutFine[Coord2Idx(coord.x, coord.y - 1)].x < _lutCoarse[Coord2Idx(coord.x, coord.y - 1)].x) aFine.z *= 2;
                    _aCoarse[ci] = aFine;
                }
            }
        }
    }

    [BurstCompile]
    private struct RestrictionJob : IJobParallelFor
    {
        [ReadOnly] private NativeArray<int2> _lutFine;
        [ReadOnly] private NativeArray<float3> _aFine;
        [ReadOnly] private NativeArray<float> _fFine;
        [ReadOnly] private NativeArray<float> _vFine;
        [ReadOnly] private NativeArray<int2> _lutCoarse;
        [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float> _rCoarse;
        [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float> _vCoarse;


        public RestrictionJob(NativeArray<float> ff, NativeArray<float3> af, NativeArray<float> vf,NativeArray<int2> lf, 
            NativeArray<float> rc, NativeArray<float> vc, NativeArray<int2> lc)
        {
            _aFine = af;
            _fFine = ff;
            _vFine = vf;
            _lutFine = lf;
            _rCoarse = rc;
            _vCoarse = vc;
            _lutCoarse = lc;
        }

        public void Execute(int i)
        {
            int2 infoFine = _lutFine[i];
            int2 infoCoarse = _lutCoarse[i];
            int levelF = infoFine.x;
            if (levelF < 0) return;
            int phf = infoFine.y;
            int levelC = infoCoarse.x;
            int phc = infoCoarse.y;

            int blockWidthF = BlockWidth(levelF);
            int haloWidth = blockWidthF + 2;
            var blockV = new NativeArray<float >(haloWidth * haloWidth, Allocator.Temp);
            var blockA = new NativeArray<float3>(haloWidth * haloWidth, Allocator.Temp);
            var blockR = new NativeArray<float >(blockWidthF * blockWidthF, Allocator.Temp);
            FillHaloBlock(_vFine, _aFine, _lutFine, blockV, blockA, Idx2Coord(i));
            for (int fy = 0; fy < blockWidthF; fy++)
            for (int fx = 0; fx < blockWidthF; fx++)
            {
                int localId = BlockCoord2Idx(fx, fy, blockWidthF);

                float3 a = blockA[BlockCoord2Idx(fx + 1, fy + 1, haloWidth)];
                if (a.x < 1e-5f) continue;
                float al = a.y;
                float ar = blockA[BlockCoord2Idx(fx + 2, fy + 1, haloWidth)].y;
                float ab = a.z;
                float at = blockA[BlockCoord2Idx(fx + 1, fy + 2, haloWidth)].z;
                float xc = blockV[BlockCoord2Idx(fx + 1, fy + 1, haloWidth)];
                float xl = blockV[BlockCoord2Idx(fx, fy + 1, haloWidth)];
                float xr = blockV[BlockCoord2Idx(fx + 2, fy + 1, haloWidth)];
                float xb = blockV[BlockCoord2Idx(fx + 1, fy, haloWidth)];
                float xt = blockV[BlockCoord2Idx(fx + 1, fy + 2, haloWidth)];
                if (a.x == 0) continue;
                blockR[localId] = _fFine[phf + localId] - (xl * al + xr * ar + xb * ab + xt * at + a.x * xc);
            }

            if (levelF < levelC)
            {
                int blockWidthC = BlockWidth(levelC);
                for (int cy = 0; cy < blockWidthC; cy++)
                for (int cx = 0; cx < blockWidthC; cx++)
                {
                    float rCoarse = 0;
                    for (int yy = 0; yy < 2; yy++)
                    for (int xx = 0; xx < 2; xx++)
                    {
                        int fx = cx * 2 + xx;
                        int fy = cy * 2 + yy;
                        float rFine = blockR[BlockCoord2Idx(fx, fy, blockWidthF)];
                        rCoarse += rFine;
                    }

                    int ci = phc + BlockCoord2Idx(cx, cy, blockWidthC);
                    _rCoarse[ci] = rCoarse;
                    _vCoarse[ci] = 0;
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
                    float rFine = blockR[BlockCoord2Idx(x, y, blockWidthF)];
                    _rCoarse[ci] = rFine;
                    _vCoarse[ci] = 0;
                }
            }
        }
    }
    
    
    private void Prolongation(NativeArray<float> vc, NativeArray<int2> infoc, NativeArray<float> vf, NativeArray<int2> infof)
    {
        new ProlongationJob(vc, infoc, vf, infof).Schedule(BlockCount, 1).Complete();
    }
    
    [BurstCompile]
    private struct ProlongationJob : IJobParallelFor
    {
        [ReadOnly] private NativeArray<int2> _lutCoarse;
        [ReadOnly] private NativeArray<int2> _lutFine;
        [ReadOnly] private NativeArray<float> _eCoarse;
        [NativeDisableParallelForRestriction] private NativeArray<float> _eFine;
        

        public ProlongationJob(NativeArray<float> ec, NativeArray<int2> lc, NativeArray<float> ef, NativeArray<int2> lf)
        {
            _eCoarse = ec;
            _lutCoarse = lc;
            _eFine = ef;
            _lutFine = lf;
        }

        public void Execute(int i)
        {
            int2 infoFine = _lutFine[i];
            int2 infoCoarse = _lutCoarse[i];
            int levelF = infoFine.x;
            if (levelF < 0) return;
            int phf = infoFine.y;
            int levelC = infoCoarse.x;
            int phc = infoCoarse.y;

            int blockWidthC = BlockWidth(levelC);
            int blockWidthF = BlockWidth(levelF);
            if (levelF < levelC)
            {
                for (int y = 0; y < blockWidthC; y++)
                for (int x = 0; x < blockWidthC; x++)
                {
                    float eCoarse = _eCoarse[phc + BlockCoord2Idx(x, y, blockWidthC)];
                    for (int yy = 0; yy < 2; yy++)
                    for (int xx = 0; xx < 2; xx++)
                    {
                        int fx = x * 2 + xx;
                        int fy = y * 2 + yy;
                        int fi = phf + BlockCoord2Idx(fx, fy, blockWidthF);

                        _eFine[fi] += eCoarse;
                    }
                }
            }
            else
            {
                int blockSize = blockWidthC * blockWidthC;
                for (int ii = 0; ii < blockSize; ii++)
                    _eFine[phf + ii] += _eCoarse[phc + ii];
            }
        }
    }
    
    private void ApplyPressure(NativeArray<int2> infos, NativeArray<float> pressure, NativeArray<float2> velocity)
    {
        new ApplyPressureJob
        {
            GridLut = infos,
            GridPressure = pressure,
            GridVelocity = velocity
        }.Schedule(BlockCount, 1).Complete();
    }

    [BurstCompile]
    private struct ApplyPressureJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int2> GridLut;
        [ReadOnly] public NativeArray<float> GridPressure;
        [NativeDisableParallelForRestriction] public NativeArray<float2> GridVelocity;
    
        public void Execute(int i)
        {
            int2 info = GridLut[i];
            int level = info.x;
            if (level < 0) return;
            int offset = info.y, width = BlockWidth(level);
            
            int haloWidth = width + 2;
            var blockV = new  NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);
            var blockA = new NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);

            // fill halo block
            FillHaloBlock(GridPressure, GridLut, blockV, blockA, Idx2Coord(i));
            
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int ii = offset + yy * width + xx;

                float p = blockV[BlockCoord2Idx(xx + 1, yy + 1, haloWidth)];
                float up = blockV[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                float ua = blockA[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                float vp = blockV[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                float va = blockA[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                
                float2 delta = new float2((p - up) * ua, (p - vp) * va);
                GridVelocity[ii] += delta;
            }
        }
    }
        
    public static float NeighborSum(NativeArray<float> block, NativeArray<float3> param, int x, int y, int blockRes)
    {
        float sum = 0;
        float3 ac = param[BlockCoord2Idx(x, y, blockRes)];
        float3 ar = param[BlockCoord2Idx(x + 1, y, blockRes)];
        float3 au = param[BlockCoord2Idx(x, y + 1, blockRes)];
        sum += ac.y * block[BlockCoord2Idx(x - 1, y, blockRes)];
        sum += ac.z * block[BlockCoord2Idx(x, y - 1, blockRes)];
        sum += ar.y * block[BlockCoord2Idx(x + 1, y, blockRes)];
        sum += au.z * block[BlockCoord2Idx(x, y + 1, blockRes)];
        
        return sum;
    }
    
    private static float NeighborSum(NativeArray<float> block, NativeArray<float3> param, out float diag, int x, int y, int blockRes)
    {
        float3 ac = param[BlockCoord2Idx(x, y, blockRes)];
        diag = ac.x;
        if (diag < 1e-5f) return 0;
        int l = BlockCoord2Idx(x - 1, y, blockRes);
        int r = BlockCoord2Idx(x + 1, y, blockRes);
        int b = BlockCoord2Idx(x, y - 1, blockRes);
        int t = BlockCoord2Idx(x, y + 1, blockRes);
        float pl = ac.y, pr = param[r].y, pb = ac.z, pt = param[t].z;
        return pl * block[l] + pr * block[r] + pb * block[b] + pt * block[t];
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
            int physicsIdx = ptr + BlockCoord2Idx(bx, by, blockWidth);
                
            block[localIdx] = v[physicsIdx];
        }
        int4 ox = new int4(-1, 1, 0, 0);
        int4 oy = new int4(0, 0, -1, 1);
        
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
    private static void FillHaloBlock(NativeArray<float> v, NativeArray<int2> infos, NativeArray<float> blockV, NativeArray<float> blockA, int2 coord)
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
            blockA[localIdx] = 1f / GetH(level);
        }

        // halo
        int4 ox = new int4(-1, 1, 0, 0);
        int4 oy = new int4(0, 0, -1, 1);
        for (int n = 0; n < 4; n++)
        {
            int2 dir = new int2(ox[n], oy[n]);
            int2 curr = coord + dir;
            if (curr.x < 0 || curr.y < 0 || curr.x >= GridWidth || curr.y >= GridWidth)
                continue;

            int2 nInfo = infos[BlockCoord2Idx(curr, GridWidth)];
            int nLevel = nInfo.x;
            if (nLevel < 0)
                continue;
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
                    blockA[paddingIdx] = 1f / GetH(nLevel);
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
                    blockA[paddingIdx] = 1f / (0.5f * (GetH(level) + GetH(nLevel)));
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
                    blockA[paddingIdx] = 1f / (0.5f * (GetH(level) + GetH(nLevel)));
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
        int4 ox = new int4(-1, 1, 0, 0);
        int4 oy = new int4(0, 0, -1, 1);
        for (int n = 0; n < 4; n++)
        {
            int2 dir = new int2(ox[n], oy[n]);
            int2 curr = coord + dir;
            if (curr.x < 0 || curr.y < 0 || curr.x >= GridWidth || curr.y >= GridWidth)
                continue;

            int2 nInfo = infos[BlockCoord2Idx(curr, GridWidth)];
            int nLevel = nInfo.x;
            if (nLevel < 0)
                continue;
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

    private static void FillHaloBlock(NativeArray<float3> p, NativeArray<int2> infos, NativeArray<float3> blockP, int2 coord)
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

            blockP[localIdx] = p[physicsIdx];
        }

        // halo
        int4 ox = new int4(-1, 1, 0, 0);
        int4 oy = new int4(0, 0, -1, 1);
        for (int n = 0; n < 4; n++)
        {
            int2 dir = new int2(ox[n], oy[n]);
            int2 curr = coord + dir;
            if (curr.x < 0 || curr.y < 0 || curr.x >= GridWidth || curr.y >= GridWidth)
                continue;

            int2 nInfo = infos[BlockCoord2Idx(curr, GridWidth)];
            int nLevel = nInfo.x;
            if (nLevel < 0)
                continue;
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
                    blockP[paddingIdx] = p[phn + nLocalIdx];
                }
            }
            else // n_level < level
            {
                for (int c = 0; c < width; c++)
                {
                    int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                    int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                    int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                    int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                    blockP[paddingIdx] = p[phn + nLocalIdx0]; // the two fine cells' params are the same
                }
            }
        }
    }

    private static int BlockWidth(int level) => 1 << (3 - level);
    private static int BlockCoord2Idx(int2 coord, int res) => coord.x + coord.y * res;
    private static int BlockCoord2Idx(int x, int y, int res) => x + y * res;
    private static int2 Idx2Coord(int idx) => new int2(idx % GridWidth, idx / GridWidth);
    private static int Coord2Idx(int x, int y) => x + y * GridWidth;
    private static int Coord2Idx(int2 coord) => coord.x + coord.y * GridWidth;
        
    private static float GetH(int level) => 1 << level;

    private static uint4 NeighborGridTypes( uint gridTypes)
    {
        return new uint4((gridTypes >> 2) & 3u, (gridTypes >> 4) & 3u, (gridTypes >> 6) & 3u, (gridTypes >> 8) & 3u);
    }
    private static bool IsSolidCell(uint gridTypes)
    {
        return (gridTypes & 3u) == SOLID;
    }
    private static bool2 IsSolidCell(uint2 gridTypes)
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
}

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace PF_FLIP
{
    public class MSBG_Solver : System.IDisposable
    {
        public NativeArray<float3>[] As; // x: center, y: left, z: down
        public NativeArray<float>[] Rs;
        public NativeArray<float>[] Zs;
        public NativeArray<int2>[] Ls;
        public NativeArray<float> F;

        public NativeArray<float> V;
        public NativeArray<int4> Lut;

        private NativeArray<float3> A => As[0];

        private NativeArray<float> R => Rs[0];
        private NativeArray<float> Z => Zs[0];
        private NativeArray<float> P;
        private NativeArray<float> Ap;
        
        private NativeReference<float> rz_old;
        private NativeReference<float> pAp;
        private NativeReference<float> rz_new;
        
        private NativeArray<int4> _sampleLut;
        private NativeArray<float> _dataDS;
        
        public NativeArray<int> LutApron;
        public NativeArray<uint> ApronIdx;

        private const int Levels = 5;
        private const int BatchSize = 64;
        
        public int ActiveGridCount;
        
        public MSBG_Solver(NativeArray<int4> gridLut, NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f)
        {
            V = v;
            F = f;

            As = new NativeArray<float3>[Levels];
            Zs = new NativeArray<float>[Levels];
            Rs = new NativeArray<float>[Levels];
            Ls = new NativeArray<int2>[Levels];
            const int poolSize = MSBGConstants.PoolSize;
            const int gridCount = MSBGConstants.GridCount;
            As[0] = a;
            Zs[0] = new NativeArray<float>(poolSize, Allocator.Persistent);
            Rs[0] = new NativeArray<float>(poolSize, Allocator.Persistent);
            Lut = gridLut;
            Ls[0]  = new NativeArray<int2>(gridCount, Allocator.Persistent);
            int[] res = { poolSize, gridCount * 64, gridCount * 16, gridCount * 4, gridCount };
            for (int i = 1; i < Levels; i++)
            {
                As[i] = new NativeArray<float3>(res[i], Allocator.Persistent);
                Zs[i] = new NativeArray<float>(res[i], Allocator.Persistent);
                Rs[i] = new NativeArray<float>(res[i], Allocator.Persistent);
                Ls[i] = new NativeArray<int2>(gridCount, Allocator.Persistent);
            }

            P = new NativeArray<float>(poolSize, Allocator.Persistent);
            Ap = new NativeArray<float>(poolSize, Allocator.Persistent);

            rz_old = new NativeReference<float>(0, Allocator.Persistent);
            pAp = new NativeReference<float>(0, Allocator.Persistent);
            rz_new = new NativeReference<float>(0, Allocator.Persistent);

            _sampleLut = new NativeArray<int4>(gridCount, Allocator.Persistent);
            _dataDS = new NativeArray<float>(res[1], Allocator.Persistent);
            
            LutApron = new NativeArray<int>(gridCount, Allocator.Persistent);
            ApronIdx = new NativeArray<uint>(poolSize * 2, Allocator.Persistent);
        }

        public void Dispose()
        {
            for (int i = 1; i < Levels; i++)
            {
                As[i].Dispose();
                Zs[i].Dispose();
                Rs[i].Dispose();
                Ls[i].Dispose();
            }

            Ls[0].Dispose();
            Zs[0].Dispose();
            Rs[0].Dispose();
            P.Dispose();
            Ap.Dispose();

            rz_old.Dispose();
            rz_new.Dispose();
            pAp.Dispose();
            
            _sampleLut.Dispose();
            _dataDS.Dispose();

            LutApron.Dispose();
            ApronIdx.Dispose();
        }

        public JobHandle BuildLutPyramid(JobHandle handle = default)
        {
            handle = new BuildLut(Lut, Ls[0], Ls[1], Ls[2], Ls[3], Ls[4], _sampleLut, LutApron)
                .Schedule(MSBGConstants.GridCount, BatchSize, handle);
            return new PrefixSum(Ls[1], Ls[2], Ls[3], Ls[4], _sampleLut, LutApron).Schedule(handle);
        }
        
        public void Solve_GS(int maxIter, out float rs)
        {
            // Z.CopyFrom(V);
            R.CopyFrom(F);
            var handle = BuildLutPyramid();
            // handle = new FindApronIndex(Ls[0], LutApron, ApronIdx).Schedule(MSBGConstants.GridCount, BatchSize, handle);
            for (int i = 0; i < maxIter; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    handle = new SmoothGaussSeidel(V, Ls[0], F, 0).Schedule(MSBGConstants.GridCount, BatchSize, handle);
                    handle = new SmoothGaussSeidel(V, Ls[0], F, 1).Schedule(MSBGConstants.GridCount, BatchSize, handle);
                    handle = new SmoothGaussSeidel(V, Ls[0], F, 1).Schedule(MSBGConstants.GridCount, BatchSize, handle);
                    handle = new SmoothGaussSeidel(V, Ls[0], F, 0).Schedule(MSBGConstants.GridCount, BatchSize, handle);
                    // handle = SmoothJob(A, Z, F, Ls[0], 0, 1, handle);
                    // handle = SmoothJob(A, Z, F, Ls[0], 1, 1, handle);
                    // handle = SmoothJob(A, Z, F, Ls[0], 2, 1, handle);
                    // handle = SmoothJob(A, Z, F, Ls[0], 1, 1, handle);
                    // handle = SmoothJob(A, Z, F, Ls[0], 0, 1, handle);
                }
                // new Residual(F, A, Z, Ls[0], R).Schedule(MSBGConstants.GridCount, BatchSize).Complete();
                // new Dot(R,R, rs_ref, ActiveGridCount).Schedule().Complete();
                // UnityEngine.Debug.Log($"[BuildLevels] iter {i*8}, rs {math.sqrt(rs_ref.Value)}");
            }
            
            // V.CopyFrom(Z);
            handle = new Residual(F, A, V, Ls[0], R).Schedule(MSBGConstants.GridCount, BatchSize, handle);
            new Dot(R, R, rz_old, ActiveGridCount).Schedule(handle).Complete();
            rs = math.sqrt(rz_old.Value);
        }
        
        [BurstCompile]
        private struct FindApronIndex : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int2> _lut;
            [ReadOnly] private NativeArray<int> _lutApron;
            [NativeDisableParallelForRestriction] private NativeArray<uint> _apronIdx;
            
            public FindApronIndex(NativeArray<int2> lut, NativeArray<int> lutApron, NativeArray<uint> apronIdx)
            {
                _lut = lut;
                _lutApron = lutApron;
                _apronIdx = apronIdx;
            }
            
            public void Execute(int i)
            {
                int2 coord = Idx2Coord(i);
                int2 info = _lut[i];
                int level = info.y;
                if (level < 0)
                    return;
                
                int ph = _lutApron[i];
                
                int blockWidth = GetBlockWidth(level);
                int haloBlockWidth = blockWidth + 2;
                int haloBlockSize = haloBlockWidth * haloBlockWidth;
                
                var valueArr = new NativeArray<uint>(haloBlockSize, Allocator.Temp);
                
                FillApronIdx(_lut, valueArr, coord, info);
                
                // copy back
                for (int by = 0; by < haloBlockWidth; by++)
                for (int bx = 0; bx < haloBlockWidth; bx++)
                {
                    int bIdx = BlockCoord2Idx(bx, by, haloBlockWidth);
                    int idx = ph + bIdx;
                    _apronIdx[idx] = valueArr[bIdx];
                }
            }
            
            private static void FillApronIdx(NativeArray<int2> lut, NativeArray<uint> block,  int2 coord, int2 info)
            {
                int level = info.y;
                int blockWidth = GetBlockWidth(level);
                int haloBlockWidth = blockWidth + 2;
                for (int by = 0; by < blockWidth; by++)
                for (int bx = 0; bx < blockWidth; bx++)
                {
                    int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
                    int physicsIdx = info.x + BlockCoord2Idx(bx, by, blockWidth);
                        
                    var flag = 1u;
                    block[localIdx] = (uint)(physicsIdx) << 2 | flag;
                }
                int4 ox = new int4(-1, 0, 1, 0);
                int4 oy = new int4(0, -1, 0, 1);
                
                for (int n = 0; n < 4; n++)
                {
                    int2 dir = new int2(ox[n], oy[n]);
                    int2 curr = coord + dir;
                    if (curr.x < 0 || curr.y < 0 || curr.x >= MSBGConstants.GridWidth || curr.y >= MSBGConstants.GridWidth)
                        continue;
                    
                    int2 neighborInfo = lut[Coord2Idx(curr)];
                    int nLevel = neighborInfo.y;
                    if (nLevel < 0)
                        continue;
                    
                    int phn = neighborInfo.x;
                    if (nLevel == level)
                    {
                        for (int c = 0; c < blockWidth; c++)
                        {
                            int2 nCoord = math.select(math.select(c, 0, dir > 0), blockWidth - 1, dir < 0);
                            int nLocalIdx = BlockCoord2Idx(nCoord, blockWidth);
                            int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                            int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                            
                            var flag = 1u;
                            block[paddingIdx] = (uint)(phn + nLocalIdx) << 2 | flag;
                        }
                    }
                    else if (nLevel > level)
                    {
                        int nBlockWidth = GetBlockWidth(nLevel);
                        for (int c = 0; c < blockWidth; c++)
                        {
                            int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                            int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                            int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                            int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                            var flag = 2u;
                            block[paddingIdx] = (uint)(phn + nLocalIdx) << 2 | flag;
                        }
                    }
                    else // n_level < level
                    {
                        int nBlockWidth = GetBlockWidth(nLevel);
                        for (int c = 0; c < blockWidth; c++)
                        {
                            int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                            int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                            int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                            int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                            int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                            int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                            var flag = 3u;
                            block[paddingIdx] = (uint)(phn + nLocalIdx0) << 17 | (uint)(phn + nLocalIdx1) << 2 | flag;
                        }
                    }
                }
            }
        }
        
        [BurstCompile]
        private struct SmoothGaussSeidel : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int2> _lut;
            [ReadOnly] private NativeArray<float> _f;
            [NativeDisableParallelForRestriction] private NativeArray<float> _values;
            private readonly int _phase;
            
            public SmoothGaussSeidel(NativeArray<float> v, NativeArray<int2> lut, NativeArray<float> f, int phase)
            {
                _values = v;
                _lut = lut;
                _f = f;
                _phase = phase;
            }

            public void Execute(int i)
            {
                int2 coord = Idx2Coord(i);
                int2 info = _lut[i];
                int level = info.y;
                if (((coord.x + coord.y) & 1) != _phase || level < 0)
                    return;
                
                int ph = info.x;
                
                int blockWidth = GetBlockWidth(level);
                int haloBlockWidth = blockWidth + 2;
                int haloBlockSize = haloBlockWidth * haloBlockWidth;
                
                var valueArr = new NativeArray<float>(haloBlockSize, Allocator.Temp);
                var paramArr = new NativeArray<float>(haloBlockSize, Allocator.Temp);
                
                FillHaloBlock1(_values, _lut, valueArr, paramArr, coord, info);
                
                // forward
                for (int by = 1; by < haloBlockWidth - 1; by++)
                for (int bx = 1; bx < haloBlockWidth - 1; bx++)
                {
                    int localIdx = BlockCoord2Idx(bx, by, haloBlockWidth);
                    float3 ac = paramArr[localIdx];
                    if (InActive(ac.x)) continue;
                    float psum = NeighborSum1(valueArr, paramArr, out var csum, bx, by, haloBlockWidth);
                    float f = _f[ph + BlockCoord2Idx(bx - 1, by - 1, blockWidth)];
                    valueArr[localIdx] = (-f + psum) / csum;
                }
                
                // backward
                for (int by = haloBlockWidth - 2; by > 0; by--)
                for (int bx = haloBlockWidth - 2; bx > 0; bx--)
                {
                    int localIdx = BlockCoord2Idx(bx, by, haloBlockWidth);
                    float3 ac = paramArr[localIdx];
                    if (InActive(ac.x)) continue;
                    float psum = NeighborSum1(valueArr, paramArr, out var csum, bx, by, haloBlockWidth);
                    float f = _f[ph + BlockCoord2Idx(bx - 1, by - 1, blockWidth)];
                    valueArr[localIdx] = (-f + psum) / csum;
                }
                
                // copy back
                for (int by = 0; by < blockWidth; by++)
                for (int bx = 0; bx < blockWidth; bx++)
                {
                    int idx = ph + BlockCoord2Idx(bx, by, blockWidth);
                    _values[idx] = valueArr[BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth)];
                }

                valueArr.Dispose();
                paramArr.Dispose();
            }
        }
        
        public void Solve_CG(int maxIter, out int iter, out float rs)
        {
            var handle = BuildLutPyramid();
            handle.Complete();
            R.CopyFrom(F);
            P.CopyFrom(R);
            
            int count = MSBGConstants.PoolSize;
            new Dot(R, R, rz_old, count).Schedule().Complete();
            // UnityEngine.Debug.Log("CG init with res: " + rz_old.Value + ". ");
            if (math.abs(rz_old.Value) > 1e-5f)
            {
                // var msg = "";
                for (iter = 0; iter < maxIter * 10; iter++)
                {
                    // for (int i = 0; i < 4; i++)
                    
                        new Laplace(A, P, Ls[0], Ap).Schedule(MSBGConstants.GridCount, BatchSize).Complete();

                        new Dot(P, Ap, pAp, count).Schedule().Complete();
                        if (math.abs(pAp.Value) < 1e-7f)
                            break;

                        new UpdateVR(P, Ap, V, R, rz_old, pAp).Schedule(count, BatchSize).Complete();

                        new Dot(R, R, rz_new, count).Schedule().Complete();
                        if (rz_new.Value < 1e-6f)
                            break;

                        new UpdateP(R, P, rz_new, rz_old).Schedule(count, BatchSize).Complete();
                        (rz_old, rz_new) = (rz_new, rz_old);
                    
                    // msg += "CG iter " + ((iter + 1) * 4) + ", res: " + rz_old.Value + ". \n";
                }
                handle.Complete();
                // UnityEngine.Debug.Log(msg);
            }
            else iter = 0;
            
            // new Residual(F, A, V, Ls[0], R).Schedule(MSBGConstants.GridCount, BatchSize).Complete();
            // new Dot(R, R, rz_old, count).Schedule().Complete();
            rs = rz_old.Value;
            // UnityEngine.Debug.Log("CG solved with rs: " + rz_old.Value + ". ");
            // Z.CopyFrom(V);
        }
        
        public void Solve_MGPCG(int maxIter, out float rs)
        {
            BuildLutPyramid().Complete();
            for (int i = 0; i < Z.Length; i++)
                Zs[0][i] = 0;
            R.CopyFrom(F);
            MultiGridVCycle().Complete();
            // Z.CopyFrom(R);
            P.CopyFrom(Z);

            new Dot(R, Z, rz_old, ActiveGridCount).Schedule().Complete();
            UnityEngine.Debug.Log("MGPCG init with rs: " + math.sqrt(rz_old.Value) + ". ");
            
            if (math.abs(rz_old.Value) > 1e-8f)
            {
                for (int iter = 0; iter < maxIter; iter++)
                {
                    new Laplace(A, P, Ls[0], Ap).Schedule(MSBGConstants.GridCount, BatchSize).Complete();
                    new Dot(P, Ap, pAp, ActiveGridCount).Schedule().Complete();
                    new UpdateVR(P, Ap, V, R, rz_old, pAp).Schedule(ActiveGridCount, BatchSize).Complete();

                    if (iter == maxIter - 1) break;

                    for (int j = 0; j < Zs[0].Length; j++)
                        Zs[0][j] = 0;
                    MultiGridVCycle().Complete();
                    // Z.CopyFrom(R);

                    new Dot(R, Z, rz_new, ActiveGridCount).Schedule().Complete();
                    float beta = rz_new.Value / rz_old.Value;
                    new UpdateP(Z, P, rz_new, rz_old).Schedule(P.Length, BatchSize).Complete();
                    (rz_old, rz_new) = (rz_new, rz_old);
                    
                    UnityEngine.Debug.Log("MGPCG iter: " + iter + ", res: " + math.sqrt(rz_old.Value) + ". ");
                }
            }
            else
                UnityEngine.Debug.Log("MGPCG early stop due to zero search direction");
            
            new Residual(F, A, V, Ls[0], R).Schedule(MSBGConstants.GridCount, BatchSize).Complete();
            new Dot(R, R, rz_old, ActiveGridCount).Schedule().Complete();
            rs = math.sqrt(rz_old.Value);
            Z.CopyFrom(V);
        }
        
        public void Solve_MG(int maxIter, out float rs)
        {
            for (int i = 0; i < Z.Length; i++)
                Zs[0][i] = 0;
            R.CopyFrom(F);
            BuildLutPyramid().Complete();
            for (int level = 0; level < maxIter; level++)
            {
                MultiGridVCycle().Complete();
                new Residual(F, A, Zs[0], Ls[0], Rs[0]).Schedule(MSBGConstants.GridCount, BatchSize).Complete();
                new Dot(R, R, rz_old, ActiveGridCount).Schedule().Complete();
                UnityEngine.Debug.Log($"[Solve_MG] iter {level}, res {math.sqrt(rz_old.Value)}");
                Rs[0].CopyFrom(F);
            }

            V.CopyFrom(Z);
            new Residual(F, A, V, Ls[0], R).Schedule(MSBGConstants.GridCount, BatchSize).Complete();
            new Dot(R, R, rz_old, ActiveGridCount).Schedule().Complete();
            rs = math.sqrt(rz_old.Value);
        }

        private JobHandle MultiGridVCycle(JobHandle handle = default)
        {
            const int topLevel = Levels - 1;
            const int smoothCount = 2;
            
            for (int level = 0; level < topLevel; level++)
            {
                handle = SmoothJob(As[level], Zs[level], Rs[level], Ls[level], level, smoothCount, handle);

                handle = new Restriction(Rs[level], As[level], Zs[level], Ls[level], 
                        Rs[level + 1], As[level + 1], Zs[level + 1], Ls[level + 1], level)
                    .Schedule(MSBGConstants.GridCount, BatchSize, handle);
            }
            
            handle = SmoothJob(As[topLevel], Zs[topLevel], Rs[topLevel], Ls[topLevel], topLevel, MSBGConstants.GridWidth, handle);
            
            for (int level = topLevel - 1; level >= 0; level--)
            {
                handle = new Prolongation(Zs[level + 1], Ls[level + 1], Zs[level], Ls[level], As[level], level)
                    .Schedule(MSBGConstants.GridCount, BatchSize, handle);
                
                handle = SmoothJob(As[level], Zs[level], Rs[level], Ls[level], level, smoothCount, handle);
            }

            return handle;
        }
        
        private JobHandle SmoothJob(NativeArray<float3> a, NativeArray<float> v, NativeArray<float> f, 
            NativeArray<int2> l, int level, int count, JobHandle handle = default)
        {
            for (int i = 0; i < count; i++)
            {
                handle = new SymmetricSmoothGaussSeidel(v, l, a, f, level, 0)
                    .Schedule(MSBGConstants.GridCount, BatchSize, handle);
                handle = new SymmetricSmoothGaussSeidel(v, l, a, f, level, 1)
                    .Schedule(MSBGConstants.GridCount, BatchSize, handle);
                handle = new SymmetricSmoothGaussSeidel(v, l, a, f, level, 1)
                    .Schedule(MSBGConstants.GridCount, BatchSize, handle);
                handle = new SymmetricSmoothGaussSeidel(v, l, a, f, level, 0)
                    .Schedule(MSBGConstants.GridCount, BatchSize, handle);
            }
            
            return handle;
        }

        public void DownSampleField()
        {
            new DownSample(F, _dataDS, _sampleLut).Schedule(MSBGConstants.GridCount, BatchSize).Complete();
        }

        public void SampleFieldBilinear(float[] field, int width)
        {
            var result = new NativeArray<float>(field.Length, Allocator.TempJob);
            new SamplerField(F, _dataDS, _sampleLut, result, width)
                .Schedule(field.Length, BatchSize).Complete();

            for (int i = 0; i < field.Length; i++)
                field[i] = result[i] * 2;

            result.Dispose();
        }

        #region Jobs

        [BurstCompile]
        private struct BuildLut : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int4> _lut;
            [WriteOnly] private NativeArray<int2> _lut0;
            [WriteOnly] private NativeArray<int2> _lut1;
            [WriteOnly] private NativeArray<int2> _lut2;
            [WriteOnly] private NativeArray<int2> _lut3;
            [WriteOnly] private NativeArray<int2> _lut4;
            [WriteOnly] private NativeArray<int4> _lutS;
            [WriteOnly] private NativeArray<int> _lutApron;
            
            public BuildLut(NativeArray<int4> lut, NativeArray<int2> lut0, NativeArray<int2> lut1, NativeArray<int2> lut2,
                NativeArray<int2> lut3, NativeArray<int2> lut4, NativeArray<int4> lutSample, NativeArray<int> lutApron)
            {
                _lut = lut;
                _lut0 = lut0;
                _lut1 = lut1;
                _lut2 = lut2;
                _lut3 = lut3;
                _lut4 = lut4;
                _lutS = lutSample;
                _lutApron = lutApron;
            }

            public void Execute(int i)
            {
                // const int top = Levels - 1;
                int levelCur = GetCurLevel(_lut[i].z);
                int levelRight = GetCurLevel(_lut[math.min(i + 1, _lut.Length - 1)].z);
                int levelUp = GetCurLevel(_lut[math.min(i + MSBGConstants.GridWidth, _lut.Length - 1)].z);
                int levelRT = GetCurLevel(_lut[math.min(i + MSBGConstants.GridWidth + 1, _lut.Length - 1)].z);
                int4 levels = new int4(levelCur,
                    levelRight < 0 ? levelCur : levelRight,
                    levelUp < 0 ? levelCur : levelUp,
                    levelRT < 0 ? levelCur : levelRT);
                int level0 = levels.x;
                if (level0 < 0)
                {
                    _lut0[i] = new int2(0, -1);
                    _lut1[i] = new int2(0, -1);
                    _lut2[i] = new int2(0, -1);
                    _lut3[i] = new int2(0, -1);
                    _lut4[i] = new int2(0, -1);
                    _lutS[i] = new int4(0, 0, -1, -1);
                    _lutApron[i] = -1;
                }
                else
                {
                    int level1 = math.max(1, level0);
                    int level2 = math.max(2, level0);
                    int level3 = math.max(3, level0);
                    int level4 = math.max(4, level0);
                    _lut0[i] = new int2(_lut[i].x, level0);
                    _lut1[i] = new int2(GetBlockSize(level1), level1);
                    _lut2[i] = new int2(GetBlockSize(level2), level2);
                    _lut3[i] = new int2(GetBlockSize(level3), level3);
                    _lut4[i] = new int2(GetBlockSize(level4), level4);
                    int haloBlockWidth = GetBlockWidth(level0) + 2;
                    _lutApron[i] = haloBlockWidth * haloBlockWidth;
                    
                    int4 levelD = math.min(2, levels + 1);
                    _lutS[i] = new int4(GetBlockSize(level0), GetBlockSize(levelD.x), 
                        PackNeighborsLevel(levels), PackNeighborsLevel(levelD));
                }
            }
        }

        [BurstCompile]
        private struct PrefixSum : IJob
        {
            private NativeArray<int2> _lut1;
            private NativeArray<int2> _lut2;
            private NativeArray<int2> _lut3;
            private NativeArray<int2> _lut4;
            private NativeArray<int4> _lutS;
            private NativeArray<int> _lutApron;
            
            public PrefixSum(NativeArray<int2> lut1, NativeArray<int2> lut2, 
                NativeArray<int2> lut3, NativeArray<int2> lut4, NativeArray<int4> lutSample, NativeArray<int> lutApron)
            {
                _lut1 = lut1;
                _lut2 = lut2;
                _lut3 = lut3;
                _lut4 = lut4;
                _lutS = lutSample;
                _lutApron = lutApron;
            }

            public void Execute()
            {
                int ptr1 = 0;
                int ptr2 = 0;
                int ptr3 = 0;
                int ptr4 = 0;
                int2 ptrS = int2.zero;
                int ptrA = 0;
                for (int i = 0; i < MSBGConstants.GridCount; i++)
                {
                    int2 info1 = _lut1[i];
                    int2 info2 = _lut2[i];
                    int2 info3 = _lut3[i];
                    int2 info4 = _lut4[i];
                    int4 infoS = _lutS[i];
                    int apronSize = _lutApron[i];

                    _lut1[i] = new int2(ptr1, info1.y);
                    ptr1 += info1.x;

                    _lut2[i] = new int2(ptr2, info2.y);
                    ptr2 += info2.x;
                    
                    _lut3[i] = new int2(ptr3, info3.y);
                    ptr3 += info3.x;
                
                    _lut4[i] = new int2(ptr4, info4.y);
                    ptr4 += info4.x;
                
                    _lutS[i] = new int4(ptrS.x, ptrS.y, infoS.z, infoS.w);
                    ptrS += new int2(infoS.x, infoS.y);
                
                    _lutApron[i] = ptrA;
                    ptrA += apronSize;
                }
            }
        }

        [BurstCompile]
        private struct SymmetricSmoothGaussSeidel : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int2> _lut;
            [ReadOnly] private NativeArray<float3> _a;
            [ReadOnly] private NativeArray<float> _f;
            [NativeDisableParallelForRestriction] private NativeArray<float> _values;
            private readonly int _level;
            private readonly int _phase;
            private readonly int _blockWidth;
            private readonly int _haloBlockWidth;
            private readonly int _haloBlockSize;
            
            public SymmetricSmoothGaussSeidel(NativeArray<float> v, NativeArray<int2> lut,
                NativeArray<float3> a, NativeArray<float> f, int level, int phase)
            {
                _values = v;
                _lut = lut;
                _a = a;
                _f = f;
                _level = level;
                _phase = phase;
                _blockWidth = GetBlockWidth(level);
                _haloBlockWidth = _blockWidth + 2;
                _haloBlockSize = _haloBlockWidth * _haloBlockWidth;
            }

            public void Execute(int i)
            {
                int2 coord = Idx2Coord(i);
                int2 info = _lut[i];
                int level = info.y;
                if (((coord.x + coord.y) & 1) != _phase || level != _level)
                    return;
                
                int ph = info.x;
                
                var valueArr = new NativeArray<float>(_haloBlockSize, Allocator.Temp);
                var paramArr = new NativeArray<float3>(_haloBlockSize, Allocator.Temp);
                var residualArr = new NativeArray<float>(_blockWidth * _blockWidth, Allocator.Temp);
                var aArr = new NativeArray<float>(_blockWidth * _blockWidth, Allocator.Temp);
                for (int by = 0; by < _blockWidth; by++)
                for (int bx = 0; bx < _blockWidth; bx++)
                {
                    residualArr[BlockCoord2Idx(bx, by, _blockWidth)] =
                        _f[ph + BlockCoord2Idx(bx - 1, by - 1, _blockWidth)];
                    aArr[BlockCoord2Idx(bx, by, _blockWidth)] =
                        _a[ph + BlockCoord2Idx(bx - 1, by - 1, _blockWidth)].x;
                }
                
                FillHaloBlock(_values, _a, _lut, valueArr, paramArr, coord, info);
                
                // forward
                for (int by = 1; by < _haloBlockWidth - 1; by++)
                for (int bx = 1; bx < _haloBlockWidth - 1; bx++)
                {
                    int localIdx = BlockCoord2Idx(bx, by, _haloBlockWidth);
                    float3 ac = aArr[BlockCoord2Idx(bx-1, by-1, _blockWidth)];
                    if (InActive(ac.x)) continue;
                    valueArr[localIdx] = (residualArr[BlockCoord2Idx(bx-1, by-1, _blockWidth)] -
                                          NeighborSum(valueArr, paramArr, bx, by, _haloBlockWidth)) / ac.x;
                }
                
                // backward
                for (int by = _haloBlockWidth - 2; by > 0; by--)
                for (int bx = _haloBlockWidth - 2; bx > 0; bx--)
                {
                    int localIdx = BlockCoord2Idx(bx, by, _haloBlockWidth);
                    float3 ac = aArr[BlockCoord2Idx(bx-1, by-1, _blockWidth)];
                    if (InActive(ac.x)) continue;
                    valueArr[localIdx] = (residualArr[BlockCoord2Idx(bx-1, by-1, _blockWidth)] -
                                          NeighborSum(valueArr, paramArr, bx, by, _haloBlockWidth)) / ac.x;
                }
                
                // copy back
                for (int by = 0; by < _blockWidth; by++)
                for (int bx = 0; bx < _blockWidth; bx++)
                {
                    int idx = ph + BlockCoord2Idx(bx, by, _blockWidth);
                    _values[idx] = valueArr[BlockCoord2Idx(bx + 1, by + 1, _haloBlockWidth)];
                }
            }
        }

        [BurstCompile]
        private struct Restriction : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int2> _lutFine;
            [ReadOnly] private NativeArray<float3> _aFine;
            [ReadOnly] private NativeArray<float> _fFine;
            [ReadOnly] private NativeArray<float> _vFine;
            [ReadOnly] private NativeArray<int2> _lutCoarse;
            [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float3> _aCoarse;
            [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float> _rCoarse;
            [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float> _vCoarse;

            private readonly int _level;
            private readonly int _blockWidthFine;
            private readonly int _blockWidthCoarse;
            private readonly int _haloBlockWidth;

            public Restriction(NativeArray<float> ff, NativeArray<float3> af, NativeArray<float> vf,NativeArray<int2> lf, 
                NativeArray<float> rc, NativeArray<float3> ac, NativeArray<float> vc, NativeArray<int2> lc, int level)
            {
                _aFine = af;
                _fFine = ff;
                _vFine = vf;
                _lutFine = lf;
                _aCoarse = ac;
                _rCoarse = rc;
                _vCoarse = vc;
                _lutCoarse = lc;
                _level = level;
                _blockWidthFine = 1 << (Levels - 1 - level);
                _haloBlockWidth = _blockWidthFine + 2;
                _blockWidthCoarse = _blockWidthFine >> 1;
            }

            public void Execute(int i)
            {
                int2 infoFine = _lutFine[i];
                int level = infoFine.y;
                if (level < 0)
                    return;
                
                int phf = infoFine.x;
                int2 infoCoarse = _lutCoarse[i];
                int phc = infoCoarse.x;
                if (level == _level)
                {
                    var haloBlockV = new NativeArray<float>(_haloBlockWidth * _haloBlockWidth, Allocator.Temp);
                    var haloBlockA = new NativeArray<float3>(_haloBlockWidth * _haloBlockWidth, Allocator.Temp);
                    
                    FillHaloBlock(_vFine, _aFine, _lutFine, haloBlockV, haloBlockA, Idx2Coord(i), infoFine);

                    for (int y = 0; y < _blockWidthCoarse; y++)
                    for (int x = 0; x < _blockWidthCoarse; x++)
                    {
                        float rCoarse = 0;
                        float3 aCoarse = float3.zero;
                        for (int yy = 0; yy < 2; yy++)
                        for (int xx = 0; xx < 2; xx++)
                        {
                            int fx = x * 2 + xx;
                            int fy = y * 2 + yy;
                            int fi = phf + BlockCoord2Idx(fx, fy, _blockWidthFine);
                            float3 aFine = haloBlockA[BlockCoord2Idx(fx + 1, fy + 1, _haloBlockWidth)];
                            if (aFine.x == 0) continue;

                            aCoarse.x += aFine.x;
                            float vFine = haloBlockV[BlockCoord2Idx(fx + 1, fy + 1, _haloBlockWidth)];
                            float neighborSum = NeighborSum(haloBlockV, haloBlockA, fx + 1, fy + 1, _haloBlockWidth);
                            float rFine = _fFine[fi] - (aFine.x * vFine + neighborSum);
                            rCoarse += rFine;

                            if (xx == 0) aCoarse.y += aFine.y;
                            else aCoarse.x += aFine.y * 2;

                            if (yy == 0) aCoarse.z += aFine.z;
                            else aCoarse.x += aFine.z * 2;
                        }

                        int ci = phc + BlockCoord2Idx(x, y, _blockWidthCoarse);
                        _rCoarse[ci] = rCoarse * 0.25f * 0.5f;
                        _aCoarse[ci] = aCoarse * 0.25f * 0.5f;
                        _vCoarse[ci] = 0;
                    }
                }
                else // level > _level
                {
                    int blockWidth = GetBlockWidth(level);
                    int haloBlockWidth = blockWidth + 2;
                    var haloBlockV = new NativeArray<float>(haloBlockWidth * haloBlockWidth, Allocator.Temp);
                    var haloBlockA = new NativeArray<float3>(haloBlockWidth * haloBlockWidth, Allocator.Temp);
                    
                    FillHaloBlock(_vFine, _aFine, _lutFine, haloBlockV, haloBlockA, Idx2Coord(i), infoFine);
                    for (int y = 0; y < blockWidth; y++)
                    for (int x = 0; x < blockWidth; x++)
                    {
                        int ii = BlockCoord2Idx(x, y, blockWidth);
                        int ci = phc + ii;
                        int fi = phf + ii;
                        float3 aFine = haloBlockA[BlockCoord2Idx(x + 1, y + 1, haloBlockWidth)];
                        if (aFine.x == 0) continue;
                    
                        float vFine = haloBlockV[BlockCoord2Idx(x + 1, y + 1, haloBlockWidth)];
                        float neighborSum = NeighborSum(haloBlockV, haloBlockA, x + 1, y + 1, haloBlockWidth);
                        float rFine = _fFine[fi] - (aFine.x * vFine + neighborSum);
                        _rCoarse[ci] = rFine;
                        _aCoarse[ci] = aFine;
                        _vCoarse[ci] = 0;
                    }
                    // int blockWidth = GetBlockWidth(level);
                    // for (int y = 0; y < blockWidth; y++)
                    // for (int x = 0; x < blockWidth; x++)
                    // {
                    //     int ii = BlockCoord2Idx(x, y, blockWidth);
                    //     int ci = phc + ii;
                    //     int fi = phf + ii;
                    //     _rCoarse[ci] = _fFine[fi];
                    //     _aCoarse[ci] = _aFine[fi];
                    //     _vCoarse[ci] = 0;
                    // }
                }
            }
        }
        
        [BurstCompile]
        private struct Prolongation : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int2> _lutCoarse;
            [ReadOnly] private NativeArray<int2> _lutFine;
            [ReadOnly] private NativeArray<float3> _aFine;
            [ReadOnly] private NativeArray<float> _eCoarse;
            [NativeDisableParallelForRestriction] private NativeArray<float> _eFine;
            
            private readonly int _level;
            private readonly int _blockWidthFine;
            private readonly int _blockWidthCoarse;

            public Prolongation(NativeArray<float> ec, NativeArray<int2> lc, NativeArray<float> ef, 
                NativeArray<int2> lf, NativeArray<float3> af,int level)
            {
                _eCoarse = ec;
                _lutCoarse = lc;
                _eFine = ef;
                _lutFine = lf;
                _aFine = af;
                
                _level = level;
                _blockWidthFine = GetBlockWidth(level);
                _blockWidthCoarse = _blockWidthFine >> 1;
            }

            public void Execute(int i)
            {
                int2 infoFine = _lutFine[i];
                int level = infoFine.y;
                if (level < 0)
                    return;
                
                int phf = infoFine.x;
                int2 infoCoarse = _lutCoarse[i];
                int phc = infoCoarse.x;
                if (level == _level)
                {
                    for (int y = 0; y < _blockWidthCoarse; y++)
                    for (int x = 0; x < _blockWidthCoarse; x++)
                    {
                        float eCoarse = _eCoarse[phc + BlockCoord2Idx(x, y, _blockWidthCoarse)];
                        for (int yy = 0; yy < 2; yy++)
                        for (int xx = 0; xx < 2; xx++)
                        {
                            int fx = x * 2 + xx;
                            int fy = y * 2 + yy;
                            int fi = phf + BlockCoord2Idx(fx, fy, _blockWidthFine);
                            float3 aFine = _aFine[fi];
                            if (aFine.x == 0) continue;

                            _eFine[fi] += eCoarse;
                        }
                    }
                }
                else // level > _level
                {
                    int blockWidth = GetBlockWidth(level);
                    int blockSize = blockWidth * blockWidth;
                    for (int ii = 0; ii < blockSize; ii++)
                        _eFine[phf + ii] += _eCoarse[phc + ii];
                }
            }
        }

        [BurstCompile]
        private struct Residual : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int2> _lut;
            [ReadOnly] private NativeArray<float3> _a;
            [ReadOnly] private NativeArray<float> _f;
            [ReadOnly] private NativeArray<float> _v;
            [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float> _r;

            public Residual(NativeArray<float> f, NativeArray<float3> a, NativeArray<float> v,
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
                int level = info.y;
                if (level < 0)
                    return;
                
                int ph = info.x;
                
                int blockWidth = GetBlockWidth(level);
                int haloBlockWidth = blockWidth + 2;
                var haloBlockV = new NativeArray<float>(haloBlockWidth * haloBlockWidth, Allocator.Temp);
                var haloBlockA = new NativeArray<float>(haloBlockWidth * haloBlockWidth, Allocator.Temp);
                
                FillHaloBlock1(_v, _lut, haloBlockV, haloBlockA, Idx2Coord(i), info);
                for (int y = 0; y < blockWidth; y++)
                for (int x = 0; x < blockWidth; x++)
                {
                    int ii = ph + BlockCoord2Idx(x, y, blockWidth);
                    float3 aFine = _a[ii];

                    float vFine = haloBlockV[BlockCoord2Idx(x + 1, y + 1, haloBlockWidth)];
                    float neighborSum = NeighborSum1(haloBlockV, haloBlockA, out float csum, x + 1, y + 1, haloBlockWidth);
                    float rFine = _f[ii] - (neighborSum - csum * vFine);
                    _r[ii] = aFine.x < 1 ? 0 : rFine;
                }
                haloBlockA.Dispose();
                haloBlockV.Dispose();
            }
        }

        [BurstCompile]
        private struct Laplace : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int2> _lut;
            [ReadOnly] private NativeArray<float3> _a;
            [ReadOnly] private NativeArray<float> _v;
            [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float> _r;

            public Laplace(NativeArray<float3> a, NativeArray<float> v, NativeArray<int2> l, NativeArray<float> r)
            {
                _a = a;
                _v = v;
                _lut = l;
                _r = r;
            }

            public void Execute(int i)
            {
                int2 info = _lut[i];
                int level = info.y;
                if (level < 0)
                    return;
                
                int ph = info.x;
                
                int blockWidth = GetBlockWidth(level);
                int haloBlockWidth = blockWidth + 2;
                var haloBlockV = new NativeArray<float>(haloBlockWidth * haloBlockWidth, Allocator.Temp);
                var haloBlockA = new NativeArray<float>(haloBlockWidth * haloBlockWidth, Allocator.Temp);
                
                FillHaloBlock1(_v, _lut, haloBlockV, haloBlockA, Idx2Coord(i), info);
                for (int y = 0; y < blockWidth; y++)
                for (int x = 0; x < blockWidth; x++)
                {
                    int ii = ph + BlockCoord2Idx(x, y, blockWidth);

                    float3 aFine = _a[ii];
                    float vFine = haloBlockV[BlockCoord2Idx(x + 1, y + 1, haloBlockWidth)];
                    float neighborSum = NeighborSum1(haloBlockV, haloBlockA, out float csum, x + 1, y + 1, haloBlockWidth);
                    _r[ii] = aFine.x < 0.001f ? 0 : neighborSum - csum * vFine;
                }
                
                haloBlockA.Dispose();
                haloBlockV.Dispose();
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
            [ReadOnly] private NativeReference<float> _rsOld;
            [ReadOnly] private NativeReference<float> _rsNew;
            
            public UpdateP(NativeArray<float> z, NativeArray<float> p,
                NativeReference<float> rsNew, NativeReference<float> rsOld)
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

        [BurstCompile]
        private struct Dot : IJob
        {
            [ReadOnly] private NativeArray<float> _lhs;
            [ReadOnly] private NativeArray<float> _rhs;
            [WriteOnly] private NativeReference<float> _result;
            private readonly int _count;
            
            public Dot(NativeArray<float> lhs, NativeArray<float> rhs, NativeReference<float> result, int count)
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

        [BurstCompile]
        private struct DownSample : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int4> _lut;
            [ReadOnly] private NativeArray<float> _vFine;
            [NativeDisableParallelForRestriction] [WriteOnly] private NativeArray<float> _vCoarse;
            
            public DownSample(NativeArray<float> vf, NativeArray<float> vc, NativeArray<int4> lut)
            {
                _vFine = vf;
                _lut = lut;
                _vCoarse = vc;
            }

            public void Execute(int i)
            {
                int4 info = _lut[i];
                int levelFine = GetCurLevel(info.z);
                int levelCoarse = GetCurLevel(info.w);
                if (levelFine < 0 || levelCoarse < 0)
                    return;
                
                int phf = info.x;
                int phc = info.y;
                if (levelFine < levelCoarse)
                {
                    int blockWidthCoarse = GetBlockWidth(levelCoarse);
                    int blockWidthFine = GetBlockWidth(levelFine);
                    for (int y = 0; y < blockWidthCoarse; y++)
                    for (int x = 0; x < blockWidthCoarse; x++)
                    {
                        float vCoarse = 0;
                        for (int yy = 0; yy < 2; yy++)
                        for (int xx = 0; xx < 2; xx++)
                        {
                            int fx = x * 2 + xx;
                            int fy = y * 2 + yy;
                            vCoarse += _vFine[phf + BlockCoord2Idx(fx, fy, blockWidthFine)];
                        }

                        _vCoarse[phc + BlockCoord2Idx(x, y, blockWidthCoarse)] = vCoarse * 0.25f;
                    }
                }
                else // levelC == _levelF
                {
                    int blockWidth = GetBlockWidth(levelFine);
                    for (int y = 0; y < blockWidth; y++)
                    for (int x = 0; x < blockWidth; x++)
                    {
                        int ii = BlockCoord2Idx(x, y, blockWidth);
                        int ci = phc + ii;
                        int fi = phf + ii;
                    
                        _vCoarse[ci] = _vFine[fi];
                    }
                }
            }
        }
        
        [BurstCompile]
        private struct SamplerField : IJobParallelFor
        {
            [ReadOnly] private NativeArray<int4> _lut;
            [ReadOnly] private NativeArray<float> _vFine;
            [ReadOnly] private NativeArray<float> _vCoarse;
            [WriteOnly] private NativeArray<float> _result;
            private readonly int _width;
            
            public SamplerField(NativeArray<float> vf, NativeArray<float> vc, NativeArray<int4> lut,
                NativeArray<float> res, int width)
            {
                _vFine = vf;
                _lut = lut;
                _vCoarse = vc;
                _result = res;
                _width = width;
            }

            public void Execute(int i)
            {
                const float width = MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth;
                // const float cellSize = 0.5f / width;
                float x = i % _width;
                float y = i / _width;
                float2 pos = ((new float2(x, y) + 0.5f) / _width * width);
                // _result[i] = SampleBilinear(_vFine, _vCoarse, _lut, pos);
                SampleGridFaceBilinear(pos, _vFine, _vCoarse, _lut, out var data);
                _result[i] = data;
            }
            private void SampleGridFaceBilinear(float2 pos, NativeArray<float> vf1, NativeArray<float> vc1, 
                NativeArray<int4> lut, out float u1)
            {
                u1 = 0;
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                const int gridSize = MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth;
                float2 basePos = pos * MSBGConstants.InvBaseCellSize;
                int2 baseCoord = (int2)math.floor(basePos);
                if (math.any(baseCoord < 0) || math.any(baseCoord >= gridSize))
                    return;
                
                int2 blockCoord = baseCoord / baseBlockWidth;
                int4 info = lut[Coord2Idx(blockCoord)];
                int blockLevel = GetCurLevel(info.z);
                if (blockLevel < 0)
                    return;
                
                float2 localPos = (basePos - blockCoord * baseBlockWidth) / (1 << blockLevel);
                int blockWidth = baseBlockWidth >> blockLevel;
                float2 offset = new float2(0.5f, 0.5f);
                float2 localUV = localPos - offset; 
                float2 weight = localUV - math.floor(localUV);

                // Inside block
                if (math.all(localUV > 0 & localUV < blockWidth - 1))
                {
                    int2 c0 = math.max(0, (int2)math.floor(localUV));
                    int2 c1 = c0 + 1;
                    int ptr = info.x;

                    if (math.all(weight < 1e-5f))
                    {
                        int idx = ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth);
                        u1 = vf1[idx];
                    }
                    else if (math.all(weight > 0.9999f))
                    {
                        int idx = ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth);
                        u1 = vf1[idx];
                    }
                    else
                    {
                        int idx00 = ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth);
                        int idx10 = ptr + BlockCoord2Idx(c1.x, c0.y, blockWidth);
                        int idx01 = ptr + BlockCoord2Idx(c0.x, c1.y, blockWidth);
                        int idx11 = ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth);
                        u1 = LerpBilinear(weight, vf1[idx00], vf1[idx10], vf1[idx01], vf1[idx11]);
                    }
                    return;
                }

                int fineLevel, coarseLevel;
                // edge between same level blocks
                {
                    int2 lbBlockCoord = blockCoord - math.select(int2.zero, 1, localUV < 0 & blockCoord > 0);
                    int4 infoLB = lut[Coord2Idx(lbBlockCoord)];
                    int4 neighborLevel = GetNeighborsLevel(infoLB.z);
                    blockLevel = neighborLevel.x;
                    bool isRight = localUV.x < 0 || localUV.x > blockWidth - 1, isTop = localUV.y < 0 || localUV.y > blockWidth - 1;
                    int levelR = math.select(blockLevel, neighborLevel.y, isRight);
                    int levelT = math.select(blockLevel, neighborLevel.z, isTop);
                    int levelRT = math.select(math.select(neighborLevel.y, neighborLevel.z, isTop), neighborLevel.w,
                        isRight && isTop);
                    fineLevel = math.min(math.min(blockLevel, levelR), math.min(levelT, levelRT));
                    coarseLevel = math.max(math.max(blockLevel, levelR), math.max(levelT, levelRT));
                    if (coarseLevel == fineLevel)
                    {
                        bool2 selector = weight > 0.5f;
                        int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                        int2 c1 = c0 + (1 << blockLevel);
                        SamplePointFine(vf1,  lut, c0.x, c0.y, out float lb1);
                        SamplePointFine(vf1,  lut, c1.x, c0.y, out float rb1);
                        SamplePointFine(vf1,  lut, c0.x, c1.y, out float lt1);
                        SamplePointFine(vf1,  lut, c1.x, c1.y, out float rt1);
                        u1 = LerpBilinear(weight, lb1, rb1, lt1, rt1);
                        return;
                    }
                }
                
                int2 faceBlockCoord = blockCoord + math.select(int2.zero, new int2(1,0), localUV.x > blockWidth - 0.5f & blockCoord > 0);
                int4 infoFace = lut[Coord2Idx(faceBlockCoord)];
            
                blockLevel = GetCurLevel(infoFace.z);
                blockCoord = faceBlockCoord;
                
                if (blockLevel == coarseLevel)
                {
                    float2 coarsePos = (basePos - blockCoord * baseBlockWidth) / (1 << coarseLevel);
                    float2 coarseUV = coarsePos - offset;
                    float2 coarseWeight = coarseUV - math.floor(coarseUV);
                    bool2 selector = coarseWeight > 0.5f;
                    int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                    int2 c1 = c0 + (1 << blockLevel);
                    SamplePointLevel(vf1, vc1, lut, c0.x, c0.y, coarseLevel, out float lb1);
                    SamplePointLevel(vf1, vc1, lut, c1.x, c0.y, coarseLevel, out float rb1);
                    SamplePointLevel(vf1, vc1, lut, c0.x, c1.y, coarseLevel, out float lt1);
                    SamplePointLevel(vf1, vc1, lut, c1.x, c1.y, coarseLevel, out float rt1);
                    u1 = LerpBilinear(weight, lb1, rb1, lt1, rt1);
                }
                else
                {
                    float2 coarsePos = (basePos - blockCoord * baseBlockWidth) / (1 << coarseLevel);
                    float2 coarseUV = coarsePos - offset;
                    float2 coarseWeight = coarseUV - math.floor(coarseUV);
                    
                    float2 finePos = (basePos - blockCoord * baseBlockWidth) / (1 << fineLevel);
                    float2 fineUV = finePos - offset;
                    float2 fineWeight = fineUV - math.floor(fineUV);
                    
                    bool2 selector = coarseWeight > 0.5f;
                    int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                    int2 c1 = c0 + (1 << coarseLevel);
                    SamplePointLevel(vf1, vc1, lut, c0.x, c0.y, coarseLevel, out float lb1);
                    SamplePointLevel(vf1, vc1, lut, c1.x, c0.y, coarseLevel, out float rb1);
                    SamplePointLevel(vf1, vc1, lut, c0.x, c1.y, coarseLevel, out float lt1);
                    SamplePointLevel(vf1, vc1, lut, c1.x, c1.y, coarseLevel, out float rt1);
                    
                    int4 neighborLevelCur = GetNeighborsLevel(infoFace.z);
                    int4 neighborLevelLB = GetNeighborsLevel(lut[Coord2Idx(math.max(0, faceBlockCoord - 1))].z);
                    int levelR = neighborLevelCur.y;
                    int levelT = neighborLevelCur.z;
                    int levelRT = neighborLevelCur.w;
                    int levelL = neighborLevelLB.z;
                    int levelB = neighborLevelLB.y;
                    int levelLB = neighborLevelLB.x;
                    int levelLT = GetCurLevel(lut[Coord2Idx(math.clamp(faceBlockCoord + new int2(-1, 1), 0, MSBGConstants.GridWidth - 1))].z);
                    int levelRB = GetCurLevel(lut[Coord2Idx(math.clamp(faceBlockCoord + new int2(1, -1), 0, MSBGConstants.GridWidth - 1))].z);
                    float dstToCoarse = 1;
                    localPos = fineUV + new float2(1, 0.5f);
                    float2 subPos = GetBlockWidth(fineLevel) - fineUV - new float2(0.5f, 0.5f);
                    // return math.cmin(localPos);
                    int levelC = GetCurLevel(infoFace.z);
                    if (levelC < levelL) dstToCoarse = math.min(dstToCoarse, localPos.x);
                    if (levelC < levelR) dstToCoarse = math.min(dstToCoarse, subPos.x);
                    if (levelC < levelT) dstToCoarse = math.min(dstToCoarse, subPos.y);
                    if (levelC < levelB) dstToCoarse = math.min(dstToCoarse, localPos.y);
                    if (levelC < levelLB) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, localPos.y));
                    if (levelC < levelRT) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, subPos.y));
                    if (levelC < levelLT) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, subPos.y));
                    if (levelC < levelRB) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, localPos.y));
                    dstToCoarse = math.min(1, dstToCoarse * 2);

                    var valueCoarse1 = LerpBilinear(coarseWeight, lb1, rb1, lt1, rt1);
                    
                    
                    selector = fineWeight > 0.5f;
                    c0 = baseCoord - math.select(int2.zero, 1 << fineLevel, selector);
                    c1 = c0 + (1 << fineLevel);

                    SamplePointFine(vf1, lut, c0.x, c0.y, out lb1);
                    SamplePointFine(vf1, lut, c1.x, c0.y, out rb1);
                    SamplePointFine(vf1, lut, c0.x, c1.y, out lt1);
                    SamplePointFine(vf1, lut, c1.x, c1.y, out rt1);
                    
                    var valueFine1 = LerpBilinear(fineWeight, lb1, rb1, lt1, rt1);
                    
                    u1 = math.lerp(valueCoarse1, valueFine1, dstToCoarse);
                }
            }
            
            private static void SamplePointFine(NativeArray<float> v1,  NativeArray<int4> lut,
                int x, int y, out float r1)
            {
                var baseCoord = math.clamp(new int2(x, y), 0, MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth - 1);
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                int2 blockCoord = baseCoord / baseBlockWidth;
                int blockIdx = Coord2Idx(blockCoord);
                int4 info = lut[blockIdx];
                int level = GetCurLevel(info.z);
                r1 = 0;
                if (level < 0)
                    return;
                int blockWidth = baseBlockWidth >> level;
                int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
                r1 = v1[info.x + BlockCoord2Idx(localCoord, blockWidth)];
            }

            private static void SamplePointLevel(NativeArray<float> vf1, NativeArray<float> vc1, 
                NativeArray<int4> lut, int x, int y, int targetLevel, out float r1)
            {
                var baseCoord = math.clamp(new int2(x, y), 0, MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth - 1);
                const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
                int2 blockCoord = baseCoord / baseBlockWidth;
                int blockIdx = Coord2Idx(blockCoord);
                int4 info = lut[blockIdx];
                int level = GetCurLevel(info.z);
                
                r1 = 0;
                if (level < 0) return;
                
                if (level == targetLevel)
                {
                    int blockWidth = baseBlockWidth >> level;
                    int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
                    r1 = vf1[info.x + BlockCoord2Idx(localCoord, blockWidth)];
                }
                else
                {
                    // UnityEngine.Debug.Assert(level < targetLevel);
                    int blockWidth = baseBlockWidth >> targetLevel;
                    int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> targetLevel;
                    r1 = vc1[info.y + BlockCoord2Idx(localCoord, blockWidth)];
                }
            }
        }
        #endregion
        
        #region HelperFunctions

        private static void FillHaloBlock(NativeArray<float> v, NativeArray<float3> a, NativeArray<int2> lut,
            NativeArray<float> block, NativeArray<float3> param, int2 coord, int2 info)
        {
            int level = info.y;
            int blockWidth = GetBlockWidth(level);
            int haloBlockWidth = blockWidth + 2;
            for (int by = 0; by < blockWidth; by++)
            for (int bx = 0; bx < blockWidth; bx++)
            {
                int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
                int physicsIdx = info.x + BlockCoord2Idx(bx, by, blockWidth);
                    
                block[localIdx] = v[physicsIdx];
                param[localIdx] = a[physicsIdx];
            }
            int4 ox = new int4(-1, 0, 1, 0);
            int4 oy = new int4(0, -1, 0, 1);
            
            for (int n = 0; n < 4; n++)
            {
                int2 dir = new int2(ox[n], oy[n]);
                int2 curr = coord + dir;
                if (curr.x < 0 || curr.y < 0 || curr.x >= MSBGConstants.GridWidth || curr.y >= MSBGConstants.GridWidth)
                    continue;
                
                int2 neighborInfo = lut[Coord2Idx(curr)];
                int nLevel = neighborInfo.y;
                if (nLevel < 0)
                    continue;
                
                int phn = neighborInfo.x;
                if (nLevel == level)
                {
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 nCoord = math.select(math.select(c, 0, dir > 0), blockWidth - 1, dir < 0);
                        int nLocalIdx = BlockCoord2Idx(nCoord, blockWidth);
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = v[phn + nLocalIdx];
                        param[paddingIdx] = a[phn + nLocalIdx];
                    }
                }
                else if (nLevel > level)
                {
                    int nBlockWidth = GetBlockWidth(nLevel);
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                        int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = v[phn + nLocalIdx];
                        param[paddingIdx] = 2f/3f;
                    }
                }
                else // n_level < level
                {
                    int nBlockWidth = GetBlockWidth(nLevel);
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                        int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                        int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                        int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = (v[phn + nLocalIdx0] + v[phn + nLocalIdx1]);
                        param[paddingIdx] = 2f/3f;
                    }
                }
            }
        }
        
        private static float SamplePoint(NativeArray<float> v, NativeArray<int2> lut, int2 baseCoord)
        {
            baseCoord = math.clamp(baseCoord, 0, MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth - 1);
            const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
            int2 blockCoord = baseCoord / baseBlockWidth;
            int blockIdx = Coord2Idx(blockCoord);
            int2 info = lut[blockIdx];
            int level = info.y;
            if (level < 0)
                return 0;
            int blockWidth = baseBlockWidth >> level;
            int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
            return v[info.x + BlockCoord2Idx(localCoord, blockWidth)];
        }

        public static float SampleBilinear(NativeArray<float> vf, NativeArray<float> vc, NativeArray<int4> lut, float2 pos)
        {
            const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
            const int gridSize = MSBGConstants.GridWidth * baseBlockWidth;
            float2 basePos = pos / MSBGConstants.BaseCellSize;
            int2 baseCoord = math.clamp((int2)math.floor(basePos), 0, gridSize - 1);
            int2 blockCoord = baseCoord / baseBlockWidth;
            int4 info = lut[Coord2Idx(blockCoord)];
            int blockLevel = GetCurLevel(info.z);
            if (blockLevel < 0)
                return 0;
            
            float2 localPos = (basePos - blockCoord * baseBlockWidth) / (1 << blockLevel);
            int blockWidth = GetBlockWidth(blockLevel);
            float2 localUV = localPos - 0.5f;
            float2 weight = localUV - math.floor(localUV);

            if (math.all(localUV > 0 & localUV < blockWidth - 1))
            {
                int2 c0 = math.max(0, (int2)math.floor(localUV));
                int2 c1 = c0 + 1;
                int ptr = info.x;
                // return 0;

                if (math.all(weight < 1e-5f))
                    return vf[ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth)];

                if (math.all(weight > 0.9999f))
                    return vf[ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth)];

                var lb = vf[ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth)];
                var rb = vf[ptr + BlockCoord2Idx(c1.x, c0.y, blockWidth)];
                var lt = vf[ptr + BlockCoord2Idx(c0.x, c1.y, blockWidth)];
                var rt = vf[ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth)];

                return LerpBilinear(weight, lb, rb, lt, rt);
            }

            int2 lbBlockCoord = blockCoord - math.select(int2.zero, 1, localUV < 0 & blockCoord > 0);
            int4 infoLB = lut[Coord2Idx(lbBlockCoord)];
            int4 neighborLevel = GetNeighborsLevel(infoLB.z);
            blockLevel = neighborLevel.x;
            bool isRight = localUV.x < 0 || localUV.x > blockWidth - 1;
            bool isTop = localUV.y < 0 || localUV.y > blockWidth - 1;
            int levelR = math.select(blockLevel, neighborLevel.y, isRight);
            int levelT = math.select(blockLevel, neighborLevel.z, isTop);
            int levelRT = math.select(math.select(neighborLevel.y, neighborLevel.z, isTop), neighborLevel.w,
                isRight && isTop);
            int fineLevel = math.min(math.min(blockLevel, levelR), math.min(levelT, levelRT));
            int coarseLevel = math.max(math.max(blockLevel, levelR), math.max(levelT, levelRT));
            if (coarseLevel == fineLevel)
            {
                int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, weight > 0.5f);
                int2 c1 = c0 + (1 << blockLevel);
                var lb = SamplePointFine(vf, lut, c0.x, c0.y);
                var rb = SamplePointFine(vf, lut, c1.x, c0.y);
                var lt = SamplePointFine(vf, lut, c0.x, c1.y);
                var rt = SamplePointFine(vf, lut, c1.x, c1.y);
                // return 0;
                return LerpBilinear(weight, lb, rb, lt, rt);
            }

            
            // return localUV.x;
            // UnityEngine.Debug.Assert(coarseLevel - fineLevel == 1, "MSBG SampleBilinear: level difference greater than 1");
            if (GetCurLevel(info.z) == coarseLevel)
            {
                int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, weight > 0.5f);
                int2 c1 = c0 + (1 << blockLevel);
                var lb = SamplePointLevel(vf, vc, lut, c0.x, c0.y, coarseLevel);
                var rb = SamplePointLevel(vf, vc, lut, c1.x, c0.y, coarseLevel);
                var lt = SamplePointLevel(vf, vc, lut, c0.x, c1.y, coarseLevel);
                var rt = SamplePointLevel(vf, vc, lut, c1.x, c1.y, coarseLevel);
                return LerpBilinear(weight, lb, rb, lt, rt);
            }
            else
            {
                // return 0;
                float2 coarsePos = (basePos - blockCoord * baseBlockWidth) / (1 << coarseLevel);
                float2 coarseUV = coarsePos - 0.5f;
                float2 coarseWeight = coarseUV - math.floor(coarseUV);
                
                int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, coarseWeight > 0.5f);
                int2 c1 = c0 + (1 << coarseLevel);
                var lb = SamplePointLevel(vf, vc, lut, c0.x, c0.y, coarseLevel);
                var rb = SamplePointLevel(vf, vc, lut, c1.x, c0.y, coarseLevel);
                var lt = SamplePointLevel(vf, vc, lut, c0.x, c1.y, coarseLevel);
                var rt = SamplePointLevel(vf, vc, lut, c1.x, c1.y, coarseLevel);
                
                int4 neighborLevelCur = GetNeighborsLevel(info.z);
                int4 neighborLevelLB = GetNeighborsLevel(lut[Coord2Idx(math.max(0, blockCoord - 1))].z);
                levelR = neighborLevelCur.y;
                levelT = neighborLevelCur.z;
                levelRT = neighborLevelCur.w;
                int levelL = neighborLevelLB.z;
                int levelB = neighborLevelLB.y;
                int levelLB = neighborLevelLB.x;
                int levelLT = GetCurLevel(lut[Coord2Idx(math.max(0, blockCoord + new int2(-1, 1)))].z);
                int levelRB = GetCurLevel(lut[Coord2Idx(math.max(0, blockCoord + new int2(1, -1)))].z);
                float dstToCoarse = 1;
                float2 subPos = blockWidth - localPos;
                int levelC = GetCurLevel(info.z);
                if (levelC < levelL) dstToCoarse = math.min(dstToCoarse, localPos.x);
                if (levelC < levelR) dstToCoarse = math.min(dstToCoarse, subPos.x);
                if (levelC < levelT) dstToCoarse = math.min(dstToCoarse, subPos.y);
                if (levelC < levelB) dstToCoarse = math.min(dstToCoarse, localPos.y);
                if (levelC < levelLB) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, localPos.y));
                if (levelC < levelRT) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, subPos.y));
                if (levelC < levelLT) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, subPos.y));
                if (levelC < levelRB) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, localPos.y));

                float valueCoarse = LerpBilinear(coarseWeight, lb, rb, lt, rt);
                
                c0 = baseCoord - math.select(int2.zero, 1 << fineLevel, weight > 0.5f);
                c1 = c0 + (1 << fineLevel);
                
                lb = SamplePointFine(vf, lut, c0.x, c0.y);
                rb = SamplePointFine(vf, lut, c1.x, c0.y);
                lt = SamplePointFine(vf, lut, c0.x, c1.y);
                rt = SamplePointFine(vf, lut, c1.x, c1.y);
                
                float valueFine = LerpBilinear(weight, lb, rb, lt, rt);
                
                dstToCoarse *= 2;
                return math.lerp(valueCoarse, valueFine, dstToCoarse);
            }
        }
        
        private static float SampleGridFaceBilinear(int axis, NativeArray<float> vf1, NativeArray<float> vc1, 
            NativeArray<int4> lut, float2 pos)
        {
            const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
            const int gridSize = MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth;
            float2 basePos = pos * MSBGConstants.InvBaseCellSize;
            int2 baseCoord = (int2)math.floor(basePos);
            if (math.any(baseCoord < 0) || math.any(baseCoord >= gridSize))
                return 0;
            
            int2 blockCoord = baseCoord / baseBlockWidth;
            int4 info = lut[Coord2Idx(blockCoord)];
            int blockLevel = GetCurLevel(info.z);
            if (blockLevel < 0)
                return 0;
            
            float2 localPos = (basePos - blockCoord * baseBlockWidth) / (1 << blockLevel);
            int blockWidth = baseBlockWidth >> blockLevel;
            float2 offset = new float2(0.5f, 0.5f);
            offset[axis] = 0;
            float2 localUV = localPos - offset; 
            float2 weight = localUV - math.floor(localUV);

            // Inside block
            if (math.all(localUV > 0 & localUV < blockWidth - 1))
            {
                int2 c0 = math.max(0, (int2)math.floor(localUV));
                int2 c1 = c0 + 1;
                int ptr = info.x;

                // return 0;
                float u1;

                if (math.all(weight < 1e-5f))
                {
                    int idx = ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth);
                    u1 = vf1[idx];
                }
                else if (math.all(weight > 0.9999f))
                {
                    int idx = ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth);
                    u1 = vf1[idx];
                }
                else
                {
                    int idx00 = ptr + BlockCoord2Idx(c0.x, c0.y, blockWidth);
                    int idx10 = ptr + BlockCoord2Idx(c1.x, c0.y, blockWidth);
                    int idx01 = ptr + BlockCoord2Idx(c0.x, c1.y, blockWidth);
                    int idx11 = ptr + BlockCoord2Idx(c1.x, c1.y, blockWidth);
                    u1 = LerpBilinear(weight, vf1[idx00], vf1[idx10], vf1[idx01], vf1[idx11]);
                }
                return u1;
            }

            // return 0;
            // return 

            int fineLevel, coarseLevel;
            
            // edge between same level blocks
            {
                int2 lbBlockCoord = blockCoord - math.select(int2.zero, 1, localUV < 0 & blockCoord > 0);
                int4 infoLB = lut[Coord2Idx(lbBlockCoord)];
                int4 neighborLevel = GetNeighborsLevel(infoLB.z);
                blockLevel = neighborLevel.x;
                bool isRight = localUV.x < 0 || localUV.x > blockWidth - 1, isTop = localUV.y < 0 || localUV.y > blockWidth - 1;
                int levelR = math.select(blockLevel, neighborLevel.y, isRight);
                int levelT = math.select(blockLevel, neighborLevel.z, isTop);
                int levelRT = math.select(math.select(neighborLevel.y, neighborLevel.z, isTop), neighborLevel.w,
                    isRight && isTop);
                fineLevel = math.min(math.min(blockLevel, levelR), math.min(levelT, levelRT));
                coarseLevel = math.max(math.max(blockLevel, levelR), math.max(levelT, levelRT));
                if (coarseLevel == fineLevel)
                {
                    // return 0;
                    bool2 selector = weight > 0.5f;
                    selector[axis] = false;
                    int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                    int2 c1 = c0 + (1 << blockLevel);
                    float lb1 = SamplePointFine(vf1, lut, c0.x, c0.y);
                    float rb1 = SamplePointFine(vf1, lut, c1.x, c0.y);
                    float lt1 = SamplePointFine(vf1, lut, c0.x, c1.y);
                    float rt1 = SamplePointFine(vf1, lut, c1.x, c1.y);
                    return LerpBilinear(weight, lb1, rb1, lt1, rt1);
                }
            }
            // return -0.1f;

            // return localUV.x > blockWidth - 0.5f ? 1 : 0;
            int2 faceBlockCoord = blockCoord + math.select(int2.zero, new int2(1,0), localUV.x > blockWidth - 0.5f & blockCoord > 0);
            int4 infoFace = lut[Coord2Idx(faceBlockCoord)];
            
            blockLevel = GetCurLevel(infoFace.z);
            blockCoord = faceBlockCoord;
            // UnityEngine.Debug.Assert(coarseLevel - fineLevel == 1, "MSBG SampleBilinear: level difference greater than 1");
            if (blockLevel == coarseLevel)
            {
                float2 coarsePos = (basePos - blockCoord * baseBlockWidth) / (1 << coarseLevel);
                float2 coarseUV = coarsePos - offset;
                float2 coarseWeight = coarseUV - math.floor(coarseUV);
                // return -0.1f;
                bool2 selector = coarseWeight > 0.5f;
                selector[axis] = false;
                int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                int2 c1 = c0 + (1 << blockLevel);
                var lb = SamplePointLevel(vf1, vc1, lut, c0.x, c0.y, coarseLevel);
                var rb = SamplePointLevel(vf1, vc1, lut, c1.x, c0.y, coarseLevel);
                var lt = SamplePointLevel(vf1, vc1, lut, c0.x, c1.y, coarseLevel);
                var rt = SamplePointLevel(vf1, vc1, lut, c1.x, c1.y, coarseLevel);
                return LerpBilinear(coarseWeight, lb, rb, lt, rt);
            }
            else
            {
                // return weight.x;
                float2 coarsePos = (basePos - blockCoord * baseBlockWidth) / (1 << coarseLevel);
                float2 coarseUV = coarsePos - offset;
                float2 coarseWeight = coarseUV - math.floor(coarseUV);
                
                float2 finePos = (basePos - blockCoord * baseBlockWidth) / (1 << fineLevel);
                float2 fineUV = finePos - offset;
                float2 fineWeight = fineUV - math.floor(fineUV);
                // return fineWeight.x;
                
                // return coarseWeight.x;
                bool2 selector = coarseWeight > 0.5f;
                selector[axis] = false;
                int2 c0 = baseCoord - math.select(int2.zero, 1 << blockLevel, selector);
                int2 c1 = c0 + (1 << coarseLevel);
                float lb1 = SamplePointLevel(vf1, vc1,  lut, c0.x, c0.y, coarseLevel);
                float rb1 = SamplePointLevel(vf1, vc1,  lut, c1.x, c0.y, coarseLevel);
                float lt1 = SamplePointLevel(vf1, vc1,  lut, c0.x, c1.y, coarseLevel);
                float rt1 = SamplePointLevel(vf1, vc1,  lut, c1.x, c1.y, coarseLevel);

                // float2 vec = basePos - c0;
                // return vec.x / (1 << coarseLevel);
                int4 neighborLevelCur = GetNeighborsLevel(infoFace.z);
                int4 neighborLevelLB = GetNeighborsLevel(lut[Coord2Idx(math.max(0, faceBlockCoord - 1))].z);
                int levelR = neighborLevelCur.y;
                int levelT = neighborLevelCur.z;
                int levelRT = neighborLevelCur.w;
                int levelL = neighborLevelLB.z;
                int levelB = neighborLevelLB.y;
                int levelLB = neighborLevelLB.x;
                int levelLT = GetCurLevel(lut[Coord2Idx(math.max(0, faceBlockCoord + new int2(-1, 1)))].z);
                int levelRB = GetCurLevel(lut[Coord2Idx(math.max(0, faceBlockCoord + new int2(1, -1)))].z);
                float dstToCoarse = 1;
                localPos = fineUV + new float2(1, 0.5f);
                float2 subPos = GetBlockWidth(fineLevel) - fineUV - new float2(0.5f, 0.5f);
                // return math.cmin(localPos);
                int levelC = GetCurLevel(infoFace.z);
                if (levelC < levelL) dstToCoarse = math.min(dstToCoarse, localPos.x);
                if (levelC < levelR) dstToCoarse = math.min(dstToCoarse, subPos.x);
                if (levelC < levelT) dstToCoarse = math.min(dstToCoarse, subPos.y);
                if (levelC < levelB) dstToCoarse = math.min(dstToCoarse, localPos.y);
                if (levelC < levelLB) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, localPos.y));
                if (levelC < levelRT) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, subPos.y));
                if (levelC < levelLT) dstToCoarse = math.min(dstToCoarse, math.max(localPos.x, subPos.y));
                if (levelC < levelRB) dstToCoarse = math.min(dstToCoarse, math.max(subPos.x, localPos.y));
                dstToCoarse = math.min(1, dstToCoarse * 2);
                // if (GetCurLevel(info.z) > GetCurLevel(infoFace.z))
                //     dstToCoarse = math.min(1, dstToCoarse*2);
                // return levelC < levelL ? localPos.x : 0;
                // return dstToCoarse;

                var valueCoarse1 = LerpBilinear(coarseWeight, lb1, rb1, lt1, rt1);
                
                // return fineWeight.y;
                selector = fineWeight > 0.5f;
                selector[axis] = false;
                c0 = baseCoord - math.select(int2.zero, 1 << fineLevel, selector);
                c1 = c0 + (1 << fineLevel);

                lb1 = SamplePointFine(vf1, lut, c0.x, c0.y);
                rb1 = SamplePointFine(vf1, lut, c1.x, c0.y);
                lt1 = SamplePointFine(vf1, lut, c0.x, c1.y);
                rt1 = SamplePointFine(vf1, lut, c1.x, c1.y);
                var valueFine1 = LerpBilinear(fineWeight, lb1, rb1, lt1, rt1);
                // return valueFine1;
                return math.lerp(valueCoarse1, valueFine1, dstToCoarse);
            }
        }
        
        private static float SamplePointFine(NativeArray<float> v, NativeArray<int4> lut, int x, int y)
        {
            var baseCoord = math.clamp(new int2(x, y), 0, MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth - 1);
            const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
            int2 blockCoord = baseCoord / baseBlockWidth;
            int blockIdx = Coord2Idx(blockCoord);
            int4 info = lut[blockIdx];
            int level = GetCurLevel(info.z);
            if (level < 0)
                return 0;
            int blockWidth = baseBlockWidth >> level;
            int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
            return v[info.x + BlockCoord2Idx(localCoord, blockWidth)];
        }

        private static float SamplePointLevel(NativeArray<float> vf, NativeArray<float> vc, NativeArray<int4> lut, int x, int y, int targetLevel)
        {
            var baseCoord = math.clamp(new int2(x, y), 0, MSBGConstants.GridWidth * MSBGConstants.BaseBlockWidth - 1);
            const int baseBlockWidth = MSBGConstants.BaseBlockWidth;
            int2 blockCoord = baseCoord / baseBlockWidth;
            int blockIdx = Coord2Idx(blockCoord);
            int4 info = lut[blockIdx];
            int level = GetCurLevel(info.z);
            if (level < 0)
                return 0;
            if (level == targetLevel)
            {
                int blockWidth = baseBlockWidth >> level;
                int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> level;
                return vf[info.x + BlockCoord2Idx(localCoord, blockWidth)];
            }
            else
            {
                UnityEngine.Debug.Assert(level < targetLevel);
                int blockWidth = baseBlockWidth >> targetLevel;
                int2 localCoord = (baseCoord - blockCoord * baseBlockWidth) >> targetLevel;
                return vc[info.y + BlockCoord2Idx(localCoord, blockWidth)];
            }
        }
        
        private static int GetCurLevel(int code) => code < 0 ? -1 : (code & 15);

        private static int4 GetNeighborsLevel(int code) =>
            new int4(code & 15, (code >> 4) & 15, (code >> 8) & 15, (code >> 12) & 15);

        private static int PackNeighborsLevel(int4 levels) => 
            (levels.x & 15) | ((levels.y & 15) << 4) | ((levels.z & 15) << 8) | ((levels.w & 15) << 12);

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
        
        public static void FillHaloBlock1(NativeArray<float> v, NativeArray<int2> lut,
            NativeArray<float> block, NativeArray<float> param, int2 coord, int2 info)
        {
            int level = info.y;
            int blockWidth = GetBlockWidth(level);
            int haloBlockWidth = blockWidth + 2;
            for (int by = 0; by < blockWidth; by++)
            for (int bx = 0; bx < blockWidth; bx++)
            {
                int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloBlockWidth);
                int physicsIdx = info.x + BlockCoord2Idx(bx, by, blockWidth);
                    
                block[localIdx] = v[physicsIdx];
                param[localIdx] = 1;
            }
            int4 ox = new int4(-1, 0, 1, 0);
            int4 oy = new int4(0, -1, 0, 1);
            
            for (int n = 0; n < 4; n++)
            {
                int2 dir = new int2(ox[n], oy[n]);
                int2 curr = coord + dir;
                if (curr.x < 0 || curr.y < 0 || curr.x >= MSBGConstants.GridWidth || curr.y >= MSBGConstants.GridWidth)
                    continue;
                
                int2 neighborInfo = lut[Coord2Idx(curr)];
                int nLevel = neighborInfo.y;
                if (nLevel < 0)
                    continue;
                
                int phn = neighborInfo.x;
                if (nLevel == level)
                {
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 nCoord = math.select(math.select(c, 0, dir > 0), blockWidth - 1, dir < 0);
                        int nLocalIdx = BlockCoord2Idx(nCoord, blockWidth);
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = v[phn + nLocalIdx];
                        param[paddingIdx] = 1;
                    }
                }
                else if (nLevel > level)
                {
                    int nBlockWidth = GetBlockWidth(nLevel);
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                        int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = v[phn + nLocalIdx];
                        param[paddingIdx] = 2f / 3f;
                    }
                }
                else // n_level < level
                {
                    int nBlockWidth = GetBlockWidth(nLevel);
                    for (int c = 0; c < blockWidth; c++)
                    {
                        int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                        int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                        int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                        int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                        int2 cCoord = math.select(math.select(c + 1, haloBlockWidth - 1, dir > 0), 0, dir < 0);
                        int paddingIdx = BlockCoord2Idx(cCoord, haloBlockWidth);
                        block[paddingIdx] = 0.5f * (v[phn + nLocalIdx0] + v[phn + nLocalIdx1]);
                        param[paddingIdx] = 4f / 3f;
                    }
                }
            }
        }
        private static float NeighborSum1(NativeArray<float> block, NativeArray<float> param, out float csum, int x, int y, int blockRes)
        {
            int l = BlockCoord2Idx(x - 1, y, blockRes);
            int b = BlockCoord2Idx(x, y - 1, blockRes);
            int r = BlockCoord2Idx(x + 1, y, blockRes);
            int t = BlockCoord2Idx(x, y + 1, blockRes);
            float pl = param[l], pr = param[r], pb = param[b], pt = param[t];
            float vl = block[l], vr = block[r], vb = block[b], vt = block[t];
            csum = pl + pr + pb + pt;
            return pl * vl + pr * vr + pb * vb + pt * vt;
        }
        
        private static bool InActive(float x) => math.abs(x) < 1e-6f;

        private static int BlockCoord2Idx(int2 coord, int res) => coord.x + coord.y * res;

        private static int BlockCoord2Idx(int x, int y, int res) => x + y * res;

        private static int2 Idx2Coord(int i) => new int2(i % MSBGConstants.GridWidth, i / MSBGConstants.GridWidth);

        private static int Coord2Idx(int2 i) => Coord2Idx(i.x, i.y);
        private static int Coord2Idx(int x, int y) => x + y * MSBGConstants.GridWidth;

        private static float GetCellSize(int level) => MSBGConstants.BaseCellSize * (1 << level);
        
        private static int GetBlockWidth(int level) => MSBGConstants.BaseBlockWidth >> level;
        private static int GetBlockSize(int level) => (MSBGConstants.BaseBlockWidth >> level) * (MSBGConstants.BaseBlockWidth >> level);
        
        private static float LerpBilinear(float2 weight, float lb, float rb, float lt, float rt)
        {
            var b = math.lerp(lb, rb, weight.x);
            var t = math.lerp(lt, rt, weight.x);
            return math.lerp(b, t, weight.y);
        }
        #endregion
    }
}

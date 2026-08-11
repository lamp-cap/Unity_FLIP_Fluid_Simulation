using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public class AdaptiveNarrowBandSolver : MonoBehaviour
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
    
    
    // ── 领域常量 ──────────────────────────────────────────────────────────
    private const int GridWidth      = 16;  // 块网格宽度（块数/边）
    private const int GridCount      = GridWidth * GridWidth;
    private const int BlockWorldSize = 8;   // level 0 块覆盖的精细格数（边长）
    private const int FineWidth      = GridWidth * BlockWorldSize; // = 128 精细格/边
    private const int FineCount      = FineWidth * FineWidth;
    private const int Band1          = 2;   // cellSDF ≥ Band1：深水，不靠粒子判类型
    private const int Band2          = 3;   // cellSDF ≥ Band2：无粒子
    private const int ChebyThresh0   = 1;   // chebySDF ≤ 此值 → level 0
    private const int ChebyThresh1   = 2;   // chebySDF = 此值 → level 1
    private const int SDF_Inf        = FineWidth; // 距离场无穷大安全上界
    private const uint SOLID = 2, AIR = 1, FLUID = 0;

    // ── 块级数据 ──────────────────────────────────────────────────────────
    private NativeArray<int>  _blockChebySDF;  // [GridCount] L∞ 距离到最近粒子块
    private NativeArray<int>  _blockLevel;     // [GridCount] 块精度 0/1/2（当前帧）
    private NativeArray<int>  _blockLevelPrev; // [GridCount] 上一帧精度（检测变化）
    private NativeArray<int2> _gridInfos;      // [GridCount] (level, flatPtr)

    // ── cell 级数据（精细格坐标，FineWidth × FineWidth）────────────────────
    // 仅 level 0 块内的格子有效；level 1/2 块位置固定写 -1（屏障）。
    // 使用独立精细格数组，使 Manhattan 扫描可以直接按行序线性遍历，
    // 无需处理 _gridInfos 的跨块稀疏索引。
    private NativeArray<int>  _fineCellSDF;  // [FineCount] 曼哈顿距离；-1 = 非流体/屏障
    private NativeArray<uint> _fineCellType; // [FineCount] FLUID/AIR/SOLID（细格类型）

    // ── 网格物理量（flat，via _gridInfos[b].y + localIdx）────────────────
    private NativeArray<float2> _gridVelocity;   // MAC 面速度：.x=左面 u，.y=下面 v
    private NativeArray<float2> _gridVelocitySwap; // 平流/FLIP 差分用旧速度快照
    private NativeArray<float3> _gridLaplacian;  // 有限体积矩阵系数 (diag, -left, -down)
    private NativeArray<float>  _pressure;
    private NativeArray<float>  _flux;           // 散度 / RHS

    // ── 粒子数据 ──────────────────────────────────────────────────────────
    // 粒子仅存在于 level 0 块内 cellSDF ∈ [0, Band2) 的格子
    private NativeArray<float2> _particlePos;
    private NativeArray<float2> _particleVel;
    private NativeArray<float2> _particlePosSwap; // 双缓冲（resample 输出写入此处）
    private NativeArray<float2> _particleVelSwap;

    // 每细格粒子区间 [begin, end)，索引入 _particleIDs（按细格哈希排序后的粒子列表）
    private NativeArray<int2> _particleRange; // [FineCount]
    private NativeArray<int>  _particleIDs;   // 排序后粒子索引（大小 = 粒子总数）

    // ── 辅助 / 临时量 ─────────────────────────────────────────────────────
    private NativeArray<int>         _blockIsSeed;  // [GridCount] 该块当帧是否含粒子
    private NativeReference<int>     _totalCells;   // flat 数组当前总格数（prefix-sum 结果）
    private NativeReference<int>     _particleCount;// 当前粒子总数
    private NativeReference<int> _cellCount;

    // ── 上一帧场快照（半拉格朗日平流的采样源）────────────────────────────
    private NativeArray<int2>   _gridInfosPrev;    // 旧场结构 (level, flatPtr)
    private int _flatCapacity;                     // 当前 flat 物理量数组容量

    // ── 仿真配置（Inspector）──────────────────────────────────────────────
    [SerializeField] private float _dt         = 1f / 60f;
    [SerializeField] private int   _substeps   = 1;
    [SerializeField, Range(0, 1)] private float _flipBlend = 0.95f; // 1=纯FLIP，0=纯PIC
    [SerializeField] private float2 _gravity   = new float2(0f, -9.81f);
    [SerializeField] private int   _maxParticles = FineCount * 4;   // 粒子容量上界
    private bool _initialized;

    // ── 工具函数（静态内联，与 MultiresSparseBlockGrids.cs 命名一致）────────
    private static int  Coord2Idx(int x, int y) => x + y * GridWidth;
    private static int2 Idx2Coord(int i)          => new int2(i % GridWidth, i / GridWidth);
    private static int  FineCoord2Idx(int x, int y) => x + y * FineWidth;
    private static int2 FineIdx2Coord(int i)          => new int2(i % FineWidth, i / FineWidth);

    // level 0 → BlockWidth=8, level 1 → 4, level 2 → 2
    private static int   BlockWidth(int level)  => 1 << (3 - level);
    // level 0 → h=1.0, level 1 → 2.0, level 2 → 4.0
    private static float GetH(int level)        => 1 << level;
    private static int   GetBlockSize(int level) { int w = BlockWidth(level); return w * w; }
    private static int   BlockCoord2Idx(int x, int y, int w) => x + y * w;

    private static int ChebySDFToLevel(int sdf) =>
        sdf <= ChebyThresh0 ? 0 : (sdf <= ChebyThresh1 ? 1 : 2);


    private static int CellsPerEdge(int L) => FineWidth >> L;

    // 判定采样点 pos 应采用的层：取其 2×2 块邻域中最粗（level 最大）的块。
    private static int DetermineSampleLevel(float2 pos, NativeArray<int> blockLevel)
    {
        int2 b0 = (int2)math.floor(pos / BlockWorldSize);
        int lvl = 0;
        for (int dy = 0; dy <= 1; dy++)
        for (int dx = 0; dx <= 1; dx++)
        {
            int2 b = math.clamp(b0 + new int2(dx, dy), 0, GridWidth - 1);
            lvl = math.max(lvl, blockLevel[Coord2Idx(b.x, b.y)]);
        }
        return lvl;
    }

    // 读取「层 L、全局格 (i,j) 的左面 u」。
    //   i==0 或 i==cellsPerEdge → 域左/右边界，诺依曼返回 0。
    //   块本身为层 L → 直读 .x；块更细 → 该粗左面 = ratio 道共线细左面平均。
    private static float ReadFaceU(int i, int j, int L,
        NativeArray<int2> gridInfos, NativeArray<float2> field)
    {
        int cpe = CellsPerEdge(L);
        if (i <= 0 || i >= cpe) return 0f;      // 域左右边界：诺依曼 0
        j = math.clamp(j, 0, cpe - 1);          // 横向轴 clamp，防越界
        int wL = BlockWidth(L);
        int bx = i / wL, by = j / wL;
        int2 gi = gridInfos[Coord2Idx(bx, by)];
        int bl = gi.x, ptr = gi.y;
        if (bl == L)
        {
            int lx = i - bx * wL, ly = j - by * wL;
            return field[ptr + BlockCoord2Idx(lx, ly, wL)].x;
        }
        // 块更细：粗左面覆盖 ratio 道细左面（同一 x，y 方向展开 ratio 道）
        int wB    = BlockWidth(bl);
        int ratio = 1 << (L - bl);
        int fi = i * ratio - bx * wL * ratio;   // 块内细格 x（左面对齐）
        int fj0 = j * ratio - by * wL * ratio;  // 块内细格 y 起点
        float acc = 0f;
        for (int k = 0; k < ratio; k++)
        {
            int ly = math.clamp(fj0 + k, 0, wB - 1);
            acc += field[ptr + BlockCoord2Idx(fi, ly, wB)].x;
        }
        return acc / ratio;
    }

    // 读取「层 L、全局格 (i,j) 的下面 v」。j==0/cellsPerEdge → 边界 0。
    private static float ReadFaceV(int i, int j, int L,
        NativeArray<int2> gridInfos, NativeArray<float2> field)
    {
        int cpe = CellsPerEdge(L);
        if (j <= 0 || j >= cpe) return 0f;
        i = math.clamp(i, 0, cpe - 1);          // 横向轴 clamp，防越界
        int wL = BlockWidth(L);
        int bx = i / wL, by = j / wL;
        int2 gi = gridInfos[Coord2Idx(bx, by)];
        int bl = gi.x, ptr = gi.y;
        if (bl == L)
        {
            int lx = i - bx * wL, ly = j - by * wL;
            return field[ptr + BlockCoord2Idx(lx, ly, wL)].y;
        }
        int wB    = BlockWidth(bl);
        int ratio = 1 << (L - bl);
        int fj = j * ratio - by * wL * ratio;   // 块内细格 y（下面对齐）
        int fi0 = i * ratio - bx * wL * ratio;
        float acc = 0f;
        for (int k = 0; k < ratio; k++)
        {
            int lx = math.clamp(fi0 + k, 0, wB - 1);
            acc += field[ptr + BlockCoord2Idx(lx, fj, wB)].y;
        }
        return acc / ratio;
    }

    // 采样 u 分量（左面网格：x 整，y 半偏移）。
    private static float SampleU(float2 pos, int L,
        NativeArray<int2> gridInfos, NativeArray<float2> field)
    {
        float h = GetH(L);
        float gx = pos.x / h;            // 左面在整数 x
        float gy = pos.y / h - 0.5f;     // 面中点在 y 半偏移
        int i0 = (int)math.floor(gx), j0 = (int)math.floor(gy);
        float fx = gx - i0, fy = gy - j0;
        float u00 = ReadFaceU(i0,   j0,   L, gridInfos, field);
        float u10 = ReadFaceU(i0+1, j0,   L, gridInfos, field);
        float u01 = ReadFaceU(i0,   j0+1, L, gridInfos, field);
        float u11 = ReadFaceU(i0+1, j0+1, L, gridInfos, field);
        return math.lerp(math.lerp(u00, u10, fx), math.lerp(u01, u11, fx), fy);
    }

    // 采样 v 分量（下面网格：x 半偏移，y 整）。
    private static float SampleV(float2 pos, int L,
        NativeArray<int2> gridInfos, NativeArray<float2> field)
    {
        float h = GetH(L);
        float gx = pos.x / h - 0.5f;
        float gy = pos.y / h;
        int i0 = (int)math.floor(gx), j0 = (int)math.floor(gy);
        float fx = gx - i0, fy = gy - j0;
        float v00 = ReadFaceV(i0,   j0,   L, gridInfos, field);
        float v10 = ReadFaceV(i0+1, j0,   L, gridInfos, field);
        float v01 = ReadFaceV(i0,   j0+1, L, gridInfos, field);
        float v11 = ReadFaceV(i0+1, j0+1, L, gridInfos, field);
        return math.lerp(math.lerp(v00, v10, fx), math.lerp(v01, v11, fx), fy);
    }

    // 简化跨级 MAC 速度采样：分量各在其错开网格上双线性插值，统一采样层由 pos 定。
    private static float2 SampleVelocitySimplified(
        float2 pos, NativeArray<int> blockLevel,
        NativeArray<int2> gridInfos, NativeArray<float2> field)
    {
        int L = DetermineSampleLevel(pos, blockLevel);
        return new float2(
            SampleU(pos, L, gridInfos, field),
            SampleV(pos, L, gridInfos, field));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ① MarkParticleBlockSeedsJob
    //    逐粒子标记：该粒子所在块写 blockIsSeed = 1。
    //    每帧在 ComputeBlockChebyshevSDFJob 之前运行，之后 _blockIsSeed 被
    //    用作切比雪夫距离场的初始种子（距离 0）。
    // ═══════════════════════════════════════════════════════════════════════
    [BurstCompile]
    private struct MarkParticleBlockSeedsJob : IJobParallelFor
    {
        [ReadOnly]  public NativeArray<float2> ParticlePos;
        // 允许并发写入（只写 1，多线程写同一块时结果相同，无竞争问题）
        [NativeDisableParallelForRestriction]
        public NativeArray<int> BlockIsSeed;

        public void Execute(int i)
        {
            float2 pos    = ParticlePos[i];
            int2   bCoord = (int2)math.floor(pos / BlockWorldSize);
            bCoord = math.clamp(bCoord, 0, GridWidth - 1);
            BlockIsSeed[Coord2Idx(bCoord.x, bCoord.y)] = 1;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ② ComputeBlockChebyshevSDFJob
    //    切比雪夫（L∞）距离变换：从含粒子块向外传播距离，8邻域 min-plus 双遍。
    //
    //    入参 BlockChebySDF 初始化规则（调用前由调度代码完成）：
    //      blockIsSeed[b] == 1 → BlockChebySDF[b] = 0（种子）
    //      blockIsSeed[b] == 0 → BlockChebySDF[b] = SDF_Inf
    //
    //    与 MultiresSparseBlockGrids.cs:718 ComputeDistanceFieldJob 结构完全相同，
    //    差别仅在种子来源：粒子块而非硬编码几何区域。
    // ═══════════════════════════════════════════════════════════════════════
    [BurstCompile]
    private struct ComputeBlockChebyshevSDFJob : IJob
    {
        public NativeArray<int> BlockChebySDF;

        public void Execute()
        {
            const int W = GridWidth;
            // 前向扫描：线性顺序 (0,0)→(W-1,W-1)，处理4个前驱8邻方向
            for (int i = 0; i < GridCount; i++)
            {
                int v = BlockChebySDF[i];
                if (v <= 0) continue;        // 种子（0）不向自身传播
                int2 c = Idx2Coord(i);
                if (c.x > 0)
                    v = math.min(v, 1 + BlockChebySDF[Coord2Idx(c.x - 1, c.y)]);
                if (c.y > 0)
                    v = math.min(v, 1 + BlockChebySDF[Coord2Idx(c.x,     c.y - 1)]);
                if (c.x > 0 && c.y > 0)
                    v = math.min(v, 1 + BlockChebySDF[Coord2Idx(c.x - 1, c.y - 1)]);
                if (c.x < W - 1 && c.y > 0)
                    v = math.min(v, 1 + BlockChebySDF[Coord2Idx(c.x + 1, c.y - 1)]);
                BlockChebySDF[i] = v;
            }
            // 后向扫描：(W-1,W-1)→(0,0)，处理另4个后继8邻方向
            for (int i = GridCount - 1; i >= 0; i--)
            {
                int v = BlockChebySDF[i];
                if (v <= 0) continue;
                int2 c = Idx2Coord(i);
                if (c.x < W - 1)
                    v = math.min(v, 1 + BlockChebySDF[Coord2Idx(c.x + 1, c.y)]);
                if (c.y < W - 1)
                    v = math.min(v, 1 + BlockChebySDF[Coord2Idx(c.x,     c.y + 1)]);
                if (c.x < W - 1 && c.y < W - 1)
                    v = math.min(v, 1 + BlockChebySDF[Coord2Idx(c.x + 1, c.y + 1)]);
                if (c.x > 0 && c.y < W - 1)
                    v = math.min(v, 1 + BlockChebySDF[Coord2Idx(c.x - 1, c.y + 1)]);
                BlockChebySDF[i] = v;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ③ AssignBlockLevelsJob
    //    从 chebySDF 映射到 level 0/1/2，检测变化并写 _blockLevelPrev。
    // ═══════════════════════════════════════════════════════════════════════
    [BurstCompile]
    private struct AssignBlockLevelsJob : IJobParallelFor
    {
        [ReadOnly]  public NativeArray<int> BlockChebySDF;
        public           NativeArray<int>  BlockLevel;
        public           NativeArray<int>  BlockLevelPrev;
        [WriteOnly] public NativeArray<int> LevelChanged; // [GridCount] 1 = 该块精度改变

        public void Execute(int i)
        {
            int prev  = BlockLevel[i];
            int next  = ChebySDFToLevel(BlockChebySDF[i]);
            BlockLevelPrev[i]  = prev;
            BlockLevel[i]      = next;
            LevelChanged[i]    = (next != prev) ? 1 : 0;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ④ RebuildGridInfosJob
    //    根据最新的 blockLevel 重算 _gridInfos 的 (level, flatPtr)。
    //    单线程顺序做前缀和，结果写入 GridInfos 和 TotalCells。
    //    注意：此 Job 只更新索引结构；实际数据的 restrict/prolongate
    //    需在此之后由单独的 RestrictProlongateJob 完成。
    // ═══════════════════════════════════════════════════════════════════════
    [BurstCompile]
    private struct RebuildGridInfosJob : IJob
    {
        [ReadOnly]  public NativeArray<int>  BlockLevel;
        [WriteOnly] public NativeArray<int2> GridInfos;
        public           NativeReference<int> TotalCells;

        public void Execute()
        {
            int ptr = 0;
            for (int i = 0; i < GridCount; i++)
            {
                int level = BlockLevel[i];
                GridInfos[i] = new int2(level, ptr);
                ptr += GetBlockSize(level);
            }
            TotalCells.Value = ptr;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ⑤ InitBlockChebySDFJob
    //    用 blockIsSeed 初始化切比雪夫距离场：种子=0，其余=SDF_Inf。
    //    在 MarkParticleBlockSeedsJob 之后、ComputeBlockChebyshevSDFJob 之前运行。
    // ═══════════════════════════════════════════════════════════════════════
    [BurstCompile]
    private struct InitBlockChebySDFJob : IJobParallelFor
    {
        [ReadOnly]  public NativeArray<int> BlockIsSeed;
        [WriteOnly] public NativeArray<int> BlockChebySDF;

        public void Execute(int i) =>
            BlockChebySDF[i] = (BlockIsSeed[i] != 0) ? 0 : SDF_Inf;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ⑥ SetCellSeedJob
    //    为 cell 级曼哈顿 SDF 设置初始种子值。
    //    仅在 level 0 块内的精细格有效；其余位置写 -1（屏障，不参与传播）。
    //
    //    种子规则（与 NarrowBand_FLIP.cs:571 SetGridLevelJob 逻辑一致）：
    //      非流体格（AIR/SOLID）→ -1
    //      液面格（FLUID 且4邻之一非 FLUID）→ 0（距离种子）
    //      内部流体格（FLUID 且全部4邻均为 FLUID）→ SDF_Inf
    // ═══════════════════════════════════════════════════════════════════════
    [BurstCompile]
    private struct SetCellSeedJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int>  BlockLevel;
        [ReadOnly] public NativeArray<uint> FineCellType; // FLUID/AIR/SOLID，精细格坐标
        [WriteOnly] public NativeArray<int> FineCellSDF;

        public void Execute(int i)
        {
            int2 fc = FineIdx2Coord(i);
            int bx = fc.x / BlockWorldSize, by = fc.y / BlockWorldSize;
            int blockIdx = Coord2Idx(bx, by);

            // 非 level 0 块 → 屏障
            if (BlockLevel[blockIdx] != 0) { FineCellSDF[i] = -1; return; }

            uint t = FineCellType[i];
            if (t != FLUID) { FineCellSDF[i] = -1; return; }

            // 检查4轴向邻居：任意一个非 FLUID → 液面种子
            bool isSurface = false;
            if (fc.x > 0)           isSurface |= FineCellType[FineCoord2Idx(fc.x-1, fc.y)] != FLUID;
            if (fc.x < FineWidth-1) isSurface |= FineCellType[FineCoord2Idx(fc.x+1, fc.y)] != FLUID;
            if (fc.y > 0)           isSurface |= FineCellType[FineCoord2Idx(fc.x, fc.y-1)] != FLUID;
            if (fc.y < FineWidth-1) isSurface |= FineCellType[FineCoord2Idx(fc.x, fc.y+1)] != FLUID;

            FineCellSDF[i] = isSurface ? 0 : SDF_Inf;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ⑦ ComputeCellManhattanSDFJob
    //    曼哈顿距离场双遍扫描（与 NarrowBand_FLIP.cs:595 逻辑相同）。
    //    扫描在 FineWidth × FineWidth 精细格坐标上进行：
    //      - level 0 格子：正常参与扫描（可读可写）
    //      - 屏障格（SDF = -1）：跳过更新，同时不向邻居传播
    //    跨 level 0/level 1 边界的格子自然被 -1 屏障阻断，无需显式处理。
    // ═══════════════════════════════════════════════════════════════════════
    [BurstCompile]
    private struct ComputeCellManhattanSDFJob : IJob
    {
        public NativeArray<int> FineCellSDF;

        public void Execute()
        {
            const int FW = FineWidth;
            // 前向扫描：(0,0)→(FW-1,FW-1)，2 个前驱方向（左、下）
            for (int i = 0; i < FineCount; i++)
            {
                int v = FineCellSDF[i];
                if (v <= 0) continue;   // 种子(0) 或屏障(-1) 不更新
                int2 fc = FineIdx2Coord(i);
                if (fc.x > 0)
                {
                    int nb = FineCellSDF[FineCoord2Idx(fc.x - 1, fc.y)];
                    if (nb >= 0) v = math.min(v, 1 + nb);
                }
                if (fc.y > 0)
                {
                    int nb = FineCellSDF[FineCoord2Idx(fc.x, fc.y - 1)];
                    if (nb >= 0) v = math.min(v, 1 + nb);
                }
                FineCellSDF[i] = v;
            }
            // 后向扫描：(FW-1,FW-1)→(0,0)，2 个后继方向（右、上）
            for (int i = FineCount - 1; i >= 0; i--)
            {
                int v = FineCellSDF[i];
                if (v <= 0) continue;
                int2 fc = FineIdx2Coord(i);
                if (fc.x < FW - 1)
                {
                    int nb = FineCellSDF[FineCoord2Idx(fc.x + 1, fc.y)];
                    if (nb >= 0) v = math.min(v, 1 + nb);
                }
                if (fc.y < FW - 1)
                {
                    int nb = FineCellSDF[FineCoord2Idx(fc.x, fc.y + 1)];
                    if (nb >= 0) v = math.min(v, 1 + nb);
                }
                FineCellSDF[i] = v;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ⑧ SetCellTypeJob
    //    根据 cell SDF + 粒子范围确定每个精细格的类型（FLUID/AIR/SOLID）。
    //
    //    判定规则（与 NarrowBand_FLIP.cs:818 SetGridTypeJob 一致）：
    //      level 1/2 块（非 level 0）→ 恒为 FLUID（深水欧拉区，不靠粒子判断）
    //      level 0 块：
    //        cellSDF >= Band1（确定深水带）→ FLUID（与粒子存在无关）
    //        cellSDF ∈ [0, Band1)（近液面）→ 看格内是否有粒子：有→FLUID，无→AIR
    //        cellSDF == -1（非流体屏障）→ AIR（或 SOLID，此处简化为 AIR）
    // ═══════════════════════════════════════════════════════════════════════
    [BurstCompile]
    private struct SetCellTypeJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int>  BlockLevel;
        [ReadOnly] public NativeArray<int>  FineCellSDF;
        [ReadOnly] public NativeArray<int2> ParticleRange;  // per 精细格 [begin, end)
        [WriteOnly] public NativeArray<uint> FineCellType;

        public void Execute(int i)
        {
            int2 fc = FineIdx2Coord(i);
            int bx = fc.x / BlockWorldSize, by = fc.y / BlockWorldSize;
            if (BlockLevel[Coord2Idx(bx, by)] != 0)
            {
                // level 1/2 块：深水欧拉，恒为 FLUID
                FineCellType[i] = FLUID;
                return;
            }

            int sdf = FineCellSDF[i];
            if (sdf < 0)           { FineCellType[i] = AIR;   return; } // 屏障/非流体
            if (sdf >= Band1)      { FineCellType[i] = FLUID; return; } // 确定深水

            // 近液面：靠粒子占据决定 FLUID/AIR
            int2 range = ParticleRange[i];
            FineCellType[i] = (range.y > range.x) ? FLUID : AIR;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ⑨ ResampleParticlesJob
    //    维持 level 0 窄带内的粒子密度。
    //    逐精细格并行：
    //      - cellSDF ∈ [0, Band2)：活跃格，目标 TargetCount 颗粒子；
    //        不足时用 farthest-point 候选采样补充（与 NarrowBand_FLIP.cs:720 一致）。
    //      - cellSDF >= Band2 或不在 level 0 → 清空该格的粒子（不拷入输出缓冲）。
    // ═══════════════════════════════════════════════════════════════════════
    [BurstCompile]
    private struct ResampleParticlesJob : IJobParallelFor
    {
        private const int TargetCount  = 4;  // 每格目标粒子数（2D；3D 建议8）
        private const int CandidateDiv = 2;  // 候选点每维数量（CandidateDiv²=4 个候选）

        [ReadOnly] public NativeArray<int>   BlockLevel;
        [ReadOnly] public NativeArray<int>   FineCellSDF;
        [ReadOnly] public NativeArray<int2>  ParticleRange;  // 输入：当前格粒子区间
        [ReadOnly] public NativeArray<float2> PosIn;
        [ReadOnly] public NativeArray<float2> VelIn;
        [ReadOnly] public NativeArray<float2> GridVelocity;  // 用于给新粒子赋速度（MAC 采样）
        [ReadOnly] public NativeArray<int2>  GridInfos;

        // 输出：紧凑写入新粒子缓冲（写入偏移由 ParticleRange 的 begin 决定）
        [NativeDisableParallelForRestriction] public NativeArray<float2> PosOut;
        [NativeDisableParallelForRestriction] public NativeArray<float2> VelOut;
        // 输出：更新后的粒子区间（begin = 输入 begin，end = 实际写入数量 + begin）
        [WriteOnly] public NativeArray<int2> ParticleRangeOut;

        public void Execute(int i)
        {
            int2 fc = FineIdx2Coord(i);
            int bx = fc.x / BlockWorldSize, by = fc.y / BlockWorldSize;
            int blockIdx = Coord2Idx(bx, by);

            // level 1/2 或 cellSDF >= Band2 → 无粒子，清空
            int sdf = FineCellSDF[i];
            if (BlockLevel[blockIdx] != 0 || sdf < 0 || sdf >= Band2)
            {
                ParticleRangeOut[i] = new int2(0, 0);
                return;
            }

            int2   range  = ParticleRange[i];
            int    count  = range.y - range.x;
            float2 origin = (float2)fc; // level 0 h=1，格左下角世界坐标 = 精细格坐标

            // 先把现有粒子拷入输出缓冲
            int outBase = range.x;
            for (int j = 0; j < count; j++)
            {
                PosOut[outBase + j] = PosIn[range.x + j];
                VelOut[outBase + j] = VelIn[range.x + j];
            }

            // 不足 TargetCount → farthest-point 候选采样补充新粒子
            int toAdd = TargetCount - count;
            for (int k = 0; k < toAdd; k++)
            {
                // CandidateDiv² 个均匀子格候选点
                float  bestDist = -1f;
                float2 bestPos  = origin + 0.5f;
                for (int cy = 0; cy < CandidateDiv; cy++)
                for (int cx = 0; cx < CandidateDiv; cx++)
                {
                    float2 cand = origin + (new float2(cx, cy) + 0.5f) / CandidateDiv;
                    // 计算到已有粒子（含本轮已添加的）的最小距离
                    float minDist = float.MaxValue;
                    int filled = count + k;
                    for (int j = 0; j < filled; j++)
                        minDist = math.min(minDist, math.lengthsq(PosOut[outBase + j] - cand));
                    if (minDist > bestDist) { bestDist = minDist; bestPos = cand; }
                }
                // 新粒子速度：在候选位置对 MAC 速度场做简化跨级采样
                float2 gv = SampleVelocitySimplified(bestPos, BlockLevel, GridInfos, GridVelocity);
                PosOut[outBase + count + k] = bestPos;
                VelOut[outBase + count + k] = gv;
            }

            int written = math.max(count, TargetCount);
            ParticleRangeOut[i] = new int2(outBase, outBase + written);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ⑩ GridAdvectionJob（半拉格朗日，MAC 交错）
    //    新场每格的 u/v 两道面各自独立 backtrace：
    //      u 在左面 (i·h, (j+0.5)·h)，v 在下面 ((i+0.5)·h, j·h)。
    //    从各面位置回溯采样旧速度场 → 完成级别切换的数据迁移（feedback #1，
    //    不用 restrict/prolongate）。跨级由 SampleVelocitySimplified 处理。
    //
    //    按块并行：每块遍历自身格，写各格 float2(.x=u, .y=v)。避免 flat→块反查。
    // ═══════════════════════════════════════════════════════════════════════
    [BurstCompile]
    private struct GridAdvectionJob : IJobParallelFor  // 索引 = 块 idx ∈ [0, GridCount)
    {
        [ReadOnly] public NativeArray<int2> GridInfos;    // 当前帧（新场）索引结构
        [ReadOnly] public NativeArray<int>  BlockLevelOld;// 旧场层（采样用）
        [ReadOnly] public NativeArray<int2> GridInfosOld; // 旧场索引结构
        [ReadOnly] public NativeArray<float2> VelOld;     // 旧速度场
        [NativeDisableParallelForRestriction]
        public NativeArray<float2> VelNew;                // 新速度场（输出，各块写自身段）
        public float Dt;

        // 从面位置 backtrace 一步，取回溯点的对应速度分量
        private float BacktraceComponent(float2 facePos, bool isU)
        {
            float2 uHere = SampleVelocitySimplified(facePos, BlockLevelOld, GridInfosOld, VelOld);
            float2 back  = facePos - Dt * uHere;
            float2 uBack = SampleVelocitySimplified(back, BlockLevelOld, GridInfosOld, VelOld);
            return isU ? uBack.x : uBack.y;
        }

        public void Execute(int blockIdx)
        {
            int2 gi     = GridInfos[blockIdx];
            int  level  = gi.x, ptr = gi.y;
            int  w      = BlockWidth(level);
            float h     = GetH(level);
            int2 bc     = Idx2Coord(blockIdx);
            float2 origin = (float2)bc * BlockWorldSize;

            for (int ly = 0; ly < w; ly++)
            for (int lx = 0; lx < w; lx++)
            {
                // 左面 u 与下面 v 各自的世界坐标
                float2 uFacePos = origin + new float2(lx,        ly + 0.5f) * h;
                float2 vFacePos = origin + new float2(lx + 0.5f, ly       ) * h;
                float uNew = BacktraceComponent(uFacePos, true);
                float vNew = BacktraceComponent(vFacePos, false);
                VelNew[ptr + BlockCoord2Idx(lx, ly, w)] = new float2(uNew, vNew);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 每帧流水线方法
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// 阶段一：更新块级切比雪夫 SDF 并重新分配块精度。
    /// 必须在粒子排序（BuildLUT）之后、cell SDF 之前调用。
    /// </summary>
    private JobHandle UpdateBlockSDF(JobHandle dep, NativeArray<int> levelChanged)
    {
        // 清空上帧种子标记
        var clearSeeds = new ClearBlockSeedsJob { BlockIsSeed = _blockIsSeed };
        dep = clearSeeds.Schedule(_blockIsSeed.Length, 64, dep);

        // 逐粒子标记所在块
        var markSeeds = new MarkParticleBlockSeedsJob
        {
            ParticlePos  = _particlePos,
            BlockIsSeed  = _blockIsSeed,
        };
        dep = markSeeds.Schedule(_particleCount.Value, 64, dep);

        // 初始化距离场（种子=0，其余=SDF_Inf）
        var initSDF = new InitBlockChebySDFJob
        {
            BlockIsSeed   = _blockIsSeed,
            BlockChebySDF = _blockChebySDF,
        };
        dep = initSDF.Schedule(_blockChebySDF.Length, 64, dep);

        // 切比雪夫双遍扫描（单线程，串行）
        dep = new ComputeBlockChebyshevSDFJob { BlockChebySDF = _blockChebySDF }.Schedule(dep);

        // 映射 chebySDF → level，记录变化
        var assign = new AssignBlockLevelsJob
        {
            BlockChebySDF  = _blockChebySDF,
            BlockLevel     = _blockLevel,
            BlockLevelPrev = _blockLevelPrev,
            LevelChanged   = levelChanged,
        };
        dep = assign.Schedule(_blockLevel.Length, 64, dep);

        // 重建 _gridInfos 前缀和（单线程）
        dep = new RebuildGridInfosJob
        {
            BlockLevel = _blockLevel,
            GridInfos  = _gridInfos,
            TotalCells = _totalCells,
        }.Schedule(dep);

        return dep;
    }

    /// <summary>
    /// 阶段三：更新 cell 级曼哈顿 SDF（在 SetCellType 之前，且需粒子范围已建好）。
    /// </summary>
    private JobHandle UpdateCellSDF(JobHandle dep)
    {
        // 设置种子：液面格=0，内部流体=SDF_Inf，非流体=-1
        var setSeeds = new SetCellSeedJob
        {
            BlockLevel   = _blockLevel,
            FineCellType = _fineCellType,
            FineCellSDF  = _fineCellSDF,
        };
        dep = setSeeds.Schedule(_fineCellSDF.Length, 128, dep);

        // 曼哈顿双遍扫描（单线程，串行）
        dep = new ComputeCellManhattanSDFJob { FineCellSDF = _fineCellSDF }.Schedule(dep);
        return dep;
    }

    /// <summary>
    /// 阶段四：从 SDF + 粒子范围确定格子类型。
    /// </summary>
    private JobHandle UpdateCellTypes(JobHandle dep)
    {
        dep = new SetCellTypeJob
        {
            BlockLevel    = _blockLevel,
            FineCellSDF   = _fineCellSDF,
            ParticleRange = _particleRange,
            FineCellType  = _fineCellType,
        }.Schedule(_fineCellType.Length, 128, dep);
        return dep;
    }

    /// <summary>
    /// 阶段五：维持窄带粒子密度（resample）。
    /// 注意：此处需要在调用前为 _particlePosSwap / _particleVelSwap 预分配足够空间。
    /// </summary>
    private JobHandle ResampleParticles(JobHandle dep, NativeArray<int2> rangeOut)
    {
        dep = new ResampleParticlesJob
        {
            BlockLevel      = _blockLevel,
            FineCellSDF     = _fineCellSDF,
            ParticleRange   = _particleRange,
            PosIn           = _particlePos,
            VelIn           = _particleVel,
            GridVelocity    = _gridVelocity,
            GridInfos       = _gridInfos,
            PosOut          = _particlePosSwap,
            VelOut          = _particleVelSwap,
            ParticleRangeOut = rangeOut,
        }.Schedule(_fineCellType.Length, 64, dep);
        return dep;
    }

    /// <summary>
    /// 阶段六：把旧速度场半拉格朗日平流到新场（同时完成级别切换迁移）。
    /// 需要 _gridInfosPrev / _gridVelocitySwap 持有上一帧的场快照。
    /// </summary>
    private JobHandle AdvectGrid(JobHandle dep)
    {
        dep = new GridAdvectionJob
        {
            GridInfos     = _gridInfos,       // 新场结构
            BlockLevelOld = _blockLevelPrev,  // 旧场层
            GridInfosOld  = _gridInfosPrev,   // 旧场结构
            VelOld        = _gridVelocitySwap,// 旧速度快照
            VelNew        = _gridVelocity,    // 写入新场
            Dt            = _dt,
        }.Schedule(GridCount, 4, dep);
        return dep;
    }

    // ── 以下阶段先留桩，逐个补齐（保持框架可编译、流程完整）──────────────

    /// <summary>阶段二：粒子按细格哈希排序，建 _particleRange / _particleIDs。
    /// TODO: 复用 MultiresSparseBlockGrids.cs 的 Hash/BuildLut/CombineLut/Shuffle。</summary>
    private JobHandle BuildParticleLUT(JobHandle dep)
    {
        // [TODO:LUT] 计数排序：按 FineCoord2Idx(floor(pos)) 分桶 → 前缀和 → shuffle。
        return dep;
    }

    /// <summary>阶段七：自适应 P2G，粒子速度散布到格心（level 2 块跳过，纯欧拉）。
    /// TODO: 复用 ParticleToGridJob 的二次 B 样条权重 + GetCellSize 缩放。</summary>
    private JobHandle ParticleToGrid(JobHandle dep)
    {
        // [TODO:P2G] scatter：逐粒子按所在块层 splat 到 3×3(或对应)邻域格心。
        return dep;
    }

    /// <summary>阶段八：施加重力等体力到网格速度。</summary>
    private JobHandle ApplyForces(JobHandle dep)
    {
        // [TODO:FORCES] VelNew += gravity * dt（仅 FLUID 格）。可直接一个 IJobParallelFor。
        return dep;
    }

    /// <summary>阶段九：压力投影（MGPCG）。TODO: 复用 MSBG_Solver 的矩阵装配 + 求解。</summary>
    private JobHandle PressureSolve(JobHandle dep)
    {
        // [TODO:PRESSURE] 装配 _gridLaplacian（2:1 边界 2/3 系数）→ 散度 → MGPCG → 减梯度。
        return dep;
    }

    /// <summary>阶段十：速度外插到窄带外一圈（保证 G2P/平流边界稳定）。</summary>
    private JobHandle ExtrapolateVelocity(JobHandle dep)
    {
        // [TODO:EXTRAP] 从 FLUID 格向 AIR 格做若干轮邻域平均外插。
        return dep;
    }

    /// <summary>阶段十一：G2P，用简化跨级采样把新旧速度差（FLIP）+ 新速度（PIC）回传粒子。</summary>
    private JobHandle GridToParticle(JobHandle dep)
    {
        // [TODO:G2P] 逐粒子：uPIC = Sample(VelNew)，uFLIP = uOld + (Sample(VelNew)-Sample(VelOld))
        //            vel = lerp(uPIC, uFLIP, _flipBlend)。采样用 SampleVelocitySimplified。
        return dep;
    }

    /// <summary>阶段十二：粒子平流 pos += vel*dt，并做边界/固体约束。</summary>
    private JobHandle AdvectParticles(JobHandle dep)
    {
        // [TODO:ADVECT] RK2 更好；先 pos += vel*dt，clamp 进域内。
        return dep;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 每帧总编排
    // ═══════════════════════════════════════════════════════════════════════
    private void Update()
    {
        if (!_initialized) return;
        for (int s = 0; s < _substeps; s++)
            StepSimulation();
    }

    private void StepSimulation()
    {
        // 保存上一帧场快照（供半拉格朗日平流从旧场采样）
        SnapshotPrevGrid();

        JobHandle dep = default;
        var levelChanged = new NativeArray<int>(GridCount, Allocator.TempJob);

        // ① 粒子排序 → 建 LUT（块 SDF 的种子标记也依赖粒子位置）
        dep = BuildParticleLUT(dep);

        // ② 块级切比雪夫 SDF + 重分层 + 重建 _gridInfos
        dep = UpdateBlockSDF(dep, levelChanged);

        // ③ 网格结构可能已变 → 保证 flat 容量充足（同步点）
        dep.Complete();
        EnsureGridCapacity();

        // ④ 半拉格朗日平流旧速度场 → 新场（含级别迁移）
        dep = AdvectGrid(default);

        // ⑤ cell 级曼哈顿 SDF（仅 level 0）
        dep = UpdateCellSDF(dep);

        // ⑥ 细格类型（FLUID/AIR/SOLID）
        dep = UpdateCellTypes(dep);

        // ⑦ resample 维持窄带粒子密度（双缓冲，之后 swap）
        var rangeOut = new NativeArray<int2>(FineCount, Allocator.TempJob);
        dep = ResampleParticles(dep, rangeOut);
        dep.Complete();
        SwapParticleBuffers(rangeOut);

        // ⑧ P2G → 受力 → 压力投影 → 外插 → G2P → 粒子平流
        dep = ParticleToGrid(default);
        dep = ApplyForces(dep);
        dep = PressureSolve(dep);
        dep = ExtrapolateVelocity(dep);
        dep = GridToParticle(dep);
        dep = AdvectParticles(dep);
        dep.Complete();

        levelChanged.Dispose();
        rangeOut.Dispose();
    }

    // 上一帧场快照：把当前 _gridInfos / _gridVelocity 拷入 Prev/Swap 供平流采样。
    private void SnapshotPrevGrid()
    {
        if (!_gridInfosPrev.IsCreated) return;
        _gridInfos.CopyTo(_gridInfosPrev);
        if (_gridVelocitySwap.IsCreated && _gridVelocity.IsCreated)
            _gridVelocity.CopyTo(_gridVelocitySwap);
    }

    // resample 后交换粒子双缓冲，并更新 _particleRange。
    private void SwapParticleBuffers(NativeArray<int2> rangeOut)
    {
        (_particlePos, _particlePosSwap) = (_particlePosSwap, _particlePos);
        (_particleVel, _particleVelSwap) = (_particleVelSwap, _particleVel);
        rangeOut.CopyTo(_particleRange);
    }

    // flat 物理量容量保证。最坏情形 = 全部块 level 0 = GridCount·64 = FineCount，
    // 故直接按 FineCount 预分配一次即可，无需逐帧扩容。
    private void EnsureGridCapacity()
    {
        if (_flatCapacity >= FineCount) return; // 已足量
        // 仅首帧走到这里（首帧 EnsureGridCapacity 前 flat 数组尚未创建）
        _flatCapacity     = FineCount;
        _gridVelocity     = new NativeArray<float2>(FineCount, Allocator.Persistent);
        _gridVelocitySwap = new NativeArray<float2>(FineCount, Allocator.Persistent);
        _gridLaplacian    = new NativeArray<float3>(FineCount, Allocator.Persistent);
        _pressure         = new NativeArray<float>(FineCount,  Allocator.Persistent);
        _flux             = new NativeArray<float>(FineCount,  Allocator.Persistent);
        _gridInfosPrev    = new NativeArray<int2>(GridCount,   Allocator.Persistent);
        // 首帧无旧场：把首帧结构作为「旧场」，速度置零 → 平流为恒等，安全。
        _gridInfos.CopyTo(_gridInfosPrev);
    }

    // ── ClearBlockSeedsJob（辅助）──────────────────────────────────────────
    [BurstCompile]
    private struct ClearBlockSeedsJob : IJobParallelFor
    {
        public NativeArray<int> BlockIsSeed;
        public void Execute(int i) => BlockIsSeed[i] = 0;
    }

    // ── 生命周期 ──────────────────────────────────────────────────────────
    private void Awake()
    {
        _blockChebySDF  = new NativeArray<int>(GridCount,  Allocator.Persistent);
        _blockLevel     = new NativeArray<int>(GridCount,  Allocator.Persistent);
        _blockLevelPrev = new NativeArray<int>(GridCount,  Allocator.Persistent);
        _gridInfos      = new NativeArray<int2>(GridCount, Allocator.Persistent);
        _blockIsSeed    = new NativeArray<int>(GridCount,  Allocator.Persistent);
        _totalCells     = new NativeReference<int>(0,      Allocator.Persistent);
        _particleCount  = new NativeReference<int>(0,      Allocator.Persistent);

        _fineCellSDF    = new NativeArray<int>(FineCount,  Allocator.Persistent);
        _fineCellType   = new NativeArray<uint>(FineCount, Allocator.Persistent);
        _particleRange  = new NativeArray<int2>(FineCount, Allocator.Persistent);

        // 粒子缓冲（容量上界；实际使用量 = _particleCount）
        _particlePos     = new NativeArray<float2>(_maxParticles, Allocator.Persistent);
        _particleVel     = new NativeArray<float2>(_maxParticles, Allocator.Persistent);
        _particlePosSwap = new NativeArray<float2>(_maxParticles, Allocator.Persistent);
        _particleVelSwap = new NativeArray<float2>(_maxParticles, Allocator.Persistent);
        _particleIDs     = new NativeArray<int>(_maxParticles,    Allocator.Persistent);

        // 块精度初始化：全部 level 2（全域粗网格），首帧由粒子驱动细化
        for (int i = 0; i < GridCount; i++) _blockLevel[i] = 2;

        // flat 物理量数组延迟到首帧 EnsureGridCapacity 分配（见 StepSimulation）。
        // [TODO:SEED] 场景初始化粒子（位置/数量）后设置 _particleCount，再置 _initialized。
        _initialized = true;
    }

    private void OnDestroy()
    {
        if (_blockChebySDF.IsCreated)  _blockChebySDF.Dispose();
        if (_blockLevel.IsCreated)     _blockLevel.Dispose();
        if (_blockLevelPrev.IsCreated) _blockLevelPrev.Dispose();
        if (_gridInfos.IsCreated)      _gridInfos.Dispose();
        if (_blockIsSeed.IsCreated)    _blockIsSeed.Dispose();
        if (_totalCells.IsCreated)     _totalCells.Dispose();
        if (_particleCount.IsCreated)  _particleCount.Dispose();
        if (_fineCellSDF.IsCreated)    _fineCellSDF.Dispose();
        if (_fineCellType.IsCreated)   _fineCellType.Dispose();
        if (_particleRange.IsCreated)  _particleRange.Dispose();
        if (_gridVelocity.IsCreated)   _gridVelocity.Dispose();
        if (_gridLaplacian.IsCreated)  _gridLaplacian.Dispose();
        if (_pressure.IsCreated)       _pressure.Dispose();
        if (_flux.IsCreated)           _flux.Dispose();
        if (_particlePos.IsCreated)    _particlePos.Dispose();
        if (_particleVel.IsCreated)    _particleVel.Dispose();
        if (_particlePosSwap.IsCreated) _particlePosSwap.Dispose();
        if (_particleVelSwap.IsCreated) _particleVelSwap.Dispose();
        if (_particleIDs.IsCreated)    _particleIDs.Dispose();
        if (_gridVelocitySwap.IsCreated) _gridVelocitySwap.Dispose();
        if (_gridInfosPrev.IsCreated)  _gridInfosPrev.Dispose();
    }
} // AdaptiveNarrowBandSolver

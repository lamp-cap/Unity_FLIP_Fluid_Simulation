using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;

public class SmallTest : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public uint seed;

    public void SoveCG()
    {
        // left as coarse (1x2), middle as fine (2x4), right as coarse (1x2)
        //      |  8  9  |  
        //  1   |  6  7  |  11
        //      |  4  5  |  
        //  0   |  2  3  |  10
        var uvs = new float2[12];
        var divs = new float[12];
        var pres = new float[12];

        const float h0 = 2; 
        const float h1 = 4;
        const float d01 = (h0 + h1) * 0.5f;

        var rnd = Random.CreateFromIndex(seed);

        void CalcDivergence()
        {
            divs[0] = ((uvs[2].x + uvs[4].x) * 0.5f - uvs[0].x + uvs[1].y - uvs[0].y) * h1;
            divs[1] = ((uvs[6].x + uvs[8].x) * 0.5f - uvs[1].x + 0 - uvs[1].y) * h1;
            divs[2] = (uvs[3].x - uvs[2].x + uvs[4].y - uvs[2].y) * h0;
            divs[3] = (uvs[10].x - uvs[3].x + uvs[5].y - uvs[3].y) * h0;
            divs[4] = (uvs[5].x - uvs[4].x + uvs[6].y - uvs[4].y) * h0;
            divs[5] = (uvs[10].x - uvs[5].x + uvs[7].y - uvs[5].y) * h0;
            divs[6] = (uvs[7].x - uvs[6].x + uvs[8].y - uvs[6].y) * h0;
            divs[7] = (uvs[11].x - uvs[7].x + uvs[9].y - uvs[7].y) * h0;
            divs[8] = (uvs[9].x - uvs[8].x + 0 - uvs[8].y) * h0;
            divs[9] = (uvs[11].x - uvs[9].x - uvs[9].y) * h0;
            divs[10] = (-uvs[10].x + uvs[11].y - uvs[10].y) * h1;
            divs[11] = (-uvs[11].x - uvs[11].y) * h1;
        }
        
        // init
        if (true)
        {
            for (int i = 0; i < 12; i++)
            {
                var vel = rnd.NextFloat2Direction();
                if (i is 0 or 2 or 3 or 10) vel.y = 0;
                if (i is 0 or 1) vel.x = 0;
                uvs[i] = vel;
            }

            CalcDivergence();
            float sum = 0;
            for (int i = 0; i < 10; i++)
                sum += divs[i] * divs[i];
            
            Debug.Log("Init divergence: " + sum);
        }

        // solve
        if (true)
        {
            var r = new float[12];
            var p = new float[12];
            var v = new float[12];
            divs.CopyTo(r, 0);
            r.CopyTo(p, 0);
            var Ap = new float[12];
            float pAp, rsNew, rsOld = 0;
            // 2f/3f between fine-coarse(h0/d01), 1f between same levels
            var matrix = new float[12, 12];
            float l = 1f, m = 2f/3f;
            matrix[0, 1] = l;  matrix[0, 2] = m;  matrix[0, 4] = m;
            matrix[1, 0] = l;  matrix[1, 6] = m;  matrix[1, 8] = m;
            matrix[2, 0] = m;  matrix[2, 3] = l;  matrix[2, 4] = l;
            matrix[3, 2] = l;  matrix[3, 5] = l;  matrix[3, 10] = m;
            matrix[4, 0] = m;  matrix[4, 2] = l;  matrix[4, 5] = l; matrix[4, 6] = l;
            matrix[5, 3] = l;  matrix[5, 4] = l;  matrix[5, 7] = l; matrix[5, 10] = m;
            matrix[6, 1] = m;  matrix[6, 4] = l;  matrix[6, 7] = l; matrix[6, 8] = l;
            matrix[7, 5] = l;  matrix[7, 6] = l;  matrix[7, 9] = l; matrix[7, 11] = m;
            matrix[8, 1] = m;  matrix[8, 6] = l;  matrix[8, 9] = l;
            matrix[9, 7] = l;  matrix[9, 8] = l;  matrix[9, 11] = m;
            matrix[10, 3] = m; matrix[10, 5] = m; matrix[10, 11] = l;
            matrix[11, 7] = m; matrix[11, 9] = m; matrix[11, 10] = l;
            for (int y = 0; y < 12; y++)
            {
                float sum = 0;
                for (int x = 0; x < 12; x++)
                {
                    sum += matrix[x, y];
                    Debug.Assert(Mathf.Abs(matrix[x, y] - matrix[y, x]) < 1e-5f, $"param[{x},{y}]!=param[{y}, {x}]]");
                }

                matrix[y, y] = -sum;
            }
            matrix[0, 0] = -3;

            var msg = "";

            for (int y = 0; y < 12; y++)
            {
                for (int x = 0; x < 12; x++)
                    msg += $"{matrix[x, y]:F2},\t";
                msg += "\n";
            }

            // Debug.Log(msg);

            for (int i = 0; i < 12; i++)
                rsOld += r[i] * r[i];

            msg = "CG init with rs" + rsOld + "\n";

            for (int iter = 0; iter < 12; iter++)
            {
                for (int i = 0; i < 12; i++)
                {
                    var dot = 0f;
                    for (var j = 0; j < 12; j++)
                        dot += matrix[i, j] * p[j];
                    Ap[i] = dot;
                }

                pAp = 0;
                for (var i = 0; i < 12; i++)
                    pAp += p[i] * Ap[i];

                float alpha = rsOld / pAp;
                for (var i = 0; i < 12; i++)
                {
                    v[i] += alpha * p[i];
                    r[i] -= alpha * Ap[i];
                }

                rsNew = 0;
                for (var i = 0; i < 12; i++)
                    rsNew += r[i] * r[i];

                msg += $"iter {iter + 1} : rs {rsNew}\n";
                if (rsNew < 1e-7f) break;

                float beta = rsNew / rsOld;
                for (int i = 0; i < 12; i++)
                    p[i] = r[i] + beta * p[i];

                rsOld = rsNew;
            }

            Debug.Log(msg);

            v.CopyTo(pres, 0);
        }

        // apply pressure
        if (true)
        {
            uvs[1] -= new float2(0, (pres[1] - pres[0]) / h1);
            uvs[2] -= new float2((pres[2] - pres[0]) / d01, 0);
            uvs[3] -= new float2((pres[3] - pres[2]) / h0, 0);
            uvs[4] -= new float2((pres[4] - pres[0]) / d01, (pres[4] - pres[2]) / h0);
            uvs[5] -= new float2((pres[5] - pres[4]) / h0, (pres[5] - pres[3]) / h0);
            uvs[6] -= new float2((pres[6] - pres[1]) / d01, (pres[6] - pres[4]) / h0);
            uvs[7] -= new float2((pres[7] - pres[6]) / h0, (pres[7] - pres[5]) / h0);
            uvs[8] -= new float2((pres[8] - pres[1]) / d01, (pres[8] - pres[6]) / h0);
            uvs[9] -= new float2((pres[9] - pres[8]) / h0, (pres[9] - pres[7]) / h0);
            uvs[10] -= new float2((pres[10] - 0.5f * (pres[3] + pres[5])) / d01, 0);
            uvs[11] -= new float2((pres[11] - 0.5f * (pres[7] + pres[9])) / d01, (pres[11] - pres[10]) / h1);
            CalcDivergence();
            float sum = 0;
            var msg = "";
            for (int i = 0; i < 12; i++)
            {
                sum += divs[i] * divs[i];
                msg += $"{divs[i]:F2}, ";
            }
            
            Debug.Log(msg + "\nSolved divergence: " + sum);
            // uvs[5] += new float2(0, divs[5] * 0.5f);
            // uvs[9] += new float2(0, divs[9] * 0.5f);
            // CalcDivergence();
            // sum = 0;
            // msg = "";
            // for (int i = 0; i < 12; i++)
            // {
            //     sum += divs[i] * divs[i];
            //     msg += $"{divs[i]:F2}, ";
            // }
            //
            // Debug.Log(msg + "\nSolved divergence: " + sum);
        }
    }

    private float2[] _deltaUV;
    private float2[] _uvs;
    private float2[] _uvsOld;
    private float[] _pressures;
    private float[] _flux;
    private int[] _levels;
    private int[] _ptrs;
    private float _meanPre;

    public bool postProcess;
    
    private const int gridWidth = 8;
    private const int gridCount = gridWidth * gridWidth;

    // level 0: 2x2, level 1: 4x4
    int GetWidth(int level) { return 2 << level; }
    // h0: 4, h1: 2, h2: 1 
    int GetH(int level) { return 4 >> level; }
    
    public void Test()
    {
        var levels = new int[gridCount];
        var ptrs = new int[gridCount];
        // levels[5] = levels[6] = levels[9] = levels[10] = 1;
        for (int y = 0; y < gridWidth; y++)
        for (int x = 0; x < gridWidth; x++)
        {
            float2 rUV = math.abs(new float2(x - 3.5f, y - 3.5f));
            float d = math.length(rUV);
            if (d < 2) levels[y * gridWidth + x] = 2;
            else if (d < 3) levels[y * gridWidth + x] = 1;
        }
        int ptr = 0;
        for (int i = 0; i < gridCount; i++)
        {
            var blockWidth = GetWidth(levels[i]);
            ptrs[i] = ptr;
            ptr += blockWidth * blockWidth;
        }

        var vels = new float2[ptr];
        var flux = new float[ptr];
        var pres = new float[ptr];

        var rnd = Random.CreateFromIndex(seed);

        // init velocity
        for (int y = 0; y < gridWidth; y++)
        for (int x = 0; x < gridWidth; x++)
        {
            int i = y * gridWidth + x;
            int level = levels[i];
            int offset = ptrs[i];
            int width = GetWidth(level);
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                var vel = rnd.NextFloat2Direction();
                if (x == 0 && xx == 0) vel.x = 0;
                if (y == 0 && yy == 0) vel.y = 0;
                int ii = yy * width + xx;
                vels[offset + ii] = vel;
            }
        }
        
        // calc flux
        void CalcFlux()
        {
            for (int y = 0; y < gridWidth; y++)
            for (int x = 0; x < gridWidth; x++)
            {
                int i = y * gridWidth + x;
                int level = levels[i], offset = ptrs[i], width = GetWidth(level);
                int haloWidth = width + 2;
                var temp = new NativeArray<float2>(haloWidth * haloWidth, Allocator.Temp);

                // fill halo block
                if (true)
                {
                    for (int by = 0; by < width; by++)
                    for (int bx = 0; bx < width; bx++)
                    {
                        int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloWidth);
                        int physicsIdx = offset + BlockCoord2Idx(bx, by, width);

                        temp[localIdx] = vels[physicsIdx];
                    }

                    int4 ox = new int4(-1, 0, 1, 0);
                    int4 oy = new int4(0, -1, 0, 1);

                    for (int n = 0; n < 4; n++)
                    {
                        int2 dir = new int2(ox[n], oy[n]);
                        int2 curr = new int2(x, y) + dir;
                        if (curr.x < 0 || curr.y < 0 || curr.x >= gridWidth || curr.y >= gridWidth)
                            continue;

                        int nLevel = levels[BlockCoord2Idx(curr, gridWidth)];
                        int phn = ptrs[BlockCoord2Idx(curr, gridWidth)];
                        int nBlockWidth = GetWidth(nLevel);
                        
                        if (nLevel == level)
                        {
                            for (int c = 0; c < width; c++)
                            {
                                int2 nCoord = math.select(math.select(c, 0, dir > 0), width - 1, dir < 0);
                                int nLocalIdx = BlockCoord2Idx(nCoord, width);
                                int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                                int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                                temp[paddingIdx] = vels[phn + nLocalIdx];
                            }
                        }
                        else if (nLevel < level)
                        {
                            for (int c = 0; c < width; c++)
                            {
                                int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                                int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                                int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                                int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                                temp[paddingIdx] = vels[phn + nLocalIdx];
                            }
                        }
                        else // n_level > level
                        {
                            for (int c = 0; c < width; c++)
                            {
                                int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                                int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                                int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                                int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                                int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                                int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                                temp[paddingIdx] = (vels[phn + nLocalIdx0] + vels[phn + nLocalIdx1]) * 0.5f;
                            }
                        }
                    }
                }

                for (int yy = 1; yy <= width; yy++)
                for (int xx = 1; xx <= width; xx++)
                {
                    float2 uv = temp[BlockCoord2Idx(xx, yy, haloWidth)];
                    float up = temp[BlockCoord2Idx(xx + 1, yy, haloWidth)].x;
                    float vp = temp[BlockCoord2Idx(xx, yy + 1, haloWidth)].y;

                    flux[offset + BlockCoord2Idx(xx - 1, yy - 1, width)] = (up - uv.x + vp - uv.y) * GetH(level);
                }

                temp.Dispose();
            }
        }

        CalcFlux();
        float fluxSum = 0f;
        for (int i = 0; i < ptrs.Length; i++)
            fluxSum += flux[i];
        
        Debug.Log("init with fluxSum: " + fluxSum + "cellcount: " + ptr);
        
        // solve
        if (true)
        {
            const float l = 1f, m = 2f / 3f; // 1f for same level, h0 / (0.5f * (h0 + h1)) between fine coarse
            // fill matrix
            var matrix = new float[ptr, ptr];
            for (int y = 0; y < gridWidth; y++)
            for (int x = 0; x < gridWidth; x++)
            {
                int i = y * gridWidth + x;
                int level = levels[i];
                int offset = ptrs[i];
                int width = GetWidth(level);
                for (int yy = 0; yy < width; yy++)
                for (int xx = 0; xx < width; xx++)
                {
                    int ii = offset + yy * width + xx;
                    // left
                    if (xx > 0) matrix[ii, offset + yy * width + (xx - 1)] = l;
                    else if (x > 0)
                    {
                        int ni = y * gridWidth + x - 1;
                        int nLevel = levels[ni], nOffset = ptrs[ni], nWidth = GetWidth(nLevel);
                        if (nLevel == level) matrix[ii, nOffset + yy * width + width - 1] = l;
                        else if (nLevel > level)
                        {
                            matrix[ii, nOffset + (yy * 2) * nWidth + nWidth - 1] = m;
                            matrix[ii, nOffset + (yy * 2 + 1) * nWidth + nWidth - 1] = m;
                        }
                        else matrix[ii, nOffset + (yy / 2) * nWidth + nWidth - 1] = m;

                    }

                    // right
                    if (xx < width - 1)
                        matrix[ii, offset + yy * width + (xx + 1)] = l;
                    else if (x < gridWidth - 1)
                    {
                        int ni = y * gridWidth + x + 1;
                        int nLevel = levels[ni], nOffset = ptrs[ni], nWidth = GetWidth(nLevel);
                        if (nLevel == level) matrix[ii, nOffset + yy * width] = l;
                        else if (nLevel > level)
                        {
                            matrix[ii, nOffset + (yy * 2) * nWidth] = m;
                            matrix[ii, nOffset + (yy * 2 + 1) * nWidth] = m;
                        }
                        else matrix[ii, nOffset + (yy / 2) * nWidth] = m;

                    }

                    // bottom
                    if (yy > 0)
                        matrix[ii, offset + (yy - 1) * width + xx] = l;
                    else if (y > 0)
                    {
                        int ni = (y - 1) * gridWidth + x;
                        int nLevel = levels[ni], nOffset = ptrs[ni], nWidth = GetWidth(nLevel);
                        if (nLevel == level) matrix[ii, nOffset + (width - 1) * width + xx] = l;
                        else if (nLevel > level)
                        {
                            matrix[ii, nOffset + (nWidth - 1) * nWidth + xx * 2] = m;
                            matrix[ii, nOffset + (nWidth - 1) * nWidth + xx * 2 + 1] = m;
                        }
                        else matrix[ii, nOffset + (nWidth - 1) * nWidth + xx / 2] = m;

                    }

                    // top
                    if (yy < width - 1) matrix[ii, offset + (yy + 1) * width + xx] = l;
                    else if (y < gridWidth - 1)
                    {
                        int ni = (y + 1) * gridWidth + x;
                        int nLevel = levels[ni], nOffset = ptrs[ni], nWidth = GetWidth(nLevel);
                        if (nLevel == level) matrix[ii, nOffset + xx] = l;
                        else if (nLevel > level)
                        {
                            matrix[ii, nOffset + xx * 2] = m;
                            matrix[ii, nOffset + xx * 2 + 1] = m;
                        }
                        else matrix[ii, nOffset + xx / 2] = m;
                    }
                }
            }

            for (int y = 0; y < ptr; y++)
            {
                float colSum = 0;
                for (int x = 0; x < ptr; x++)
                {
                    colSum += matrix[x, y];
                    Debug.Assert(Mathf.Abs(matrix[x, y] - matrix[y, x]) < 1e-5f, $"param[{x},{y}]!=param[{y}, {x}]]");
                }

                matrix[y, y] = -colSum;
            }

            // var mt = ptr + "x" + ptr + " matrix:\n";
            // for (int y = 0; y < ptr; y++)
            // {
            //     for (int x = 0; x < ptr; x++)
            //         mt += $"{matrix[x, y]}, ";
            //     mt += "\n";
            // }
            // Debug.Log(mt);
            matrix[0, 0] -= 0.5f;

            var r = new float[ptr];
            var p = new float[ptr];
            var v = new float[ptr];
            flux.CopyTo(r, 0);
            r.CopyTo(p, 0);
            var Ap = new float[ptr];
            float pAp, rsNew, rsOld = 0;

            for (int i = 0; i < ptr; i++)
                rsOld += r[i] * r[i];

            var msg = "CG init with rs" + rsOld + "\n";

            for (int iter = 0; iter < ptr; iter++)
            {
                // for (int i = 0; i < ptr; i++)
                // {
                //     var dot = 0f;
                //     for (var j = 0; j < ptr; j++)
                //         dot += matrix[i, j] * p[j];
                //     Ap[i] = dot;
                // }
                for (int y = 0; y < gridWidth; y++)
                for (int x = 0; x < gridWidth; x++)
                {
                    int i = y * gridWidth + x;
                    int level = levels[i], offset = ptrs[i], width = GetWidth(level);
                    
                    int haloWidth = width + 2;
                    var temp = new NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);
                    var param = new NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);

                    // fill halo block
                    if (true)
                    {
                        for (int by = 0; by < width; by++)
                        for (int bx = 0; bx < width; bx++)
                        {
                            int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloWidth);
                            int physicsIdx = offset + BlockCoord2Idx(bx, by, width);

                            temp[localIdx] = p[physicsIdx];
                            param[localIdx] = l;
                        }

                        int4 ox = new int4(-1, 0, 1, 0);
                        int4 oy = new int4(0, -1, 0, 1);

                        for (int n = 0; n < 4; n++)
                        {
                            int2 dir = new int2(ox[n], oy[n]);
                            int2 curr = new int2(x, y) + dir;
                            if (curr.x < 0 || curr.y < 0 || curr.x >= gridWidth || curr.y >= gridWidth)
                                continue;

                            int nLevel = levels[BlockCoord2Idx(curr, gridWidth)];
                            int phn = ptrs[BlockCoord2Idx(curr, gridWidth)];
                            int nBlockWidth = GetWidth(nLevel);
                            
                            if (nLevel == level)
                            {
                                for (int c = 0; c < width; c++)
                                {
                                    int2 nCoord = math.select(math.select(c, 0, dir > 0), width - 1, dir < 0);
                                    int nLocalIdx = BlockCoord2Idx(nCoord, width);
                                    int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                                    int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                                    temp[paddingIdx] = p[phn + nLocalIdx];
                                    param[paddingIdx] = l;
                                }
                            }
                            else if (nLevel < level)
                            {
                                for (int c = 0; c < width; c++)
                                {
                                    int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                                    int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                                    int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                                    int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                                    temp[paddingIdx] = p[phn + nLocalIdx];
                                    param[paddingIdx] = m;
                                }
                            }
                            else // n_level > level
                            {
                                for (int c = 0; c < width; c++)
                                {
                                    int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                                    int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                                    int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                                    int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                                    int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                                    int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                                    temp[paddingIdx] = (p[phn + nLocalIdx0] + p[phn + nLocalIdx1]) * 0.5f;
                                    param[paddingIdx] = m * 2;
                                }
                            }
                        }
                    }
                    
                    for (int yy = 0; yy < width; yy++)
                    for (int xx = 0; xx < width; xx++)
                    {
                        int ii = offset + yy * width + xx;

                        float c = temp[BlockCoord2Idx(xx + 1, yy + 1, haloWidth)];
                        float pl = temp[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                        float pr = temp[BlockCoord2Idx(xx + 2, yy + 1, haloWidth)];
                        float pb = temp[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                        float pt = temp[BlockCoord2Idx(xx + 1, yy + 2, haloWidth)];
                        float al = param[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                        float ar = param[BlockCoord2Idx(xx + 2, yy + 1, haloWidth)];
                        float ab = param[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                        float at = param[BlockCoord2Idx(xx + 1, yy + 2, haloWidth)];
                        float off = x == 0 && y == 0 && xx == 0 && yy == 0 ? 1 : 0;
                        Ap[ii] = pl * al + pr * ar + pb * ab + pt * at - (al + ar + ab + at + off) * c;
                    }
                }

                pAp = 0;
                for (var i = 0; i < ptr; i++)
                    pAp += p[i] * Ap[i];

                float alpha = rsOld / pAp;
                for (var i = 0; i < ptr; i++)
                {
                    v[i] += alpha * p[i];
                    r[i] -= alpha * Ap[i];
                }

                rsNew = 0;
                for (var i = 0; i < ptr; i++)
                    rsNew += r[i] * r[i];

                msg += $"iter{iter + 1} \trsNew:{rsNew}\n";
                if (rsNew < 1e-7f) break;

                float beta = rsNew / rsOld;
                for (int i = 0; i < ptr; i++)
                    p[i] = r[i] + beta * p[i];

                rsOld = rsNew;
            }

            Debug.Log(msg);

            v.CopyTo(pres, 0);
        }

        _pressures = pres;
        _deltaUV = new float2[ptr];
        _uvsOld = (float2[])vels.Clone();
        
        // apply pressure
        if (true)
        {
            for (int y = 0; y < gridWidth; y++)
            for (int x = 0; x < gridWidth; x++)
            {
                int i = y * gridWidth + x, level = levels[i], offset = ptrs[i], width = GetWidth(level);
                
                int haloWidth = width + 2;
                var temp = new NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);
                var param = new NativeArray<float>(haloWidth * haloWidth, Allocator.Temp);

                // fill halo block
                if (true)
                {
                    for (int by = 0; by < width; by++)
                    for (int bx = 0; bx < width; bx++)
                    {
                        int localIdx = BlockCoord2Idx(bx + 1, by + 1, haloWidth);
                        int physicsIdx = offset + BlockCoord2Idx(bx, by, width);

                        temp[localIdx] = pres[physicsIdx];
                        param[localIdx] = GetH(level);
                    }

                    int4 ox = new int4(-1, 0, 1, 0);
                    int4 oy = new int4(0, -1, 0, 1);

                    for (int n = 0; n < 4; n++)
                    {
                        int2 dir = new int2(ox[n], oy[n]);
                        int2 curr = new int2(x, y) + dir;
                        if (curr.x < 0 || curr.y < 0 || curr.x >= gridWidth || curr.y >= gridWidth)
                            continue;

                        int nLevel = levels[BlockCoord2Idx(curr, gridWidth)];
                        int phn = ptrs[BlockCoord2Idx(curr, gridWidth)];
                        int nBlockWidth = GetWidth(nLevel);
                        
                        if (nLevel == level)
                        {
                            for (int c = 0; c < width; c++)
                            {
                                int2 nCoord = math.select(math.select(c, 0, dir > 0), width - 1, dir < 0);
                                int nLocalIdx = BlockCoord2Idx(nCoord, width);
                                int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                                int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                                temp[paddingIdx] = pres[phn + nLocalIdx];
                                param[paddingIdx] = GetH(level);
                            }
                        }
                        else if (nLevel < level)
                        {
                            for (int c = 0; c < width; c++)
                            {
                                int2 nCoord = math.select(math.select(c >> 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                                int nLocalIdx = BlockCoord2Idx(nCoord, nBlockWidth);
                                int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                                int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                                temp[paddingIdx] = pres[phn + nLocalIdx];
                                param[paddingIdx] = 0.5f * (GetH(level) + GetH(nLevel));
                            }
                        }
                        else // n_level > level
                        {
                            for (int c = 0; c < width; c++)
                            {
                                int2 nCoord0 = math.select(math.select(c << 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                                int nLocalIdx0 = BlockCoord2Idx(nCoord0, nBlockWidth);
                                int2 nCoord1 = math.select(math.select((c << 1) + 1, 0, dir > 0), nBlockWidth - 1, dir < 0);
                                int nLocalIdx1 = BlockCoord2Idx(nCoord1, nBlockWidth);
                                int2 cCoord = math.select(math.select(c + 1, haloWidth - 1, dir > 0), 0, dir < 0);
                                int paddingIdx = BlockCoord2Idx(cCoord, haloWidth);
                                temp[paddingIdx] = (pres[phn + nLocalIdx0] + pres[phn + nLocalIdx1]) * 0.5f;
                                param[paddingIdx] = 0.5f * (GetH(level) + GetH(nLevel));
                            }
                        }
                    }
                }
                
                for (int yy = 0; yy < width; yy++)
                for (int xx = 0; xx < width; xx++)
                {
                    int ii = offset + yy * width + xx;

                    float p = temp[BlockCoord2Idx(xx + 1, yy + 1, haloWidth)];
                    float up = temp[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                    float ua = param[BlockCoord2Idx(xx, yy + 1, haloWidth)];
                    float vp = temp[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                    float va = param[BlockCoord2Idx(xx + 1, yy, haloWidth)];
                    
                    float2 delta = new float2(ua < 1e-5f ? 0 : (p - up) / ua, va < 1e-5f ? 0 : (p - vp) / va);
                    _deltaUV[ii] = delta;
                    vels[ii] -= delta;
                }

                temp.Dispose();
                param.Dispose();
            }
            CalcFlux();
            float sum = 0;
            var msg = "\nremain: ";
            for (int i = 0; i < ptr; i++)
            {
                sum += flux[i] * flux[i];
                if (math.abs(flux[i]) > 0.001f) msg += $"{i}: {flux[i]:F3},  ";
            }
            
            Debug.Log("Solved divergence: " + sum + msg);
            if (postProcess)
            {
                for (int y = 0; y < gridWidth; y++)
                for (int x = 0; x < gridWidth; x++)
                {
                    int i = y * gridWidth + x, level = levels[i], offset = ptrs[i], width = GetWidth(level);
                    float ih = 1.0f / GetH(level);
                    if (x < gridWidth - 1 && level > levels[y * gridWidth + x + 1])
                    {
                        for (int yy = 1; yy < width; yy += 2)
                        {
                            int ii = offset + yy * width + width - 1;
                            vels[ii] -= new float2(0, ih * flux[offset + (yy - 1) * width + width - 1]);
                        }
                    }

                    if (y < gridWidth - 1 && level > levels[(y + 1) * gridWidth + x])
                    {
                        for (int xx = 1; xx < width; xx += 2)
                        {
                            int ii = offset + (width - 1) * width + xx;
                            vels[ii] -= new float2(ih * flux[offset + (width - 1) * width + xx - 1], 0);
                        }

                    }
                }

                CalcFlux();

                sum = 0;
                msg = "\nremain: ";
                for (int i = 0; i < ptr; i++)
                {
                    sum += flux[i] * flux[i];
                    if (math.abs(flux[i]) > 0.001f) msg += $"{i}: {flux[i]:F3},  ";
                }

                Debug.Log("After postProcess divergence: " + sum + msg);
            }
        }
        _levels = levels;
        _ptrs = ptrs;
        _meanPre = 0;
        for (int i = 0; i < ptr; i++)
            _meanPre += _pressures[i];
        _meanPre /= ptr;
        _flux = flux;
        _uvs = vels;
    }

    private void OnDrawGizmos()
    {
        if (_pressures == null || _deltaUV == null) return;
        
        for (int y = 0; y < gridWidth; y++)
        for (int x = 0; x < gridWidth; x++)
        {
            int i =  y * gridWidth + x;
            int level = _levels[i], width = GetWidth(level), h = GetH(level), offset = _ptrs[i];
            int2 blockMin = new int2(x, y) * width * h;
            // Gizmos.DrawWireCube(new Vector3(blockMin.x, 0, blockMin.y) + new Vector3(0.5f, 0, 0.5f) * (width * h),
            //     new Vector3(1,0,1) * (width * h));
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                int ii = offset + yy * width + xx;
                float f = _flux[ii];
                Gizmos.color = new Color(f, -f, 0);
                Gizmos.DrawCube(new Vector3(blockMin.x, 0, blockMin.y) + new Vector3(xx + 0.5f, 0, yy + 0.5f) * h,
                    new Vector3(1, 0, 1) * h);
            }
            Gizmos.color = Color.white;
            for (int yy = 0; yy < width; yy++)
            for (int xx = 0; xx < width; xx++)
            {
                Gizmos.color = Color.white;
                int ii = offset + yy * width + xx;
                float p = 0.1f * (_pressures[ii] - _meanPre);
                Gizmos.DrawWireCube(new Vector3(blockMin.x, 0, blockMin.y) + new Vector3(xx + 0.5f, p*0.5f, yy + 0.5f) * h,
                    new Vector3(1, Mathf.Abs(p), 1) * h);
                float2 delta = -_deltaUV[ii] * 0.5f;
                Vector3 left = new Vector3(blockMin.x, 0, blockMin.y) + new Vector3(xx, 0.1f, yy + 0.5f) * h;
                Vector3 bottom = new Vector3(blockMin.x, 0, blockMin.y) + new Vector3(xx + 0.5f, 0.1f, yy) * h;
                left += Vector3.forward * 0.15f;
                bottom += Vector3.right * 0.15f;
                // Gizmos.DrawLine(left, left + delta.x * Vector3.left);
                // Gizmos.DrawLine(bottom, bottom + delta.y * Vector3.forward);
                float2 uv = _uvs[ii] * 0.5f;
                left += Vector3.forward * 0.05f;
                bottom += Vector3.right * 0.05f;
                Gizmos.color = Color.cyan;
                Gizmos.DrawLine(left, left + uv.x * Vector3.left);
                Gizmos.DrawLine(bottom, bottom + uv.y * Vector3.forward);
                // left -= Vector3.forward * 0.1f;
                // bottom -= Vector3.right * 0.1f;
                // uv = _uvsOld[ii] * 0.5f;
                // Gizmos.color = Color.magenta;
                // Gizmos.DrawLine(left, left + uv.x * Vector3.left);
                // Gizmos.DrawLine(bottom, bottom + uv.y * Vector3.forward);
            }
        }
    }
    private static int BlockCoord2Idx(int2 coord, int res) => coord.x + coord.y * res;
    private static int BlockCoord2Idx(int x, int y, int res) => x + y * res;
}

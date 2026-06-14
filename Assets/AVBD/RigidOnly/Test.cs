using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

namespace AVBD.RigidOnly
{
    [ExecuteAlways]
    public unsafe class Test : MonoBehaviour
    {
        private void OnEnable()
        {
            const int size = 512;
            var mat = new bool[size,size];
            var csr = new int4[size];
            var data = new List<int4>();
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    var rnd = Random.Range(0, size);
                    if (rnd != i)
                    {
                        mat[i, rnd] = true;
                        mat[rnd, i] = true;
                    }
                }
            }

            for (int i = 0; i < size; i++)
            {
                int start = data.Count;
                for (int j = 0; j < size; j++)
                {
                    if (mat[i, j])
                        data.Add(new int4(i, j, 0, 0));
                }
                int end = data.Count;
                csr[i] = new int4(start, end, -1, 0);
            }

            var edges = data.ToArray();
            GreedyColoring(csr, edges);
            ValidateColoring(csr, edges);
        }

        /// <summary>
        /// 单线程贪心着色
        /// </summary>
        public void GreedyColoring(int4[] Csr, int4[] Data)
        {
            // 计算最大度数，用于优化
            int vertexCount = Csr.Length;

            // 按顶点顺序着色
            for (int v = 0; v < vertexCount; v++)
            {
                int4 vertexInfo = Csr[v];
                int start = vertexInfo.x;
                int end = vertexInfo.y;
                
                // 重置标记数组
                uint usedColors = 0;
                
                // 标记邻居使用的颜色
                for (int idx = start; idx < end; idx++)
                {
                    int4 edge = Data[idx];
                    int neighbor = edge.y;  // 目标节点
                    int4 neighborInfo = Csr[neighbor];
                    int neighborColor = neighborInfo.z;
                    
                    if (neighborColor != -1)
                    {
                        usedColors |= 1u << neighborColor;
                    }
                }
                
                // 选择最小可用颜色
                int color = usedColors == 0 ? 0 : math.tzcnt(~usedColors);
                
                // 赋值颜色
                vertexInfo.z = color;
                Csr[v] = vertexInfo;
            }
            
            // 统计使用的颜色数
            int colorCount = 0;
            for (int v = 0; v < vertexCount; v++)
            {
                int c = Csr[v].z;
                if (c > colorCount) colorCount = c;
            }
            Debug.Log($"着色完成！使用了 {colorCount} 种颜色");
        }
        
        /// <summary>
        /// 验证着色正确性
        /// </summary>
        public bool ValidateColoring(int4[] Csr, int4[] Data)
        {
            int vertexCount = Csr.Length;
            for (int v = 0; v < vertexCount; v++)
            {
                int4 vertexInfo = Csr[v];
                int myColor = vertexInfo.z;
                int start = vertexInfo.x;
                int end = vertexInfo.y;
                
                if (myColor == -1)
                {
                    Debug.LogError($"顶点 {v} 未着色");
                    return false;
                }
                
                // 检查所有邻居
                for (int idx = start; idx < end; idx++)
                {
                    int4 edge = Data[idx];
                    int neighbor = edge.y;
                    int neighborColor = Csr[neighbor].z;
                    
                    if (myColor == neighborColor)
                    {
                        Debug.LogError($"冲突！顶点 {v} 和 {neighbor} 颜色相同 ({myColor})");
                        return false;
                    }
                }
            }
            
            Debug.Log("验证通过！所有相邻顶点颜色不同");
            return true;
        }
    }
}

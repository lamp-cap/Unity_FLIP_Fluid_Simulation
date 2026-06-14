using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace AVBD.RigidOnly
{
    public class BVH
    {
        public struct BvhNode
        {
            public Bounds bounds;
            public int leftChild;        // 左子节点索引（-1表示叶子）
            public int rightChild;       // 右子节点索引
            public bool IsLeaf => leftChild < 0;   // 叶子：起始物体索引
            public int DataID => rightChild;
        }
        public struct BvhPrimitive
        {
            public Bounds bounds;          // 物体包围盒
            public int index;            // 原始物体索引
        }
        private NativeList<BvhNode> _nodes;
        public NativeList<BvhNode> BvhNodes => _nodes;

        private struct BvhPrimitiveComparer : IComparer<BvhPrimitive>
        {
            private int _axis;

            public BvhPrimitiveComparer(int axis)
            {
                _axis = axis;
            }
            public int Compare(BvhPrimitive x, BvhPrimitive y)
            {
                _axis = math.clamp(_axis, 0, 2);
                return x.bounds.center[_axis].CompareTo(y.bounds.center[_axis]);
            }
        }
        
        [BurstCompile]
        private struct BuildBvhJob : IJob
        {
            [ReadOnly] public NativeArray<OBB> ObjList;
            public NativeArray<BvhNode> Nodes;
    
            public void Execute()
            {
                var primitives = new NativeArray<BvhPrimitive>(ObjList.Length, Allocator.Temp);
                for (int i = 0; i < ObjList.Length; i++)
                    primitives[i] = new BvhPrimitive
                    {
                        bounds = ObjList[i].ToAABB_Fast(),
                        index = i
                    };
            
                primitives.Sort(new BvhPrimitiveComparer(0));
        
                var stack = new NativeList<int3>(Allocator.Temp);
                stack.Add(new int3(0, primitives.Length, 0));

                int nodeCount = 1;
                // 简化：实际需要更完整的节点构建逻辑
                while (stack.Length > 0)
                {
                    var frame = stack[^1];
                    stack.RemoveAtSwapBack(stack.Length - 1);
                
                    int start = frame.x;
                    int end = frame.y;
                    int nodeIdx = frame.z;
                
                    var bs = CalculateBounds(primitives, start, end);
                    int sliceCount = end - start;
                    if (sliceCount <= 1)
                    {
                        Nodes[nodeIdx] = new BvhNode
                        {
                            bounds = bs,
                            leftChild = -1,
                            rightChild = start,
                        };
                        continue;
                    }
                
                    float3 size = bs.size;
                    int axis = 0;
                    if (size.y > size.x && size.y > size.z) axis = 1;
                    if (size.z > size.x && size.z > size.y) axis = 2;
                
                    primitives.Slice(start, sliceCount).Sort(new BvhPrimitiveComparer(axis));
                
                    int mid = (start + end) / 2;
                
                    int leftIdx = nodeCount++;
                    int rightIdx = nodeCount++;
                
                    Nodes[nodeIdx] = new BvhNode
                    {
                        bounds = bs,
                        leftChild = leftIdx,
                        rightChild = rightIdx
                    };
                
                    if (end > mid)
                        stack.Add(new int3(mid, end, rightIdx));
                    stack.Add(new int3(start, mid, leftIdx));
                }
            
                // 4. 提取重排后的物体索引
                for (int i = 0; i < nodeCount; i++)
                {
                    var node = Nodes[i];
                    if (node.IsLeaf)
                        node.rightChild = primitives[node.rightChild].index;
                    Nodes[i] = node;
                }
            }
        
            private static Bounds CalculateBounds(NativeArray<BvhPrimitive> primitives, int start, int end)
            {
                if (end - start == 1) 
                    return primitives[start].bounds;
            
                var bs = primitives[start].bounds;
                for (int i = start + 1; i < end; i++)
                {
                    bs.Encapsulate(primitives[i].bounds);
                }
                return bs;
            }
        }
    
        [BurstCompile]
        private struct BvhOverlapJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<OBB> ObjList;
            [ReadOnly] public NativeArray<BvhNode> Nodes;
            public NativeList<int2>.ParallelWriter Results;
    
            public void Execute(int i)
            {
                var obj = ObjList[i];
                var queryBs = obj.ToAABB_Fast();
        
                // 遍历BVH（使用栈，避免递归）
                var stack = new NativeList<int>(Allocator.Temp);
                stack.Add(0);
        
                while (stack.Length > 0)
                {
                    int nodeIdx = stack[^1];
                    stack.RemoveAt(stack.Length - 1);
            
                    var node = Nodes[nodeIdx];
            
                    // Bounds相交测试
                    if (!queryBs.Intersects(node.bounds))
                        continue;
            
                    if (node.IsLeaf)
                    {
                        var id = node.DataID;
                        if (i < id && Collision.Collide(obj, ObjList[id]))
                            Results.AddNoResize(new int2(i, id));
                    }
                    else
                    {
                        stack.Add(node.leftChild);
                        stack.Add(node.rightChild);
                    }
                }
            }
        }
        
        public BVH()
        {
            _nodes = new NativeList<BvhNode>(16, Allocator.Persistent);
        }
    
        public void Collide(NativeArray<OBB> objs, NativeList<int2> pairs)
        {
            if (_nodes.Length < objs.Length * 2 - 1)
                _nodes.Length = objs.Length * 2 - 1;
            var nodePool = _nodes.AsArray();
            new BuildBvhJob
            {
                ObjList = objs,
                Nodes = nodePool,
            }.Run();
        
            // 3. 重叠检测
            new BvhOverlapJob
            {
                ObjList = objs,
                Nodes = nodePool,
                Results = pairs.AsParallelWriter()
            }.Schedule(objs.Length, 1).Complete();
        }
    
        public void Dispose()
        {
            _nodes.Dispose();
        }
    }
}
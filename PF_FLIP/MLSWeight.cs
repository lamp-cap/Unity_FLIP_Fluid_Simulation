using System;
using System.Collections.Generic;
using System.Diagnostics;
using Sirenix.OdinInspector;
using Unity.Collections;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;
using Debug = UnityEngine.Debug;
using Random = UnityEngine.Random;

[ExecuteInEditMode]
public class MLSWeight : MonoBehaviour
{
    private float2[] _weights = new float2[64];
    public Transform trans;
    [Range(1, 3)]
    public float radius;

    public float2 sum;
    public float max;

    // Update is called once per frame
    void Update()
    {
        if (trans == null) return;

        var pos = trans.position;
        float2 p = new float2(pos.x, pos.z);
        sum = 0;
        max = 0;
        for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
        {
            float2 cellLeft = new float2(x, y + 0.5f);
            float2 cellBottom = new float2(x + 0.5f, y);
            // _weights[x + y * 8] = new float2(KernelFunc(math.lengthsq(p - cellLeft), radius * radius),
            //                             KernelFunc(math.lengthsq(p - cellBottom), radius * radius));
            _weights[x + y * 8] = new float2(GetWeight(p, cellLeft, 1f/radius),
                GetWeight(p, cellBottom, 1f/radius))/radius;
            sum += _weights[x + y * 8];
            max = math.max(max, math.max(_weights[x + y * 8].x, _weights[x + y * 8].y));
        }
    }
    float GetWeight(float2 p_pos, float2 c_pos, float grid_inv_spacing)
    {
        float2 dist = math.abs((p_pos - c_pos) * grid_inv_spacing);

        float2 weight = GetQuadraticWeight(dist);

        return weight.x * weight.y;
    }
    private static float2 GetQuadraticWeight(float2 abs_x)
    {
        float2 dst = math.saturate(1.5f - abs_x);
        return math.select(0.5f * dst * dst, 0.75f - abs_x * abs_x, abs_x < 0.5f);
    }

    private float KernelFunc(float r2, float h2)
    {
        float k = math.max(0, 1 - r2 / h2);
        return k * k * k;
    }

    private void OnDrawGizmos()
    {
        if (trans == null) return;
        for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            Gizmos.DrawWireCube(new Vector3(x + 0.5f, 0, y + 0.5f), new Vector3(1, 0, 1));
        for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
        {
            var cellLeft = new Vector3(x, 0, y + 0.5f);
            var cellBottom = new Vector3(x + 0.5f, 0, y);
            float2 weights = _weights[x + y * 8];
            Gizmos.DrawLine(cellLeft, cellLeft + Vector3.up * weights.x);
            Gizmos.DrawLine(cellBottom, cellBottom + Vector3.up * weights.y);
        }
        
        Gizmos.color = Color.red;
        var pos = trans.position;
        Gizmos.DrawWireSphere(new Vector3(pos.x, 0, pos.z), radius);
    }
    
    // Number of valuse to use for testing.
    const int NUM_VALUES = 65536;

    public struct DefaultComparer<T> : IComparer<T>
        where T : struct, IComparable<T>
    {
        public int Compare(T x, T y)
        {
            return x.CompareTo(y);
        }
    }

    [Button]
    public void SortPerformance()
    {
        var r = new System.Random();

        var a = new NativeArray<int>(NUM_VALUES, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var cmp = new DefaultComparer<int>();

        for (int i = 0; i < a.Length; ++i)
            a[i] = r.Next();

        var sw = Stopwatch.StartNew();
        a.Sort(cmp);
        sw.Stop();
        Debug.Log($"Sort: {sw.ElapsedMilliseconds} ticks");

        a.Dispose();
    }

    [Button]
    public void SortJobPerformance()
    {
        var r = new System.Random();

        var a = new NativeArray<int>(NUM_VALUES, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var cmp = new DefaultComparer<int>();

        for (int i = 0; i < a.Length; ++i)
        {
            a[i] = r.Next();
        }

        var sw = Stopwatch.StartNew();
        a.SortJob(cmp).Schedule().Complete();
        sw.Stop();
        Debug.Log($"Sort: {sw.ElapsedMilliseconds} ticks");

        a.Dispose();
    }

    [Button]
    public void MergeSortPerformance()
    {
        var r = new System.Random();

        var a = new NativeArray<int>(NUM_VALUES, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var cmp = new DefaultComparer<int>();

        for (int i = 0; i < a.Length; ++i)
        {
            a[i] = r.Next();
        }

        var sw = Stopwatch.StartNew();
        Sort.MergeSort(a, cmp);
        sw.Stop();
        Debug.Log($"Sort: {sw.ElapsedMilliseconds} ticks");
        
        a.Dispose();
    }

    [Button]
    public void MergeSortTest()
    {
        var r = new System.Random();

        var a = new NativeArray<int>(NUM_VALUES, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var cmp = new DefaultComparer<int>();

        for (int i = 0; i < a.Length; ++i)
        {
            a[i] = r.Next();
        }

        Sort.MergeSort(a, cmp);

        bool isSorted = true;
        for (int i = 0; i < a.Length - 1 && isSorted; ++i)
        {
            isSorted &= a[i] <= a[i + 1];
        }

        Assert.IsTrue(isSorted);

        a.Dispose();
    }

    public ComputeShader shader;
    [Button]
    public void SumTest()
    {
        var id = new[] { 0, 0, 0, 0, 0, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 2, 2, 2, 2, 2, 2,
                         2, 2, 3, 3, 2, 2, 2, 2 };
        var data = new float[32];
        for (int i = 0; i < data.Length; i++)
            data[i] = 1;
        
        var idBuffer = new ComputeBuffer(id.Length, sizeof(int));
        var dataBuffer = new ComputeBuffer(data.Length, sizeof(float));
        idBuffer.SetData(id);
        dataBuffer.SetData(data);
        
        // UnityEditorInternal.RenderDoc.BeginCaptureRenderDoc(EditorWindow.focusedWindow);
        shader.SetBuffer(0, "s_id", idBuffer);
        shader.SetBuffer(0, "s_data", dataBuffer);
        shader.Dispatch(0, 1, 1, 1);
        // UnityEditorInternal.RenderDoc.EndCaptureRenderDoc(EditorWindow.focusedWindow);

        var msg = "Raw data: \n";
        for (int i = 0; i < 32; i++)
            msg += data[i] + ", ";
        
        var result = new float[32];
        dataBuffer.GetData(result);
        msg += "\nGPU result: \n";
        for (int i = 0; i < 32; i++)
            msg += result[i] + ", ";

        float sum = 0;
        for (int i = id.Length - 1; i >= 0; --i)
        {
            int curID = id[i];
            int lastId = i > 0 ? id[i - 1] : -1;
            if (curID != lastId)
            {
                sum += data[i];
                result[i] = sum;
                sum = 0;
            }
            else
            {
                result[i] = 0;
                sum += data[i];
            }
        }
        msg += "\nCPU result: \n";
        for (int i = 0; i < 32; i++)
            msg += result[i] + ", ";
        Debug.Log(msg);
        
        idBuffer.Dispose();
        dataBuffer.Dispose();
    }
}

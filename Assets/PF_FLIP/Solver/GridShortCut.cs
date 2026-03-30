using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class GridShortCut : ScriptableObject
{
    public float3[] laplacian = new float3[64];
    public float[] divergence = new float[64];
}

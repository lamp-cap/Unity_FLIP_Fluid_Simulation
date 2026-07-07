using System;
using Abecombe.GPUUtil;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

public class NB_FLIP : MonoBehaviour
{
    private struct Particle
    {
        public uint2 Position;
        public uint2 Velocity;
    }

    public enum DrawType
    {
        Particles,
        Mesh,
    }

    public ComputeShader initCs;
    public ComputeShader buildLutCs;
    public ComputeShader P2GCs;
    public ComputeShader NBCs;
    public ComputeShader JFACs;
    public ComputeShader projectionCs;
    public ComputeShader G2PCs;
    public ComputeShader sortCs;
    
    public ComputeShader mcCs;
    public ComputeShader solverCs;

    [Range(0.5f, 1)]
    public float damping = 0.6667f;
    [Range(0, 1)]
    public float flipness;
    [Range(0f, 5f)] 
    public float mouseForce = 1;
    [Range(0f, 5f)] 
    public float mouseForceRange = 2;
    
    [Range(0, 2)]
    public float threshold = 0.5f;

    public DrawType drawType;

    public Material meshMat;
    
    private Material _material;
    
    private readonly GPUDoubleBuffer<Particle> _particles = new();
    private readonly GPUBuffer<float4> _particleRendering = new();
    
    private readonly GPUDoubleBuffer<uint> _particleID = new();
    private readonly GPUDoubleBuffer<uint> _particleHash = new();
    private readonly GPUDoubleBuffer<uint2> _gridParticleRange = new();
    
    private readonly GPUBuffer<uint> _scanFlags = new();
    private readonly GPUBuffer<int> _gridWeightsTemp = new();
    
    private readonly GPUTexture3D _gridTypes = new();
    private readonly GPUTexture3D _gridVelocity = new();
    private readonly GPUTexture3D _gridOldVelocity = new();
    private readonly GPUTexture3D[] _gridCoefficientPymaid = new GPUTexture3D[MGLevel];
    private readonly GPUTexture3D[] _gridDivergencePymaid = new GPUTexture3D[MGLevel];
    private readonly GPUTexture3D[] _gridPressurePymaid = new GPUTexture3D[MGLevel];
    private readonly GPUTexture3D _gridLaplace = new();
    private readonly GPUTexture3D _gridP = new();
    private readonly GPUTexture3D _gridSDF = new();
    
    private readonly GPUDoubleTexture3D _gridSeed = new();
    
    private ComputeBuffer globalHist;
    private ComputeBuffer passHist;
    private ComputeBuffer _argsBuffer;
    private ComputeBuffer _verticesBuffer;
    
    private ComputeBuffer _dotBuffer;
    private ComputeBuffer _counterBuffer;
    
    private const int ParticlesBufferSize = 256 * 128 * 128;
    private static readonly int3 GridSize = new int3(256, 128, 128);
    private const float GridSpacing = 0.2f;
    private int NumGrids => GridSize.x * GridSize.y * GridSize.z;

    private int _kernelInitParticles;
    private int _kernelInitSDF;
    private int _kernelParticlesCounterInit;
    
    private int _kernelUpsweep;
    private int _kernelScan;
    private int _kernelDownsweep;

    private int _kernelMakePair;
    private int _kernelClearGrid;
    private int _kernelSetRange;
    private int _kernelRearrange;
    
    private int _kernelSetGridType;
    private int _kernelSetGridLevel;
    private int _kernelClearCounter;
    private int _kernelClearScanFlags;
    private int _kernelParticlesCounter;
    private int _kernelResampleParticles;
    
    private int _kernelInitSeeds;
    private int _kernelJFA6Pass;
    private int _kernelFinalDistance;
    
    private int _kernelP2G;
    private int _kernelExternalForce;
    private int _kernelDivergence;
    private int _kernelProject;
    private int _kernelUpdateVelocity;
    private int _kernelG2P;
    private int _kernelAdvection;
    private int _kernelGridAdvection;

    private int _kernelRendering;
    
    private const int _pGroupThreadsX = ((ParticlesBufferSize) + 127) / 128;
    private readonly int3 _gGroupThreads = (GridSize + new int3(7, 7, 7)) / new int3(8, 8, 8);
    
    private const int k_radix = 256;
    private const int k_radixPasses = 4;
    private const int k_partitionSize = 3840;
    
    private MaterialPropertyBlock _mpb;
    private float ParticleRadius => GridSpacing * 0.25f;
    
    private ComputeBuffer _particleRenderingBufferWithArgs;
    private Camera _cam;
    private OrbitCamera _orbitCamera;
    private float2 _lastMousePlane = float2.zero;
    
    private int[] _particlesCount = new int[1];

    private int _vertBufferSize;
    private Bounds _bounds;
    private bool _slowDown;
    private bool _pause;

    private const int MGLevel = 5;
    
    void Start()
    {
        _cam = Camera.main;
        _orbitCamera = _cam.GetComponent<OrbitCamera>();
        
        _particles.Init(ParticlesBufferSize);
        _particleRendering.Init(ParticlesBufferSize);
        _particleID.Init(ParticlesBufferSize);
        _particleHash.Init(ParticlesBufferSize);
        
        _gridParticleRange.Init(NumGrids);
        _scanFlags.Init(DivRoundUp(NumGrids, 512) + 2);
        _gridVelocity.Init(GridSize, RenderTextureFormat.ARGBHalf);
        _gridOldVelocity.Init(GridSize, RenderTextureFormat.ARGBHalf);
        _gridTypes.Init(GridSize, RenderTextureFormat.RInt);
        _gridLaplace.Init(GridSize, RenderTextureFormat.RHalf);
        _gridP.Init(GridSize, RenderTextureFormat.RHalf);
        _gridSDF.Init(GridSize, RenderTextureFormat.RHalf);
        _gridSeed.Init(GridSize, RenderTextureFormat.RInt);
        
        for (int i = 0; i < MGLevel; i++)
        {
            _gridPressurePymaid[i] = new GPUTexture3D();
            _gridPressurePymaid[i].Init(GridSize >> i, RenderTextureFormat.RHalf);
            _gridDivergencePymaid[i] = new GPUTexture3D();
            _gridDivergencePymaid[i].Init(GridSize >> i, RenderTextureFormat.RHalf);
            _gridCoefficientPymaid[i] = new GPUTexture3D();
            _gridCoefficientPymaid[i].Init(GridSize >> i, RenderTextureFormat.ARGBHalf);
        }
        
        _gridWeightsTemp.Init(NumGrids * 7);
        _dotBuffer = new ComputeBuffer(2, sizeof(uint));
        _counterBuffer = new ComputeBuffer(1, sizeof(uint));
        _counterBuffer.SetData(new uint[] { 0 });
        
        globalHist = new ComputeBuffer(k_radix * k_radixPasses, 4);
        passHist = new ComputeBuffer(k_radix * DivRoundUp(ParticlesBufferSize, k_partitionSize) * k_radixPasses, 4);
        
        _kernelInitParticles = initCs.FindKernel("InitParticles");
        _kernelInitSDF = initCs.FindKernel("InitLevel");
        _kernelParticlesCounterInit = initCs.FindKernel("ParticlesCounter");

        _kernelMakePair = buildLutCs.FindKernel("MakePair");
        
        _kernelUpsweep = sortCs.FindKernel("UpSweep");
        _kernelScan = sortCs.FindKernel("Scan");
        _kernelDownsweep = sortCs.FindKernel("DownSweep");
        
        _kernelClearGrid = buildLutCs.FindKernel("ClearGrid");
        _kernelSetRange = buildLutCs.FindKernel("SetRange");
        _kernelRearrange = buildLutCs.FindKernel("Rearrange");
        
        _kernelP2G = P2GCs.FindKernel("P2G");
        _kernelExternalForce = P2GCs.FindKernel("AddForce");
        
        _kernelSetGridType = NBCs.FindKernel("SetGridType");
        _kernelSetGridLevel = NBCs.FindKernel("SetGridLevel");
        _kernelParticlesCounter = NBCs.FindKernel("ParticlesCounter");
        _kernelClearCounter = NBCs.FindKernel("ClearCounter");
        _kernelClearScanFlags = NBCs.FindKernel("ClearScanFlags");
        _kernelResampleParticles = NBCs.FindKernel("ResampleParticles");
        
        _kernelInitSeeds = JFACs.FindKernel("InitSeeds");
        _kernelJFA6Pass = JFACs.FindKernel("JFA6Pass");
        _kernelFinalDistance = JFACs.FindKernel("FinalDistance");
        
        _kernelDivergence = projectionCs.FindKernel("CalcDivergence");
        _kernelProject = projectionCs.FindKernel("Projection");
        _kernelUpdateVelocity = projectionCs.FindKernel("UpdateVelocity");
        
        _kernelG2P = G2PCs.FindKernel("G2P");
        _kernelAdvection = G2PCs.FindKernel("Advection");
        _kernelGridAdvection = G2PCs.FindKernel("GridAdvection");
        _kernelRendering = initCs.FindKernel("PrepareForRendering");

        // UnityEditorInternal.RenderDoc.BeginCaptureRenderDoc(EditorWindow.focusedWindow);
        InitParticles();
        // UnityEditorInternal.RenderDoc.EndCaptureRenderDoc(EditorWindow.focusedWindow);
        
        
        _particleRenderingBufferWithArgs = new ComputeBuffer(1, 5*sizeof(uint), ComputeBufferType.IndirectArguments);
        var args = new uint[5];
        args[0] = (uint)_particlesCount[0];
        args[1] = 1;
        _particleRenderingBufferWithArgs.SetData(args);
        
        _mpb = new MaterialPropertyBlock();
        
        _mpb.SetBuffer("_ParticleRenderingBuffer", _particleRendering);
        _mpb.SetFloat("_Radius", ParticleRadius);
        _mpb.SetFloat("_NearClipPlane", Camera.main.nearClipPlane);
        _mpb.SetFloat("_FarClipPlane", Camera.main.farClipPlane);
        _mpb.SetVector("_SlowColor", new Color(0f, 0.3891521f, 0.7735849f, 1f));
        _mpb.SetVector("_FastColor", new Color(0.5999911f, 0.7552593f, 0.9150943f, 1f));
        _mpb.SetVector("_VelocityRange", new Vector2(2f, 8f));
        _mpb.SetFloat("_FresnelPower", 0.3f);
        
        _material = new Material(Shader.Find("ParticleRendering/ParticleInstance"));

        float3 size = (float3)GridSize * GridSpacing;
        _vertBufferSize = Mathf.RoundToInt(Mathf.Pow(GridSize.x, 2.6f)) * 3;
        _verticesBuffer = new ComputeBuffer(_vertBufferSize, sizeof(float) * 4);
        meshMat.SetBuffer("_Buffer", _verticesBuffer);
        meshMat.SetVector("_Size", new Vector4(size.x, size.y, size.z));
        meshMat.SetTexture("_Density", _gridVelocity);

        _argsBuffer = new ComputeBuffer(5, sizeof(uint), ComputeBufferType.IndirectArguments);
        _argsBuffer.SetData(new[] {0, 1, 0, 0, 0});

        _bounds = new Bounds(size * 0.5f, size);

        _labelStyle = new GUIStyle()
        {
            alignment = TextAnchor.UpperLeft,
            fontSize = 32,
            normal = { textColor = Color.white }
        };
        
        Debug.Log($"Initializing GPU flip with particles: {ParticlesBufferSize}, GridSize: {GridSize}, numCells: {NumGrids}, bufferSize: {_vertBufferSize}");
        Debug.Log($"Initializing GPU flip with particlesT: {_pGroupThreadsX}, GridSizeT: {_gGroupThreads}");
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyUp(KeyCode.Space))
            _slowDown = !_slowDown;
        if (Input.GetKeyUp(KeyCode.P))
            _pause = !_pause;
        
        Simulation();
        
        if (drawType == DrawType.Mesh)
            Graphics.DrawProceduralIndirect(meshMat, _bounds, MeshTopology.Triangles, _argsBuffer);
        else
            Graphics.DrawProceduralIndirect(_material,
                _bounds,
                MeshTopology.Points,
                _particleRenderingBufferWithArgs,
                0,
                null,
                _mpb,
                ShadowCastingMode.Off,
                false
            );
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(_bounds.center, _bounds.size);
    }

    private GUIStyle _labelStyle;

    private void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 300, 20), 
            $"Particles: {_particlesCount[0]} / {ParticlesBufferSize}", _labelStyle);
    }

    private void Simulation()
    {
        // if (Time.frameCount > 20) return;
        // _counterBuffer.GetData(_particlesCount);
        // Debug.Log($"Particles: {_particlesCount[0]} / {ParticlesBufferSize}");
        // if (_particlesCount[0] <= 0 || _particlesCount[0] >= ParticlesBufferSize) return;
        // if (Time.frameCount == 10)
        // UnityEditorInternal.RenderDoc.BeginCaptureRenderDoc(EditorWindow.focusedWindow);
        var cmd = CommandBufferPool.Get("FLIP");
        cmd.Clear();
        
        cmd.BeginSample("Rendering");
        if (drawType == DrawType.Mesh)
            PrepareForRenderingMesh(cmd);
        else
            PrepareForRenderParticles(cmd);
        cmd.EndSample("Rendering");
        Graphics.ExecuteCommandBuffer(cmd);
        cmd.Clear();
        
        if (!_pause)
        {
            cmd.BeginSample("BuildLUT");
            BuildLut(cmd);
            cmd.EndSample("BuildLUT");
            Graphics.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            
            cmd.BeginSample("Resample");
            ResampleParticles(cmd);
            cmd.EndSample("Resample");
            Graphics.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            
            cmd.BeginSample("P2G");
            ParticleToGrid(cmd);
            cmd.EndSample("P2G");
            Graphics.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            
            cmd.BeginSample("Projection");
            Projection(cmd);
            cmd.EndSample("Projection");
            Graphics.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            
            cmd.BeginSample("G2P");
            GridToParticle(cmd);
            cmd.EndSample("G2P");
            Graphics.ExecuteCommandBuffer(cmd);
            cmd.Clear();
        }
        
        Graphics.ExecuteCommandBuffer(cmd);
        cmd.Clear();
        CommandBufferPool.Release(cmd);
        // if (Time.frameCount == 10)
        // UnityEditorInternal.RenderDoc.EndCaptureRenderDoc(EditorWindow.focusedWindow);
    }

    private void InitParticles()
    {
        initCs.SetFloat("_Scale", 0.7f);
        initCs.SetFloat("_CellSize", GridSpacing);
        initCs.SetFloat("_InvCellSize", 1f / GridSpacing);
        initCs.SetInt("_NumParticles", ParticlesBufferSize);
        initCs.SetVector("_InitMin0", new float4((float3)GridSize * new float3(0.02f, 0.25f, 0.04f) * GridSpacing, 1));
        initCs.SetVector("_InitMin1", new float4((float3)GridSize * new float3(0.6f, 0.25f, 0.26f) * GridSpacing, 1));
        initCs.SetVector("_GridSize", new Vector4(GridSize.x, GridSize.y, GridSize.z, 1));
        initCs.SetBuffer(_kernelInitParticles, "_ParticlesW", _particles.Read);
        initCs.SetBuffer(_kernelInitParticles, "_ParticlesIDW", _particleID.Read);
        initCs.Dispatch(_kernelInitParticles, ParticlesBufferSize / 128, 1, 1);
        
        var cmd = CommandBufferPool.Get("FLIP");
        cmd.Clear();
        
        SetParams(cmd, initCs);
        
        cmd.SetComputeVectorParam(initCs, "_Start", new Vector4(20, 1, 1, 0));
        cmd.SetComputeVectorParam(initCs, "_End", new Vector4(240, 80, 120, 0));
        cmd.SetComputeTextureParam(initCs, _kernelInitSDF, "_GridSDFW", _gridSDF);
        cmd.DispatchCompute(initCs, _kernelInitSDF, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        var cs = buildLutCs;
        SetParams(cmd, cs);
        int kernel = _kernelClearGrid;
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeW", _gridParticleRange.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_PartitionScanDescriptors", _scanFlags);
        cmd.SetComputeTextureParam(cs, kernel, "_PressureW", _gridPressurePymaid[0]);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        cs = JFACs;
        SetParams(cmd, cs);
        kernel = _kernelInitSeeds;
        cmd.SetComputeTextureParam(cs, kernel, "_GridSDFR", _gridSDF);
        cmd.SetComputeTextureParam(cs, kernel, "_SeedBufferW", _gridSeed.Read);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        kernel = _kernelJFA6Pass;
        int stepSize = 8;
        while(stepSize > 0)
        {
            cmd.SetComputeIntParam(cs, "_StepSize", stepSize);
            cmd.SetComputeTextureParam(cs, kernel, "_SeedBufferR", _gridSeed.Read);
            cmd.SetComputeTextureParam(cs, kernel, "_SeedBufferW", _gridSeed.Write);
            cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
            _gridSeed.Swap();
            stepSize >>= 1;
        }

        kernel = _kernelFinalDistance;
        cmd.SetComputeTextureParam(cs, kernel, "_SeedBufferR", _gridSeed.Read);
        cmd.SetComputeTextureParam(cs, kernel, "_GridSDFW", _gridSDF);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        cs = NBCs;
        SetParams(cmd, cs);
        
        kernel = _kernelClearCounter;
        cmd.SetComputeBufferParam(cs, kernel, "_CounterW", _counterBuffer);
        cmd.DispatchCompute(cs, kernel, 1, 1, 1);

        kernel = _kernelClearScanFlags;
        cmd.SetComputeBufferParam(cs, kernel, "_PartitionScanDescriptors", _scanFlags);
        cmd.DispatchCompute(cs, kernel, DivRoundUp(NumGrids / 512 + 2, 128), 1, 1);

        kernel = _kernelParticlesCounterInit;
        cmd.SetComputeTextureParam(initCs, kernel, "_GridSDFR", _gridSDF);
        cmd.SetComputeBufferParam(initCs, kernel, "_PartitionScanDescriptors", _scanFlags);
        cmd.SetComputeBufferParam(initCs, kernel, "_CounterW", _counterBuffer);
        cmd.SetComputeBufferParam(initCs, kernel, "_ParticlesIDW", _particleID.Read);
        cmd.SetComputeBufferParam(initCs, kernel, "_ParticlesRangeW", _gridParticleRange.Write);
        cmd.DispatchCompute(initCs, kernel, DivRoundUp(NumGrids, 512), 1, 1);
        
        _gridParticleRange.Swap();
        
        kernel = _kernelResampleParticles;;
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityR", _gridVelocity);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeR", _gridParticleRange.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesIDR", _particleID.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesR", _particles.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesW", _particles.Write);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        _particles.Swap();
        
        kernel = _kernelSetGridType;
        cmd.SetComputeTextureParam(cs, kernel, "_GridSDFR", _gridSDF);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeR", _gridParticleRange.Read);
        cmd.SetComputeTextureParam(cs, kernel, "_GridTypesW", _gridTypes);
        cmd.SetComputeTextureParam(cs, kernel, "_GridCoefficientW", _gridCoefficientPymaid[0]);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        Graphics.ExecuteCommandBuffer(cmd);
        cmd.Clear();
        CommandBufferPool.Release(cmd);
        
        _counterBuffer.GetData(_particlesCount);
        Debug.Log( "Init with particles count: " + _particlesCount[0]);
    }

    private void BuildLut(CommandBuffer cmd)
    {
        var cs = buildLutCs;
        SetParams(cmd, cs);
        
        // clear grid data
        int kernel = _kernelClearGrid;
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeW", _gridParticleRange.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_PartitionScanDescriptors", _scanFlags);
        cmd.SetComputeTextureParam(cs, kernel, "_PressureW", _gridPressurePymaid[0]);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        // make pair
        kernel = _kernelMakePair;
        cmd.SetComputeBufferParam(cs, kernel, "_CounterR", _counterBuffer);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesR", _particles.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesIDW", _particleID.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesHashW", _particleHash.Read);
        cmd.DispatchCompute(cs, kernel, _pGroupThreadsX, 1, 1);
        
        // sort
        Sort(cmd, _particleHash, _particleID, 24);
        
        // set range
        kernel = _kernelSetRange;
        cmd.SetComputeBufferParam(cs, kernel, "_CounterR", _counterBuffer);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeW", _gridParticleRange.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesHashR", _particleHash.Read);
        cmd.DispatchCompute(cs, kernel, _pGroupThreadsX, 1, 1);
        
        // rearrange
        kernel = _kernelRearrange;
        cmd.SetComputeBufferParam(cs, kernel, "_CounterR", _counterBuffer);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesIDR", _particleID.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesR", _particles.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesW", _particles.Write);
        cmd.DispatchCompute(cs, kernel, _pGroupThreadsX, 1, 1);
        _particles.Swap();
    }

    private void ResampleParticles(CommandBuffer cmd)
    {
        var cs = NBCs;
        SetParams(cmd, cs);
        
        int kernel = _kernelSetGridType;
        cmd.SetComputeTextureParam(cs, kernel, "_GridSDFR", _gridSDF);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeR", _gridParticleRange.Read);
        cmd.SetComputeTextureParam(cs, kernel, "_GridTypesW", _gridTypes);
        cmd.SetComputeTextureParam(cs, kernel, "_GridCoefficientW", _gridCoefficientPymaid[0]);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        kernel = _kernelSetGridLevel;
        cmd.SetComputeTextureParam(cs, kernel, "_GridTypesR", _gridTypes);
        cmd.SetComputeTextureParam(cs, kernel, "_GridSDFW", _gridSDF);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);

        cs = JFACs;
        SetParams(cmd, cs);
        kernel = _kernelInitSeeds;
        cmd.SetComputeTextureParam(cs, kernel, "_GridSDFR", _gridSDF);
        cmd.SetComputeTextureParam(cs, kernel, "_SeedBufferW", _gridSeed.Read);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        kernel = _kernelJFA6Pass;
        int stepSize = 8;
        while(stepSize > 0)
        {
            cmd.SetComputeIntParam(cs, "_StepSize", stepSize);
            cmd.SetComputeTextureParam(cs, kernel, "_SeedBufferR", _gridSeed.Read);
            cmd.SetComputeTextureParam(cs, kernel, "_SeedBufferW", _gridSeed.Write);
            cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
            _gridSeed.Swap();
            stepSize >>= 1;
        }

        kernel = _kernelFinalDistance;
        cmd.SetComputeTextureParam(cs, kernel, "_SeedBufferR", _gridSeed.Read);
        cmd.SetComputeTextureParam(cs, kernel, "_GridSDFW", _gridSDF);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        cs = NBCs;
        SetParams(cmd, cs);
        
        kernel = _kernelClearCounter;
        cmd.SetComputeBufferParam(cs, kernel, "_CounterW", _counterBuffer);
        cmd.DispatchCompute(cs, kernel, 1, 1, 1);

        kernel = _kernelClearScanFlags;
        cmd.SetComputeBufferParam(cs, kernel, "_PartitionScanDescriptors", _scanFlags);
        cmd.DispatchCompute(cs, kernel, DivRoundUp(NumGrids / 512 + 2, 128), 1, 1);

        kernel = _kernelParticlesCounter;

        cmd.SetComputeTextureParam(cs, kernel, "_GridSDFR", _gridSDF);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeR", _gridParticleRange.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_PartitionScanDescriptors", _scanFlags);
        cmd.SetComputeBufferParam(cs, kernel, "_CounterW", _counterBuffer);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesIDW", _particleID.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeW", _gridParticleRange.Write);
        cmd.DispatchCompute(cs, kernel, DivRoundUp(NumGrids, 512), 1, 1);
        _gridParticleRange.Swap();
        
        kernel = _kernelResampleParticles;
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityR", _gridVelocity);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeR", _gridParticleRange.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesIDR", _particleID.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesR", _particles.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesW", _particles.Write);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        _particles.Swap();
    }

    private void ParticleToGrid(CommandBuffer cmd)
    {
        var cs = P2GCs;
        SetParams(cmd, cs);
        
        int kernel = _kernelP2G;
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRange", _gridParticleRange.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_Particles", _particles.Read);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityOldW", _gridOldVelocity);
        cmd.SetComputeTextureParam(cs, kernel, "_GridSDFR", _gridSDF);
        
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        var mouseRay = _cam.ScreenPointToRay(Input.mousePosition);
        cmd.SetComputeVectorParam(cs,"_RayOrigin", mouseRay.origin);
        cmd.SetComputeVectorParam(cs,"_RayDirection", mouseRay.direction);

        var height = Mathf.Tan(_cam.fieldOfView * 0.5f * Mathf.Deg2Rad) * 2f;
        var width = height * Screen.width / Screen.height;
        var mousePlane = ((float3)Input.mousePosition).xy / new float2(Screen.width, Screen.height) - 0.5f;
        mousePlane *= new float2(width, height);
        mousePlane *= _orbitCamera.Distance;
        var cameraViewMatrix = _cam.worldToCameraMatrix;
        var cameraRight = new float3(cameraViewMatrix[0], cameraViewMatrix[4], cameraViewMatrix[8]);
        var cameraUp = new float3(cameraViewMatrix[1], cameraViewMatrix[5], cameraViewMatrix[9]);
        var mouseVelocity = (mousePlane - _lastMousePlane) / Time.smoothDeltaTime;
        if (Input.GetMouseButton(0) || Input.GetMouseButton(1) || Input.GetMouseButton(2) || Time.frameCount <= 1)
            mouseVelocity = float2.zero;
        _lastMousePlane = mousePlane;
        var mouseAxisVelocity = mouseVelocity.x * cameraRight + mouseVelocity.y * cameraUp;
        cmd.SetComputeVectorParam(cs, "_MouseForceParameter", new float4(mouseAxisVelocity * mouseForce, mouseForceRange));
        
        cmd.SetComputeVectorParam(cs, "_Gravity", new Vector4(0f, -9f, 0f, 0f));
        kernel = _kernelExternalForce;
        cmd.SetComputeTextureParam(cs, kernel, "_GridTypesR", _gridTypes);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityOldR", _gridOldVelocity);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityW", _gridVelocity);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
    }

    private void Projection(CommandBuffer cmd)
    {
        var cs = projectionCs;
        SetParams(cmd, cs);
        cmd.SetComputeFloatParam(cs, "_Damping", damping);

        int kernel = _kernelDivergence;
        
        cmd.SetComputeTextureParam(cs, kernel, "_GridTypesR", _gridTypes);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityR", _gridVelocity);
        cmd.SetComputeTextureParam(cs, kernel, "_DivergenceW", _gridDivergencePymaid[0]);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);

        MGPCG(cmd);
        
        kernel = _kernelUpdateVelocity;
        cmd.SetComputeTextureParam(cs, kernel, "_GridTypesR", _gridTypes);
        cmd.SetComputeTextureParam(cs, kernel, "_PressureR", _gridPressurePymaid[0]);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityW", _gridVelocity);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
    }

    private void GridToParticle(CommandBuffer cmd)
    {
        var cs = G2PCs;
        SetParams(cmd, cs);
        cmd.SetComputeFloatParam(cs, "_Flipness", flipness);
        int kernel = _kernelG2P;
        cmd.SetComputeBufferParam(cs, kernel, "_CounterR", _counterBuffer);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityR", _gridVelocity);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityOldR", _gridOldVelocity);
        cmd.SetComputeBufferParam(cs, kernel, "_Particles", _particles.Read);
        cmd.DispatchCompute(cs, kernel, _pGroupThreadsX, 1, 1);
        
        kernel = _kernelAdvection;
        cmd.SetComputeBufferParam(cs, kernel, "_CounterR", _counterBuffer);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityR", _gridVelocity);
        cmd.SetComputeBufferParam(cs, kernel, "_Particles", _particles.Read);
        cmd.DispatchCompute(cs, kernel, _pGroupThreadsX, 1, 1);
        
        kernel = _kernelGridAdvection;
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityR", _gridVelocity);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityOldW", _gridOldVelocity);
        cmd.SetComputeTextureParam(cs, kernel, "_GridTypesR", _gridTypes);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
    }

    private void PrepareForRenderParticles(CommandBuffer cmd)
    {
        var cs = initCs;
        SetParams(cmd, cs);
        int kernel = _kernelRendering;
        cmd.SetComputeBufferParam(cs, kernel, "_CounterR", _counterBuffer);
        cmd.SetComputeBufferParam(cs, kernel, "_Args", _particleRenderingBufferWithArgs);
        cmd.SetComputeBufferParam(cs, kernel, "_CounterR", _counterBuffer);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesR", _particles.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRender", _particleRendering);
        cmd.DispatchCompute(cs, kernel, _pGroupThreadsX, 1, 1);
    }

    private void PrepareForRenderingMesh(CommandBuffer cmd)
    {
        var cs = initCs;
        cmd.SetComputeVectorParam(cs, "_Size", new Vector4(GridSize.x, GridSize.y, GridSize.z, 0));
        cmd.SetComputeTextureParam(cs, 2, "_Src", _gridOldVelocity);
        cmd.SetComputeTextureParam(cs, 2, "_Dst", _gridVelocity);
        cmd.DispatchCompute(cs, 2, GridSize.x / 8, GridSize.y / 8, GridSize.z / 8);
        
        cs = mcCs;
        
        cmd.SetComputeVectorParam(cs, "_Size", new Vector4(GridSize.x, GridSize.y, GridSize.z, 0));
        cmd.SetComputeIntParam(cs, "_BufferSize", _vertBufferSize);
        cmd.SetComputeFloatParam(cs, "_Target", threshold);
        cmd.SetComputeFloatParam(cs, "_CellSize", GridSpacing);
        
        // clear vertex counter
        cmd.SetComputeBufferParam(cs, 0, "_Counter", _argsBuffer);
        cmd.DispatchCompute(cs, 0, 1, 1, 1);
        
        //Make the mesh verts
        
        cmd.SetComputeTextureParam(cs, 1, "_Voxels", _gridOldVelocity);
        cmd.SetComputeBufferParam(cs, 1, "_Buffer", _verticesBuffer);
        cmd.SetComputeBufferParam(cs, 1, "_Counter", _argsBuffer);
        
        cmd.DispatchCompute(cs, 1, GridSize.x / 8 + 1, GridSize.y / 8, GridSize.z / 8 + 1);
    }

    private void Sort(CommandBuffer cmd, GPUDoubleBuffer<uint> toSort, GPUDoubleBuffer<uint> payload, int maxDigit = 32)
    {
        int sortSize = toSort.Size;
        int numThreadBlocks = (sortSize + k_partitionSize) / k_partitionSize;
        
        cmd.SetComputeIntParam(sortCs, "e_numKeys", sortSize);
        cmd.SetComputeIntParam(sortCs, "e_threadBlocks", numThreadBlocks);

        cmd.SetComputeBufferParam(sortCs, 0, "b_globalHist", globalHist);

        cmd.SetComputeBufferParam(sortCs, _kernelUpsweep, "b_passHist", passHist);
        cmd.SetComputeBufferParam(sortCs, _kernelUpsweep, "b_globalHist", globalHist);

        cmd.SetComputeBufferParam(sortCs, _kernelScan, "b_passHist", passHist);

        cmd.SetComputeBufferParam(sortCs, _kernelDownsweep, "b_passHist", passHist);
        cmd.SetComputeBufferParam(sortCs, _kernelDownsweep, "b_globalHist", globalHist);
        
        cmd.DispatchCompute(sortCs, 0, 1, 1, 1);
        
        for (int radixShift = 0; radixShift < maxDigit; radixShift += 8)
        {
            cmd.SetComputeIntParam(sortCs, "e_radixShift", radixShift);

            cmd.SetComputeBufferParam(sortCs, _kernelUpsweep, "b_sort", toSort.Read);
            cmd.DispatchCompute(sortCs, _kernelUpsweep, numThreadBlocks, 1, 1);

            cmd.DispatchCompute(sortCs, _kernelScan, k_radix, 1, 1);

            cmd.SetComputeBufferParam(sortCs, _kernelDownsweep, "b_sort", toSort.Read);
            cmd.SetComputeBufferParam(sortCs, _kernelDownsweep, "b_sortPayload", payload.Read);
            cmd.SetComputeBufferParam(sortCs, _kernelDownsweep, "b_alt", toSort.Write);
            cmd.SetComputeBufferParam(sortCs, _kernelDownsweep, "b_altPayload", payload.Write);
            cmd.DispatchCompute(sortCs, _kernelDownsweep, numThreadBlocks, 1, 1);

            toSort.Swap();
            payload.Swap();
        }
        
    }

    private void MGPCG(CommandBuffer cmd)
    {
        var cs = solverCs;
        
        cmd.SetComputeFloatParam(cs, "_h", GridSpacing);
        cmd.SetComputeFloatParam(cs, "_h2", GridSpacing * GridSpacing);
        cmd.SetComputeFloatParam(cs, "_ih2", 1f / GridSpacing / GridSpacing);
        
        // Multigrid preconditioner
        int top = MGLevel - 1;
        for (int i = 0; i < top; i++)
        {
            int3 res = GridSize >> i;
            cmd.SetComputeVectorParam(cs, "_size", new Vector3(res.x, res.y, res.z));
            PreSmooth(cmd, 3, i);
            
            cmd.SetComputeTextureParam(cs, 2, "_af", _gridCoefficientPymaid[i]);
            cmd.SetComputeTextureParam(cs, 2, "_ac", _gridCoefficientPymaid[i + 1]);
            cmd.SetComputeTextureParam(cs, 2, "_rf", _gridDivergencePymaid[i]);
            cmd.SetComputeTextureParam(cs, 2, "_rc", _gridDivergencePymaid[i + 1]);
            cmd.SetComputeTextureParam(cs, 2, "_efR", _gridPressurePymaid[i]);
            cmd.SetComputeTextureParam(cs, 2, "_ecW", _gridPressurePymaid[i + 1]);
            cmd.DispatchCompute(cs, 2,
                res.x / 16, res.y / 16, res.z / 16);
        }

        cmd.SetComputeBufferParam(cs, 4, "_counterW", _dotBuffer);
        cmd.SetComputeTextureParam(cs, 4, "_x", _gridPressurePymaid[top]);
        cmd.SetComputeTextureParam(cs, 4, "_b", _gridDivergencePymaid[top]);
        cmd.SetComputeTextureParam(cs, 4, "_coefficients", _gridCoefficientPymaid[top]);
        cmd.DispatchCompute(cs, 4, 1, 1, 1);

        for (int i = top - 1; i >= 0; i--)
        {
            int3 res = GridSize >> i;
            cmd.SetComputeVectorParam(cs, "_size", new Vector3(res.x, res.y, res.z));
            cmd.SetComputeTextureParam(cs, 3, "_af", _gridCoefficientPymaid[i]);
            cmd.SetComputeTextureParam(cs, 3, "_efW", _gridPressurePymaid[i]);
            cmd.SetComputeTextureParam(cs, 3, "_ecR", _gridPressurePymaid[i + 1]);
            cmd.DispatchCompute(cs, 3,
                res.x / 8, res.y / 8, res.z / 8);
            PostSmooth(cmd, 3, i);
        }
        
        // copy Z to P
        cmd.CopyTexture(_gridPressurePymaid[0], _gridP);
        
        // dot R Z
        cmd.SetComputeIntParam(cs, "_Index", 0);
        cmd.SetComputeTextureParam(cs, 5, "_lhs", _gridPressurePymaid[0]);
        cmd.SetComputeTextureParam(cs, 5, "_rhs", _gridDivergencePymaid[0]);
        cmd.SetComputeBufferParam(cs, 5, "_counterW", _dotBuffer);
        cmd.DispatchCompute(cs, 5,
            GridSize.x / 8, GridSize.y / 8, GridSize.z / 8);
        
        // Laplace
        cmd.SetComputeTextureParam(cs, 6, "_p", _gridP);
        cmd.SetComputeTextureParam(cs, 6, "_coefficients", _gridCoefficientPymaid[0]);
        cmd.SetComputeTextureParam(cs, 6, "_Ap", _gridLaplace);
        cmd.DispatchCompute(cs, 6,
            GridSize.x / 8, GridSize.y / 8, GridSize.z / 8);
        
        // dot p Ap
        cmd.SetComputeIntParam(cs, "_Index", 1);
        cmd.SetComputeTextureParam(cs, 5, "_lhs", _gridPressurePymaid[0]);
        cmd.SetComputeTextureParam(cs, 5, "_rhs", _gridLaplace);
        cmd.SetComputeBufferParam(cs, 5, "_counterW", _dotBuffer);
        cmd.DispatchCompute(cs, 5,
            GridSize.x / 8, GridSize.y / 8, GridSize.z / 8);
        
        // Update V
        cmd.SetComputeTextureParam(cs, 7, "_x", _gridPressurePymaid[top]);
        cmd.SetComputeTextureParam(cs, 7, "_p", _gridP);
        cmd.SetComputeBufferParam(cs, 7, "_counterR", _dotBuffer);
        cmd.DispatchCompute(cs, 7,
            GridSize.x / 8, GridSize.y / 8, GridSize.z / 8);
    }

    private void PreSmooth(CommandBuffer cmd, int iter, int level)
    {
        int3 res = GridSize >> level;
        var cs = solverCs;
        cmd.SetComputeTextureParam(cs, 0, "_x", _gridPressurePymaid[level]);
        cmd.SetComputeTextureParam(cs, 0, "_b", _gridDivergencePymaid[level]);
        cmd.SetComputeTextureParam(cs, 0, "_coefficients", _gridCoefficientPymaid[level]);
        cmd.SetComputeTextureParam(cs, 1, "_x", _gridPressurePymaid[level]);
        cmd.SetComputeTextureParam(cs, 1, "_b", _gridDivergencePymaid[level]);
        cmd.SetComputeTextureParam(cs, 1, "_coefficients", _gridCoefficientPymaid[level]);
        
        for (int i = 0; i < iter; i++)
        {
            cmd.DispatchCompute(cs, 0,
                res.x / 8, res.y / 8, res.z / 8);
            cmd.DispatchCompute(cs, 1,
                res.x / 8, res.y / 8, res.z / 8);
        }
    }
    
    private void PostSmooth(CommandBuffer cmd, int iter, int level)
    {
        int3 res = GridSize >> level;
        var cs = solverCs;
        cmd.SetComputeTextureParam(cs, 0, "_x", _gridPressurePymaid[level]);
        cmd.SetComputeTextureParam(cs, 0, "_b", _gridDivergencePymaid[level]);
        cmd.SetComputeTextureParam(cs, 0, "_coefficients", _gridCoefficientPymaid[level]);
        cmd.SetComputeTextureParam(cs, 1, "_x", _gridPressurePymaid[level]);
        cmd.SetComputeTextureParam(cs, 1, "_b", _gridDivergencePymaid[level]);
        cmd.SetComputeTextureParam(cs, 1, "_coefficients", _gridCoefficientPymaid[level]);
        
        for (int i = 0; i < iter; i++)
        {
            cmd.DispatchCompute(cs, 1,
                res.x / 8, res.y / 8, res.z / 8);
            cmd.DispatchCompute(cs, 0,
                res.x / 8, res.y / 8, res.z / 8);
        }
    }
    private void SetParams(CommandBuffer cmd, ComputeShader cs)
    {
        cmd.SetComputeVectorParam(cs, "_GridMin", new Vector3(0, 0, 0));
        cmd.SetComputeVectorParam(cs, "_GridSize", new Vector3(GridSize.x, GridSize.y, GridSize.z));
        cmd.SetComputeIntParam(cs, "_NumParticles", ParticlesBufferSize);
        cmd.SetComputeIntParam(cs, "_NumCells", GridSize.x * GridSize.y * GridSize.z);
        cmd.SetComputeFloatParam(cs, "_CellSize", GridSpacing);
        cmd.SetComputeFloatParam(cs, "_InvCellSize", 1f / GridSpacing);
        cmd.SetComputeFloatParam(cs, "_DeltaTime", 1f / (_slowDown ? 600f : 60f));
        cmd.SetComputeIntParam(cs, "_NumPartitions", DivRoundUp(NumGrids, 512));
    }
    
    private static int DivRoundUp(int x, int y)
    {
        return (x + y - 1) / y;
    }

    private void OnDestroy()
    {
        _particles.Dispose();
        _particleRendering.Dispose();
        _particleID.Dispose();
        _particleHash.Dispose();
        
        _gridParticleRange.Dispose();
        _gridVelocity.Dispose();
        _gridOldVelocity.Dispose();
        _gridTypes.Dispose();
        _gridLaplace.Dispose();
        _gridP.Dispose();
        _gridSDF.Dispose();
        _counterBuffer.Dispose();
        _scanFlags.Dispose();
        
        _gridSeed.Dispose();
        
        foreach (var buffer in _gridDivergencePymaid)
            buffer.Dispose();
        
        foreach (var buffer in _gridPressurePymaid)
            buffer.Dispose();
        
        foreach (var buffer in _gridCoefficientPymaid)
            buffer.Dispose();
        
        _gridWeightsTemp.Dispose();
        _dotBuffer.Dispose();
        
        globalHist.Dispose();
        passHist.Dispose();
        
        _particleRenderingBufferWithArgs.Dispose();
        
        _argsBuffer.Dispose();
        _verticesBuffer.Dispose();
        
        Destroy(_material);
    }
}

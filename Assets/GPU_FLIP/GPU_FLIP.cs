using Abecombe.GPUUtil;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

public class GPU_FLIP : MonoBehaviour
{
    private struct Particle
    {
        public uint2 Position;
        public uint2 Velocity;
    }

    public ComputeShader initCs;
    public ComputeShader buildLutCs;
    public ComputeShader P2GCs;
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

    public Material meshMat;
    
    private Material _material;
    
    private readonly GPUDoubleBuffer<Particle> _particles = new();
    private readonly GPUBuffer<float4> _particleRendering = new();
    
    private readonly GPUDoubleBuffer<uint> _particleID = new();
    private readonly GPUDoubleBuffer<uint> _particleHash = new();

    private readonly GPUBuffer<uint2> _gridParticleRange = new();
    private readonly GPUBuffer<uint2> _blockParticleRange = new();
    private readonly GPUBuffer<int> _gridWeightsTemp = new();
    
    private readonly GPUTexture3D _gridTypes = new();
    private readonly GPUTexture3D _gridVelocity = new();
    private readonly GPUTexture3D _gridOldVelocity = new();
    private readonly GPUTexture3D[] _gridCoefficientPymaid = new GPUTexture3D[MGLevel];
    private readonly GPUTexture3D[] _gridDivergencePymaid = new GPUTexture3D[MGLevel];
    private readonly GPUTexture3D[] _gridPressurePymaid = new GPUTexture3D[MGLevel];
    private readonly GPUTexture3D _gridLaplace = new();
    private readonly GPUTexture3D _gridP = new();
    
    private ComputeBuffer globalHist;
    private ComputeBuffer passHist;
    private ComputeBuffer _argsBuffer;
    private ComputeBuffer _verticesBuffer;
    
    private ComputeBuffer _dotBuffer;
    
    private const int NumParticles = 256 * 128 * 128;
    private static readonly int3 GridSize = new int3(256, 128, 128);
    private const float GridSpacing = 0.2f;
    private int NumGrids => GridSize.x * GridSize.y * GridSize.z;

    private int _kernelInitParticles;
    
    private int _kernelUpsweep;
    private int _kernelScan;
    private int _kernelDownsweep;

    private int _kernelMakePair;
    private int _kernelClearGrid;
    private int _kernelSetRange;
    private int _kernelRearrange;
    private int _kernelBlockRange;
    
    private int _kernelMyGridType;
    
    private int _kernelP2G;
    private int _kernelNormalize;
    private int _kernelExternalForce;
    private int _kernelDivergence;
    private int _kernelProject;
    private int _kernelUpdateVelocity;
    private int _kernelG2P;
    private int _kernelAdvection;

    private int _kernelRendering;
    
    private const int PGroupThreadsX = (NumParticles + 127) / 128;
    private readonly int3 _gGroupThreads = (GridSize + new int3(7, 3, 3)) / new int3(8, 4, 4);
    
    private const int k_radix = 256;
    private const int k_radixPasses = 4;
    private const int k_partitionSize = 3840;
    
    private MaterialPropertyBlock _mpb;
    private float ParticleRadius => GridSpacing * 0.25f;
    
    private ComputeBuffer _particleRenderingBufferWithArgs;
    private Camera _cam;
    private OrbitCamera _orbitCamera;
    private float2 _lastMousePlane = float2.zero;

    private int _vertBufferSize;
    private Bounds _bounds;
    private bool _slowDown;
    private bool _pause;

    private const int MGLevel = 5;
    
    void Start()
    {
        _cam = Camera.main;
        _orbitCamera = _cam.GetComponent<OrbitCamera>();
        
        _particles.Init(NumParticles);
        _particleRendering.Init(NumParticles);
        _particleID.Init(NumParticles);
        _particleHash.Init(NumParticles);
        
        _gridParticleRange.Init(NumGrids);
        _blockParticleRange.Init(NumGrids / 512);
        _gridVelocity.Init(GridSize, RenderTextureFormat.ARGBHalf);
        _gridOldVelocity.Init(GridSize, RenderTextureFormat.ARGBHalf);
        _gridTypes.Init(GridSize, RenderTextureFormat.RInt);
        _gridLaplace.Init(GridSize, RenderTextureFormat.RHalf);
        _gridP.Init(GridSize, RenderTextureFormat.RHalf);
        
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
        
        globalHist = new ComputeBuffer(k_radix * k_radixPasses, 4);
        passHist = new ComputeBuffer(k_radix * DivRoundUp(NumParticles, k_partitionSize) * k_radixPasses, 4);
        
        _kernelInitParticles = initCs.FindKernel("InitParticles");
        
        _kernelMakePair = buildLutCs.FindKernel("MakePair");
        
        _kernelUpsweep = sortCs.FindKernel("UpSweep");
        _kernelScan = sortCs.FindKernel("Scan");
        _kernelDownsweep = sortCs.FindKernel("DownSweep");
        
        _kernelClearGrid = buildLutCs.FindKernel("ClearGrid");
        _kernelSetRange = buildLutCs.FindKernel("SetRange");
        _kernelRearrange = buildLutCs.FindKernel("Rearrange");
        _kernelBlockRange = buildLutCs.FindKernel("BlockRange");
        
        _kernelP2G = P2GCs.FindKernel("P2G");
        _kernelMyGridType = P2GCs.FindKernel("SetGridType");
        _kernelNormalize = P2GCs.FindKernel("Normalize");
        
        _kernelExternalForce = P2GCs.FindKernel("AddForce");
        
        _kernelDivergence = projectionCs.FindKernel("CalcDivergence");
        _kernelProject = projectionCs.FindKernel("Projection");
        _kernelUpdateVelocity = projectionCs.FindKernel("UpdateVelocity");
        
        _kernelG2P = G2PCs.FindKernel("G2P");
        _kernelAdvection = G2PCs.FindKernel("Advection");
        
        _kernelRendering = initCs.FindKernel("PrepareForRendering");

        InitParticles();
        
        _particleRenderingBufferWithArgs = new ComputeBuffer(1, 5*sizeof(uint), ComputeBufferType.IndirectArguments);
        var args = new uint[5];
        args[0] = NumParticles;
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
        
        Debug.Log($"Initializing GPU flip with particles: {NumParticles}, GridSize: {GridSize}, numCells: {NumGrids}, bufferSize: {_vertBufferSize}");
        Debug.Log($"Initializing GPU flip with particlesT: {PGroupThreadsX}, GridSizeT: {_gGroupThreads}");
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyUp(KeyCode.Space))
            _slowDown = !_slowDown;
        if (Input.GetKeyUp(KeyCode.P))
            _pause = !_pause;
        if (!_pause)
            Simulation();
        
        Graphics.DrawProceduralIndirect(meshMat, _bounds, MeshTopology.Triangles, _argsBuffer);
        
        // Graphics.DrawProceduralIndirect(_material,
        //     _bounds,
        //     MeshTopology.Points,
        //     _particleRenderingBufferWithArgs,
        //     0,
        //     null,
        //     _mpb,
        //     ShadowCastingMode.Off,
        //     false
        // );
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(_bounds.center, _bounds.size);
    }

    private void Simulation()
    {
        var cmd = CommandBufferPool.Get("FLIP");
        cmd.Clear();
        
        cmd.BeginSample("BuildLUT");
        BuildLut(cmd);
        cmd.EndSample("BuildLUT");
        
        cmd.BeginSample("P2G");
        ParticleToGrid(cmd);
        cmd.EndSample("P2G");
        
        cmd.BeginSample("Projection");
        Projection(cmd, 64);
        cmd.EndSample("Projection");
        
        cmd.BeginSample("G2P");
        GridToParticle(cmd);
        cmd.EndSample("G2P");
        
        cmd.BeginSample("Rendering");
        // PrepareForRenderParticles(cmd);
        RenderingMesh(cmd);
        cmd.EndSample("Rendering");
        
        Graphics.ExecuteCommandBuffer(cmd);
        cmd.Clear();
        CommandBufferPool.Release(cmd);
    }

    private void InitParticles()
    {
        initCs.SetFloat("_Scale", 0.7f);
        initCs.SetFloat("_CellSize", GridSpacing);
        initCs.SetFloat("_InvCellSize", 1f / GridSpacing);
        initCs.SetInt("_NumParticles", NumParticles);
        initCs.SetVector("_InitMin0", new float4((float3)GridSize * new float3(0.02f, 0.25f, 0.04f) * GridSpacing, 1));
        initCs.SetVector("_InitMin1", new float4((float3)GridSize * new float3(0.6f, 0.25f, 0.26f) * GridSpacing, 1));
        initCs.SetVector("_GridSize", new Vector4(GridSize.x, GridSize.y, GridSize.z, 1));
        initCs.SetBuffer(_kernelInitParticles, "_ParticlesW", _particles.Read);
        initCs.Dispatch(_kernelInitParticles, PGroupThreadsX, 1, 1);
    }

    private void BuildLut(CommandBuffer cmd)
    {
        var cs = buildLutCs;
        SetParams(cmd, cs);
        
        // clear grid data
        int kernel = _kernelClearGrid;
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeW", _gridParticleRange);
        cmd.SetComputeTextureParam(cs, kernel, "_PressureW", _gridPressurePymaid[0]);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
        // make pair
        kernel = _kernelMakePair;
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesR", _particles.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesIDW", _particleID.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesHashW", _particleHash.Read);
        cmd.DispatchCompute(cs, kernel, PGroupThreadsX, 1, 1);
        
        // sort
        Sort(cmd, _particleHash, _particleID, 24);
        
        // set range
        kernel = _kernelSetRange;
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeW", _gridParticleRange);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesHashR", _particleHash.Read);
        cmd.DispatchCompute(cs, kernel, PGroupThreadsX, 1, 1);
        
        // rearrange
        kernel = _kernelRearrange;
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesIDR", _particleID.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesR", _particles.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesW", _particles.Write);
        cmd.DispatchCompute(cs, kernel, PGroupThreadsX, 1, 1);
        _particles.Swap();
        
        // 8*8*8 block range
        kernel = _kernelBlockRange;
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeR", _gridParticleRange);
        cmd.SetComputeBufferParam(cs, kernel, "_BlockParticleRangeW", _blockParticleRange);
        cmd.DispatchCompute(cs, kernel, _blockParticleRange.Size, 1, 1);
    }

    private void ParticleToGrid(CommandBuffer cmd)
    {
        var cs = P2GCs;
        SetParams(cmd, cs);
        
        int kernel = _kernelMyGridType;
        cmd.SetComputeTextureParam(cs, kernel, "_GridTypesW", _gridTypes);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRange", _gridParticleRange);
        cmd.SetComputeTextureParam(cs, kernel, "_GridCoefficientW", _gridCoefficientPymaid[0]);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
        
#if GATHER
        kernel = _kernelP2G;
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRange", _gridParticleRange);
        cmd.SetComputeBufferParam(cs, kernel, "_Particles", _particles.Read);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityOldW", _gridOldVelocity);
        
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
#else
        cmd.SetComputeBufferParam(cs, 4, "_WeightsSumW", _gridWeightsTemp);
        cmd.DispatchCompute(cs, 4, 
            GridSize.x*GridSize.y*GridSize.z/256, 1, 1);
        kernel = _kernelP2G;
        
        cmd.SetComputeBufferParam(cs, kernel, "_BlockParticleRangeR", _blockParticleRange);
        cmd.SetComputeBufferParam(cs, kernel, "_Particles", _particles.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_WeightsSumW", _gridWeightsTemp);
        cmd.DispatchCompute(cs, kernel, 
            GridSize.x/8, GridSize.y/8, GridSize.z/8);
        
        kernel = _kernelNormalize;
        cmd.SetComputeTextureParam(cs, kernel, "_GridTypesR", _gridTypes);
        cmd.SetComputeBufferParam(cs, kernel, "_WeightsSumR", _gridWeightsTemp);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityOldW", _gridOldVelocity);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
#endif
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

    private void Projection(CommandBuffer cmd, int iteration)
    {
        var cs = projectionCs;
        SetParams(cmd, cs);
        cmd.SetComputeFloatParam(cs, "_Damping", damping);

        int kernel = _kernelDivergence;
        
        cmd.SetComputeTextureParam(cs, kernel, "_GridTypesR", _gridTypes);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityR", _gridVelocity);
        cmd.SetComputeTextureParam(cs, kernel, "_DivergenceW", _gridDivergencePymaid[0]);
        cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
#if true
        MGPCG(cmd);
        // MGPCG(cmd);
        // MultiGridVCycle(cmd);
#else
        SetParams(cmd, solverCs);
        for (int i = 0; i < iteration; i++)
        {
            cmd.SetComputeTextureParam(solverCs, 0, "_GridCoefficientR", _gridCoefficientPymaid[0]);
            // cmd.SetComputeTextureParam(solverCs, 0, "_GridTypes", _gridTypes);
            cmd.SetComputeTextureParam(solverCs, 0, "_Divergence", _gridDivergencePymaid[0]);
            cmd.SetComputeTextureParam(solverCs, 0, "_Pressure", _gridPressurePymaid[0]);
            cmd.SetComputeIntParam(solverCs,  "_Color", 0);
            cmd.DispatchCompute(solverCs, 0, GridSize.x/8, GridSize.y/8, GridSize.z/8);
            cmd.SetComputeIntParam(solverCs,  "_Color", 1);
            cmd.DispatchCompute(solverCs, 0, GridSize.x/8, GridSize.y/8, GridSize.z/8);
        }
#endif
        
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
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityR", _gridVelocity);
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityOldR", _gridOldVelocity);
        cmd.SetComputeBufferParam(cs, kernel, "_Particles", _particles.Read);
        cmd.DispatchCompute(cs, kernel, PGroupThreadsX, 1, 1);
        
        kernel = _kernelAdvection;
        cmd.SetComputeTextureParam(cs, kernel, "_VelocityR", _gridVelocity);
        cmd.SetComputeBufferParam(cs, kernel, "_Particles", _particles.Read);
        cmd.DispatchCompute(cs, kernel, PGroupThreadsX, 1, 1);
    }

    private void PrepareForRenderParticles(CommandBuffer cmd)
    {
        var cs = initCs;
        SetParams(cmd, cs);
        int kernel = _kernelRendering;
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesR", _particles.Read);
        cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRender", _particleRendering);
        cmd.DispatchCompute(cs, kernel, PGroupThreadsX, 1, 1);
    }

    private void RenderingMesh(CommandBuffer cmd)
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
        
        // cmd.DrawProceduralIndirect(Matrix4x4.identity, _meshMat, 0, MeshTopology.Triangles, _argsBuffer);
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
        cmd.SetComputeIntParam(cs, "_NumParticles", NumParticles);
        cmd.SetComputeIntParam(cs, "_NumCells", GridSize.x * GridSize.y * GridSize.z);
        cmd.SetComputeFloatParam(cs, "_CellSize", GridSpacing);
        cmd.SetComputeFloatParam(cs, "_InvCellSize", 1f / GridSpacing);
        cmd.SetComputeFloatParam(cs, "_DeltaTime", 1f / (_slowDown ? 600f : 60f));
    }
    
    private static int DivRoundUp(int x, int y)
    {
        return (x + y - 1) / y;
    }

    #region Debug
    
    float2 OctWrap( float2 v )
    {
        return ( 1.0f - math.abs(v.yx) ) * math.select( -1.0f, 1.0f, v >= 0.0f);
    }
     
    float2 EncodeNormal(float3 n)
    {
        n /= ( math.abs( n.x ) + math.abs( n.y ) + math.abs( n.z ) );
        n.xy = n.z >= 0.0f ? n.xy : OctWrap( n.xy );
        n.xy = n.xy * 0.5f + 0.5f;
        return n.xy;
    }
     
    float3 DecodeNormal( float2 f )
    {
        f = f * 2.0f - 1.0f;
     
        // https://twitter.com/Stubbesaurus/status/937994790553227264
        float3 n = math.float3( f.x, f.y, 1.0f - math.abs( f.x ) - math.abs( f.y ) );
        float t = math.saturate( -n.z );
        n.xy += math.select(t, -t, n.xy >= 0.0f);
        return math.normalize( n );
    }

    uint Morton3DGetThirdBits(uint num) {
        uint x = num        & 0x49249249;
        x = (x ^ (x >> 2))  & 0xc30c30c3;
        x = (x ^ (x >> 4))  & 0x0f00f00f;
        x = (x ^ (x >> 8))  & 0xff0000ff;
        x = (x ^ (x >> 16)) & 0x0000ffff;
        return x;
    }

    uint3 MortonD3Decode(uint code)
    {
        return math.uint3(Morton3DGetThirdBits(code), Morton3DGetThirdBits(code >> 1), Morton3DGetThirdBits(code >> 2));
    }

    uint Morton3DSplitBy3Bits(uint num) 
    {
        uint x = num & 1023u;
        x = (x | (x << 16)) & 0xff0000ff;
        x = (x | (x << 8))  & 0x0f00f00f;
        x = (x | (x << 4))  & 0xc30c30c3;
        x = (x | (x << 2))  & 0x49249249;
        return x;
    }

    uint Morton3DEncode(uint x, uint y, uint z)
    {
        return Morton3DSplitBy3Bits(x) | (Morton3DSplitBy3Bits(y) << 1) | (Morton3DSplitBy3Bits(z) << 2);
    }
    uint Morton3DEncode(uint3 v)
    {
        return Morton3DSplitBy3Bits(v.x) | (Morton3DSplitBy3Bits(v.y) << 1) | (Morton3DSplitBy3Bits(v.z) << 2);
    }

    uint Coord2Idx(uint x, uint y, uint z)
    {
        return Morton3DEncode(x, y, z);
        // return z * _GridSize.x * _GridSize.y + y * _GridSize.x + x;
    }
    uint Coord2Idx(uint3 coord)
    {
        // return Morton3DEncode(coord.x, coord.y, coord.z);
        return Coord2Idx(coord.x, coord.y, coord.z);
    }

    uint PackUint3(uint3 v)
    {
        return v.x | (v.y << 10) | (v.z << 20);
    }

    uint3 UnpackUint3(uint v)
    {
        return math.uint3(v & 1023u, (v >> 10) & 1023u, (v >> 20) & 1023u);
    }

    uint2 EncodePosition(float3 pos)
    {
        float3 cellPos = pos / GridSpacing;
        return math.uint2(Morton3DEncode((uint3)math.floor(cellPos)), PackUint3((uint3)math.round(math.frac(cellPos) * 1023)));
    }

    uint3 PositionCoord(uint2 packedPos)
    {
        return MortonD3Decode(packedPos.x);
    }

    float3 DecodePosition(uint2 packedPos)
    {
        float3 coord = MortonD3Decode(packedPos.x);
        float3 localPos = (float3)UnpackUint3(packedPos.y) / 1023.0f;
        return (coord + localPos) * GridSpacing;
    }

    uint PackUNorm2(float2 v)
    {
        uint2 coord = (uint2)math.round(v * 65535) & 65535u;
        return coord.x | (coord.y << 16);
    }

    float2 UnpackUNorm2(uint packed)
    {
        return math.float2((packed & 65535u) / 65535.0f, (packed >> 16) / 65535.0f);
    }

    float2 EncodeVelocity(float3 vel)
    {
        float len = math.length(vel);
        return math.float2(math.asuint(len), len > 1e-8 ? PackUNorm2(EncodeNormal(vel / len)) : 0);
    }

    float3 DecodeVelocity(uint2 packedVel)
    {
        float len = math.asfloat(packedVel.x);
        return len > 1e-8 ? len * DecodeNormal(UnpackUNorm2(packedVel.y)) : 0;
    }
    float3 GetLinearWeight(float3 abs_x)
    {
        return math.saturate(1.0f - abs_x);
    }

    float3 GetQuadraticWeight(float3 abs_x)
    {
        return math.select(0.5f * math.saturate(1.5f - abs_x) * math.saturate(1.5f - abs_x), 0.75f - abs_x * abs_x, abs_x < 0.5f);
    }

// #define USE_LINEAR_KERNEL

    float GetWeight(float3 p_pos, float3 c_pos, float grid_inv_spacing)
    {
        float3 dist = math.abs((p_pos - c_pos) * grid_inv_spacing);

#if USE_LINEAR_KERNEL
    const float3 weight = GetLinearWeight(dist);
#else // defined(USE_QUADRATIC_KERNEL)
        float3 weight = GetQuadraticWeight(dist);
#endif

        return weight.x * weight.y * weight.z;
    }
    const float KernelPoly6 = 315.0f / (64.0f * 3.14159265f);

    float SmoothingKernelPoly6(float r2)
    {
        if (r2 < 1)
        {
            float v = 1 - r2;
            return v * v * v * KernelPoly6;
        }

        return 0;
    }

    #endregion

    private void OnDestroy()
    {
        _particles.Dispose();
        _particleRendering.Dispose();
        _particleID.Dispose();
        _particleHash.Dispose();
        
        _gridParticleRange.Dispose();
        _blockParticleRange.Dispose();
        _gridVelocity.Dispose();
        _gridOldVelocity.Dispose();
        _gridTypes.Dispose();
        _gridLaplace.Dispose();
        _gridP.Dispose();
        
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

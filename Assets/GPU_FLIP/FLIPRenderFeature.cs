using System;
using Abecombe.GPUUtil;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class FLIPRenderFeature : ScriptableRendererFeature
{
    class CustomRenderPass : ScriptableRenderPass
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
        
        private Material _material;
        private Material _meshMat;
        
        private readonly GPUDoubleBuffer<Particle> _particles = new();
        private readonly GPUBuffer<float4> _particleRendering = new();
        
        private readonly GPUDoubleBuffer<uint> _particleID = new();
        private readonly GPUDoubleBuffer<uint> _particleHash = new();

        private readonly GPUBuffer<uint2> _gridParticleRange = new();
        // private readonly GPUBuffer<uint2> _blockParticleRange = new();
        // private readonly GPUBuffer<int> _gridWeightsTemp = new();
        
        private readonly GPUTexture3D _gridType = new();
        private readonly GPUTexture3D _gridVelocity = new();
        private readonly GPUTexture3D _gridOldVelocity = new();
        private readonly GPUTexture3D _gridDivergence = new();
        private readonly GPUDoubleTexture3D _gridPressure = new();
        
        private ComputeBuffer globalHist;
        private ComputeBuffer passHist;
        private ComputeBuffer _argsBuffer;
        private ComputeBuffer _verticesBuffer;
        
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
        // private int _kernelBlockRange;
        
        private int _kernelMyGridType;
        
        private int _kernelP2G;
        // private int _kernelNormalize;
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
    
        
        public CustomRenderPass(FLIPSettings settings)
        {
            initCs = settings.initCs;
            buildLutCs = settings.buildLutCs;
            P2GCs = settings.P2GCs;
            projectionCs = settings.projectionCs;
            G2PCs = settings.G2PCs;
            sortCs = settings.sortCs;
            mcCs = settings.mcCs;
            damping = settings.damping;
            flipness = settings.flipness;
            mouseForce = settings.mouseForce;
            mouseForceRange = settings.mouseForceRange;
            threshold = settings.threshold;
            
            _cam = Camera.main;
            _orbitCamera = _cam.GetComponent<OrbitCamera>();
            
            _particles.Init(NumParticles);
            _particleRendering.Init(NumParticles);
            _particleID.Init(NumParticles);
            _particleHash.Init(NumParticles);
            
            _gridParticleRange.Init(NumGrids);
            // _blockParticleRange.Init(NumGrids / 512);
            _gridType.Init(GridSize, RenderTextureFormat.RInt);
            _gridVelocity.Init(GridSize, RenderTextureFormat.ARGBHalf);
            _gridOldVelocity.Init(GridSize, RenderTextureFormat.ARGBHalf);
            _gridDivergence.Init(GridSize, RenderTextureFormat.RHalf);
            _gridPressure.Init(GridSize, RenderTextureFormat.RHalf);
            // _gridWeights.Init(GridSize, RenderTextureFormat.ARGBHalf);
            // _gridWeightsTemp.Init(NumGrids * 7);
            
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
            // _kernelBlockRange = buildLutCs.FindKernel("BlockRange");
            
            _kernelP2G = P2GCs.FindKernel("P2G");
            _kernelMyGridType = P2GCs.FindKernel("SetGridType");
            // _kernelNormalize = P2GCs.FindKernel("Normalize");
            
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

            _vertBufferSize = Mathf.RoundToInt(Mathf.Pow(GridSize.x, 2.5f)) * 3;
            _verticesBuffer = new ComputeBuffer(_vertBufferSize, sizeof(uint) * 4);
            _meshMat = new Material(Shader.Find("Custom/DrawStructuredBuffer"));
            _meshMat.SetBuffer("_Buffer", _verticesBuffer);
            _meshMat.SetPass(0);

            _argsBuffer = new ComputeBuffer(5, sizeof(uint), ComputeBufferType.IndirectArguments);
            _argsBuffer.SetData(new[] {0, 1, 0, 0, 0});

            float3 size = (float3)GridSize * GridSpacing;
            _bounds = new Bounds(size * 0.5f, size);
            
            Debug.Log($"Initializing GPU flip with particles: {NumParticles}, GridSize: {GridSize}, numCells: {NumGrids}, bufferSize: {_vertBufferSize}");
            Debug.Log($"Initializing GPU flip with particlesT: {PGroupThreadsX}, GridSizeT: {_gGroupThreads}");
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
        
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
        }
        
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
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
        
            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
        
            cmd.BeginSample("Rendering");
            // PrepareForRendering(cmd);
            RenderingMesh(cmd);
            cmd.EndSample("Rendering");
        
            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);
        }

        public override void OnCameraCleanup(CommandBuffer cmd)
        {
        }
        
        private static int DivRoundUp(int x, int y)
        {
            return (x + y - 1) / y;
        }
    
        private void BuildLut(CommandBuffer cmd)
        {
            var cs = buildLutCs;
            SetParams(cmd, cs);
            
            // clear grid data
            int kernel = _kernelClearGrid;
            cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeW", _gridParticleRange);
            cmd.SetComputeTextureParam(cs, kernel, "_PressureW", _gridPressure.Write);
            // cmd.SetComputeBufferParam(cs, kernel, "_WeightsSumW", _gridWeightsTemp);
            cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
            _gridPressure.Swap();
            
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
            // kernel = _kernelBlockRange;
            // cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRangeR", _gridParticleRange);
            // cmd.SetComputeBufferParam(cs, kernel, "_BlockParticleRangeW", _blockParticleRange);
            // cmd.DispatchCompute(cs, kernel, _blockParticleRange.Size, 1, 1);
        }

        private void ParticleToGrid(CommandBuffer cmd)
        {
            var cs = P2GCs;
            SetParams(cmd, cs);
            
            int kernel = _kernelMyGridType;
            cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRange", _gridParticleRange);
            cmd.SetComputeTextureParam(cs, kernel, "_GridTypesW", _gridType.Data);
            cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
            
            kernel = _kernelP2G;
            cmd.SetComputeBufferParam(cs, kernel, "_ParticlesRange", _gridParticleRange);
            // cmd.SetComputeBufferParam(cs, kernel, "_BlockParticleRangeR", _blockParticleRange);
            cmd.SetComputeBufferParam(cs, kernel, "_Particles", _particles.Read);
            cmd.SetComputeTextureParam(cs, kernel, "_VelocityOldW", _gridOldVelocity);
            
            cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
            
            // cmd.SetComputeBufferParam(cs, kernel, "_WeightsSumW", _gridWeightsTemp);
            // cmd.DispatchCompute(cs, kernel, 
            //     _gGroupThreads.x/2, _gGroupThreads.y/2, _gGroupThreads.z/2);
            //
            // kernel = _kernelNormalize;
            // cmd.SetComputeTextureParam(cs, kernel, "_GridTypesR", _gridType.Data);
            // cmd.SetComputeBufferParam(cs, kernel, "_WeightsSumR", _gridWeightsTemp);
            // cmd.SetComputeTextureParam(cs, kernel, "_VelocityOldW", _gridOldVelocity);
            // cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);

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
            cmd.SetComputeTextureParam(cs, kernel, "_GridTypesR", _gridType.Data);
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
            
            cmd.SetComputeTextureParam(cs, kernel, "_GridTypes", _gridType.Data);
            cmd.SetComputeTextureParam(cs, kernel, "_VelocityR", _gridVelocity);
            cmd.SetComputeTextureParam(cs, kernel, "_DivergenceW", _gridDivergence);
            cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);

            kernel = _kernelProject;
            for (int i = 0; i < iteration; i++)
            {
                cmd.SetComputeTextureParam(cs, kernel, "_GridTypes", _gridType.Data);
                cmd.SetComputeTextureParam(cs, kernel, "_DivergenceR", _gridDivergence);
                cmd.SetComputeTextureParam(cs, kernel, "_PressureR", _gridPressure.Read);
                cmd.SetComputeTextureParam(cs, kernel, "_PressureW", _gridPressure.Write);
                cmd.DispatchCompute(cs, kernel, _gGroupThreads.x, _gGroupThreads.y, _gGroupThreads.z);
                _gridPressure.Swap();
            }
            
            kernel = _kernelUpdateVelocity;
            cmd.SetComputeTextureParam(cs, kernel, "_GridTypes", _gridType.Data);
            cmd.SetComputeTextureParam(cs, kernel, "_PressureR", _gridPressure.Read);
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

        private void PrepareForRendering(CommandBuffer cmd)
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
            var cs = mcCs;
            
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
            
            cmd.DispatchCompute(cs, 1, GridSize.x / 8, GridSize.y / 8, GridSize.z / 8);
            
            cmd.DrawProceduralIndirect(Matrix4x4.identity, _meshMat, 0, MeshTopology.Triangles, _argsBuffer);
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

        private void SetParams(CommandBuffer cmd, ComputeShader cs)
        {
            cmd.SetComputeVectorParam(cs, "_GridMin", new Vector3(0, 0, 0));
            cmd.SetComputeVectorParam(cs, "_GridSize", new Vector3(GridSize.x, GridSize.y, GridSize.z));
            cmd.SetComputeIntParam(cs, "_NumParticles", NumParticles);
            cmd.SetComputeFloatParam(cs, "_CellSize", GridSpacing);
            cmd.SetComputeFloatParam(cs, "_InvCellSize", 1f / GridSpacing);
            cmd.SetComputeFloatParam(cs, "_DeltaTime", 1f / 50f);
        }

        public void Dispose()
        {
            _particles.Dispose();
            _particleRendering.Dispose();
            _particleID.Dispose();
            _particleHash.Dispose();
        
            _gridParticleRange.Dispose();
            // _blockParticleRange.Dispose();
            _gridVelocity.Dispose();
            _gridOldVelocity.Dispose();
            _gridDivergence.Dispose();
            _gridPressure.Dispose();
        
            // _gridWeightsTemp.Dispose();
        
            globalHist.Dispose();
            passHist.Dispose();
        
            _particleRenderingBufferWithArgs.Dispose();
        
            _argsBuffer.Dispose();
            _verticesBuffer.Dispose();
            
            _gridType.Dispose();
        
            DestroyImmediate(_material);
            DestroyImmediate(_meshMat);
        }
    }
    
    [System.Serializable]
    public class FLIPSettings
    {
        public ComputeShader initCs;
        public ComputeShader buildLutCs;
        public ComputeShader P2GCs;
        public ComputeShader projectionCs;
        public ComputeShader G2PCs;
        public ComputeShader sortCs;
    
        public ComputeShader mcCs;

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
    }

    public FLIPSettings settings;
    CustomRenderPass m_ScriptablePass;

    /// <inheritdoc/>
    public override void Create()
    {
        if (!Application.isPlaying || !settings.initCs ||
            !settings.buildLutCs ||
            !settings.P2GCs ||
            !settings.projectionCs ||
            !settings.G2PCs ||
            !settings.sortCs ||
            !settings.mcCs) return;
        m_ScriptablePass = new CustomRenderPass(settings);

        // Configures where the render pass should be injected.
        m_ScriptablePass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;
    }

    // Here you can inject one or multiple render passes in the renderer.
    // This method is called when setting up the renderer once per-camera.
    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        var camType = renderingData.cameraData.camera.cameraType;
        if (m_ScriptablePass == null || !Application.isPlaying || (camType != CameraType.Game )) return;
        renderer.EnqueuePass(m_ScriptablePass);
    }
    
    protected override void Dispose(bool disposing)
    {
        m_ScriptablePass?.Dispose();
    }

    public void OnDestroy()
    {
        OnDisable();
    }
    
    public void OnDisable()
    {
        m_ScriptablePass?.Dispose();
    }
}



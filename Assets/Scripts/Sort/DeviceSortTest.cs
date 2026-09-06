using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

public class DeviceSortTest : MonoBehaviour
{
    private int m_kernelInit = -1;
    private int m_kernelUpsweep = -1;
    private int m_kernelScan = -1;
    private int m_kernelDownsweep = -1;
    private const int k_radix = 256;
    private const int k_radixPasses = 4;
    private const int k_partitionSize = 3840;

    // private const int k_minSize = 1;
    // private const int k_maxSize = 65535 * k_partitionSize;

    [SerializeField]
    ComputeShader m_cs;
    [SerializeField]
    private ComputeShader m_util;
    
    private ComputeBuffer toTest;
    private ComputeBuffer toTestPayload;
    
    private ComputeBuffer alt;
    private ComputeBuffer altPayload;
    
    private ComputeBuffer globalHist;
    private ComputeBuffer passHist;
    
    private ComputeBuffer index;
    private ComputeBuffer errCount;

    private CommandBuffer m_cmd;

    private const int k_validatePartSize = 4096;

    private bool m_isValid;
    private int m_kernelInitRandom = -1;
    private int m_kernelClearErrors = -1;
    private int m_kernelValidate = -1;


    // private int _counter = 0;
    public int errorCount;
    private const int TestSize = 26042 * k_partitionSize;
    public int preSortTime;
    public int sortTime;
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Initialize(TestSize);
        
        alt = new ComputeBuffer(TestSize, 4);
        altPayload = new ComputeBuffer(TestSize, 4);
        globalHist = new ComputeBuffer(k_radix * k_radixPasses, 4);
        passHist = new ComputeBuffer(k_radix * DivRoundUp(TestSize, k_partitionSize) * k_radixPasses, 4);
    }
    
    private static int DivRoundUp(int x, int y)
    {
        return (x + y - 1) / y;
    }

    private void OnDestroy()
    {
        var data = new uint [TestSize];
        toTest?.GetData(data);
        var msg = "result:  ";
        uint last = 0;
        for (int i = 0; i < TestSize; i += k_partitionSize * 2 / 3)
        {
            if (data[i] < last)
                Debug.LogError($"Data out of order at index {i}: {data[i]} < {last}");
            
            msg += data[i] + ", ";
            last = data[i];
        }
        
        Debug.Log(msg);
        
        toTest?.Dispose();
        toTestPayload?.Dispose();
        alt?.Dispose();
        altPayload?.Dispose();
        globalHist?.Dispose();
        passHist?.Dispose();
        errCount?.Dispose();
        m_cmd?.Release();
        index?.Dispose();
    }

    private void Initialize(int bufferSize)
    {
        m_kernelInitRandom = m_util.FindKernel("InitSortInput");
        m_kernelClearErrors = m_util.FindKernel("ClearErrorCount");
        m_kernelValidate = m_util.FindKernel("Validate");
        
        m_isValid = m_kernelInitRandom >= 0 && m_kernelClearErrors >= 0 && m_kernelValidate >= 0;

        if (m_isValid)
        {
            if (!m_util.IsSupported(m_kernelInitRandom) ||
                !m_util.IsSupported(m_kernelClearErrors) ||
                !m_util.IsSupported(m_kernelValidate))
                m_isValid = false;
        }

        Assert.IsTrue(m_isValid);
        m_cmd = new CommandBuffer();
        m_cmd.name = "DeviceSort";
        toTest = new ComputeBuffer(bufferSize, sizeof(uint));
        toTestPayload = new ComputeBuffer(bufferSize, sizeof(uint));
        errCount = new ComputeBuffer(1, sizeof(uint));
        
        m_kernelInit = m_cs.FindKernel("InitDeviceRadixSort");
        m_kernelUpsweep = m_cs.FindKernel("UpSweep");
        m_kernelScan = m_cs.FindKernel("Scan");
        m_kernelDownsweep = m_cs.FindKernel("DownSweep");
        
        var isValid = m_kernelInit >= 0 && m_kernelUpsweep >= 0 &&
                      m_kernelScan >= 0 && m_kernelDownsweep >= 0;

        if (isValid)
        {
            if (!m_cs.IsSupported(m_kernelInit) ||
                !m_cs.IsSupported(m_kernelUpsweep) ||
                !m_cs.IsSupported(m_kernelScan) ||
                !m_cs.IsSupported(m_kernelDownsweep))
                isValid = false;
        }

        Assert.IsTrue(isValid);
        Debug.Log($"Test initialization success. size: {bufferSize}");
        
    }

    // Update is called once per frame
    void Update()
    {
        // if (_counter++ > 0) return;
        errorCount = 0;
        
        var sw = System.Diagnostics.Stopwatch.StartNew();
        PreSort(TestSize, 17, false);
        sw.Stop();
        preSortTime = (int)sw.ElapsedTicks;
        
        sw.Restart();
        m_cmd.Clear();
        
        Sort(m_cmd,
            TestSize,
            toTest,
            toTestPayload,
            alt,
            altPayload,
            globalHist,
            passHist);
        Graphics.ExecuteCommandBuffer(m_cmd);
        // PostSort(TestSize,  false);
        sw.Stop();
        sortTime = (int)sw.ElapsedTicks;
    }

    private void Sort(CommandBuffer cmd,
        int sortSize,
        ComputeBuffer toSort,
        ComputeBuffer toSortPayload,
        ComputeBuffer tempKeyBuffer,
        ComputeBuffer tempPayloadBuffer,
        ComputeBuffer tempGlobalHistBuffer,
        ComputeBuffer tempPassHistBuffer)
    {
        int threadBlocks = DivRoundUp(sortSize, k_partitionSize);
        SetStaticRootParameters(
            sortSize,
            threadBlocks,
            cmd,
            tempPassHistBuffer,
            tempGlobalHistBuffer);
        
        Dispatch(threadBlocks, cmd, toSort, toSortPayload, tempKeyBuffer, tempPayloadBuffer);
    }
    
    private void Dispatch(
        int numThreadBlocks,
        CommandBuffer _cmd,
        ComputeBuffer _toSort,
        ComputeBuffer _toSortPayload,
        ComputeBuffer _alt,
        ComputeBuffer _altPayload)
    {
        _cmd.DispatchCompute(m_cs, m_kernelInit, 1, 1, 1);

        for (int radixShift = 0; radixShift < 32; radixShift += 8)
        {
            _cmd.SetComputeIntParam(m_cs, "e_radixShift", radixShift);

            _cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_sort", _toSort);
            _cmd.DispatchCompute(m_cs, m_kernelUpsweep, numThreadBlocks, 1, 1);

            _cmd.DispatchCompute(m_cs, m_kernelScan, k_radix, 1, 1);

            _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_sort", _toSort);
            _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_sortPayload", _toSortPayload);
            _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_alt", _alt);
            _cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_altPayload", _altPayload);
            _cmd.DispatchCompute(m_cs, m_kernelDownsweep, numThreadBlocks, 1, 1);

            (_toSort, _alt) = (_alt, _toSort);
            (_toSortPayload, _altPayload) = (_altPayload, _toSortPayload);
        }
    }
    private void SetStaticRootParameters(
        int numKeys,
        int numThreadBlocks,
        CommandBuffer cmd,
        ComputeBuffer passHistBuffer,
        ComputeBuffer globalHistBuffer)
    {
        cmd.SetComputeIntParam(m_cs, "e_numKeys", numKeys);
        cmd.SetComputeIntParam(m_cs, "e_threadBlocks", numThreadBlocks);

        cmd.SetComputeBufferParam(m_cs, m_kernelInit, "b_globalHist", globalHistBuffer);

        cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_passHist", passHistBuffer);
        cmd.SetComputeBufferParam(m_cs, m_kernelUpsweep, "b_globalHist", globalHistBuffer);

        cmd.SetComputeBufferParam(m_cs, m_kernelScan, "b_passHist", passHistBuffer);

        cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_passHist", passHistBuffer);
        cmd.SetComputeBufferParam(m_cs, m_kernelDownsweep, "b_globalHist", globalHistBuffer);
    }
    
    private void PreSort(int testSize, int seed, bool keysOnly)
    {
        m_util.SetInt("e_numKeys", testSize);
        m_util.SetInt("e_seed", seed);

        m_util.SetBuffer(m_kernelInitRandom, "b_sort", toTest);

        m_util.SetBuffer(m_kernelInitRandom, "b_sortPayload", toTestPayload);
        m_util.Dispatch(m_kernelInitRandom, 256, 1, 1);
    }
    
    private bool PostSort(int testSize, bool shouldPrint)
    {
        m_util.SetBuffer(m_kernelClearErrors, "b_errorCount", errCount);
        m_util.Dispatch(m_kernelClearErrors, 1, 1, 1);

        m_util.SetInt("e_threadBlocks", DivRoundUp(testSize, k_validatePartSize));
        m_util.SetBuffer(m_kernelValidate, "b_sort", toTest);
        m_util.SetBuffer(m_kernelValidate, "b_sortPayload", toTestPayload);
        m_util.SetBuffer(m_kernelValidate, "b_errorCount", errCount);
        m_util.Dispatch(m_kernelValidate, DivRoundUp(testSize, k_validatePartSize), 1, 1);
        uint[] errors = new uint[1];
        errCount.GetData(errors);
        
        errorCount = (int)errors[0];
        
        if (errors[0] == 0)
            return true;

        if (shouldPrint)
            Debug.LogError("Test Failed: " + errors[0] + " errors.");
        return false;
    }
}


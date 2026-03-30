using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;

namespace PF_FLIP
{
    public struct Particle
    {
        public float2 Pos;
        public int Level;
        public int Counter;
    }
    public class Adpative_FLIP : MonoBehaviour
    {
        private NativeArray<Particle> _particlePool;
        private NativeArray<float2> _particleVelPool;
        private NativeArray<Particle> _particlePoolCopy;
        private NativeArray<float2> _particleVelPoolCopy;
        private NativeArray<int2> _particleHash;
        private NativeReference<int> _particleCount;
        
        private MultiResSparseBlockGrids _sbg;
        private int[] _levels;

        public const int ParticlePoolSize = 24000;

        private float _rs;
        private int _blockCount;
        public Mesh mesh;
        public Material mat;

        private ComputeBuffer _posBuffer;
        private Bounds _bounds;
        private float _oldFPS;

        private void OnGUI()
        {
            GUI.Label(new Rect(0, 0, 100, 32), $"Residual: {_rs}", new GUIStyle()
            {
                alignment = TextAnchor.UpperLeft,
                fontSize = 28,
                normal =
                {
                    textColor = math.isfinite(_rs) ? (_rs > 1 ? Color.yellow : Color.green) :  Color.red,
                }
            });
            GUI.Label(new Rect(0, 32, 100, 32), $"grid pool: {_blockCount}/{MSBGConstants.PoolSize}", new GUIStyle()
            {
                alignment = TextAnchor.UpperLeft,
                fontSize = 28,
                normal =
                {
                    textColor = (_blockCount < (MSBGConstants.PoolSize * 0.9f))
                        ? (_blockCount > (MSBGConstants.PoolSize * 0.6f) ? Color.yellow : Color.green)
                        : Color.red
                }
            });
            GUI.Label(new Rect(0, 64, 100, 32), $"particle pool: {_particleCount.Value}/{ParticlePoolSize}", new GUIStyle()
            {
                alignment = TextAnchor.UpperLeft,
                fontSize = 28,
                normal =
                {
                    textColor = (_particleCount.Value < (ParticlePoolSize * 0.9f))
                        ? (_particleCount.Value > (ParticlePoolSize * 0.8f) ? Color.yellow : Color.green)
                        : Color.red
                }
            });
            float fps = 1f / Time.deltaTime;
            if (_oldFPS == 0)
                _oldFPS = fps;
            else
            {
                _oldFPS = Mathf.Lerp(_oldFPS, fps, 0.01f);
                fps = _oldFPS;
            }
            GUI.Label(new Rect(0, 96, 100, 32), $"FPS: {fps:F1}", new GUIStyle()
            {
                alignment = TextAnchor.UpperLeft,
                fontSize = 28,
                normal =
                {
                    textColor = fps < 60 ? Color.yellow : Color.green,
                }
            });
        }

        void Start()
        {
            _particlePool = new NativeArray<Particle>(ParticlePoolSize, Allocator.Persistent);
            _particleVelPool = new NativeArray<float2>(ParticlePoolSize, Allocator.Persistent);
            _particlePoolCopy = new NativeArray<Particle>(ParticlePoolSize, Allocator.Persistent);
            _particleVelPoolCopy = new NativeArray<float2>(ParticlePoolSize, Allocator.Persistent);
            _particleHash = new NativeArray<int2>(ParticlePoolSize, Allocator.Persistent);
            
            _particleCount = new NativeReference<int>(Allocator.Persistent);
            _particleCount.Value = 100 * 100;
            
            _posBuffer = new ComputeBuffer(ParticlePoolSize, sizeof(float) * 4);
            mat.SetBuffer("_ParticleBuffer", _posBuffer);
            _bounds = new Bounds(Vector3.zero, Vector3.one*128);

            for (int y = 0; y < 100; y++)
            for (int x = 0; x < 100; x++)
            {
                _particlePool[x + y * 100] = new Particle
                {
                    Pos = MSBGConstants.BaseCellSize * (new float2(x, y) * 0.7f + new float2(20, 20)),
                    Level = 1,
                    Counter = 0,
                };
            }
            
            _sbg = new MultiResSparseBlockGrids();
            _levels = new int[MSBGConstants.GridCount];
        }

        void Update()
        {
            Profiler.BeginSample("Clear Grid");
            _sbg.ClearGrid();
            Profiler.EndSample();
            
            Profiler.BeginSample("Build Lut");
            _sbg.BuildSpatialLookup(_particleHash, _particlePool, _particleVelPool,
                _particlePoolCopy, _particleVelPoolCopy, _particleCount.Value);
            Profiler.EndSample();
            
            (_particlePool, _particlePoolCopy) = (_particlePoolCopy, _particlePool);
            (_particleVelPool, _particleVelPoolCopy) = (_particleVelPoolCopy, _particleVelPool);

            Profiler.BeginSample("AllocateBlocks");
            bool success = _sbg.AllocateBlocks(out _blockCount);
            Profiler.EndSample();
            if (success)
            {
                Profiler.BeginSample("P2G");
                _sbg.ParticleToGrid(_particlePool, _particleVelPool);
                Profiler.EndSample();

                Profiler.BeginSample("Solve");
                _sbg.SolveMultiGridPressure(new float2(0, -8f), out _rs);
                Profiler.EndSample();
                
                Profiler.BeginSample("Resample");
                if (_sbg.ResampleParticles(_particlePool, _particleVelPool, _particlePoolCopy,
                        _particleVelPoolCopy, _particleCount))
                {
                    (_particlePool, _particlePoolCopy) = (_particlePoolCopy, _particlePool);
                    (_particleVelPool, _particleVelPoolCopy) = (_particleVelPoolCopy, _particleVelPool);
                }
                Profiler.EndSample();

                Profiler.BeginSample("G2P");
                _sbg.GridToParticle(_particlePool, _particleVelPool, _particleCount.Value, 0.99f);
                Profiler.EndSample();
            }
            else
            {
                _blockCount = -1;
                Debug.LogError("Failed to solve");
            }

            _sbg.GetLevels(_levels);
            
            _posBuffer.SetData(_particlePool);
            Graphics.DrawMeshInstancedProcedural(mesh, 0, mat, _bounds, _particleCount.Value);

        }
        
        private readonly Color[] _colors =
        {
            new Color(1, 0, 0, 0.3f), new Color(1, 0.5f, 0, 0.3f), new Color(0.5f, 0.5f, 0.5f, 0.3f), 
        };

        private void OnDrawGizmos()
        {
            if (_levels == null || _levels.Length != MSBGConstants.GridCount) return;
            const int w = MSBGConstants.GridWidth;
            float blockSize = 0.1f * MSBGConstants.BaseBlockWidth;
            Vector3 size = new Vector3(blockSize, blockSize, 0);
            for (int y = 0; y < w; y++)
            for (int x = 0; x < w; x++)
            {
                int level = _levels[x + y * w];
                if (level < 0) continue;
                Gizmos.color = _colors[level % _colors.Length];
                // Gizmos.DrawCube(new Vector3((x + 0.5f) * blockSize, (y + 0.5f) * blockSize, 0), size);
                Gizmos.DrawWireCube(new Vector3((x + 0.5f) * blockSize, (y + blockSize) * blockSize, 0), size);
            }
            Gizmos.color = Color.white;
            Gizmos.DrawWireCube(new Vector3(0.5f * blockSize * w, 0.5f * blockSize * w, 0),
                new Vector3(blockSize * w, blockSize * w, 0));
            // _sbg.DrawGridType();
        }

        private void OnDestroy()
        {
            _particlePool.Dispose();
            _particlePoolCopy.Dispose();
            _particleVelPool.Dispose();
            _particleVelPoolCopy.Dispose();
            _particleHash.Dispose();

            _particleCount.Dispose();
            
            _sbg.Dispose();
            _posBuffer.Dispose();
        }
    }
}

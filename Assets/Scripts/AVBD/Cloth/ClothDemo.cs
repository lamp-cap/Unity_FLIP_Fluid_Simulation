using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;

namespace AVBD.Cloth
{
    /// <summary>
    /// Demo:两块布料落在一个横放的胶囊体上。
    ///
    /// - 程序化生成两块平面网格(可调分辨率/尺寸),分别放在不同高度。
    /// - 用解析胶囊体(横放,沿 X 轴)作为碰撞体。
    /// - 每帧驱动 ClothSolver,把求解后的顶点回写到 Unity Mesh 并渲染。
    ///
    /// 求解管线对应 VBDClothPhysics.cpp 的 runStep_CDEveryIter:
    ///   惯性初值 -> (按需)碰撞检测+安全距离 -> 图着色 GS 迭代 -> 速度更新。
    /// 布料-布料碰撞走 LBVH(三角形树 + 边树);布料-胶囊走解析碰撞体。
    /// </summary>
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public class ClothDemo : MonoBehaviour
    {
        [Header("布料网格(程序化平面)")]
        public int resX = 24;
        public int resZ = 24;
        public float sizeX = 1.0f;
        public float sizeZ = 1.0f;

        [Header("两块布料的初始高度/水平偏移")]
        public float cloth0Height = 1.2f;
        public float cloth1Height = 1.6f;
        public float cloth1OffsetZ = 0.15f;

        [Header("横放胶囊体(沿 X 轴)")]
        public float capsuleHalfLength = 0.6f;   // 沿 X 的半长
        public float capsuleRadius = 0.25f;
        public float3 capsuleCenter = new float3(0, 0.5f, 0);

        [Header("求解参数")]
        public int iterations = 20;
        public int substeps = 1;
        public float density = 1.0f;
        public float miu = 1e4f;
        public float lambda = 1e4f;
        public float bendingStiffness = 1e-3f;
        public float contactStiffness = 1e5f;
        public float thickness = 0.01f;
        public float contactRadius = 0.02f;
        public float queryRadius = 0.06f;
        public bool selfCollision = true;
        public bool useFixedTimestep = true;
        public bool useBurst = true;

        // ---- 运行时 ----
        private ClothTopology _topo;
        private ClothState _state;
        private ClothCollision _collision;
        private ColliderSet _colliders;
        private ClothSolver _solver;
        private ClothSolverParameters _params;

        private Mesh _mesh;
        private Vector3[] _renderVerts;
        private int _cloth0VertCount;

        void Start()
        {
            BuildMeshesAndSolver();
        }

        void BuildMeshesAndSolver()
        {
            // ---- 生成两块平面网格(世界空间顶点) ----
            GeneratePlane(resX, resZ, sizeX, sizeZ, cloth0Height, 0f,
                out float3[] v0, out int[] t0);
            GeneratePlane(resX, resZ, sizeX, sizeZ, cloth1Height, cloth1OffsetZ,
                out float3[] v1, out int[] t1);

            _cloth0VertCount = v0.Length;

            var meshVerts = new List<float3[]> { v0, v1 };
            var meshTris = new List<int[]> { t0, t1 };

            // ---- 拓扑 ----
            _topo = new ClothTopology();
            // 不固定任何点(自由下落);如需悬挂可在此返回 true
            _topo.Build(meshVerts, meshTris, density, fixedPredicate: null);

            // ---- 状态 ----
            _state = new ClothState();
            _state.Allocate(_topo.NumVertices, _topo.RestPositions);

            // ---- 碰撞(布料自/间碰撞 LBVH) ----
            if (selfCollision)
            {
                _collision = new ClothCollision();
                _collision.Allocate(_topo.NumVertices, _topo.NumFaces, _topo.NumEdges);
            }

            // ---- 解析碰撞体:横放胶囊(沿 X) ----
            _colliders = new ColliderSet();
            float3 a = capsuleCenter + new float3(-capsuleHalfLength, 0, 0);
            float3 b = capsuleCenter + new float3(+capsuleHalfLength, 0, 0);
            _colliders.Set(new[]
            {
                AnalyticCollider.MakeCapsule(a, b, capsuleRadius, friction: 0.3f),
                // 地面平面,接住滑落的布料
                AnalyticCollider.MakePlane(new float3(0, -0.5f, 0), new float3(0, 1, 0), friction: 0.4f),
            });

            // ---- 参数 ----
            _params = new ClothSolverParameters
            {
                dt = 1f / 120f,
                numSubsteps = substeps,
                iterations = iterations,
                miu = miu,
                lambda = lambda,
                bendingStiffness = bendingStiffness,
                contactStiffness = contactStiffness,
                thickness = thickness,
                contactRadius = contactRadius,
                queryRadius = queryRadius,
                handleCollision = selfCollision,
                useBurst = useBurst,
            };

            // ---- 求解器 ----
            _solver = new ClothSolver();
            _solver.Initialize(_topo, _state, _params, _colliders, _collision);

            // ---- 渲染网格(把两块布料合并到一个 Mesh 显示) ----
            BuildRenderMesh(v0, t0, v1, t1);
        }

        void BuildRenderMesh(float3[] v0, int[] t0, float3[] v1, int[] t1)
        {
            _mesh = new Mesh { name = "ClothDemoMesh" };
            _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

            int nv = v0.Length + v1.Length;
            _renderVerts = new Vector3[nv];
            for (int i = 0; i < v0.Length; i++) _renderVerts[i] = (Vector3)v0[i];
            for (int i = 0; i < v1.Length; i++) _renderVerts[v0.Length + i] = (Vector3)v1[i];

            var tris = new int[t0.Length + t1.Length];
            System.Array.Copy(t0, 0, tris, 0, t0.Length);
            for (int i = 0; i < t1.Length; i++) tris[t0.Length + i] = t1[i] + v0.Length;

            _mesh.vertices = _renderVerts;
            _mesh.triangles = tris;
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();

            var mf = GetComponent<MeshFilter>();
            mf.mesh = _mesh;

            var mr = GetComponent<MeshRenderer>();
            if (mr.sharedMaterial == null)
            {
                var sh = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                var mat = new Material(sh);
                mat.SetFloat("_Cull", 0); // 双面(若 shader 支持)
                mr.sharedMaterial = mat;
            }
            // 顶点已是世界坐标,渲染对象置于原点
            transform.position = Vector3.zero;
            transform.rotation = Quaternion.identity;
            transform.localScale = Vector3.one;
        }

        void Update()
        {
            if (_solver == null) return;

            _params.iterations = iterations;
            _params.numSubsteps = substeps;
            _params.useBurst = useBurst;
            _params.dt = 1f / 500f;

            _solver.Step();

            // 回写到渲染网格
            var pos = _state.Positions;
            for (int i = 0; i < pos.Length; i++) _renderVerts[i] = (Vector3)pos[i];
            _mesh.vertices = _renderVerts;
            _mesh.RecalculateNormals();
            _mesh.RecalculateBounds();
        }

        // 程序化生成 XZ 平面三角网格(世界空间,居中于 x=0,z=offsetZ,高度 y=height)
        static void GeneratePlane(int resX, int resZ, float sizeX, float sizeZ,
            float height, float offsetZ, out float3[] verts, out int[] tris)
        {
            int nx = resX + 1, nz = resZ + 1;
            verts = new float3[nx * nz];
            for (int z = 0; z < nz; z++)
            {
                for (int x = 0; x < nx; x++)
                {
                    float fx = (x / (float)resX - 0.5f) * sizeX;
                    float fz = (z / (float)resZ - 0.5f) * sizeZ + offsetZ;
                    verts[z * nx + x] = new float3(fx, height, fz);
                }
            }

            tris = new int[resX * resZ * 6];
            int ti = 0;
            for (int z = 0; z < resZ; z++)
            {
                for (int x = 0; x < resX; x++)
                {
                    int i0 = z * nx + x;
                    int i1 = z * nx + x + 1;
                    int i2 = (z + 1) * nx + x;
                    int i3 = (z + 1) * nx + x + 1;
                    // 两个三角形(逆时针朝上)
                    tris[ti++] = i0; tris[ti++] = i2; tris[ti++] = i1;
                    tris[ti++] = i1; tris[ti++] = i2; tris[ti++] = i3;
                }
            }
        }

        void OnDestroy()
        {
            _solver?.Dispose();
            _collision?.Dispose();
            _colliders?.Dispose();
            _state?.Dispose();
            _topo?.Dispose();
        }

        // 可视化胶囊体与地面
        void OnDrawGizmos()
        {
            Gizmos.color = new Color(0.3f, 0.7f, 1f, 0.6f);
            float3 a = capsuleCenter + new float3(-capsuleHalfLength, 0, 0);
            float3 b = capsuleCenter + new float3(+capsuleHalfLength, 0, 0);
            Gizmos.DrawWireSphere((Vector3)a, capsuleRadius);
            Gizmos.DrawWireSphere((Vector3)b, capsuleRadius);
            Gizmos.DrawLine((Vector3)(a + new float3(0, capsuleRadius, 0)), (Vector3)(b + new float3(0, capsuleRadius, 0)));
            Gizmos.DrawLine((Vector3)(a - new float3(0, capsuleRadius, 0)), (Vector3)(b - new float3(0, capsuleRadius, 0)));
            Gizmos.DrawLine((Vector3)(a + new float3(0, 0, capsuleRadius)), (Vector3)(b + new float3(0, 0, capsuleRadius)));
            Gizmos.DrawLine((Vector3)(a - new float3(0, 0, capsuleRadius)), (Vector3)(b - new float3(0, 0, capsuleRadius)));
        }
    }
}

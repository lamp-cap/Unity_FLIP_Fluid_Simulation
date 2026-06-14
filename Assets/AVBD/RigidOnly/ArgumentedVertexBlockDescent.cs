using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Mathematics.math;
using Random = UnityEngine.Random;

namespace AVBD.RigidOnly
{
    public class ArgumentedVertexBlockDescent : MonoBehaviour
    {
        public SceneInitType type = SceneInitType.Ground;
        public bool drawContact;
        private Camera _camera;
        [Range(0, 10)] public float v0 = 5;
        public Material mat;
        public Mesh cube;
        private ComputeBuffer _buffer;

        private Stack<(float3, float3)> _toAdd;

        private Solver _solver;
        // Start is called before the first frame update
        void Start()
        {
            _solver = new Solver();
            func[(int)type].Invoke(_solver);
            _camera = Camera.main;
            _buffer = new ComputeBuffer(8192, sizeof(float) * 12);
            _style = new GUIStyle()
            {
                fontSize = 32,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleLeft,
            };
            _toAdd = new Stack<(float3, float3)>();
        }

        // Update is called once per frame
        void Update()
        {
            if (Input.GetKeyDown(KeyCode.Space))
            {
                Vector3 pos = _camera.ViewportToWorldPoint(new Vector3(0.5f, 0.5f, 3));
                var forward = _camera.transform.forward;
                // _toAdd.Push((pos, forward));
                _solver.CreateRigid(float3(1, 1, 1), 1, 1f, float3(pos).xzy, float3(forward).xzy * v0);
            }
            _solver.step();
        }

        // private void FixedUpdate()
        // {
        //     while (_toAdd.Count > 0)
        //     {
        //         var (pos, forward) = _toAdd.Pop();
        //         _solver.CreateRigid(float3(1, 1, 1), 1, 1f, float3(pos).xzy, float3(forward).xzy * v0);
        //     }
        //     _solver.step();
        //     // _solver.step();
        // }

        private void LateUpdate()
        {
            _solver.Complete();
            if (mat != null && cube != null)
            {
                _buffer.SetData(_solver.cubes.AsArray());
                mat.SetBuffer("_cubes", _buffer);
                Graphics.DrawMeshInstancedProcedural(cube, 0, mat, new Bounds(Vector3.zero, Vector3.one * 1000), _solver.cubes.Length);
            }
        }

        private void OnDestroy()
        {
            _solver.Dispose();
            _buffer.Dispose();
        }

        private void OnDrawGizmos()
        {
            if (_solver != null)
                DrawSolver(_solver);
        }

        private GUIStyle _style;

        private void OnGUI()
        {
            GUI.Label(new Rect(10, 20, 100, 40), $"body count: {_solver.BodiesCount}", _style);
            GUI.Label(new Rect(10, 60, 100, 40), $"contact count: {_solver.ForcesCount}",_style);
            GUI.Label(new Rect(10, 100, 100, 40), $"color count: {_solver.ColorCount}",_style);
        }

        #region Draw Scene

        static float3[] V = {
            new (-0.5f, -0.5f, -0.5f),
            new (+0.5f, -0.5f, -0.5f),
            new (+0.5f, +0.5f, -0.5f),
            new (-0.5f, +0.5f, -0.5f),
            new (-0.5f, -0.5f, +0.5f),
            new (+0.5f, -0.5f, +0.5f),
            new (+0.5f, +0.5f, +0.5f),
            new (-0.5f, +0.5f, +0.5f)};

        static uint3[] T = {
            new (0, 1, 2), new (0, 2, 3), // -Z
            new (4, 6, 5), new (4, 7, 6), // +Z
            new (1, 5, 6), new (1, 6, 2), // +X
            new (4, 0, 3), new (4, 3, 7), // -X
            new (3, 2, 6), new (3, 6, 7), // +Y
            new (4, 5, 1), new (4, 1, 0) // -Y
        };

        static uint2[] E = {
            new (0, 1), new (1, 2), new (2, 3), new (3, 0),
            new (4, 5), new (5, 6), new (6, 7), new (7, 4),
            new (0, 4), new (1, 5), new (2, 6), new (3, 7)};

        float3 bodyVertexWorld(Rigid body, float3 v)
        {
            float3 local = new(v[0] * body.size.x, v[1] * body.size.y, v[2] * body.size.z);
            return Utils.transform(body.positionLin, body.positionAng, local);
        }

        void DrawBody(Rigid body)
        {
            Gizmos.color = new(0.80f, 0.84f, 0.90f);
            // for (int i = 0; i < 12; ++i)
            // {
            //     float3 a = bodyVertexWorld(body, V[T[i][0]]);
            //     float3 b = bodyVertexWorld(body, V[T[i][1]]);
            //     float3 c = bodyVertexWorld(body, V[T[i][2]]);
            // }

            // Gizmos.color = new(0.10f, 0.12f, 0.14f, 1.0f);
            for (int i = 0; i < 12; ++i)
            {
                float3 a = bodyVertexWorld(body, V[E[i][0]]).xzy;
                float3 b = bodyVertexWorld(body, V[E[i][1]]).xzy;
                Gizmos.DrawLine(a, b);
            }
        }
        
        void DrawManifold(Manifold manifold)
        {
            Gizmos.color = new Color(0.75f, 0.0f, 0.0f);
            for (int i = 0; i < manifold.NumContacts; ++i)
            {
                float3 v0 = Utils.transform(_solver.bodies[manifold.BodyA].positionLin, _solver.bodies[manifold.BodyA].positionAng, manifold.contacts[i].rA).xzy;
                float3 v1 = Utils.transform(_solver.bodies[manifold.BodyB].positionLin, _solver.bodies[manifold.BodyB].positionAng, manifold.contacts[i].rB).xzy;
                Gizmos.DrawSphere(v0, 0.05f);
                Gizmos.DrawSphere(v1, 0.05f);
            }
        }

        void DrawSolver(Solver state)
        {
            // Draw dynamic bodies after shadows so they appear cleanly on top.
            // foreach (Rigid body in state.bodies)
            //     DrawBody(body);
            
            if (drawContact) 
                foreach (var force in state.forces)
                    DrawManifold(force);
        }
        
        #endregion

        #region Scene Init
        
        static void sceneEmpty(Solver solver)
        {
            solver.clear();
        }

        static void sceneGround(Solver solver)
        {
            solver.clear();
            solver.CreateRigid(new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0), float3(0, 0, 0));
            solver.CreateRigid(float3(1,1,1), 1.0f, 0.5f, float3(0, 0, 4));
        }

        static void sceneDynamicFriction(Solver solver)
        {
            solver.clear();
            solver.CreateRigid(new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0), float3(0, 0, 0));
            for (int x = 0; x <= 10; x++)
                solver.CreateRigid(float3(1, 1, 0.5f), 1.0f, 5.0f - (x / 10.0f * 5.0f), float3(0, -30.0f + x * 2.0f, 0.75f), float3(10.0f, 0, 0));
        }

        static void sceneStaticFriction(Solver solver)
        {
            solver.clear();
            solver.CreateRigid(new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0));

            const float angle = TORADIANS * 30.0f;
            var ramp = solver.CreateRigid(float3(40, 24, 1), 0.0f, 1.0f, float3(0, 0, 3));
            ramp.positionAng = quaternion(0, sin(angle * 0.5f), 0, cos(angle * 0.5f));

            float3 rampTangent = normalize(rotate(ramp.positionAng, float3(1, 0, 0)));
            float3 rampNormal = normalize(rotate(ramp.positionAng, float3(0, 0, 1)));

            for (int i = 0; i <= 10; i++)
            {
                float friction = i / 10.0f * 0.25f + 0.25f;
                float y = -10.0f + i * 2.0f;
                float3 pos = ramp.positionLin + rampTangent * -12.0f + float3(0, y, 0) + rampNormal * 1.05f;
                solver.CreateRigid(float3(1,1,1), 1.0f, friction, pos);
            }
        }

        static void scenePyramid(Solver solver)
        {
            const int SIZE = 30;
            solver.clear();
            solver.CreateRigid(new float3(100, 100, 1), 0.0f, 0.5f, float3(0.0f, 0.0f, -0.5f));

            for (int y = 0; y < SIZE; y++)
                for (int x = 0; x < SIZE - y; x++)
                    solver.CreateRigid(float3(1, 1f, 0.5f), 1.0f, 1f, float3(x * 1.01f + y * 0.5f - SIZE / 2.0f, 0.0f, y * 0.7f + 0.5f));
        }

        static void sceneStack(Solver solver)
        {
            solver.clear();
            solver.CreateRigid(new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0));
            for (int i = 0; i < 10; i++)
                solver.CreateRigid(float3(1,1,1), 1.0f, 0.5f, float3(0, 0, i * 1.5f + 1.0f));
        }

        static void sceneStackRatio(Solver solver)
        {
            solver.clear();
            const float groundThickness = 1.0f;
            solver.CreateRigid(float3(100, 100, groundThickness), 0.0f, 0.5f, float3(0, 0, 0));

            float topZ = groundThickness * 0.5f;
            float s = 1.0f;
            for (int i = 0; i < 4; i++)
            {
                float half = s * 0.5f;
                float centerZ = topZ + half;
                solver.CreateRigid(float3(s, s, s), 1.0f, 0.5f, float3(0, 0, centerZ));
                topZ = centerZ + half;
                s *= 2.0f;
            }
        }

        static void sceneLotsOfCubes(Solver solver)
        {
            solver.clear();
            solver.CreateRigid(new float3(150, 150, 1), 0.0f, 1f, float3(0, 0, 0));
            for (int z = 0; z < 50; z++)
            {
                for (int x = 0; x < 10; x++)
                for (int y = 0; y < 10; y++)
                    solver.CreateRigid(float3(1, 1, 1), 1.0f, 1.0f,
                        float3(x * 1.5f - 7.5f, y * 1.5f - 7.5f, z * 1.5f + 1.0f));
                        // Random.onUnitSphere);
            }
        }
        
        delegate void SceneInitFunc(Solver solver);

        static SceneInitFunc[] func = {
            sceneEmpty,
            sceneGround,
            sceneDynamicFriction,
            sceneStaticFriction,
            scenePyramid,
            sceneStack,
            sceneStackRatio,
            sceneLotsOfCubes
        };

        public enum SceneInitType
        {
            Empty,
            Ground,
            DynamicFriction,
            StaticFriction,
            Pyramid,
            Stack,
            StackRatio,
            LotsOfCubes
        };

        #endregion
    }
}
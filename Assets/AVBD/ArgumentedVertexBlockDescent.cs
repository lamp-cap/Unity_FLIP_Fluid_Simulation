using System;
using UnityEngine;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD
{
    public class ArgumentedVertexBlockDescent : MonoBehaviour
    {
        public SceneInitType type = SceneInitType.Ground;

        private Solver _solver;
        // Start is called before the first frame update
        void Start()
        {
            _solver = new Solver();
            func[(int)type].Invoke(_solver);
        }

        // Update is called once per frame
        void Update()
        {
            _solver.step();
        }

        private void OnDestroy()
        {
            _solver.Dispose();
        }

        private void OnDrawGizmos()
        {
            if (_solver != null)
                drawSolver(_solver);
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

        void drawBody(Rigid body)
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

        void drawJoint(Joint joint)
        {
            float3 v0 = joint.bodyA != null ? Utils.transform(joint.bodyA.positionLin, joint.bodyA.positionAng, joint.rA).xzy : joint.rA.xzy;
            float3 v1 = Utils.transform(joint.bodyB.positionLin, joint.bodyB.positionAng, joint.rB).xzy;

            Gizmos.color = new(0.75f, 0.0f, 0.0f);
            Gizmos.DrawLine(v0, v1);
        }

        void drawSpring(Spring spring)
        {
            float3 v0 = Utils.transform(spring.bodyA.positionLin, spring.bodyA.positionAng, spring.rA).xzy;
            float3 v1 = Utils.transform(spring.bodyB.positionLin, spring.bodyB.positionAng, spring.rB).xzy;

            Gizmos.color = new(0.75f, 0.0f, 0.0f);
            Gizmos.DrawLine(v0, v1);
        }

        void drawManifold(Manifold manifold)
        {

            Gizmos.color = new Color(0.75f, 0.0f, 0.0f);
            for (int i = 0; i < manifold.numContacts; ++i)
            {
                float3 v0 = Utils.transform(manifold.bodyA.positionLin, manifold.bodyA.positionAng, manifold.contacts[i].rA).xzy;
                float3 v1 = Utils.transform(manifold.bodyB.positionLin, manifold.bodyB.positionAng, manifold.contacts[i].rB).xzy;
                Gizmos.DrawSphere(v0, 0.05f);
                Gizmos.DrawSphere(v1, 0.05f);
            }
        }

        void drawSolver(Solver state)
        {
            // Draw dynamic bodies after shadows so they appear cleanly on top.
            foreach (Rigid body in state.bodies)
                drawBody(body);

            foreach (Force force in state.forces)
            {
                switch (force)
                {
                    case (Joint joint):
                        drawJoint(joint); break;
                    case (Spring spring):
                        drawSpring(spring); break;
                    case (Manifold manifold):
                        drawManifold(manifold); break;
                }
            }
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
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0), float3(0, 0, 0));
            new Rigid(solver, float3(1,1,1), 1.0f, 0.5f, float3(0, 0, 4));
        }

        static void sceneDynamicFriction(Solver solver)
        {
            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0), float3(0, 0, 0));
            for (int x = 0; x <= 10; x++)
                new Rigid(solver, float3(1, 1, 0.5f), 1.0f, 5.0f - (x / 10.0f * 5.0f), float3(0, -30.0f + x * 2.0f, 0.75f), float3(10.0f, 0, 0));
        }

        static void sceneStaticFriction(Solver solver)
        {
            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0));

            const float angle = TORADIANS * 30.0f;
            Rigid ramp = new Rigid(solver, float3(40, 24, 1), 0.0f, 1.0f, float3(0, 0, 3));
            ramp.positionAng = quaternion(0, sin(angle * 0.5f), 0, cos(angle * 0.5f));

            float3 rampTangent = normalize(rotate(ramp.positionAng, float3(1, 0, 0)));
            float3 rampNormal = normalize(rotate(ramp.positionAng, float3(0, 0, 1)));

            for (int i = 0; i <= 10; i++)
            {
                float friction = i / 10.0f * 0.25f + 0.25f;
                float y = -10.0f + i * 2.0f;
                float3 pos = ramp.positionLin + rampTangent * -12.0f + float3(0, y, 0) + rampNormal * 1.05f;
                new Rigid(solver, float3(1,1,1), 1.0f, friction, pos);
            }
        }

        static void scenePyramid(Solver solver)
        {
            const int SIZE = 16;
            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0.0f, 0.0f, -0.5f));

            for (int y = 0; y < SIZE; y++)
                for (int x = 0; x < SIZE - y; x++)
                    new Rigid(solver, float3(1, 0.5f, 0.5f), 1.0f, 0.5f, float3(x * 1.01f + y * 0.5f - SIZE / 2.0f, 0.0f, y * 0.85f + 0.5f));
        }

        static void sceneRope(Solver solver)
        {
            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, -20));

            Rigid prev = null;
            for (int i = 0; i < 20; i++)
            {
                Rigid curr = new Rigid(solver, float3(1, 0.5f, 0.5f), i == 0 ? 0.0f : 1.0f, 0.5f, float3(i, 0.0f, 10.0f));
                if (prev != null)
                    new Joint(solver, prev, curr, float3(0.5f, 0, 0), float3(-0.5f, 0, 0));
                prev = curr;
            }
        }

        static void sceneHeavyRope(Solver solver)
        {
            const int N = 20;
            const float SIZE = 5;
            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, -20));

            Rigid prev = null;
            for (int i = 0; i < N; i++)
            {
                Rigid curr = new Rigid(solver, i == N - 1 ? float3(SIZE, SIZE, SIZE) : float3(1, 0.5f, 0.5f),
                                        i == 0 ? 0.0f : 1.0f, 0.5f, float3(i + (i == N - 1 ? SIZE / 2 : 0), 0.0f, 10.0f));
                if (prev!=null)
                    new Joint(solver, prev, curr, float3(0.5f, 0, 0), i == N - 1 ? float3(-SIZE / 2, 0, 0) : float3(-0.5f, 0, 0));
                prev = curr;
            }
        }

        static void sceneSpring(Solver solver)
        {
            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0));

            Rigid anchor = new Rigid(solver, float3(1,1,1), 0.0f, 0.5f, float3(0, 0, 14.0f));
            Rigid block = new Rigid(solver, float3(2, 2, 2), 1.0f, 0.5f, float3(0, 0, 8.0f));
            new Spring(solver, anchor, block, float3(0, 0, 0), float3(0, 0, 0), 100.0f, 4.0f);
        }

        static void sceneSpringsRatio(Solver solver)
        {
            const int N = 8;
            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, -10));

            Rigid prev = null;
            for (int i = 0; i < N; i++)
            {
                float x = (i - (N - 1) * 0.5f) * 3.0f;
                Rigid curr = new Rigid(solver, float3(1, 0.75f, 0.75f), i == 0 || i == N - 1 ? 0.0f : 1.0f, 0.5f, float3(x, 0.0f, 12.0f));
                if (prev != null)
                    new Spring(solver, prev, curr, float3(0.5f, 0, 0), float3(-0.5f, 0, 0), i % 2 == 0 ? 10.0f : 10000.0f, 3.0f);
                prev = curr;
            }
        }

        static void sceneStack(Solver solver)
        {
            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0));
            for (int i = 0; i < 10; i++)
                new Rigid(solver, float3(1,1,1), 1.0f, 0.5f, float3(0, 0, i * 1.5f + 1.0f));
        }

        static void sceneStackRatio(Solver solver)
        {
            solver.clear();
            const float groundThickness = 1.0f;
            new Rigid(solver, float3(100, 100, groundThickness), 0.0f, 0.5f, float3(0, 0, 0));

            float topZ = groundThickness * 0.5f;
            float s = 1.0f;
            for (int i = 0; i < 4; i++)
            {
                float half = s * 0.5f;
                float centerZ = topZ + half;
                new Rigid(solver, float3(s, s, s), 1.0f, 0.5f, float3(0, 0, centerZ));
                topZ = centerZ + half;
                s *= 2.0f;
            }
        }

        static void sceneSoftBody(Solver solver)
        {
            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0));

            const float Klin = 1000.0f;
            const float Kang = 250.0f;
            const int W = 4;
            const int D = 4;
            const int H = 4;
            const int N = 3;
            const float size = 0.8f;
            const float half = size * 0.5f;
            const float baseZ = 8.0f;
            const float stackGap = 2.0f;

            for (int i = 0; i < N; i++)
            {
                var grid = new Rigid[W,D,H];
                float stackZ = i * (H * size + stackGap);

                for (int x = 0; x < W; x++)
                for (int y = 0; y < D; y++)
                for (int z = 0; z < H; z++)
                {
                    float px = (x - (W - 1) * 0.5f) * size;
                    float py = (y - (D - 1) * 0.5f) * size;
                    float pz = baseZ + stackZ + z * size;
                    grid[x,y,z] = new Rigid(solver, float3(size), 1.0f, 0.5f, float3(px, py, pz));
                }
                
                for (int x = 1; x < W; x++)
                for (int y = 0; y < D; y++)
                for (int z = 0; z < H; z++)
                {
                    new Joint(solver, grid[x - 1,y,z], grid[x,y,z], float3(half, 0, 0), float3(-half, 0, 0), Klin, Kang);
                }

                for (int x = 0; x < W; x++)
                for (int y = 1; y < D; y++)
                for (int z = 0; z < H; z++)
                {
                    new Joint(solver, grid[x,y - 1,z], grid[x,y,z], float3(0, half, 0), float3(0, -half, 0), Klin, Kang);
                }
                
                for (int x = 0; x < W; x++)
                for (int y = 0; y < D; y++)
                for (int z = 1; z < H; z++)
                {
                    new Joint(solver, grid[x,y,z - 1], grid[x,y,z], float3(0, 0, half), float3(0, 0, -half), Klin, Kang);
                }
                
                for (int x = 1; x < W; x++)
                for (int y = 0; y < D; y++)
                for (int z = 1; z < H; z++)
                {
                    new IgnoreCollision(solver, grid[x - 1,y,z - 1], grid[x,y,z]);
                    new IgnoreCollision(solver, grid[x,y,z - 1], grid[x - 1,y,z]);
                }

                for (int x = 0; x < W; x++)
                for (int y = 1; y < D; y++)
                for (int z = 1; z < H; z++)
                {
                    new IgnoreCollision(solver, grid[x,y - 1,z - 1], grid[x,y,z]);
                    new IgnoreCollision(solver, grid[x,y,z - 1], grid[x,y - 1,z]);
                }
                
                for (int x = 1; x < W; x++)
                for (int y = 1; y < D; y++)
                for (int z = 0; z < H; z++)
                {
                    new IgnoreCollision(solver, grid[x - 1,y - 1,z], grid[x,y,z]);
                    new IgnoreCollision(solver, grid[x,y - 1,z], grid[x - 1,y,z]);
                }
            }
        }

        static void sceneBridge(Solver solver)
        {
            const int N = 40;
            const float plankLength = 1.0f;
            const float plankWidth = 4.0f;
            const float plankHeight = 0.5f;
            const float halfLength = plankLength * 0.5f;
            const float halfWidth = plankWidth * 0.5f;

            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0));

            Rigid prev = null;
            for (int i = 0; i < N; i++)
            {
                Rigid curr = new Rigid(solver, float3(plankLength, plankWidth, plankHeight), i == 0 || i == N - 1 ? 0.0f : 1.0f, 0.5f, float3(i - N / 2.0f, 0.0f, 10.0f));
                if (prev!=null)
                {
                    new Joint(solver, prev, curr, float3(halfLength,  halfWidth, 0), float3(-halfLength,  halfWidth, 0), float.PositiveInfinity, 0.0f);
                    new Joint(solver, prev, curr, float3(halfLength, -halfWidth, 0), float3(-halfLength, -halfWidth, 0), float.PositiveInfinity, 0.0f);
                }
                prev = curr;
            }

            for (int x = 0; x < N / 4; x++)
            {
                for (int y = 0; y < N / 8; y++)
                {
                    new Rigid(solver, float3(1,1,1), 1.0f, 0.5f, float3((float)x - N / 8.0f, 0.0f, (float)y + 12.0f));
                }
            }
        }

        static void sceneBreakable(Solver solver)
        {
            const int N = 10;
            const int M = 5;
            const float breakForce = 90.0f;

            solver.clear();
            new Rigid(solver, new float3(100, 100, 1), 0.0f, 0.5f, float3(0, 0, 0));

            Rigid prev = null;
            for (int i = 0; i <= N; i++)
            {
                Rigid curr = new Rigid(solver, float3(1, 1, 0.5f), 1.0f, 0.5f, float3(i - N / 2.0f, 0.0f, 6.0f));
                if (prev  != null)
                    new Joint(solver, prev, curr, float3(0.5f, 0, 0), float3(-0.5f, 0, 0), INFINITY, INFINITY, breakForce);
                prev = curr;
            }

            new Rigid(solver, float3(1, 1, 5), 0.0f, 0.5f, float3(-N / 2.0f, 0, 2.5f));
            new Rigid(solver, float3(1, 1, 5), 0.0f, 0.5f, float3( N / 2.0f, 0, 2.5f));

            for (int i = 0; i < M; i++)
                new Rigid(solver, float3(2, 1, 1), 1.0f, 0.5f, float3(0, 0, i * 2.0f + 8.0f));
        }

        delegate void SceneInitFunc(Solver solver);

        static SceneInitFunc[] func = {
            sceneEmpty,
            sceneGround,
            sceneDynamicFriction,
            sceneStaticFriction,
            scenePyramid,
            sceneRope,
            sceneHeavyRope,
            sceneSpring,
            sceneSpringsRatio,
            sceneStack,
            sceneStackRatio,
            sceneSoftBody,
            sceneBridge,
            sceneBreakable};

        public enum SceneInitType
        {
            Empty,
            Ground,
            DynamicFriction,
            StaticFriction,
            Pyramid,
            Rope,
            HeavyRope,
            Spring,
            SpringRatio,
            Stack,
            StackRatio,
            SoftBody,
            Bridge,
            Breakable
            
        };

        #endregion
    }
}
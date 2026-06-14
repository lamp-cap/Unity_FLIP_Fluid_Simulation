using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Profiling;
using static Unity.Mathematics.math;

namespace AVBD.RigidOnly
{
    public class Solver
    {
        public const float dt = 1.0f / 120.0f;       // Timestep
        public const float gravity = -9.8f;  // Gravity
        public const int iterations = 6; // Solver iterations

        // Note: in the paper, beta is suggested to be [1, 1000]. Technically, the best choice will
        // depend on the length, mass, and constraint function scales (ie units) of your simulation,
        // along with your strategy for incrementing the penalty parameters.
        // If the value is not in the right range, you may see slower convergance for complex scenes.
        // A minor upgrade from the paper is using separate betas for constraints of different units (eg linear vs angular).
        public const float betaLin = 10000.0f;  // Penalty ramping parameter for linear constraints
        public const float betaAng = 100.0f;  // Penalty ramping parameter for angular constraints
        
        // Alpha controls how much stabilization is applied. Higher values give slower and smoother
        // error correction, and lower values are more responsive and energetic. Tune this depending
        // on your desired constraint error response.
        public const float alpha = 0.99f; // Stabilization parameter
        
        // Gamma controls how much the penalty and lambda values are decayed each step during warmstarting.
        // This should always be < 1 so that the penalty values can decrease (unless you use a different
        // penalty parameter strategy which does not require decay).
        public const float gamma = 0.999f; // Warmstarting decay parameter

        public int BodiesCount;
        public int ForcesCount;
        public int ColorCount;
        public NativeList<Rigid> bodies => _bodies;
        public NativeList<Manifold> forces => _forces;
        public NativeList<Box> cubes => _cubes;
        
        private CompressedSparseRows _csr;

        private NativeList<Rigid> _bodies;
        private NativeList<OBB> _boxes;
        private NativeList<Manifold> _forces;
        private NativeList<Manifold> _forcesAlt;
        private NativeList<int2> _pair;
        private NativeList<Box> _cubes;
        private NativeReference<int> _colorCount;
        private BVH _bvh;

        private JobHandle _handle = default;

        public Solver()
        {
            _csr = new CompressedSparseRows();
            _bodies = new NativeList<Rigid>(Allocator.Persistent);
            _forces = new NativeList<Manifold>(Allocator.Persistent);
            _forcesAlt = new NativeList<Manifold>(Allocator.Persistent);
            _boxes = new NativeList<OBB>(Allocator.Persistent);
            _pair = new NativeList<int2>(Allocator.Persistent);
            _cubes = new NativeList<Box>(Allocator.Persistent);
            _colorCount = new NativeReference<int>(Allocator.Persistent);
            _bvh = new BVH();
        }

        public Rigid pick(float3 origin, float3 dir, out float3 local)
        {
            local = float3(0);
            const float epsilon = 1.0e-6f;
            float bestT = INFINITY;
            Rigid bestBody = default;
            float3 bestLocal = float3(0, 0, 0);

            // Ray-cast against each OBB by transforming the ray into body local space.
            foreach (Rigid body in bodies)
            {
                if (body.mass <= 0.0f)
                    continue;

                quaternion invRot = conjugate(body.positionAng);
                float3 o = rotate(invRot, origin - body.positionLin);
                float3 d = rotate(invRot, dir);
                float3 half = body.size * 0.5f;

                float tEnter = 0.0f;
                float tExit = INFINITY;
                bool hit = true;

                for (int i = 0; i < 3; ++i)
                {
                    if (abs(d[i]) < epsilon)
                    {
                        if (o[i] < -half[i] || o[i] > half[i])
                        {
                            hit = false;
                            break;
                        }
                        continue;
                    }

                    float invD = 1.0f / d[i];
                    float t0 = (-half[i] - o[i]) * invD;
                    float t1 = (half[i] - o[i]) * invD;
                    if (t0 > t1)
                    {
                        (t0, t1) = (t1, t0);
                    }

                    tEnter = max(tEnter, t0);
                    tExit = min(tExit, t1);
                    if (tEnter > tExit)
                    {
                        hit = false;
                        break;
                    }
                }

                if (!hit)
                    continue;

                float tHit = tEnter >= 0.0f ? tEnter : tExit;
                if (tHit < 0.0f)
                    continue;

                if (tHit < bestT)
                {
                    bestT = tHit;
                    bestBody = body;
                    bestLocal = o + d * tHit;
                }
            }

            if (!bestBody.IsCreate)
                return default;

            local = bestLocal;
            return bestBody;
        }

        public void CreateRigid(float3 size, float density, float friction, float3 position)
            => CreateRigid(size, density, friction, position, float3(0));

        public void CreateRigid(float3 size, float density, float friction, float3 position, float3 velocity)
        {
            var body = new Rigid(bodies.Length + 1, size, density, friction, position, velocity);
            _bodies.Add(body);
            _csr.Csr.Add(int4(0));
            _boxes.Add(new OBB());
            _cubes.Add(new Box());
        }
        
        public void CreateRigid(float3 size, float density, float friction, float3 position, float3 velocity, quaternion rotation)
        {
            var body = new Rigid(bodies.Length + 1, size, density, friction, position, velocity);
            body.positionAng = rotation;
            _bodies.Add(body);
            _csr.Csr.Add(int4(0));
            _boxes.Add(new OBB());
            _cubes.Add(new Box());
        }

        public void clear()
        {
            forces.Clear();
            bodies.Clear();
            _csr.Csr.Clear();
            _boxes.Clear();
            _cubes.Clear();
        }

        public void Dispose()
        {
            _csr.Dispose();
            _bodies.Dispose();
            _forces.Dispose();
            _forcesAlt.Dispose();
            _boxes.Dispose();
            _pair.Dispose();
            _cubes.Dispose();
            _colorCount.Dispose();
            _bvh.Dispose();
        }

        public void step()
        {
            _handle.Complete();
            var bodiesArr = _bodies.AsArray();
            var csrArr = _csr.Csr.AsArray();
            var boxesArr = _boxes.AsArray();
            
            Profiler.BeginSample("Collision");
            new Collision.MakeObbJob()
            {
                Boxes = boxesArr,
                Bodies = bodiesArr,
            }.Schedule(_bodies.Length, 16).Complete();

            int expect = boxesArr.Length * 3;
            expect = expect / 64 * 64 + 64;
            if (_pair.Capacity < boxesArr.Length)
                _pair.Capacity = expect;
            _pair.Length = 0;
            _bvh.Collide(boxesArr, _pair);
            
            // _pair.Clear();
            // new Collision.PreCheckCollisionJob()
            // {
            //     Boxes = boxesArr,
            //     Bodies = bodiesArr,
            //     Pair = _pair
            // }.Run();
            Profiler.EndSample();

            Profiler.BeginSample("Initialize forces");
            if (_forcesAlt.Capacity < _pair.Length)
                _forcesAlt.Capacity = _pair.Length;
            _forcesAlt.Length = 0;
            new Collision.CheckCollisionJob()
            {
                Boxes = boxesArr,
                Bodies = bodiesArr,
                Csr = csrArr,
                CsrPool = _csr.Data.AsArray(),
                Forces = _forces.AsArray(),
                Pair = _pair.AsArray(),
                Writer = _forcesAlt.AsParallelWriter()
            }.Schedule(_pair.Length, 1).Complete();
            Profiler.EndSample();

            (_forces, _forcesAlt) = (_forcesAlt, _forces);
            
            var forcesArr = _forces.AsArray();
            
            Profiler.BeginSample("Build CSR");
            var counter = new NativeArray<int>(_bodies.Length, Allocator.TempJob);
            new CompressedSparseRows.CounterEdgesJob()
            {
                Forces = forcesArr,
                Counter = counter,
            }.Schedule(_forces.Length, 16).Complete();
            
            new CompressedSparseRows.BuildCSRJob()
            {
                Counter = counter,
                Csr = csrArr,
            }.Run();
            unsafe
            {
                UnsafeUtility.MemClear(counter.GetUnsafePtr(), counter.Length * sizeof(int));
            }
            if (_csr.Data.Length != _forces.Length * 2) _csr.Data.Length = _forces.Length * 2;
            
            var dataArr = _csr.Data.AsArray();
            
            new CompressedSparseRows.FillDataJob()
            {
                Forces = forcesArr,
                Counter = counter,
                Csr = csrArr,
                Data = dataArr,
            }.Schedule(_forces.Length, 16).Complete();
            counter.Dispose();
            
            new CompressedSparseRows.GreedyColoringJob()
            {
                Csr = csrArr,
                Data = dataArr,
                ColorCount = _colorCount
            }.Run();
            Profiler.EndSample();
            ColorCount = _colorCount.Value;

            Profiler.BeginSample("Initialize bodies");
            // Initialize and warmstart bodies (ie primal variables)
            _handle = new InitializeBodiesJob()
            {
                Bodies = bodiesArr
            }.Schedule(bodiesArr.Length, 8);
            Profiler.EndSample();

            Profiler.BeginSample("update");
            // Main solver loop
            for (int it = 0; it < iterations; it++)
            {
                for (int i = 0; i < ColorCount; i++)
                {
                    _handle = new UpdatePrimalJob()
                    {
                        Bodies = bodiesArr,
                        Forces = forcesArr,
                        Csr = csrArr,
                        Pool = dataArr,
                        Color = i
                    }.Schedule(bodiesArr.Length, 1, _handle);
                }

                _handle = new UpdateDualJob()
                {
                    Bodies = bodiesArr,
                    Forces = forcesArr,
                }.Schedule(forcesArr.Length, 1, _handle);
            }
            Profiler.EndSample();

            Profiler.BeginSample("Final");
            // Compute velocities (BDF1) after the final iteration
            _handle = new ComputeVelocitiesJob()
            {
                Bodies = bodiesArr
            }.Schedule(bodiesArr.Length, 16, _handle);

            Profiler.EndSample();
        }

        public void Complete()
        {
            _handle.Complete();
            new PrepareRenderingDataJob()
            {
                Bodies = _bodies.AsArray(),
                Data = cubes.AsArray(),
            }.Schedule(_bodies.Length, 16).Complete();
            BodiesCount = _bodies.Length;
            ForcesCount = _forces.Length;
        }

        [BurstCompile]
        private struct InitializeBodiesJob : IJobParallelFor
        {
            public NativeArray<Rigid> Bodies;
            public void Execute(int i)
            {
                var body = Bodies[i];
                // Compute inertial position (Eq 2)
                body.inertialLin = body.positionLin + body.velocityLin * dt;
                if (body.mass > 0)
                    body.inertialLin += float3(0, 0, gravity) * (dt * dt);
                body.inertialAng = body.positionAng.add(body.velocityAng * dt);

                // Adaptive warmstart (See original VBD paper)
                float3 accel = (body.velocityLin - body.prevVelocityLin) / dt;
                float accelExt = accel.z * sign(gravity);
                float accelWeight = clamp(accelExt / abs(gravity), 0.0f, 1.0f);
                if (!isfinite(accelWeight))
                    accelWeight = 0.0f;

                // Save initial position (x-) and compute warmstarted position (See original VBD paper)
                body.initialLin = body.positionLin;
                body.initialAng = body.positionAng;
                if (body.mass > 0)
                {
                    body.positionLin = body.positionLin + body.velocityLin * dt + float3(0, 0, gravity) * (accelWeight * dt * dt);
                    body.positionAng = body.positionAng.add(body.velocityAng * dt);
                }
                Bodies[i] = body;
            }
        }

        [BurstCompile]
        private struct UpdatePrimalJob : IJobParallelFor
        {
            [NativeDisableContainerSafetyRestriction] public NativeArray<Manifold> Forces;
            [NativeDisableContainerSafetyRestriction] public NativeArray<Rigid> Bodies;
            [ReadOnly] public NativeArray<int4> Csr;
            [NativeDisableContainerSafetyRestriction] [ReadOnly] public NativeArray<int4> Pool;
            public int Color;
            
            public void Execute(int i)
            {
                var body = Bodies[i];
                var range = Csr[body.ID];
                // Skip static / kinematic bodies
                if (body.mass <= 0 || Color != range.z)
                    return;

                // Initialize left and right hand sides of the linear system (Eqs. 5, 6)
                float3x3 mLin = Utils.diagonal(body.mass, body.mass, body.mass);
                float3x3 mAng = Utils.diagonal(body.moment.x, body.moment.y, body.moment.z);

                var eq6 = new Equation6()
                {
                    lhsLin = mLin / (dt * dt),
                    lhsAng = mAng / (dt * dt),
                    lhsCross = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0),
                    rhsLin = mul(mLin / (dt * dt), body.positionLin - body.inertialLin),
                    rhsAng = mul(mAng / (dt * dt), body.positionAng.sub(body.inertialAng))
                };
                    
                // Iterate over all forces acting on the body
                for (int j = range.x; j < range.y; j++)
                {
                    var data = Pool[j];
                    var force = Forces[data.z];
                    // Stamp the force and hessian into the linear system.
                    // Always pass the manifold's true A/B (data.x/data.y is swapped for the B row),
                    // otherwise the jacobian sign and rA/rB association are wrong for half the bodies.
                    force.updatePrimal(body.ID, Bodies[force.BodyA], Bodies[force.BodyB], alpha, ref eq6);
                    Forces[data.z] = force;
                }
                    
                // Solve the SPD linear system using LDL and apply the update (Eq. 4)
                eq6.Solve(out var dxLin, out var dxAng);
                    
                body.positionLin = body.positionLin + dxLin;
                body.positionAng = body.positionAng.add(dxAng);
                
                Bodies[i] = body;
            }
        }

        [BurstCompile]
        private struct UpdateDualJob : IJobParallelFor
        {
            public NativeArray<Manifold> Forces;
            [ReadOnly]public NativeArray<Rigid> Bodies;
            
            public void Execute(int i)
            {
                var force = Forces[i];
                force.updateDual(Bodies[force.BodyA], Bodies[force.BodyB], alpha);
                Forces[i] = force;
            }
        }
        
        [BurstCompile]
        private struct ComputeVelocitiesJob : IJobParallelFor
        {
            public NativeArray<Rigid> Bodies;
            
            public void Execute(int i)
            {
                var body = Bodies[i];
                body.prevVelocityLin = body.velocityLin;
                if (body.mass > 0)
                {
                    body.velocityLin = (body.positionLin - body.initialLin) / dt;
                    body.velocityAng = (body.positionAng.sub(body.initialAng)) / dt;
                }

                Bodies[i] = body;
            }
        }
        
        public struct Box
        {
            public float3 size;
            public float4 rotation;
            public float3 pos;
            public uint color;
            public float padding;
        };
        
        [BurstCompile]
        private struct PrepareRenderingDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Rigid> Bodies;
            [WriteOnly] public NativeArray<Box> Data;
            
            public void Execute(int index)
            {
                var body = Bodies[index];
                Data[index] = new Box()
                {
                    size = body.size,
                    rotation = body.positionAng.value,
                    pos = body.positionLin,
                };
            }
        }
    }
    
}

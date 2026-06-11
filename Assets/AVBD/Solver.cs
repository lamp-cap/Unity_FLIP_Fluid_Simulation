using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using UnityEngine;
using UnityEngine.Profiling;

namespace AVBD
{
    public class Solver
    {
        public const float dt = 1.0f / 200.0f;       // Timestep
        public const float gravity = -10.0f;  // Gravity
        public const int iterations = 10; // Solver iterations

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

        public List<Rigid> bodies;
        public List<Force> forces;
        
        private Stack<Manifold> _pool;
        private NativeReference<Equation6> _eq6;

        public Solver()
        {
            bodies = new List<Rigid>();
            forces = new List<Force>();
            _pool = new Stack<Manifold>();
            _eq6 = new NativeReference<Equation6>(Allocator.Persistent);
        }

        public Rigid pick(float3 origin, float3 dir, out float3 local)
        {
            local = float3(0);
            const float epsilon = 1.0e-6f;
            float bestT = INFINITY;
            Rigid bestBody = null;
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

            if (bestBody == null)
                return null;

            local = bestLocal;
            return bestBody;
        }

        public void clear()
        {
            foreach (var f in forces)
                f.Dispose();
            
            forces.Clear();
            bodies.Clear();
        }

        public void Dispose()
        {
            foreach (var f in forces)
                f.Dispose();

            foreach (var m in _pool)
                m.Dispose();
            
            _eq6.Dispose();
            _pool.Clear();
            forces.Clear();
            bodies.Clear();
        }

        private void CreateManifold(Rigid bodyA, Rigid bodyB)
        {
            Manifold m;
            if (_pool.Count > 0)
            {
                m = _pool.Pop();
                m.Reset(this, bodyA, bodyB);
            }
            else 
                m = new Manifold(this, bodyA, bodyB);
        }

        public void step()
        {
            Profiler.BeginSample("Collision");
            // Perform broadphase collision detection
            // This is a naive O(n^2) approach, but it is sufficient for small numbers of bodies in this sample.
            for (int i=0; i<bodies.Count; ++i)
            {
                var bodyA = bodies[i];
                for (int j = i + 1; j < bodies.Count; ++j)
                {
                    var bodyB = bodies[j];
                    float3 dp = bodyA.positionLin - bodyB.positionLin;
                    float r = bodyA.radius + bodyB.radius;
                    if (dot(dp, dp) <= r * r && !bodyA.constrainedTo(bodyB))
                        CreateManifold(bodyA, bodyB);
                }
            }
            Profiler.EndSample();

            Profiler.BeginSample("Initialize forces");
            // Initialize and warmstart forces
            for (int i = 0; i < forces.Count; ++i)
            {
                var f = forces[i];
                // Initialization can including caching anything that is constant over the step
                if (f.initialize()) continue;
                // Force has returned false meaning it is inactive, so remove it from the solver
                f.Dispose();
                forces.RemoveAtSwapBack(i);
                if (f is Manifold m) _pool.Push(m);
                i--;
            }
            Profiler.EndSample();

            Profiler.BeginSample("Initialize bodies");
            // Initialize and warmstart bodies (ie primal variables)
            foreach (Rigid body in bodies)
            {
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
            }
            Profiler.EndSample();

            // Main solver loop
            for (int it = 0; it < iterations; it++)
            {
                Profiler.BeginSample("Primal update");
                // Primal update
                foreach (Rigid body in bodies)
                {
                    // Skip static / kinematic bodies
                    if (body.mass <= 0)
                        continue;

                    // Initialize left and right hand sides of the linear system (Eqs. 5, 6)
                    float3x3 MLin = Utils.diagonal(body.mass, body.mass, body.mass);
                    float3x3 MAng = Utils.diagonal(body.moment.x, body.moment.y, body.moment.z);

                    var eq6 = new Equation6();
                    eq6.lhsLin = MLin / (dt * dt);
                    eq6.lhsAng = MAng / (dt * dt);
                    eq6.lhsCross = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);

                    eq6.rhsLin = mul(MLin / (dt * dt), body.positionLin - body.inertialLin);
                    eq6.rhsAng = mul(MAng / (dt * dt), body.positionAng.sub(body.inertialAng));
                    _eq6.Value = eq6;

                    // Iterate over all forces acting on the body
                    foreach (Force force in body.forces)
                    {
                        // Stamp the force and hessian into the linear system
                        force.updatePrimal(body, alpha, _eq6);
                    }

                    eq6 = _eq6.Value;

                    // Solve the SPD linear system using LDL and apply the update (Eq. 4)
                    eq6.solve(out var dxLin, out var dxAng);
                    body.positionLin = body.positionLin + dxLin;
                    body.positionAng = body.positionAng.add(dxAng);
                }
                Profiler.EndSample();

                Profiler.BeginSample("Dual update");
                // Dual update
                foreach (Force force in forces)
                {
                    force.updateDual(alpha);
                }
                Profiler.EndSample();
            }

            Profiler.BeginSample("Final");
            // Compute velocities (BDF1) after the final iteration
            foreach (Rigid body in bodies)
            {
                body.prevVelocityLin = body.velocityLin;
                if (body.mass > 0)
                {
                    body.velocityLin = (body.positionLin - body.initialLin) / dt;
                    body.velocityAng = (body.positionAng.sub(body.initialAng)) / dt;
                }
            }
            Profiler.EndSample();
        }
    }
    
}

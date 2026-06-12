using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Profiling;
using static Unity.Mathematics.math;

namespace AVBD
{
    public class Manifold : Force
    {
        // Used to track contact features between frames
        public struct FeaturePair
        {
            // struct
            // {
            //     char inR;
            //     char outR;
            //     char inI;
            //     char outI;
            // };

            public int key;
        };

        // Contact point information for a single contact
        public struct Contact
        {
            public FeaturePair feature;
            public float3 rA; // contact offset in A's local space (relative to center)
            public float3 rB; // contact offset in B's local space (relative to center)
            public float3 C0;
            public float3 penalty;
            public float3 lambda;
            public bool stick;
        };

        public NativeArray<Contact> contacts;
        public float3x3 basis; // Normal in the first row (pointing from B to A), and tangents in the second and third rows
        public int numContacts;
        public float friction;

        public Manifold(Solver solver, Rigid bodyA, Rigid bodyB) : base(solver, bodyA, bodyB)
        {
            numContacts = 0;
        }

        public void Reset(Solver solver, Rigid vBodyA, Rigid vBodyB)
        {
            bodyA = vBodyA;
            bodyB = vBodyB;
            // Add to solver linked list
            solver.forces.Add(this);

            // Add to body linked lists
            vBodyA?.forces.Add(this);
            vBodyB?.forces.Add(this);
            numContacts = 0;
        }
        
        public override bool initialize(Collision collision)
        {
            // Compute friction
            friction = sqrt(bodyA.friction * bodyB.friction);

            // Compute new contacts
            var newContacts = new NativeArray<Contact>(8, Allocator.Temp);
            Profiler.BeginSample("Collide");
            int newNumContacts = collision.collide(bodyA, bodyB, newContacts, out basis);
            Profiler.EndSample();

            // Merge old contact data with new contacts
            for (int i = 0; i < newNumContacts; i++)
            for (int j = 0; j < numContacts; j++)
            {
                if (newContacts[i].feature.key == contacts[j].feature.key)
                {
                    float3 newRA = newContacts[i].rA;
                    float3 newRB = newContacts[i].rB;
                    var nc = contacts[j];

                    // If no static friction in last frame, use the new contact point locations
                    if (!contacts[j].stick)
                    {
                        nc.rA = newRA;
                        nc.rB = newRB;
                    }
                    newContacts[i] = nc;
                    break;
                }
            }

            // Copy new contacts to the manifold
            numContacts = newNumContacts;
            if (numContacts > 0 && !contacts.IsCreated)
                contacts = new NativeArray<Contact>(8, Allocator.Persistent);
            contacts.CopyFrom(newContacts);

            newContacts.Dispose();

            // Compute error at q- and update penalty and lambdas
            for (int i = 0; i < numContacts; i++)
            {
                // Error at q-
                var c = contacts[i];
                float3 xA = Utils.transform(bodyA.positionLin, bodyA.positionAng, c.rA);
                float3 xB = Utils.transform(bodyB.positionLin, bodyB.positionAng, c.rB);
                c.C0 = mul(basis, xA - xB) + float3(Utils.COLLISION_MARGIN, 0, 0);

                // Warmstart the dual variables and penalty parameters (Eq. 19)
                // Penalty is safely clamped to a minimum and maximum value
                c.lambda = c.lambda * Solver.alpha * Solver.gamma;
                c.penalty = clamp(c.penalty * Solver.gamma, Utils.PENALTY_MIN, Utils.PENALTY_MAX);
                contacts[i] = c;
            }

            return numContacts > 0;
        }

        public override void Dispose()
        {
            base.Dispose();
            if (contacts.IsCreated)
                contacts.Dispose();
        }

        public override void updatePrimal(Rigid body, float alpha, NativeReference<Equation6> eq6)
        {
            new UpdatePrimalJob()
            {
                PositionLinA = bodyA.positionLin,
                PositionAngA = bodyA.positionAng,
                InitialLinA = bodyA.initialLin,
                InitialAngA = bodyA.initialAng,
                PositionLinB = bodyB.positionLin,
                PositionAngB = bodyB.positionAng,
                InitialLinB = bodyB.initialLin,
                InitialAngB = bodyB.initialAng,
                NumContacts = numContacts,
                Contacts = contacts,
                Basis = basis,
                Alpha = alpha,
                Friction = friction,
                IsBodyA = body == bodyA,
                Eq6 = eq6
            }.Run();
        }

        public override void updateDual(float alpha)
        {
            new UpdateDualJob()
            {
                PositionLinA = bodyA.positionLin,
                PositionAngA = bodyA.positionAng,
                InitialLinA = bodyA.initialLin,
                InitialAngA = bodyA.initialAng,
                PositionLinB = bodyB.positionLin,
                PositionAngB = bodyB.positionAng,
                InitialLinB = bodyB.initialLin,
                InitialAngB = bodyB.initialAng,
                NumContacts = numContacts,
                Contacts = contacts,
                Basis = basis,
                Alpha = alpha,
                Friction = friction
            }.Run();
        }
        
        [BurstCompile]
        private struct UpdatePrimalJob : IJob
        {
            public float3 PositionLinA;
            public float3 PositionLinB;
            public quaternion PositionAngA;
            public quaternion PositionAngB;
            public float3 InitialLinA;
            public float3 InitialLinB;
            public quaternion InitialAngA;
            public quaternion InitialAngB;
            public int NumContacts;
            public NativeArray<Contact> Contacts;
            public float3x3 Basis;
            public float Alpha;
            public float Friction;
            public bool IsBodyA;
            public NativeReference<Equation6> Eq6;

            public void Execute()
            {
                float3 dqALin = PositionLinA - InitialLinA;
                float3 dqAAng = PositionAngA.sub(InitialAngA);
                float3 dqBLin = PositionLinB - InitialLinB;
                float3 dqBAng = PositionAngB.sub(InitialAngB);

                for (int i = 0; i < NumContacts; i++)
                {
                    float3 rAWorld = rotate(PositionAngA, Contacts[i].rA);
                    float3 rBWorld = rotate(PositionAngB, Contacts[i].rB);

                    // Compute the Taylor series approximation of the constraint function C(x) (Sec 4)
                    float3x3 jALin = Basis;
                    float3x3 jBLin = -Basis;
                    float3x3 jAAng = mul(jALin, Utils.skew(-rAWorld));
                    float3x3 jBAng = mul(jBLin, Utils.skew(-rBWorld));

                    float3x3 k = Utils.diagonal(Contacts[i].penalty.x, Contacts[i].penalty.y, Contacts[i].penalty.z);
                    float3 c = Contacts[i].C0 * (1 - Alpha) + mul(jALin, dqALin) + mul(jBLin, dqBLin) +
                               mul(jAAng, dqAAng)
                               + mul(jBAng, dqBAng);

                    // Compute force
                    float3 f = mul(k, c) + Contacts[i].lambda;

                    // Clamp normal force
                    f[0] = min(f[0], 0.0f);

                    // Clamp norm of friction forces to achieve a friction cone
                    float bounds = abs(f[0]) * Friction;
                    float frictionScale = length(f.yz);
                    if (frictionScale > bounds && frictionScale > 0)
                    {
                        f[1] *= bounds / frictionScale;
                        f[2] *= bounds / frictionScale;
                    }

                    // Choose jacobian depending on input body
                    float3x3 jLin = IsBodyA ? jALin : jBLin;
                    float3x3 jAng = IsBodyA ? jAAng : jBAng;

                    // Stamp into LHS
                    float3x3 jLinT = transpose(jLin);
                    float3x3 jAngT = transpose(jAng);
                    float3x3 jAngTk = mul(jAngT, k);

                    var e = Eq6.Value;
                    e.lhsLin += mul(mul(jLinT, k), jLin);
                    e.lhsAng += mul(jAngTk, jAng);
                    e.lhsCross += mul(jAngTk, jLin);

                    // Stamp into RHS
                    e.rhsLin += mul(jLinT, f);
                    e.rhsAng += mul(jAngT, f);
                    Eq6.Value = e;
                }
            }
        }
        
        [BurstCompile]
        private struct UpdateDualJob : IJob
        {
            public float3 PositionLinA;
            public float3 PositionLinB;
            public quaternion PositionAngA;
            public quaternion PositionAngB;
            public float3 InitialLinA;
            public float3 InitialLinB;
            public quaternion InitialAngA;
            public quaternion InitialAngB;
            public int NumContacts;
            public NativeArray<Contact> Contacts;
            public float3x3 Basis;
            public float Alpha;
            public float Friction;
            
            public void Execute()
            {
                float3 dqALin = PositionLinA - InitialLinA;
                float3 dqAAng = PositionAngA.sub(InitialAngA);
                float3 dqBLin = PositionLinB - InitialLinB;
                float3 dqBAng = PositionAngB.sub(InitialAngB);

                for (int i = 0; i < NumContacts; i++)
                {
                    var c = Contacts[i];
                    float3 rAWorld = rotate(PositionAngA, c.rA);
                    float3 rBWorld = rotate(PositionAngB, c.rB);

                    // Compute the Taylor series approximation of the constraint function C(x) (Sec 4)
                    float3x3 jALin = Basis;
                    float3x3 jBLin = -Basis;
                    float3x3 jAAng = mul(jALin, Utils.skew(-rAWorld));
                    float3x3 jBAng = mul(jBLin, Utils.skew(-rBWorld));
                    
                    float3x3 K = Utils.diagonal(c.penalty.x, c.penalty.y, c.penalty.z);
                    float3 C = c.C0 * (1 - Alpha) + mul(jALin, dqALin) + mul(jBLin, dqBLin) + mul(jAAng, dqAAng) + mul(jBAng, dqBAng);

                    // Compute force
                    float3 F = mul(K, C) + c.lambda;

                    // Clamp normal force
                    F[0] = min(F[0], 0.0f);

                    // Clamp norm of friction forces to achieve a friction cone
                    float bounds = abs(F[0]) * Friction;
                    float frictionScale = length(F.yz);
                    if (frictionScale > bounds && frictionScale > 0)
                    {
                        F[1] *= bounds / frictionScale;
                        F[2] *= bounds / frictionScale;
                    }

                    // Store updated force
                    c.lambda = F;

                    // Update the penalty parameter and clamp to material stiffness if we are within the force bounds (Eq. 16)
                    if (F[0] < 0)
                        c.penalty[0] = min(c.penalty[0] + Solver.betaLin * abs(C[0]), Utils.PENALTY_MAX);
                    if (frictionScale <= bounds)
                    {
                        c.penalty[1] = min(c.penalty[1] + Solver.betaLin * abs(C[1]), Utils.PENALTY_MAX);
                        c.penalty[2] = min(c.penalty[2] + Solver.betaLin * abs(C[2]), Utils.PENALTY_MAX);
                        c.stick = length(float2(C[1], C[2])) < Utils.STICK_THRESH;
                    }

                    Contacts[i] = c;
                }
            }
        }
    }
}

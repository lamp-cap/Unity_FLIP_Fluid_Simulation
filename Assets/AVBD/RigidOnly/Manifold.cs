using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD.RigidOnly
{
    public struct Manifold
    {
        // Used to track contact features between frames
        public struct FeaturePair
        {
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
        public struct Contacts
        {
            private Contact _c0;
            private Contact _c1;
            private Contact _c2;
            private Contact _c3;
            private Contact _c4;
            private Contact _c5;
            private Contact _c6;
            private Contact _c7;

            public Contact this[int i]
            {
                get => i switch
                {
                    0 => _c0, 1 => _c1, 2 => _c2, 3 => _c3,
                    4 => _c4, 5 => _c5, 6 => _c6, _ => _c7
                };
                set
                {
                    switch (i)
                    {
                        case 0: _c0 = value; break;
                        case 1: _c1 = value; break;
                        case 2: _c2 = value; break;
                        case 3: _c3 = value; break;
                        case 4: _c4 = value; break;
                        case 5: _c5 = value; break;
                        case 6: _c6 = value; break;
                        default: _c7 = value; break;
                    }
                }
            }
        };

        private int bodyAID;
        private int bodyBID;
        public int BodyA => bodyAID - 1;
        public int BodyB => bodyBID - 1;
        public Contacts contacts;
        public float3x3 Basis; // Normal in the first row (pointing from B to A), and tangents in the second and third rows
        public int NumContacts;
        public float Friction;

        public Manifold(Rigid bodyA, Rigid bodyB)
        {
            bodyAID = bodyA.ID + 1;
            bodyBID = bodyB.ID + 1;
            Basis = new float3x3();
            contacts = new Contacts();
            NumContacts = 0;
            // Compute friction
            Friction = sqrt(bodyA.friction * bodyB.friction);
        }
        
        public bool initialize(OBB boxA, OBB boxB)
        {
            // Compute new contacts
            var newContacts = new Contacts();
            int newNumContacts = Collision.Collide(boxA, boxB, ref newContacts, out Basis);

            // Merge old contact data with new contacts
            for (int i = 0; i < newNumContacts; i++)
            for (int j = 0; j < NumContacts; j++)
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
            NumContacts = newNumContacts;
            for (int i = 0; i < NumContacts; i++)
                contacts[i] = newContacts[i];
            
            // Compute error at q- and update penalty and lambdas
            for (int i = 0; i < NumContacts; i++)
            {
                // Error at q-
                var c = contacts[i];
                float3 xA = Utils.transform(boxA.center, boxA.rotation, c.rA);
                float3 xB = Utils.transform(boxB.center, boxB.rotation, c.rB);
                c.C0 = mul(Basis, xA - xB) + float3(Utils.COLLISION_MARGIN, 0, 0);

                // Warmstart the dual variables and penalty parameters (Eq. 19)
                // Penalty is safely clamped to a minimum and maximum value
                c.lambda = c.lambda * Solver.alpha * Solver.gamma;
                c.penalty = clamp(c.penalty * Solver.gamma, Utils.PENALTY_MIN, Utils.PENALTY_MAX);
                contacts[i] = c;
            }

            return NumContacts > 0;
        }

        public void updatePrimal(int body, in Rigid bodyA, in Rigid bodyB, float alpha, ref Equation6 e)
        {
            float3 dqALin = bodyA.positionLin - bodyA.initialLin;
            float3 dqAAng = bodyA.positionAng.sub(bodyA.initialAng);
            float3 dqBLin = bodyB.positionLin - bodyB.initialLin;
            float3 dqBAng = bodyB.positionAng.sub(bodyB.initialAng);

            for (int i = 0; i < NumContacts; i++)
            {
                var ct = contacts[i];
                float3 rAWorld = rotate(bodyA.positionAng, ct.rA);
                float3 rBWorld = rotate(bodyB.positionAng, ct.rB);

                // Compute the Taylor series approximation of the constraint function C(x) (Sec 4)
                float3x3 jALin = Basis;
                float3x3 jBLin = -Basis;
                float3x3 jAAng = mul(jALin, Utils.skew(-rAWorld));
                float3x3 jBAng = mul(jBLin, Utils.skew(-rBWorld));

                float3x3 k = Utils.diagonal(ct.penalty.x, ct.penalty.y, ct.penalty.z);
                float3 c = ct.C0 * (1 - alpha) + mul(jALin, dqALin) + mul(jBLin, dqBLin) +
                           mul(jAAng, dqAAng)
                           + mul(jBAng, dqBAng);

                // Compute force
                float3 f = mul(k, c) + ct.lambda;

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
                float3x3 jLin = body == bodyA.ID ? jALin : jBLin;
                float3x3 jAng = body == bodyA.ID ? jAAng : jBAng;

                // Stamp into LHS
                float3x3 jLinT = transpose(jLin);
                float3x3 jAngT = transpose(jAng);
                float3x3 jAngTk = mul(jAngT, k);

                e.lhsLin += mul(mul(jLinT, k), jLin);
                e.lhsAng += mul(jAngTk, jAng);
                e.lhsCross += mul(jAngTk, jLin);

                // Stamp into RHS
                e.rhsLin += mul(jLinT, f);
                e.rhsAng += mul(jAngT, f);
            }
        }

        public void updateDual(in Rigid rigidA, in Rigid rigidB, float alpha)
        {
            float3 dqALin = rigidA.positionLin - rigidA.initialLin;
            float3 dqAAng = rigidA.positionAng.sub(rigidA.initialAng);
            float3 dqBLin = rigidB.positionLin - rigidB.initialLin;
            float3 dqBAng = rigidB.positionAng.sub(rigidB.initialAng);

            for (int i = 0; i < NumContacts; i++)
            {
                var c = contacts[i];
                float3 rAWorld = rotate(rigidA.positionAng, c.rA);
                float3 rBWorld = rotate(rigidB.positionAng, c.rB);

                // Compute the Taylor series approximation of the constraint function C(x) (Sec 4)
                float3x3 jALin = Basis;
                float3x3 jBLin = -Basis;
                float3x3 jAAng = mul(jALin, Utils.skew(-rAWorld));
                float3x3 jBAng = mul(jBLin, Utils.skew(-rBWorld));
                
                float3x3 K = Utils.diagonal(c.penalty.x, c.penalty.y, c.penalty.z);
                float3 C = c.C0 * (1 - alpha) + mul(jALin, dqALin) + mul(jBLin, dqBLin) + mul(jAAng, dqAAng) + mul(jBAng, dqBAng);

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

                contacts[i] = c;
            }
        }
    }
}

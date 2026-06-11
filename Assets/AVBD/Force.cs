using Unity.Collections;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD
{
    public abstract class Force : System.IDisposable
    {
        public Rigid bodyA;
        public Rigid bodyB;

        public Force(Solver solver, Rigid bodyA, Rigid bodyB)
        {
            this.bodyA = bodyA;
            this.bodyB = bodyB;
            // Add to solver linked list
            solver.forces.Add(this);

            // Add to body linked lists
            bodyA?.forces.Add(this);
            bodyB?.forces.Add(this);
        }

        public abstract bool initialize();
        public abstract void updatePrimal(Rigid body, float alpha, NativeReference<Equation6> eq6);
        public abstract void updateDual(float alpha);

        public virtual void Dispose()
        {
            // Remove from body linked lists
            bodyA?.forces.Remove(this);
            bodyB?.forces.Remove(this);
        }
    }
    
    public class Joint : Force
    {
        public float3 rA, rB;
        public float3 C0Lin, C0Ang;
        public float3 penaltyLin, penaltyAng;
        public float3 lambdaLin, lambdaAng;
        public float stiffnessLin, stiffnessAng, fracture;
        public float torqueArm;
        public bool broken;

        private static float3x3 geometricStiffnessBallSocket(int k, float3 v)
        {
            float3x3 m = Utils.diagonal(-v[k], -v[k], -v[k]);

            m[k][0] += v[0];
            m[k][1] += v[1];
            m[k][2] += v[2];

            return m;
        }

        public Joint(Solver solver, Rigid bodyA, Rigid bodyB, float3 rA, float3 rB, 
            float stiffnessLin = float.PositiveInfinity, float stiffnessAng = 0.0f, float fracture = float.PositiveInfinity)
            : base(solver, bodyA, bodyB)
        {
            this.rA = rA;
            this.rB = rB;
            this.stiffnessLin = stiffnessLin;
            this.stiffnessAng = stiffnessAng;
            this.fracture = fracture;
            broken = false;
            C0Lin = C0Ang = penaltyLin = penaltyAng = lambdaLin = lambdaAng = float3(0, 0, 0);
            torqueArm = lengthsq((bodyA != null ? bodyA.size : float3(0, 0, 0)) + bodyB.size);
        }

        public override bool initialize()
        {
            // Store constraint function at beginnning of timestep C(x-)
            // Note: if bodyA is null, it is assumed that the joint connects a body to the world space position rA
            C0Lin = (bodyA !=null ? Utils.transform(bodyA.positionLin, bodyA.positionAng, rA) : rA)
                    - Utils.transform(bodyB.positionLin, bodyB.positionAng, rB);
            C0Ang = ((bodyA != null ? bodyA.positionAng : quaternion(0,0,0,1)).sub(bodyB.positionAng)) * torqueArm;

            // Warmstart the dual variables and penalty parameters (Eq. 19)
            // Penalty is safely clamped to a minimum and maximum value
            lambdaLin = lambdaLin * Solver.alpha * Solver.gamma;
            lambdaAng = lambdaAng * Solver.alpha * Solver.gamma;
            penaltyLin = clamp(penaltyLin * Solver.gamma, Utils.PENALTY_MIN, Utils.PENALTY_MAX);
            penaltyAng = clamp(penaltyAng * Solver.gamma, Utils.PENALTY_MIN, Utils.PENALTY_MAX);

            // Clamp penalty to material stiffness
            penaltyLin = min(penaltyLin, stiffnessLin);
            penaltyAng = min(penaltyAng, stiffnessAng);

            return !broken;
        }

        public override void updatePrimal(Rigid body, float alpha, NativeReference<Equation6> eq6)
        {
            var e = eq6.Value;
            // Linear constraint
            if (lengthsq(penaltyLin) > 0)
            {
                // Compute constraint and jacobians
                float3x3 K = Utils.diagonal(penaltyLin.x, penaltyLin.y, penaltyLin.z);
                float3 C = (bodyA !=null ? Utils.transform(bodyA.positionLin, bodyA.positionAng, rA) : rA)
                           - Utils.transform(bodyB.positionLin, bodyB.positionAng, rB);
                
                // Stabilization
                if (isinf(stiffnessLin))
                    C -= C0Lin * alpha;

                // Compute force
                float3 F = mul(K, C) + lambdaLin;

                // Choose jacobian depending on input body
                float3x3 jLin = body == bodyA ? float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1) : float3x3(-1, 0, 0, 0, -1, 0, 0, 0, -1);
                float3x3 jAng = body == bodyA ? Utils.skew(-rotate(bodyA.positionAng, rA)) : Utils.skew(rotate(bodyB.positionAng, rB));

                // Stamp into LHS
                float3x3 jLinT = transpose(jLin);
                float3x3 jAngT = transpose(jAng);
                float3x3 jAngTk = mul(jAngT, K);

                e.lhsLin += mul(mul(jLinT, K), jLin);
                e.lhsAng += mul(jAngTk, jAng);
                e.lhsCross += mul(jAngTk, jLin);

                // Diagonal approximation for higher order terms
                float3 r = body == bodyA ? rotate(bodyA.positionAng, rA) : -rotate(bodyB.positionAng, rB);
                float3x3 H = 
                    geometricStiffnessBallSocket(0, r) * F[0] +
                    geometricStiffnessBallSocket(1, r) * F[1] +
                    geometricStiffnessBallSocket(2, r) * F[2];
                e.lhsAng += Utils.diagonalize(H);

                // Stamp into RHS
                e.rhsLin += mul(jLinT, F);
                e.rhsAng += mul(jAngT, F);
            }

            // Angular constraint
            if (lengthsq(penaltyAng) > 0)
            {
                // Compute constraint and jacobians
                float3x3 K = Utils.diagonal(penaltyAng.x, penaltyAng.y, penaltyAng.z);
                float3 C = ((bodyA != null ? bodyA.positionAng : quaternion(0,0,0,1)).sub(bodyB.positionAng)) * torqueArm;

                // Stabilization
                if (isinf(stiffnessAng))
                    C -= C0Ang * alpha;

                // Compute force
                float3 F = mul(K, C) + lambdaAng;

                // Choose jacobian depending on input body
                float3x3 jAng = (body == bodyA ? float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1) : float3x3(-1, 0, 0, 0, -1, 0, 0, 0, -1)) * torqueArm;

                // Stamp into LHS
                e.lhsAng += mul(mul(transpose(jAng), K), jAng);

                // Stamp into RHS
                e.rhsAng += mul(transpose(jAng), F);
            }
            eq6.Value = e;
        }

        public override void updateDual(float alpha)
        {
            // Linear constraint
            if (lengthsq(penaltyLin) > 0)
            {
                // Compute constraint and jacobians
                float3x3 K = Utils.diagonal(penaltyLin.x, penaltyLin.y, penaltyLin.z);
                float3 C = (bodyA != null ? Utils.transform(bodyA.positionLin, bodyA.positionAng, rA) : rA) - Utils.transform(bodyB.positionLin, bodyB.positionAng, rB);

                if (isinf(stiffnessLin))
                {
                    // Stabilization
                    C -= C0Lin * alpha;

                    // Compute force
                    float3 F = mul(K, C) + lambdaLin;

                    // Store updated force
                    lambdaLin = F;
                }

                // Update the penalty parameter and clamp to material stiffness if we are within the force bounds (Eq. 16)
                penaltyLin = min(penaltyLin + abs(C) * Solver.betaLin, min(stiffnessLin, Utils.PENALTY_MAX));
            }

            // Angular constraint
            if (lengthsq(penaltyAng) > 0)
            {
                // Compute constraint and jacobians
                float3x3 K = Utils.diagonal(penaltyAng.x, penaltyAng.y, penaltyAng.z);
                float3 C = ((bodyA != null ? bodyA.positionAng : quaternion(0,0,0,1)).sub(bodyB.positionAng)) * torqueArm;

                if (isinf(stiffnessAng))
                {
                    // Stabilization
                    C -= C0Ang * alpha;

                    // Compute force
                    float3 F = mul(K, C) + lambdaAng;

                    // Store updated force
                    lambdaAng = F;
                }

                // Update the penalty parameter and clamp to material stiffness if we are within the force bounds (Eq. 16)
                penaltyAng = min(penaltyAng + abs(C) * Solver.betaAng, min(stiffnessAng, Utils.PENALTY_MAX));
            }

            // Fracture test
            if (lengthsq(lambdaAng) > fracture * fracture)
            {
                penaltyLin = float3(0);
                penaltyAng = float3(0);
                lambdaLin = float3(0);
                lambdaAng = float3(0);
                broken = true;
            }
        }
    };
    
    public class Spring : Force
    {
        public float3 rA, rB;
        public float rest;
        public float stiffness;

        public Spring(Solver solver, Rigid bodyA, Rigid bodyB, float3 rA, float3 rB, float stiffness, float rest = -1)
        : base(solver, bodyA, bodyB)
        {
            this.rA = rA;
            this.rB = rB;
            this.rest = rest;
            this.stiffness = stiffness;
            if (this.rest < 0.0f)
            {
                float3 pA = Utils.transform(bodyA.positionLin, bodyA.positionAng, this.rA);
                float3 pB = Utils.transform(bodyB.positionLin, bodyB.positionAng, this.rB);
                this.rest = length(pA - pB);
            }
            
        }

        public override bool initialize()
        {
            return true;
        }

        public override void updatePrimal(Rigid body, float alpha, NativeReference<Equation6>  eq6)
        {

            float3 pA = Utils.transform(bodyA.positionLin, bodyA.positionAng, rA);
            float3 pB = Utils.transform(bodyB.positionLin, bodyB.positionAng, rB);
            float3 d = pA - pB;
            float dLen = length(d);
            if (dLen <= 1.0e-6f)
                return;

            float3 n = d / dLen;
            float C = dLen - rest;
            float f = stiffness * C;

            float3 rWorld;
            float3 jLin;
            float3 jAng;
            if (body == bodyA)
            {
                rWorld = rotate(bodyA.positionAng, rA);
                jLin = n;
                jAng = cross(rWorld, n);
            }
            else
            {
                rWorld = rotate(bodyB.positionAng, rB);
                jLin = -n;
                jAng = -cross(rWorld, n);
            }

            float3 F = jLin * f;
            float3 Tau = jAng * f;
            float3x3 Kll = Utils.outer(jLin, jLin) * stiffness;
            float3x3 Kla = Utils.outer(jAng, jLin) * stiffness;
            float3x3 Kaa = Utils.outer(jAng, jAng) * stiffness;

            var e = eq6.Value;
            e.lhsLin += Kll;
            e.lhsAng += Kaa;
            e.lhsCross += Kla;
            e.rhsLin += F;
            e.rhsAng += Tau;
            eq6.Value = e;
        }

        public override void updateDual(float alpha)
        {
            
        }
    };

    // Force which has no physical effect, but is used to ignore collisions between two bodies
    public class IgnoreCollision : Force
    {
        public IgnoreCollision(Solver solver, Rigid bodyA, Rigid bodyB)
            : base(solver, bodyA, bodyB)
        {
            
        }

        public override bool initialize(){ return true; }

        public override void updatePrimal(Rigid body, float alpha, NativeReference<Equation6>  eq6)
        {
            
        }
        public override void updateDual(float alpha){}
    }
}

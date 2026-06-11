using System.Collections.Generic;
using Unity.Mathematics;

namespace AVBD
{
    // Holds all the state for a single rigid body that is needed by AVBD
    public class Rigid
    {
        public List<Force> forces;
        public float3 positionLin;
        public quaternion positionAng;
        public float3 initialLin;
        public quaternion initialAng;
        public float3 inertialLin;
        public quaternion inertialAng;
        public float3 velocityLin;
        public float3 velocityAng;
        public float3 prevVelocityLin;
        public float3 size; // Full widths in each dimension
        public float mass;
        public float3 moment;
        public float friction;
        public float radius;

        public Rigid(Solver solver, float3 size, float density, float friction, float3 position)
            : this(solver, size, density, friction, position, float3.zero){}

        public Rigid(Solver solver, float3 size, float density, float friction, float3 position, float3 velocity)
        {
            this.size = size;
            this.friction = friction;
            positionLin = position;
            velocityLin = velocity;
            positionAng = quaternion.identity;
            velocityAng = float3.zero;
            // Add to linked list

            // Compute mass properties and bounding radius
            mass = size.x * size.y * size.z * density;
            moment = new float3(
                (size.y * size.y + size.z * size.z) / 12.0f * mass,
                (size.x * size.x + size.z * size.z) / 12.0f * mass,
                (size.x * size.x + size.y * size.y) / 12.0f * mass
            );
            radius = math.length(size * 0.5f);
            
            solver.bodies.Add(this);
            forces = new List<Force>();
        }

        public bool constrainedTo(Rigid other)
        {
            // Check if this body is constrained to the other body
            foreach (Force f in forces)
                if ((f.bodyA == this && f.bodyB == other) || (f.bodyA == other && f.bodyB == this))
                    return true;
            return false;
            
        }
    }
}

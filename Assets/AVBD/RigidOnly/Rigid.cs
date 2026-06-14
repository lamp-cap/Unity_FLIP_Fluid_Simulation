using Unity.Mathematics;

namespace AVBD.RigidOnly
{
    // Holds all the state for a single rigid body that is needed by AVBD
    public struct Rigid
    {
        private readonly int _id;
        public int ID => _id - 1;
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

        public bool IsCreate => _id > 0;
        
        public Rigid(int id, float3 size, float density, float friction, float3 position, float3 velocity)
        {
            _id = id;
            this.size = size;
            this.friction = friction;
            positionLin = position;
            inertialLin = position;
            initialLin = position;
            velocityLin = velocity;
            positionAng = quaternion.identity;
            inertialAng = quaternion.identity;
            initialAng = quaternion.identity;
            velocityAng = float3.zero;
            prevVelocityLin = velocityLin;
            // Add to linked list

            // Compute mass properties and bounding radius
            mass = size.x * size.y * size.z * density;
            moment = new float3(
                (size.y * size.y + size.z * size.z) / 12.0f * mass,
                (size.x * size.x + size.z * size.z) / 12.0f * mass,
                (size.x * size.x + size.y * size.y) / 12.0f * mass
            );
            radius = math.length(size * 0.5f);
        }
    }
}

using Unity.Collections;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD.Cloth
{
    /// <summary>
    /// 解析碰撞体(胶囊 / 球 / 无限平面)。
    /// 参考实现里布料-刚体碰撞走的是 collider mesh + BVH;
    /// 这里按 Prop.md 第二节"添加一个函数处理胶囊体、球体这种可解析的碰撞体"的要求,
    /// 用解析最近点 + 法向排斥力直接处理,避免给静态刚体也建 BVH。
    ///
    /// 统一接口:给定查询点 p,返回 (该点到表面的有符号外推信息):
    ///   - closest: 表面最近点
    ///   - normal : 由表面指向 p 的单位法向(p 在外侧时指向 p)
    ///   - dist   : p 到表面的距离(p 在内部为负)
    /// 求解器据此施加排斥(penalty)力 + 摩擦,等价于参考的
    /// computeContactRepulsiveForce / computeFriction。
    /// </summary>
    public enum ColliderType { Sphere = 0, Capsule = 1, Plane = 2 }

    public struct AnalyticCollider
    {
        public ColliderType Type;

        // Sphere: Center=A, Radius
        // Capsule: 线段 A-B, Radius(横放胶囊体即 A,B 沿某轴展开)
        // Plane: 过点 A,法向 Normal(单位)
        public float3 A;
        public float3 B;
        public float3 Normal;
        public float Radius;

        public float FrictionDynamic;
        public float FrictionEpsV;

        public static AnalyticCollider MakeSphere(float3 center, float radius, float friction = 0.3f, float epsV = 1e-3f)
            => new AnalyticCollider { Type = ColliderType.Sphere, A = center, Radius = radius, FrictionDynamic = friction, FrictionEpsV = epsV };

        public static AnalyticCollider MakeCapsule(float3 a, float3 b, float radius, float friction = 0.3f, float epsV = 1e-3f)
            => new AnalyticCollider { Type = ColliderType.Capsule, A = a, B = b, Radius = radius, FrictionDynamic = friction, FrictionEpsV = epsV };

        public static AnalyticCollider MakePlane(float3 point, float3 normal, float friction = 0.3f, float epsV = 1e-3f)
            => new AnalyticCollider { Type = ColliderType.Plane, A = point, Normal = normalizesafe(normal), FrictionDynamic = friction, FrictionEpsV = epsV };

        /// <summary>
        /// 查询点 p 到碰撞体表面的最近点 / 法向 / 距离。
        /// dist > 0:在外侧;dist < 0:已穿透。
        /// </summary>
        public void Query(float3 p, out float3 closest, out float3 normal, out float dist)
        {
            switch (Type)
            {
                case ColliderType.Sphere:
                {
                    float3 d = p - A;
                    float len = length(d);
                    normal = len > 1e-8f ? d / len : new float3(0, 1, 0);
                    closest = A + normal * Radius;
                    dist = len - Radius;
                    break;
                }
                case ColliderType.Capsule:
                {
                    // p 到线段 AB 的最近点
                    float3 ab = B - A;
                    float t = dot(p - A, ab) / max(dot(ab, ab), 1e-12f);
                    t = saturate(t);
                    float3 onAxis = A + t * ab;
                    float3 d = p - onAxis;
                    float len = length(d);
                    normal = len > 1e-8f ? d / len : new float3(0, 1, 0);
                    closest = onAxis + normal * Radius;
                    dist = len - Radius;
                    break;
                }
                default: // Plane
                {
                    dist = dot(p - A, Normal);
                    normal = Normal;
                    closest = p - Normal * dist;
                    break;
                }
            }
        }
    }

    /// <summary>持有一组解析碰撞体的容器。</summary>
    public class ColliderSet
    {
        public NativeArray<AnalyticCollider> Colliders;
        public int Count;
        private bool _allocated;

        public void Set(AnalyticCollider[] colliders)
        {
            Dispose();
            Count = colliders.Length;
            Colliders = new NativeArray<AnalyticCollider>(max(Count, 1), Allocator.Persistent);
            for (int i = 0; i < Count; i++) Colliders[i] = colliders[i];
            _allocated = true;
        }

        public void Dispose()
        {
            if (_allocated && Colliders.IsCreated) Colliders.Dispose();
            _allocated = false;
        }
    }
}

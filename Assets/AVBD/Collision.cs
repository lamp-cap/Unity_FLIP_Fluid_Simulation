using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace AVBD
{
    public class Collision
    {
        const int MAX_CONTACTS = 8;
        const int MAX_POLY_VERTS = 16;
        const float SAT_AXIS_EPSILON = 1.0e-6f;
        const float PLANE_EPSILON = 1.0e-5f;
        const float CONTACT_MERGE_DIST_SQ = 1.0e-6f;

        enum AxisType
        {
            AXIS_FACE_A = 0,
            AXIS_FACE_B = 1,
            AXIS_EDGE = 2
        };

        struct SatAxis
        {
            public AxisType type;
            public int indexA;
            public int indexB;
            public float separation;
            public float3 normalAB;
            public bool valid;
        };

        struct FaceFrame
        {
            public int axisIndex;
            public float3 normal;
            public float3 center;
            public float3 u;
            public float3 v;
            public float extentU;
            public float extentV;
        };

        private NativeReference<CollideResult> _result;
        private NativeReference<bool> _bool;

        public Collision()
        {
            _result = new NativeReference<CollideResult>(Allocator.Persistent);
            _bool = new NativeReference<bool>(Allocator.Persistent);
        }

        public void Dispose()
        {
            _result.Dispose();
            _bool.Dispose();
        }

        private static bool TestAxis(in OBB boxA, in OBB boxB, in float3 delta, in float3 axis)
        {
            float lenSq = lengthsq(axis);
            if (lenSq < SAT_AXIS_EPSILON)
                return true;

            float3 n = axis * rsqrt(lenSq);
            if (dot(n, delta) < 0.0f)
                n = -n;

            float distance = abs(dot(delta, n));

            float rA = dot(boxA.half, abs(mul(n, boxA.axis)));
            float rB = dot(boxB.half, abs(mul(n, boxB.axis)));

            float separation = distance - (rA + rB);
            return separation <= 0.0f;
        }

        public bool Collide(Rigid bodyA, Rigid bodyB)
        {
            new CollideJob(bodyA, bodyB, _bool).Run();
            return _bool.Value;
        }

        [BurstCompile]
        struct CollideJob : IJob
        {
            private OBB _boxA;
            private OBB _boxB;
            private NativeReference<bool> _result;
            
            public CollideJob(Rigid a, Rigid b, NativeReference<bool> result)
            {
                _boxA = OBB.makeOBB(a);
                _boxB = OBB.makeOBB(b);
                result.Value = false;
                _result = result;
            }
            
            public void Execute()
            {
                float3 delta = _boxB.center - _boxA.center;
            
                for (int i = 0; i < 3; ++i)
                    if (!TestAxis(_boxA, _boxB, delta, _boxA.axis[i]))
                        return;

                for (int i = 0; i < 3; ++i)
                    if (!TestAxis(_boxA, _boxB, delta, _boxB.axis[i]))
                        return;

                for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    float3 axis = cross(_boxA.axis[i], _boxB.axis[j]);
                    if (!TestAxis(_boxA, _boxB, delta, axis))
                        return;
                }

                _result.Value = true;
            }
        }
        
        public int Collide(Rigid bodyA, Rigid bodyB, NativeArray<Manifold.Contact> contacts, out float3x3 basisOut)
        {
            basisOut = new float3x3();
            OBB boxA = OBB.makeOBB(bodyA);
            OBB boxB = OBB.makeOBB(bodyB);
            
            new SatJob(boxA, boxB, _result).Run();
            var r = _result.Value;

            if (!r.collided)
            {
                return 0;
            }

            basisOut = Utils.orthonormal(-r.best.normalAB);

            int contactCount = r.best.type switch
            {
                AxisType.AXIS_EDGE => buildEdgeContact(bodyA, bodyB, boxA, boxB, r.best.indexA, r.best.indexB,
                    r.best.normalAB, contacts),
                AxisType.AXIS_FACE_A => buildFaceManifold(bodyA, bodyB, boxA, boxB, true, r.best.indexA, r.best.normalAB,
                    contacts),
                _ => buildFaceManifold(bodyA, bodyB, boxA, boxB, false, r.best.indexB, r.best.normalAB, contacts)
            };
            return contactCount;
        }

        private static float3 supportPoint(in OBB box, in float3 dir)
        {
            var s = select(-box.half, box.half, mul(dir, box.axis) >= 0.0f);

            return box.center
                   + box.axis[0] * s.x
                   + box.axis[1] * s.y
                   + box.axis[2] * s.z;
        }

        private static void getFaceAxes(in OBB box, int axisIndex, out float3 u, out float3 v, out float extentU, out float extentV)
        {
            int a = axisIndex > 0 ? 0 : 1;
            int b = axisIndex > 1 ? 1 : 2;
            u = box.axis[a];
            v = box.axis[b];
            extentU = box.half[a];
            extentV = box.half[b];
        }

        private static void buildFaceFrame(in OBB box, int axisIndex,in float3 outwardNormal, out FaceFrame frame)
        {
            float sign = dot(outwardNormal, box.axis[axisIndex]) >= 0.0f ? 1.0f : -1.0f;
            frame.axisIndex = axisIndex;
            frame.normal = box.axis[axisIndex] * sign;
            frame.center = box.center + frame.normal * box.half[axisIndex];
            getFaceAxes(box, axisIndex, out frame.u, out frame.v, out frame.extentU, out frame.extentV);
        }

        private static int chooseIncidentFaceAxis(in OBB box, in float3 referenceNormal)
        {
            int axis = 0;
            float best = -float.MaxValue;

            for (int i = 0; i < 3; ++i)
            {
                float d = abs(dot(box.axis[i], referenceNormal));
                if (d > best)
                {
                    best = d;
                    axis = i;
                }
            }

            return axis;
        }

        private static void buildIncidentFace(in OBB box, int axisIndex, in float3 referenceNormal, NativeArray<float3> outVerts)
        {
            float sign = dot(box.axis[axisIndex], referenceNormal) > 0.0f ? -1.0f : 1.0f;
            float3 faceNormal = box.axis[axisIndex] * sign;
            float3 faceCenter = box.center + faceNormal * box.half[axisIndex];

            getFaceAxes(box, axisIndex, out var u, out var v, out var extentU, out var extentV);

            u *= extentU;
            v *= extentV;
            
            outVerts[0] = faceCenter + u + v;
            outVerts[1] = faceCenter - u + v;
            outVerts[2] = faceCenter - u - v;
            outVerts[3] = faceCenter + u - v;
        }

        private static int clipPolygonAgainstPlane(in NativeArray<float3> inVerts, int inCount, in float3 planeNormal, float planeOffset, NativeArray<float3> outVerts)
        {
            if (inCount <= 0)
                return 0;

            int outCount = 0;
            float3 a = inVerts[inCount - 1];
            float da = dot(planeNormal, a) - planeOffset;

            for (int i = 0; i < inCount; ++i)
            {
                float3 b = inVerts[i];
                float db = dot(planeNormal, b) - planeOffset;

                bool aInside = da <= PLANE_EPSILON;
                bool bInside = db <= PLANE_EPSILON;

                if (aInside != bInside)
                {
                    float t = 0.0f;
                    float denom = da - db;
                    if (abs(denom) > SAT_AXIS_EPSILON)
                        t = saturate(da / denom);

                    if (outCount < MAX_POLY_VERTS)
                        outVerts[outCount++] = a + (b - a) * t;
                }

                if (bInside && outCount < MAX_POLY_VERTS)
                    outVerts[outCount++] = b;

                a = b;
                da = db;
            }

            return outCount;
        }

        private static bool addContact(Rigid bodyA, Rigid bodyB, NativeArray<Manifold.Contact> contacts, ref int contactCount,
            NativeArray<float3> contactMidpoints, float3 xA, float3 xB, int featureKey)
        {
            float3 midpoint = (xA + xB) * 0.5f;

            for (int i = 0; i < contactCount; ++i)
            {
                float3 d = midpoint - contactMidpoints[i];
                if (lengthsq(d) < CONTACT_MERGE_DIST_SQ)
                    return false;
            }

            if (contactCount >= MAX_CONTACTS)
                return false;

            Manifold.FeaturePair feature;
            feature.key = featureKey;

            Manifold.Contact c = contacts[contactCount];
            c.feature = feature;
            c.rA = rotate(conjugate(bodyA.positionAng), xA - bodyA.positionLin);
            c.rB = rotate(conjugate(bodyB.positionAng), xB - bodyB.positionLin);
            contacts[contactCount] = c;
            contactMidpoints[contactCount] = midpoint;
            ++contactCount;

            return true;
        }

        private static bool testAxis(in OBB boxA, in OBB boxB, in float3 delta, in float3 axis, AxisType type,
            int indexA, int indexB, ref SatAxis best)
        {
            float lenSq = lengthsq(axis);
            if (lenSq < SAT_AXIS_EPSILON)
                return true;

            float3 n = axis * rsqrt(lenSq);
            if (dot(n, delta) < 0.0f)
                n = -n;

            float distance = abs(dot(delta, n));

            float rA = dot(boxA.half, abs(mul(n, boxA.axis)));
            float rB = dot(boxB.half, abs(mul(n, boxB.axis)));

            float separation = distance - (rA + rB);
            if (separation > 0.0f)
                return false;

            if (!best.valid || separation > best.separation)
            {
                best.valid = true;
                best.type = type;
                best.indexA = indexA;
                best.indexB = indexB;
                best.separation = separation;
                best.normalAB = n;
            }

            return true;
        }

        private static void supportEdge(in OBB box, int axisIndex, in float3 dir, out float3 edgeA, out float3 edgeB)
        {
            int axis1 = (axisIndex + 1) % 3;
            int axis2 = (axisIndex + 2) % 3;

            float sign1 = dot(dir, box.axis[axis1]) >= 0.0f ? 1.0f : -1.0f;
            float sign2 = dot(dir, box.axis[axis2]) >= 0.0f ? 1.0f : -1.0f;

            float3 edgeCenter = box.center
                                + box.axis[axis1] * (box.half[axis1] * sign1)
                                + box.axis[axis2] * (box.half[axis2] * sign2);

            edgeA = edgeCenter - box.axis[axisIndex] * box.half[axisIndex];
            edgeB = edgeCenter + box.axis[axisIndex] * box.half[axisIndex];
        }

        private static void closestPointsOnSegments(in float3 p0, in float3 p1, in float3 q0, in float3 q1, out float3 c0, out float3 c1)
        {
            float3 d1 = p1 - p0;
            float3 d2 = q1 - q0;
            float3 r = p0 - q0;
            float a = dot(d1, d1);
            float e = dot(d2, d2);
            float f = dot(d2, r);

            float s = 0.0f;
            float t = 0.0f;

            if (a <= SAT_AXIS_EPSILON && e <= SAT_AXIS_EPSILON)
            {
                c0 = p0;
                c1 = q0;
                return;
            }

            if (a <= SAT_AXIS_EPSILON)
            {
                t = saturate(f / e);
            }
            else
            {
                float c = dot(d1, r);
                if (e <= SAT_AXIS_EPSILON)
                {
                    s = saturate(-c / a);
                }
                else
                {
                    float b = dot(d1, d2);
                    float denom = a * e - b * b;

                    if (abs(denom) > SAT_AXIS_EPSILON)
                        s = saturate((b * f - c * e) / denom);

                    t = (b * s + f) / e;

                    if (t < 0.0f)
                    {
                        t = 0.0f;
                        s = saturate(-c / a);
                    }
                    else if (t > 1.0f)
                    {
                        t = 1.0f;
                        s = saturate((b - c) / a);
                    }
                }
            }

            c0 = p0 + d1 * s;
            c1 = q0 + d2 * t;
        }

        private static int buildFaceManifold(Rigid bodyA, Rigid bodyB, in OBB boxA, in OBB boxB,
            bool referenceIsA, int referenceAxis, float3 normalAB, NativeArray<Manifold.Contact> contacts)
        {
            OBB referenceBox = referenceIsA ? boxA : boxB;
            OBB incidentBox = referenceIsA ? boxB : boxA;
            float3 referenceOutward = referenceIsA ? normalAB : -normalAB;

            FaceFrame referenceFace;
            buildFaceFrame(referenceBox, referenceAxis, referenceOutward, out referenceFace);

            int incidentAxis = chooseIncidentFaceAxis(incidentBox, referenceFace.normal);

            var clip0 = new NativeArray<float3>(MAX_POLY_VERTS, Allocator.Temp);
            var clip1 = new NativeArray<float3>(MAX_POLY_VERTS, Allocator.Temp);
            buildIncidentFace(incidentBox, incidentAxis, referenceFace.normal, clip0);
            int count = 4;

            float3 n0 = referenceFace.u;
            float o0 = dot(n0, referenceFace.center) + referenceFace.extentU;
            count = clipPolygonAgainstPlane(clip0, count, n0, o0, clip1);
            if (count == 0)
                return 0;

            float3 n1 = -referenceFace.u;
            float o1 = dot(n1, referenceFace.center) + referenceFace.extentU;
            count = clipPolygonAgainstPlane(clip1, count, n1, o1, clip0);
            if (count == 0)
                return 0;

            float3 n2 = referenceFace.v;
            float o2 = dot(n2, referenceFace.center) + referenceFace.extentV;
            count = clipPolygonAgainstPlane(clip0, count, n2, o2, clip1);
            if (count == 0)
                return 0;

            float3 n3 = -referenceFace.v;
            float o3 = dot(n3, referenceFace.center) + referenceFace.extentV;
            count = clipPolygonAgainstPlane(clip1, count, n3, o3, clip0);
            if (count == 0)
                return 0;

            int contactCount = 0;
            var contactMidpoints = new NativeArray<float3>(MAX_CONTACTS, Allocator.Temp);
            int featurePrefix = (referenceIsA ? (int)AxisType.AXIS_FACE_A : (int)AxisType.AXIS_FACE_B) << 24;
            featurePrefix |= (referenceAxis & 0xFF) << 16;
            featurePrefix |= (incidentAxis & 0xFF) << 8;

            for (int i = 0; i < count && contactCount < MAX_CONTACTS; ++i)
            {
                float3 pIncident = clip0[i];
                float distance = dot(pIncident - referenceFace.center, referenceFace.normal);
                if (distance > PLANE_EPSILON)
                    continue;

                float3 pReference = pIncident - referenceFace.normal * distance;
                float3 xA = referenceIsA ? pReference : pIncident;
                float3 xB = referenceIsA ? pIncident : pReference;

                addContact(bodyA, bodyB, contacts, ref contactCount, contactMidpoints, xA, xB, featurePrefix | (i & 0xFF));
            }

            if (contactCount == 0)
            {
                float3 xA = supportPoint(boxA, normalAB);
                float3 xB = supportPoint(boxB, -normalAB);
                addContact(bodyA, bodyB, contacts, ref contactCount, contactMidpoints, xA, xB, featurePrefix);
            }

            return contactCount;
        }

        private static int buildEdgeContact(Rigid bodyA, Rigid bodyB, in OBB boxA, in OBB boxB, int axisA, int axisB,
            float3 normalAB, NativeArray<Manifold.Contact> contacts)
        {
            supportEdge(boxA, axisA, normalAB, out var a0, out var a1);
            supportEdge(boxB, axisB, -normalAB, out var b0, out var b1);

            float3 xA;
            float3 xB;
            closestPointsOnSegments(a0, a1, b0, b1, out xA,  out xB);

            int contactCount = 0;
            var contactMidpoints= new NativeArray<float3>(MAX_CONTACTS,Allocator.Temp);
            int featureKey = ((int)AxisType.AXIS_EDGE << 24) | ((axisA & 0xFF) << 8) | (axisB & 0xFF);
            addContact(bodyA, bodyB, contacts, ref contactCount, contactMidpoints, xA, xB, featureKey);

            if (contactCount == 0)
            {
                xA = supportPoint(boxA, normalAB);
                xB = supportPoint(boxB, -normalAB);
                addContact(bodyA, bodyB, contacts, ref contactCount, contactMidpoints, xA, xB, featureKey);
            }

            return contactCount;
        }

        [BurstCompile]
        private struct SatJob : IJob
        {
            private OBB _boxA;
            private OBB _boxB;
            private NativeReference<CollideResult> _result;

            public SatJob(OBB boxA,
                OBB boxB,
                NativeReference<CollideResult> result)
            {
                _boxA = boxA;
                _boxB = boxB;
                result.Value = new CollideResult();
                _result = result;
            }
            
            public void Execute()
            {
                float3 delta = _boxB.center - _boxA.center;

                SatAxis bestFace = new SatAxis();
                bestFace.separation = -float.MaxValue;
                bestFace.valid = false;

                SatAxis bestEdge = new SatAxis();
                bestEdge.separation = -float.MaxValue;
                bestEdge.valid = false;

                for (int i = 0; i < 3; ++i)
                    if (!testAxis(_boxA, _boxB, delta, _boxA.axis[i], AxisType.AXIS_FACE_A, i, -1, ref bestFace))
                        return;
                
                for (int i = 0; i < 3; ++i)
                    if (!testAxis(_boxA, _boxB, delta, _boxB.axis[i], AxisType.AXIS_FACE_B, -1, i, ref bestFace))
                        return;
                
                for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    float3 axis = cross(_boxA.axis[i], _boxB.axis[j]);
                    if (!testAxis(_boxA, _boxB, delta, axis, AxisType.AXIS_EDGE, i, j, ref bestEdge))
                        return;
                }

                if (!bestFace.valid)
                    return;

                SatAxis best = bestFace;
                if (bestEdge.valid)
                {
                    float edgeRelTol = 0.95f;
                    float edgeAbsTol = 0.01f;
                    if (edgeRelTol * bestEdge.separation > bestFace.separation + edgeAbsTol)
                        best = bestEdge;
                }

                var r = _result.Value;
                r.collided = true;
                r.best = best;
                _result.Value = r;
            }
        }
        
        private struct CollideResult
        {
            public bool collided;
            public SatAxis best;
        }
    }
}

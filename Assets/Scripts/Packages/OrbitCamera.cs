using UnityEngine;

public class OrbitCamera : MonoBehaviour
{
    public float Distance { get; private set; } = 80f;

    [SerializeField]
    private float _rotationSensitivity = 5f;
    [SerializeField]
    private float _upDownSensitivity = 0.5f;
    [SerializeField]
    private float _zoomWheelSensitivity = 12f;
    [SerializeField]
    private float _zoomMiddleClickSensitivity = 1f;

    [SerializeField] private Transform target;

    private float _lookAtHeight = 0f;
    private float _azimuthalAngle = 70f;
    private float _polarAngle = 45f;
    private Camera _camera;

    private void Start()
    {
        _camera = Camera.main;
        _upDownSensitivity = 0.5f;
    }

    private void Update()
    {
        if (Input.GetMouseButton(0))
        {
            UpdateAngle(Input.GetAxis("Mouse X") * _rotationSensitivity, Input.GetAxis("Mouse Y") * _rotationSensitivity);
        }

        if (Input.GetMouseButton(1))
        {
            UpdateUpDown(Input.GetAxis("Mouse Y") * _upDownSensitivity);
        }

        UpdateDistance(Input.GetAxis("Mouse ScrollWheel") * _zoomWheelSensitivity);
        if (Input.GetMouseButton(2))
        {
            UpdateDistance(Input.GetAxis("Mouse Y") * _zoomMiddleClickSensitivity);
        }

        UpdatePosition();
        UpdateDirection();
    }

    private void UpdateAngle(float x, float y)
    {
        x = _azimuthalAngle - x;
        _azimuthalAngle = Mathf.Repeat(x, 360);

        y = _polarAngle + y;
        _polarAngle = Mathf.Clamp(y, 1f, 179f);
    }

    private void UpdateUpDown(float y)
    {
        _lookAtHeight -= y;
        _lookAtHeight = Mathf.Clamp(_lookAtHeight, -20f, 20f);
    }

    private void UpdateDistance(float scroll)
    {
        Distance -= scroll;
        Distance = Mathf.Clamp(Distance, 0.1f, 100f);
    }

    private void UpdatePosition()
    {
        var da = _azimuthalAngle * Mathf.Deg2Rad;
        var dp = _polarAngle * Mathf.Deg2Rad;
        _camera.transform.position =
            (target == null ? Vector3.zero : target.position)
            + new Vector3(
                Distance * Mathf.Sin(dp) * Mathf.Cos(da),
                Distance * Mathf.Cos(dp) + _lookAtHeight,
                Distance * Mathf.Sin(dp) * Mathf.Sin(da));
    }

    private void UpdateDirection()
    {
        _camera.transform.LookAt((target == null ? Vector3.zero : target.position)
                                 + new Vector3(0f, _lookAtHeight, 0f));
    }
}
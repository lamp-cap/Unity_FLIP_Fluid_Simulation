using UnityEngine;

namespace SVT
{
    public class CameraController : MonoBehaviour
    {
        public float mouseSensitivity = 2f;       // 鼠标灵敏度
        public float moveSpeed = 5f;              // 基础移动速度
        public float shiftMultiplier = 2f;        // Shift 加速倍率

        private float rotationX = 0f;
        private float rotationY = 0f;

        void Start()
        {
            Vector3 angles = transform.eulerAngles;
            rotationX = angles.y;
            rotationY = angles.x;

            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        void Update()
        {
            HandleMouseLook();
            HandleMovement();
        }

        void HandleMouseLook()
        {
            if (Input.GetKey(KeyCode.LeftAlt))
            {
                Cursor.lockState = CursorLockMode.None;
                Cursor.visible = true;
            }
            else
            {
                Cursor.lockState = CursorLockMode.Locked;
                Cursor.visible = false;

                float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
                float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;

                rotationX += mouseX;
                rotationY -= mouseY;
                rotationY = Mathf.Clamp(rotationY, -89f, 89f);

                transform.rotation = Quaternion.Euler(rotationY, rotationX, 0);
            }
        }

        void HandleMovement()
        {
            Vector3 move = Vector3.zero;

            if (Input.GetKey(KeyCode.W)) move += transform.forward;
            if (Input.GetKey(KeyCode.S)) move -= transform.forward;
            if (Input.GetKey(KeyCode.A)) move -= transform.right;
            if (Input.GetKey(KeyCode.D)) move += transform.right;
            if (Input.GetKey(KeyCode.E)) move += transform.up;
            if (Input.GetKey(KeyCode.Q)) move -= transform.up;

            float speed = moveSpeed;
            if (Input.GetKey(KeyCode.LeftShift))
            {
                speed *= shiftMultiplier;
            }

            transform.position += move.normalized * speed * Time.deltaTime;
        }
    }
}

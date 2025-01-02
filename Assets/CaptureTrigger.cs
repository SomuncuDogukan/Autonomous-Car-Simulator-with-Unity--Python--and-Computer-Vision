using UnityEngine;

public class CaptureTrigger : MonoBehaviour
{
    public CameraCapture cameraCapture;  // CameraCapture script'ine referans

    void Start()
    {
        // Kamera Capture script'ini sahne üzerinden atayın
        if (cameraCapture == null)
        {
            Debug.LogError("CameraCapture script'i atanmamış!");
        }
    }

    void Update()
    {
        // Space tuşuna basıldığında tetiklenir
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("CaptureAndSendImage çağrılıyor...");
            cameraCapture.CaptureAndSendImage();
        }
    }
}

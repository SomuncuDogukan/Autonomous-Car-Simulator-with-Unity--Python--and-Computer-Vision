using UnityEngine;

public class CaptureTrigger : MonoBehaviour
{
    public CameraCapture cameraCapture;  // Reference to the CameraCapture script

    void Start()
    {
        // Assign the CameraCapture script from the scene if it's not already assigned
        if (cameraCapture == null)
        {
            Debug.LogError("CameraCapture script is not assigned!");
        }
    }

    void Update()
    {
        // Trigger the image capture and send when the Space key is pressed
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("Calling CaptureAndSendImage...");
            cameraCapture.CaptureAndSendImage();  // Call the CaptureAndSendImage method from CameraCapture script
        }
    }
}

using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class CameraCapture : MonoBehaviour
{
    public Camera camera;  // Unity camera to capture the image from
    public string apiUrl = "http://localhost:8000/predict/";  // FastAPI URL for prediction

    // Method to capture and send the image
    public void CaptureAndSendImage()
    {
        StartCoroutine(CaptureAndSend());
    }

    private IEnumerator CaptureAndSend()
    {
        // Capture the camera image using RenderTexture
        RenderTexture renderTexture = new RenderTexture(256, 256, 24); // Image resolution (256x256)
        camera.targetTexture = renderTexture; // Assign the render texture to the camera
        Texture2D texture = new Texture2D(256, 256, TextureFormat.RGB24, false);
        camera.Render();  // Capture the camera view
        RenderTexture.active = renderTexture;
        texture.ReadPixels(new Rect(0, 0, 256, 256), 0, 0);  // Transfer the image to the texture
        camera.targetTexture = null;  // Clear the render texture
        RenderTexture.active = null;

        // Convert the captured image to byte array in JPEG format
        byte[] imageBytes = texture.EncodeToJPG();

        // Send a POST request to the FastAPI server
        UnityWebRequest www = UnityWebRequest.Post(apiUrl, "POST");
        www.uploadHandler = new UploadHandlerRaw(imageBytes);  // Attach the byte array as the payload
        www.downloadHandler = new DownloadHandlerBuffer();  // Handle the server's response
        www.SetRequestHeader("Content-Type", "application/octet-stream");  // Set the content type header

        // Send the request and wait for the response
        yield return www.SendWebRequest();  // Send the web request

        // Handle the response from the server
        if (www.result == UnityWebRequest.Result.Success)
        {
            // Print the response from the API
            string jsonResponse = www.downloadHandler.text;
            Debug.Log("API Response: " + jsonResponse);
        }
        else
        {
            // Print an error message if the request fails
            Debug.LogError("Error: " + www.error);
        }
    }
}

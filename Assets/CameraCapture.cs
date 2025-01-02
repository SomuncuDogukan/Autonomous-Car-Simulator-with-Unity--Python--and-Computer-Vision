using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class CameraCapture : MonoBehaviour
{
    public Camera camera;  // Unity kamera
    public string apiUrl = "http://localhost:8000/predict/";  // FastAPI URL

    public void CaptureAndSendImage()
    {
        StartCoroutine(CaptureAndSend());
    }

    private IEnumerator CaptureAndSend()
    {
        // RenderTexture ile kamera görüntüsünü almak
        RenderTexture renderTexture = new RenderTexture(256, 256, 24); // Görüntü boyutu
        camera.targetTexture = renderTexture;
        Texture2D texture = new Texture2D(256, 256, TextureFormat.RGB24, false);
        camera.Render();  // Kamera görüntüsünü al
        RenderTexture.active = renderTexture;
        texture.ReadPixels(new Rect(0, 0, 256, 256), 0, 0);  // Görüntüyü texture'a aktar
        camera.targetTexture = null;
        RenderTexture.active = null;

        // Görüntüyü byte dizisine dönüştür (JPEG formatında)
        byte[] imageBytes = texture.EncodeToJPG();

        // FastAPI'ye POST isteği gönderme
        UnityWebRequest www = UnityWebRequest.Post(apiUrl, "POST");
        www.uploadHandler = new UploadHandlerRaw(imageBytes);  // Byte dizisini gönder
        www.downloadHandler = new DownloadHandlerBuffer();  // Yanıtı al
        www.SetRequestHeader("Content-Type", "application/octet-stream");

        // Başlık ve diğer parametreleri ayarla
        www.SetRequestHeader("Content-Type", "application/octet-stream");

        yield return www.SendWebRequest();  // İsteği gönder

        if (www.result == UnityWebRequest.Result.Success)
        {
            // API'den gelen yanıtı yazdır
            string jsonResponse = www.downloadHandler.text;
            Debug.Log("API Response: " + jsonResponse);
        }
        else
        {
            Debug.LogError("Error: " + www.error);
        }
    }
}

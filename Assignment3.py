import jetson.inference
import jetson.utils

# Load the SSD-Mobilenet-V2 detection network
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Load the image for analysis (replace 'image.png' with the correct image file path)
image_path = "/mnt/data/image.png"  # Replace with your image path
img = jetson.utils.loadImage(image_path)

# Perform object detection on the image
detections = net.Detect(img)

# Print detection results
print("Detected objects in", image_path)
for detection in detections:
    print(f"-- ClassID: {detection.ClassID}")
    print(f"-- Confidence: {detection.Confidence:.6f}")
    print(f"-- Left: {detection.Left:.3f}")
    print(f"-- Top: {detection.Top:.3f}")
    print(f"-- Right: {detection.Right:.3f}")
    print(f"-- Bottom: {detection.Bottom:.3f}")
    print(f"-- Width: {detection.Width:.3f}")
    print(f"-- Height: {detection.Height:.3f}")
    print(f"-- Area: {detection.Area:.0f}")
    print(f"-- Center: ({detection.Center[0]:.3f}, {detection.Center[1]:.3f})")

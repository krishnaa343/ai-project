"""
Generate a realistic-looking synthetic tomato leaf image for testing.
Saved as: test_leaf.jpg
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

def make_leaf_image(output_path="test_leaf.jpg", size=(256, 256)):
    # Start with a green background simulating a leaf
    img = Image.new("RGB", size, color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw an elliptical green leaf body
    cx, cy = size[0] // 2, size[1] // 2
    leaf_color = (58, 120, 40)       # dark green
    draw.ellipse([30, 20, size[0]-30, size[1]-20], fill=leaf_color)

    # Add veins in lighter green
    vein_color = (100, 180, 70)
    # Main midrib
    draw.line([(cx, 25), (cx, size[1]-25)], fill=vein_color, width=2)
    # Side veins
    for i in range(1, 5):
        y = 30 + i * (size[1] - 60) // 5
        spread = int(40 + i * 5)
        draw.line([(cx, y), (cx - spread, y + 20)], fill=vein_color, width=1)
        draw.line([(cx, y), (cx + spread, y + 20)], fill=vein_color, width=1)

    # Add brown/yellow disease spots simulating late blight
    spot_color = (140, 80, 20)
    dark_spot   = (80, 40, 10)
    spots = [
        (80, 90, 25), (160, 130, 18), (100, 170, 20),
        (145, 80, 14), (70, 150, 16), (185, 100, 12),
    ]
    for sx, sy, sr in spots:
        draw.ellipse([sx-sr, sy-sr, sx+sr, sy+sr], fill=spot_color)
        # Darker core
        inner = sr // 2
        draw.ellipse([sx-inner, sy-inner, sx+inner, sy+inner], fill=dark_spot)

    # Yellow halo around some spots (disease symptom)
    yellow = (210, 200, 50)
    for sx, sy, sr in spots[:3]:
        outer = sr + 6
        draw.ellipse([sx-outer, sy-outer, sx+outer, sy+outer], fill=yellow, outline=yellow)
        draw.ellipse([sx-sr, sy-sr, sx+sr, sy+sr], fill=spot_color)
        inner = sr // 2
        draw.ellipse([sx-inner, sy-inner, sx+inner, sy+inner], fill=dark_spot)

    # Light grain noise
    arr = np.array(img)
    noise = np.random.randint(-12, 12, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Slight Gaussian blur to smooth out
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    img.save(output_path, "JPEG", quality=92)
    print(f"Saved synthetic leaf image to: {os.path.abspath(output_path)}")
    print(f"Size: {os.path.getsize(output_path):,} bytes")

make_leaf_image("test_leaf.jpg")

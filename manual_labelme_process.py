import os
import subprocess
from pathlib import Path

# Directory containing your .mp4 files
video_dir = Path("/Users/arielmalowany/Desktop/Learning/Cupybara/Videos de camaras trampa")
os.chdir(video_dir)

# Process all .mp4 files
for video_path in sorted(video_dir.glob("*.AVI")):
    folder_name = video_path.stem  # e.g. "03140097"
    
    print(f"\n🔧 Converting {video_path.name} to images...")
    subprocess.run(["video-toimg", video_path.name], check=True)
    
    print(f"✏️ Opening labelme on folder: {folder_name}")
    try:
        subprocess.run(["labelme", folder_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Labelme closed with error: {e}")
        continue  # Skip to next video
    
    
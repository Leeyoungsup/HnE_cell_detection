"""
Fix critical bug in P2PNetDataset.__getitem__()

Bug: In while loop, image is cropped repeatedly from already-cropped image,
but points still reference original image coordinates, causing misalignment.

Fix: Always crop from original_image instead of modified image.
"""
import json

notebook_path = 'detail_p2pnet.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Bug pattern
old_pattern = '''    "    def __getitem__(self, index):\\n",
    "        image = self.images[index].copy()\\n",
    "        points = self.labels[index]['points'].copy()\\n",
    "        classes = self.labels[index]['classes'].copy()\\n",
    "        crop_points = []\\n",
    "        crop_classes = []\\n",
    "        while(len(crop_points)<1):\\n",
    "            image, h1, w1 = self.crop_padding_image(image)\\n",'''

# Fixed pattern
new_pattern = '''    "    def __getitem__(self, index):\\n",
    "        original_image = self.images[index].copy()  # Keep original image reference\\n",
    "        points = self.labels[index]['points'].copy()\\n",
    "        classes = self.labels[index]['classes'].copy()\\n",
    "        crop_points = []\\n",
    "        crop_classes = []\\n",
    "        while(len(crop_points)<1):\\n",
    "            image, h1, w1 = self.crop_padding_image(original_image)  # Always crop from original!\\n",'''

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    print("✅ Fixed critical dataset bug!")
    print("   - Changed: image = self.images[index].copy()")
    print("   - To:      original_image = self.images[index].copy()")
    print("   - Changed: self.crop_padding_image(image)")
    print("   - To:      self.crop_padding_image(original_image)")
    print("\n⚠️  This bug caused label-image misalignment in while loop!")
else:
    print("❌ Pattern not found - file may have changed")

with open(notebook_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Notebook updated successfully!")

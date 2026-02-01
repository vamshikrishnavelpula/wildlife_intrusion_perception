import os
import shutil

# Paths
source_root = os.path.expanduser("~/roboflow_raw")  # Original unzipped datasets
merged_root = os.path.expanduser("~/wildlife/dataset")

class_map = {"elephant": 0, "boar": 1, "monkey": 2}
splits = ["train", "valid", "test"]

for animal, class_id in class_map.items():
    for split in splits:
        img_src = os.path.join(source_root, animal, split, "images")
        lbl_src = os.path.join(source_root, animal, split, "labels")
        
        img_dst = os.path.join(merged_root, "images", split)
        lbl_dst = os.path.join(merged_root, "labels", split)
        
        if not os.path.exists(img_src):
            print(f"⚠️ Missing folder: {img_src}")
            continue
        
        for img_file in os.listdir(img_src):
            if not img_file.endswith((".jpg", ".png", ".jpeg")):
                continue
            
            # Copy image
            shutil.copy2(os.path.join(img_src, img_file), os.path.join(img_dst, img_file))
            
            # Copy & fix label
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            lbl_path_src = os.path.join(lbl_src, lbl_file)
            lbl_path_dst = os.path.join(lbl_dst, lbl_file)
            
            if os.path.exists(lbl_path_src):
                with open(lbl_path_src, "r") as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    parts[0] = str(class_id)  # Assign correct class ID
                    new_lines.append(" ".join(parts) + "\n")
                
                with open(lbl_path_dst, "w") as f:
                    f.writelines(new_lines)
            else:
                # Create empty label if missing
                open(lbl_path_dst, "w").close()

print("✅ Dataset merged and class IDs fixed successfully!")












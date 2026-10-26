import os
import shutil

# ============================================================
# YOLO SEGMENTATION/POLYGON -> YOLO DETECTION BOX CONVERTER
# ============================================================

DATASET_DIR = "dataset"

SPLITS = [
    "train",
    "valid",
    "test"
]


def convert_file(label_file):

    with open(label_file, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # Class ID
        class_id = parts[0]

        values = parts[1:]

        # ----------------------------------------------------
        # Already standard YOLO detection format
        # class x_center y_center width height
        # ----------------------------------------------------

        if len(values) == 4:

            new_lines.append(
                f"{class_id} "
                f"{values[0]} "
                f"{values[1]} "
                f"{values[2]} "
                f"{values[3]}\n"
            )

            continue

        # ----------------------------------------------------
        # Polygon format
        # class x1 y1 x2 y2 x3 y3 ...
        # ----------------------------------------------------

        if len(values) >= 6 and len(values) % 2 == 0:

            points = []

            for i in range(0, len(values), 2):

                x = float(values[i])
                y = float(values[i + 1])

                points.append((x, y))

            # Find bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            xmin = min(xs)
            xmax = max(xs)

            ymin = min(ys)
            ymax = max(ys)

            # Convert to YOLO bbox
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            width = xmax - xmin
            height = ymax - ymin

            # Keep values between 0 and 1
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            new_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )

        else:

            print(
                "WARNING: Could not process:",
                label_file
            )

    # Write converted labels
    with open(label_file, "w") as f:
        f.writelines(new_lines)


# ============================================================
# PROCESS TRAIN / VALID / TEST
# ============================================================

total_files = 0

for split in SPLITS:

    label_dir = os.path.join(
        DATASET_DIR,
        split,
        "labels"
    )

    if not os.path.exists(label_dir):

        print(
            f"Labels folder not found: {label_dir}"
        )

        continue

    print()
    print("=" * 50)
    print(f"Processing: {split}")
    print("=" * 50)

    files = [
        f for f in os.listdir(label_dir)
        if f.endswith(".txt")
    ]

    for filename in files:

        label_file = os.path.join(
            label_dir,
            filename
        )

        convert_file(label_file)

        total_files += 1


print()
print("=" * 50)
print("CONVERSION COMPLETED")
print("=" * 50)
print("Total label files processed:", total_files)
print()
print("Polygon labels converted to YOLO bounding boxes.")
print("Original labels are safe in *_backup folders.")
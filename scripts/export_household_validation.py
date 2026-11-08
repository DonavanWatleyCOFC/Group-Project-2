"""Export fully reviewed household validation JSON to a 63-class YOLO evaluation set.

This deliberately refuses partial reviews and keeps the seven reserved final-test
photos out of the export.  It only creates an evaluation artifact; it never adds
photos to a training dataset or launches training.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

SCHEMA = "food63-no-spices-v1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yolo_line(box):
    x1, y1, x2, y2 = box["xyxy"]
    return "{} {:.10f} {:.10f} {:.10f} {:.10f}".format(
        box["class_id"], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    project = args.project.resolve()
    review_path = args.review.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    names = review.get("names")
    if review.get("format") != "household-box-review-v1" or review.get("class_schema") != SCHEMA:
        raise SystemExit("Unexpected review format or class schema")
    if not isinstance(names, list) or len(names) != 63:
        raise SystemExit("Expected the fixed 63-class name order")
    images = review.get("images")
    if not isinstance(images, list) or len(images) != 48:
        raise SystemExit("Expected exactly 48 household validation images")
    pending = [image.get("filename", "<unnamed>") for image in images if image.get("status") != "reviewed"]
    if pending:
        raise SystemExit("Refusing to export unreviewed images: " + ", ".join(pending))

    final_test = project / "data" / "household_final_test"
    final_names = {path.name for path in final_test.glob("*.jpg") if path.is_file()}
    if len(final_names) != 7:
        raise SystemExit("Expected exactly seven reserved final-test photos")
    if final_names & {image["filename"] for image in images}:
        raise SystemExit("Final-test photo appears in validation review")

    (output / "images").mkdir(parents=True)
    (output / "labels").mkdir()
    rows = []
    boxes = 0
    for image in images:
        filename = image["filename"]
        source = Path(image["original_image"])
        if source.name != filename or not source.is_file() or sha256(source) != image.get("original_sha256"):
            raise SystemExit(f"Source identity check failed for {filename}")
        lines = []
        for box in image.get("boxes", []):
            class_id = box.get("class_id")
            coords = box.get("xyxy")
            if not isinstance(class_id, int) or not 0 <= class_id < len(names):
                raise SystemExit(f"Invalid class ID in {filename}")
            if not isinstance(coords, list) or len(coords) != 4 or not all(isinstance(x, (int, float)) for x in coords):
                raise SystemExit(f"Invalid box in {filename}")
            x1, y1, x2, y2 = coords
            if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                raise SystemExit(f"Out-of-range box in {filename}")
            lines.append(yolo_line(box))
        shutil.copy2(source, output / "images" / filename)
        (output / "labels" / Path(filename).with_suffix(".txt").name).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        rows.append({"image": f"images/{filename}", "label": f"labels/{Path(filename).stem}.txt",
                     "split": "household_validation", "group_id": "household_validation_20261104",
                     "original_sha256": image["original_sha256"], "status": "reviewed"})
        boxes += len(lines)

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    yaml_lines = [f"path: {output}", "images: images", "labels: labels", "nc: 63",
                  f"class_schema: {SCHEMA}", "names:"]
    yaml_lines.extend(f"- {name}" for name in names)
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    report = {"images": len(rows), "boxes": boxes, "class_count": len(names), "class_schema": SCHEMA,
              "source_review": str(review_path), "all_images_reviewed": True,
              "final_test_excluded": sorted(final_names), "training_started": False}
    (output / "export-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Reviewed household validation ground truth\n\n"
        "All 48 images were manually marked reviewed before export. This is evaluation-only ground truth; "
        "do not add it to training. The seven files in `data/household_final_test` are excluded.\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

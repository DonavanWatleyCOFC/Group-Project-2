# Training/new-photo comparison — 2026-10-26

Visually inspected five training images selected at evenly spaced indices in the sorted 2,994-file list: DSC_5680, DSC_5793, DSC_5891, DSC_6009, DSC_6206. Also inspected all six new-photos with EXIF orientation applied. Originals unchanged.

All five training samples show similar refrigerator interiors, relatively sparse contents, speckled noise and black rectangular patches present in the files. The corruption source was not determined. New photos show varied refrigerator layouts, dense packaging and condiments, door storage, occlusion, and wider views with kitchen/floor area. fridge_02 has a pink cast. fridge_04/05 are closer shelf/door views, so distance alone cannot explain failures. This is a small sample, not a dataset-wide audit or causal experiment.

There are 362 distinct prefixes before .rf. among 2,994 training filenames. This suggests repeated source variants; it does not establish exact augmentation lineage or capture independence. Checkpoint metadata confirms yolo11n.pt, epochs=50, imgsz=640, and a Windows dataset path named fridge-detection-v1.local.yaml. Exact identity of the Windows training files and local export was not independently verified.

## Proposed next steps, not adopted data changes

1. Correct workbook image IDs and complete supported-object annotations, including misses and boxes. Unsupported foods are not false negatives for this 30-class model; predicting a supported class on them is a false positive. Preserve ambiguous/opaque contents rather than guessing hidden ingredients.
2. Prioritize real scenes from different refrigerators, households, sessions, brands, cameras and lighting. Include full-fridge and closer shelf views, visible foods, varied packaging and occlusion. Use the team's roughly 120-photo collection target as a pilot, not a proven sufficient count; track object counts per class.
3. Split whole refrigerators/capture sessions before augmentation. Reserve unseen refrigerators for final testing. If existing new-photos guide tuning, treat them as development data and collect fresh final-test photos. Label every visible in-scope object. Genuine no-target scenes help teach background rejection; never leave targets unlabeled to manufacture negatives.
4. Supplement weak classes with individual-item images and varied packaging, but prioritize realistic scenes. Grocery single-item source data stays training-only under project rules. Freeze class definitions/version and do not silently mix 10- and 30-class datasets.
5. Compare baseline with a run adding representative data while holding model/settings fixed. Then test realistic lighting/blur augmentation separately. Audit corruption and label visibility before any decision to replace old images; do not delete based on five samples.
6. Ultralytics handles pixel scaling; manual normalization is not the first intervention. Verify orientation/resize consistency. Test closer captures or higher resolution if small-object errors persist. Gaussian blur is an augmentation hypothesis, binary thresholding loses color/texture, HOG/Haar are alternative baselines, and segmentation requires mask labels/model changes without guaranteed detection gains.
7. Choose thresholds/experiments on validation data; report precision, recall, mAP50/mAP50-95 and a matched confusion matrix once full ground truth exists. Raising confidence can suppress false positives while increasing misses; it is not a recognition improvement by itself.

Existing iteration report records strong supplied-test metrics but unverified refrigerator/session independence. Those results do not establish generalization to household scenes.

Sources:
- https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results/ (diversity/background images)
- https://docs.ultralytics.com/guides/yolo-data-augmentation/
- https://docs.ultralytics.com/guides/preprocessing-annotated-data/
- https://docs.ultralytics.com/tasks/segment/

Verification/AI: visually reviewed contact sheets, read paired label counts and checkpoint metadata, counted filename roots. No new training, inference, performance measurement, source edits or label audit was performed. Recommendations are hypotheses; reviewer labels and exact training provenance remain independently unverified.

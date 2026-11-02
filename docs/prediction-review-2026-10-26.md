# Prediction review — 2026-10-26

Source: `/Users/donavan/school/fall2026/csci270/Book1.xlsx`, Sheet1 A1:F57.
Original workbook unchanged. There are 56 detection rows, 55 reviewed: one `good`,
54 `wrong class`, one blank issue. Reviewed correctness is 1/55 = 1.82%, a
provisional precision proxy, not IoU-validated precision or overall accuracy.

| Predicted class | Good | Wrong | Unreviewed | Precision proxy |
|---|---:|---:|---:|---:|
| banana | 0 | 8 | 0 | 0% |
| butter | 0 | 7 | 1 | 0% |
| carrot | 0 | 3 | 0 | 0% |
| cheese | 1 | 4 | 0 | 20% |
| chocolate | 0 | 1 | 0 | 0% |
| corn | 0 | 2 | 0 | 0% |
| flour | 0 | 2 | 0 | 0% |
| heavy cream | 0 | 1 | 0 | 0% |
| lime | 0 | 1 | 0 | 0% |
| milk | 0 | 19 | 0 | 0% |
| onion | 0 | 1 | 0 | 0% |
| potato | 0 | 1 | 0 | 0% |
| spinach | 0 | 2 | 0 | 0% |
| sugar | 0 | 2 | 0 | 0% |

Proxy = good/(good+wrong); unreviewed excluded. Classes without predictions have
undefined precision. Parsing normalized Soinach to spinach and banna to banana,
removed trailing confidence numbers, and trimmed whitespace. The classes column
is an inventory, not row-aligned ground truth. No source cells were changed.

Recall is unavailable without all actual objects, including misses. mAP and a
standard detection confusion matrix require prediction/ground-truth box matching;
the workbook has no coordinates and includes ambiguous or multiple actual objects.
No recall, mAP, or standard class confusion matrix was fabricated.

Image identity also needs confirmation. Existing external-review-01 predictions
contain six images with counts 8,4,5,14,6,19; workbook image IDs fridge_01–07 have
counts 8,4,5,2,12,6,19. A possible shift after the first two fridge_04 rows is
unconfirmed; do not automatically remap or join these rows.

Next: confirm image/box IDs, finish blank review, annotate all in-scope objects
with classes and boxes, then calculate precision/recall, mAP50/mAP50-95, and a
matched detection confusion matrix. These are review photos, not established test data.

Verification: read all rows with bundled Python/openpyxl; tallied labels and
classes; compared counts with prediction JSON. No training/inference or visual
annotation validation ran. AI computed summaries; human review labels remain
independently unverified. Alternative of treating every description as a single
ground-truth class was rejected because it would invent unsupported annotations.

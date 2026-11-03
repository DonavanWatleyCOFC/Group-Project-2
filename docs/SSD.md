# Software Design Document

## Project Information

**Project title:** Fridge-to-Recipe: An AI-Based Food Recognition and Recipe Recommendation System  
**Team members:** Adwoa A, Owen D, Donavan W, Carsyn W  
**Date:** 06OCT26

**Dataset section updated:** 01NOV26

**OCR design and approval section updated:** 03NOV26

## 1. Statement of Purpose

The purpose of this project is to develop a mobile application that identifies food items in a photograph of a refrigerator and recommends recipes that can be prepared using those ingredients.

Many people, particularly college students, have limited time, money, ingredients, and cooking experience. They may have food available but not know what meals can be made with it. This can lead to unnecessary grocery purchases and food waste.

The application will allow a user to photograph the contents of a refrigerator or upload an existing image. A computer vision model will detect recognizable food items, and the user will confirm or correct the detected ingredient list. The system will compare the confirmed ingredients with a recipe collection and return:

- **Make Now:** recipes requiring no additional major ingredients.
- **Missing 1–2 Items:** recipes requiring only one or two additional major ingredients.

Users may also provide pantry items, dietary restrictions, food allergies, and preferred cuisines. These inputs will filter and rank recommendations. The system is intended to help users plan meals, save money, reduce food waste, and make better use of food they already own.

## 2. Data to Be Used for Training

### Image Datasets Used in the Detection Experiments

The implemented detector uses bounding-box annotations from the following sources.
Counts below are image files actually imported, not object counts. Exported
augmentations are included in those counts and are not independent photographs.

| Dataset and version | Declared license | Imported training images | Imported validation images | Contribution |
| --- | --- | ---: | ---: | --- |
| [Fridge Detection v1](https://universe.roboflow.com/fridge-ydzni/fridge-detection-es0ja/dataset/1), exported through the team fork | MIT | 2,994 | 103 | Original 30 classes; also supplies 51 held-out internal test images. |
| [AICook v1, original images](https://universe.roboflow.com/karel-cornelis-q2qqg/aicook-lcv4d/dataset/1) | MIT | 362 | 103 | Supplemental examples for the same 30 classes; its 51 test images are excluded. |
| [5K Groceries Object Detection, train-val branch](https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset/tree/train-val) | No standalone license verified in the downloaded checkout; confirm reuse terms before redistribution | 3,067 | 0 | Selected 18 categories; 14 new classes and four existing classes. |
| [Types of Food v3](https://universe.roboflow.com/test-ol1sl/types-of-food/dataset/3) | Public Domain, as declared by the source | 104 | 0 | Six new condiment classes plus existing milk, soda, and water examples. |
| [Vegetables v1](https://universe.roboflow.com/vegetables/vegetables-el4g6/dataset/1) | CC BY 4.0 | 6,886 | 575 | Thirteen new classes plus existing carrot, corn, onion, potato, and tomato examples. |
| [Fridge Objects v12](https://universe.roboflow.com/fooddetection-essdj/fridge-objects/dataset/12) | CC BY 4.0 | 812 | 45 | Dairy-positive images; adds yogurt and supplements milk, cheese, and butter. |

Preserve the source attribution and export notices. License declarations do not
independently establish ownership of every source image. For the two CC BY sources,
retain attribution to **Vegetables (2022)** and **fooddetection (2023)** respectively,
link the source and license, and record annotation conversions and label changes.

#### Class IDs and Incoming Label Mappings

The original 30 labels keep their original order and zero-based IDs:

```text
0 apple, 1 banana, 2 beef, 3 blueberries, 4 bread, 5 butter,
6 carrot, 7 cheese, 8 chicken, 9 chicken_breast, 10 chocolate,
11 corn, 12 eggs, 13 flour, 14 goat_cheese, 15 green_beans,
16 ground_beef, 17 ham, 18 heavy_cream, 19 lime, 20 milk,
21 mushrooms, 22 onion, 23 potato, 24 shrimp, 25 spinach,
26 strawberries, 27 sugar, 28 sweet_potato, 29 tomato
```

Incoming annotations are remapped by label name rather than by concatenating YAML
files or assuming source numeric IDs match the project IDs. New labels are appended:

| Source | Appended IDs and labels | Existing-label mapping and exclusions |
| --- | --- | --- |
| 5K Groceries | 30 beans, 31 fish, 32 honey, 33 jam, 34 juice, 35 nuts, 36 oil, 37 pasta, 38 rice, 39 soda, 40 spices, 41 tomato_sauce, 42 vinegar, 43 water | corn → 11, flour → 13, milk → 20, sugar → 27. Import only the 18 selected categories. Keep packaged/dry beans distinct from green_beans. |
| Types of Food | 44 ketchup, 45 lemon_juice, 46 mayonnaise, 47 mustard, 48 relish, 49 salsa | Coke and Root-Beer → soda; Milk-Jug → milk; Water-Bottle → water. Exclude whole images containing Yogurt at this stage. Ketchup remains distinct from tomato_sauce. |
| Vegetables | 50 broccoli, 51 cucumber, 52 cabbage, 53 avocado, 54 eggplant, 55 garlic, 56 pumpkin, 57 celery, 58 peas, 59 bell_pepper, 60 beetroot, 61 cauliflower, 62 hot_pepper | bell pepper → bell_pepper; beet → beetroot; cayliflower → cauliflower; hot pepper → hot_pepper. Existing carrot, corn, onion, potato, and tomato retain their IDs. |
| Fridge Objects dairy subset | 63 yogurt | milk, cheese, butter, avocado, and eggs retain existing IDs; tomatoes → tomato. Only dairy-positive images with fully supported annotations are imported. |

Vegetable images containing any deferred source label are excluded as whole images:
`beans`, `brus capusta`, `fasol`, `rediska`, `redka`, `salad`, `squash-patisson`,
and `vegetable marrow`. This avoids merging ambiguous categories or leaving their
objects unlabeled. The dairy subset similarly excludes images containing `bacon`,
`lemon`, or ambiguous `pepper`. It preserves supported non-dairy annotations in
admitted images. One exact duplicate was skipped during the dairy import.

5K annotations are converted from Pascal VOC boxes to normalized YOLO boxes;
zero-area source boxes are omitted. Types of Food polygon annotations are converted
to enclosing detection boxes. The importers also support polygon-to-box conversion
when required by subsequent exports.

#### Dataset Progression and Evaluation Boundaries

| Combined dataset directory under data/ | Classes | Train | Validation | Internal test |
| --- | ---: | ---: | ---: | ---: |
| fridge-detection-v1 | 30 | 2,994 | 103 | 51 |
| combined-aicook-v1 | 30 | 3,356 | 206 | 51 |
| combined-aicook-5k-groceries | 44 | 6,423 | 206 | 51 |
| combined-aicook-5k-condiments | 50 | 6,527 | 206 | 51 |
| combined-condiments-vegetables | 63 | 13,413 | 781 | 51 |
| combined-condiments-vegetables-dairy | 64 | 14,225 | 826 | 51 |

The selected 5K source training and validation images were both imported into
project training. Selected Types of Food training, validation, and test images
were also imported into project training. Those source partitions can therefore
never be used as held-out tests for these models. The 14 new grocery classes and
six condiment classes have no dedicated validation or internal test examples in
these combined datasets; their generalization is not measured by the original
fridge validation/test scores.

Vegetables and dairy preserve their source validation images as validation and
exclude their source test partitions. All combined datasets retain the same 51
baseline internal test images. These public images do not provide external
unseen-fridge verification, and the test split does not cover all 64 classes.

Import reports, data.yaml, manifest.csv, and validation-summary.json are stored
with the generated datasets. Validation checks image readability, class-ID order,
box bounds, manifest coverage, and exact duplicate image content across splits.
Visual near duplicates, related augmented images, annotation completeness, and
unknown capture sessions still require review; session independence is not proven
for the public exports. Raw datasets, checkpoints, and run outputs remain local
and are excluded from Git by the project's ignore rules.

### Custom Refrigerator Images and External Evaluation

Personal photos in `data/new_photos`, including `fridge_15` through `fridge_20`,
have been used for qualitative prediction review and have not been imported into
training. They currently lack complete ground-truth boxes, so their predictions
cannot establish precision, recall, or mAP. Because these photos have informed
development decisions, reserve additional, independently labeled refrigerator
photos for final external evaluation.

Collect varied refrigerators, camera angles, lighting, packaging, arrangements,
and partial obstructions. Label every supported visible item, keep each known
refrigerator/session in a single split, and preserve a held-out external test set.
The proposed [CollectData Inside the Fridge dataset](https://huggingface.co/datasets/CollectDataIO/inside-the-fridge)
has not been imported into training; sampled frames require labeling before use
as an external evaluation source.

### Previously Considered or Unused Image Sources

The [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset)
originally proposed here is a classification source and has not been merged into
the YOLO detection datasets. It is distinct from the 5K Groceries bounding-box
dataset actually used. The Kaggle fruit/vegetable classification source and the
additional meat and beverage datasets discussed with the team have not been
imported. Searching or recommending a dataset does not make it a training source.

### RecipeNLG

Source: [RecipeNLG project](https://recipenlg.cs.put.poznan.pl/)  
Repository: [Official RecipeNLG repository](https://github.com/Glorf/recipenlg)  
Paper: [RecipeNLG paper](https://aclanthology.org/2020.inlg-1.4/)

RecipeNLG contains more than 2.2 million cooking recipes, with titles, ingredient lists, directions, source information, and extracted food entities. The project should use a manageable subset relevant to the supported food classes. The MVP will retrieve and rank recipes rather than train a large recipe-generation model.

### Alternative Recipe Dataset: Recipe1M

Source: [Recipe1M project](https://pic2recipe.csail.mit.edu/)

Recipe1M contains 1,029,720 structured recipes and 887,706 associated images. It is an alternative if its format and access better fit the implementation.

### Expected Preprocessing

Image preprocessing may include:

- Removing blurry, duplicate, unusable, or incorrectly labeled images.
- Resizing images and converting annotations to the model's format.
- Separating data into training, validation, and held-out testing sets.
- Keeping images from one refrigerator session in one split to avoid leakage.
- Applying suitable augmentation, such as cropping and brightness changes.
- Retaining color and using the implemented training settings `hsv_h=0.015`,
  `hsv_s=0.35`, and `hsv_v=0.25`; some public exports already contain augmented copies.
- Checking class balance and collecting additional examples for weak classes.

Recipe preprocessing may include:

- Removing duplicate or incomplete recipes.
- Converting ingredient names to lowercase.
- Separating ingredient names from quantities and measurement units.
- Normalizing aliases, spelling variants, and singular/plural forms.
- Identifying pantry staples such as salt, pepper, oil, and water.
- Adding dietary and cuisine tags when supported by the data.
- Filtering recipes that conflict with declared restrictions or allergens.

## 3. AI Tools and Packages

### Programming Languages

- **Python:** training, inference, preprocessing, recommendation logic, and backend services.
- **JavaScript/TypeScript or Dart:** mobile development, depending on whether React Native or Flutter is selected.

### Frameworks and Libraries

- **PyTorch:** deep-learning framework for model training and fine-tuning.
- **Ultralytics YOLO:** object detection using transfer learning.
- **PaddleOCR:** recognition of visible food-package text to support ingredient identification alongside YOLO.
- **OpenCV and Pillow:** image processing and visualization.
- **pandas and NumPy:** recipe and numeric data processing.
- **scikit-learn:** data splitting, evaluation metrics, and recommendation baselines.
- **spaCy or NLTK:** optional ingredient-text normalization.
- **Matplotlib or Seaborn:** training and evaluation charts.

The initial recommendation system will use deterministic filtering and scoring. It will consider ingredient coverage, missing-item count, dietary compatibility, declared allergens, and cuisine preference. More advanced similarity methods may be evaluated later if simple matching is insufficient.

### Optical Character Recognition (OCR)

The system uses PaddleOCR alongside the YOLO object-detection model to read visible
food-package labels. YOLO identifies items from their visual appearance, while OCR
provides supporting text evidence, such as "milk," "butter," or "kidney beans."

Recognized text is normalized and matched to supported ingredient names. Nearby,
aligned text lines may be combined when a product name spans multiple lines.
Text confidence and location are used to associate label information with detected
objects or suggest ingredients YOLO missed. OCR scans the image independently of
YOLO's boxes so labels on visually undetected products can still be considered.

OCR results remain subject to user confirmation. Small, obscured, curved, or poorly
lit labels may be unreadable or misinterpreted. OCR confidence represents confidence
in the recognized text, not certainty about the food's identity. Text-region boxes
are distinguished from whole-object detection boxes. Ingredient-list text and
ambiguous matches should not automatically establish a product's identity.

The combined approach will be evaluated against YOLO alone using the same held-out
images to measure recognition quality and the number of user corrections required.
The current prototype runs OCR locally in a separate Python environment; this
integration does not require retraining the YOLO checkpoint. The implementation and
comparison artifacts originated in the separate yoloPlus project folder. The OCR
integration is now available in Group-Project-2 through the `predict --ocr` command
in `scripts/workflow.py`, with `--rerun` providing a side-by-side comparison. YOLO-only inference
remains available, and an isolated OCR interpreter can be selected using
`--ocr-python`. Setup and output details are documented in `docs/ocr-guide.md`.

### Professor Approval for OCR Integration

The professor approved using OCR alongside the YOLO model for this project, as
reported by team member Donavan W. and recorded in this document on November 3,
2026. This approval covers adding OCR as supporting evidence for ingredient
identification. It does not establish the combined system's accuracy; evaluation
and user confirmation remain part of the design. The approval date and professor's
name are not specified in this record.

### Supporting Tools

- Python scripts run from VS Code or PowerShell for experiments and training on the Windows RTX 3060 Ti.
- GitHub for version control and collaboration.
- Roboflow or CVAT for bounding-box annotation.
- React Native or Flutter for the mobile application.
- FastAPI or Flask for the Python API; the team should select one.

## 4. Process and Workflow Overview

1. Select the supported ingredient labels for the experiment; preserve the original 30 IDs and append approved classes, currently through the 64-class dairy dataset.
2. Collect public and custom images, label visible items, and clean the data.
3. Select a manageable recipe subset and normalize its ingredients.
4. Split image data into training, validation, and held-out test sets.
5. Fine-tune a pretrained YOLO object detector.
6. Select the best model with validation results and measure final test performance.
7. Let the user confirm, add, remove, or correct detected ingredients.
8. Collect pantry items, allergies, dietary restrictions, and cuisine preferences.
9. Filter incompatible recipes before calculating match scores.
10. Present ranked recipes with matched and missing ingredients.

```text
Camera or photo upload
        |
        v
Image preprocessing and object detection
        |
        v
Editable ingredient confirmation
        |
        +---- Pantry items and user preferences
        |
        v
Allergen and dietary filtering
        |
        v
Ingredient matching and recipe ranking
        |
        v
"Make Now" and "Missing 1–2 Items" results
```

## 5. Expected Output

The system will produce ingredient detections containing a class, confidence score, and bounding box. After confirmation, it will return ranked recipes containing a title, matched ingredients, missing ingredients, ingredient-coverage score, compatibility information, and instructions or a source link.

Example:

### Make Now

**Vegetable and Cheese Omelet**

- Matched: eggs, milk, tomato, bell pepper, cheese
- Missing: none

### Missing 1–2 Items

**Vegetable Breakfast Burrito**

- Matched: eggs, tomato, bell pepper, cheese
- Missing: tortillas

Computer vision success may be measured using precision, recall, mean Average Precision, per-class results, a confusion matrix, and the average number of user corrections. Recommendation success may be measured using ingredient-match accuracy, missing-item accuracy, restriction compliance, and user ratings of relevance and practicality.

Usability testing with college students may measure task completion, correction effort, clarity, ease of use, and likelihood of reuse. The application must not claim that image recognition or filtering guarantees allergen safety; users must verify product labels and original recipe sources.

## 6. Final Envisioned System of Deployment

The final product is envisioned as a mobile application for Android and iOS. Users will photograph refrigerator contents or select an image from the photo library. The application will display detected ingredients for correction, collect preferences, and show recipe recommendations.

A cross-platform framework such as React Native or Flutter may support both platforms from one codebase. Python may provide AI inference, recommendation logic, and backend processing through FastAPI or Flask.

For the class demonstration, the mobile application and AI services may run locally or over the same network. A future version could host the model, recipe data, and backend in a cloud environment. The final framework and hosting choices remain to be selected by the team.

## 7. Prototype: Minimum Viable Product

The MVP is a working mobile prototype that:

- Captures or uploads one refrigerator image.
- Detects approximately 10–20 supported food classes.
- Displays bounding boxes, labels, and confidence scores.
- Allows the user to add, remove, or correct ingredients.
- Accepts basic pantry items and a small set of dietary/allergy filters.
- Recommends recipes that can be made immediately.
- Recommends recipes missing no more than one or two major ingredients.
- Explains the matched and missing ingredients for every result.

The MVP will be tested using refrigerator images excluded from training. Team members and a small group of college students will review detections and evaluate recommendation usefulness. The demonstration will cover the complete workflow from image capture through recipe recommendation.

## 8. References

- [PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR general OCR pipeline documentation](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [Fridge Detection v1, original source](https://universe.roboflow.com/fridge-ydzni/fridge-detection-es0ja/dataset/1)
- [Fridge Detection v1, team export](https://app.roboflow.com/donavans-workspace/fridge-detection-es0ja-ck6ks/1)
- [AICook v1, original images](https://universe.roboflow.com/karel-cornelis-q2qqg/aicook-lcv4d/dataset/1)
- [5K Groceries Object Detection, train-val branch](https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset/tree/train-val)
- [Types of Food v3](https://universe.roboflow.com/test-ol1sl/types-of-food/dataset/3)
- [Vegetables v1](https://universe.roboflow.com/vegetables/vegetables-el4g6/dataset/1)
- [Fridge Objects v12](https://universe.roboflow.com/fooddetection-essdj/fridge-objects/dataset/12)
- [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)
- [CollectData Inside the Fridge, proposed external evaluation source](https://huggingface.co/datasets/CollectDataIO/inside-the-fridge)
- [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset)
- [Grocery Store Dataset paper](https://arxiv.org/abs/1901.00711)
- [RecipeNLG project](https://recipenlg.cs.put.poznan.pl/)
- [RecipeNLG paper](https://aclanthology.org/2020.inlg-1.4/)
- [Recipe1M project](https://pic2recipe.csail.mit.edu/)
- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV documentation](https://docs.opencv.org/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)


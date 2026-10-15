# Software Design Document

## Project Information

**Project title:** Fridge-to-Recipe: An AI-Based Food Recognition and Recipe Recommendation System  
**Team members:** Adwoa A, Owen D, Donavan W, Carsyn W  
**Date:** 06OCT26

## 1. Statement of Purpose

The purpose of this project is to develop a mobile application that identifies food items in a photograph of a refrigerator and recommends recipes that can be prepared using those ingredients.

Many people, particularly college students, have limited time, money, ingredients, and cooking experience. They may have food available but not know what meals can be made with it. This can lead to unnecessary grocery purchases and food waste.

The application will allow a user to photograph the contents of a refrigerator or upload an existing image. A computer vision model will detect recognizable food items, and the user will confirm or correct the detected ingredient list. The system will compare the confirmed ingredients with a recipe collection and return:

- **Make Now:** recipes requiring no additional major ingredients.
- **Missing 1–2 Items:** recipes requiring only one or two additional major ingredients.

Users may also provide pantry items, dietary restrictions, food allergies, and preferred cuisines. These inputs will filter and rank recommendations. The system is intended to help users plan meals, save money, reduce food waste, and make better use of food they already own.

## 2. Data to Be Used for Training

### Grocery Store Dataset

Source: [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset)

The Grocery Store Dataset contains 5,125 natural smartphone images of grocery products from 81 fine-grained classes organized into 42 broader classes. It includes fruits, vegetables, and carton items such as milk, juice, and yogurt.

Because the dataset primarily supports classification of individual grocery items, it will be supplemented with custom images containing multiple items in realistic refrigerators.

### Custom Refrigerator Image Dataset

The team will collect refrigerator photographs for approximately 10–20 common food classes. Candidate classes include apples, bananas, bell peppers, broccoli, carrots, cheese, eggs, lettuce, milk, onions, oranges, tomatoes, and yogurt.

Each supported item will be labeled with a class and bounding box. Images should represent different refrigerators, camera angles, distances, lighting conditions, arrangements, and partial obstructions. A reasonable initial target is 50–100 labeled examples per class, subject to the available schedule.

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
- **OpenCV and Pillow:** image processing and visualization.
- **pandas and NumPy:** recipe and numeric data processing.
- **scikit-learn:** data splitting, evaluation metrics, and recommendation baselines.
- **spaCy or NLTK:** optional ingredient-text normalization.
- **Matplotlib or Seaborn:** training and evaluation charts.

The initial recommendation system will use deterministic filtering and scoring. It will consider ingredient coverage, missing-item count, dietary compatibility, declared allergens, and cuisine preference. More advanced similarity methods may be evaluated later if simple matching is insufficient.

### Supporting Tools

- Jupyter Notebook or Google Colab for experiments and training.
- GitHub for version control and collaboration.
- Roboflow or CVAT for bounding-box annotation.
- React Native or Flutter for the mobile application.
- FastAPI or Flask for the Python API; the team should select one.

## 4. Process and Workflow Overview

1. Select approximately 10–20 common refrigerator ingredient classes.
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

- [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset)
- [Grocery Store Dataset paper](https://arxiv.org/abs/1901.00711)
- [RecipeNLG project](https://recipenlg.cs.put.poznan.pl/)
- [RecipeNLG paper](https://aclanthology.org/2020.inlg-1.4/)
- [Recipe1M project](https://pic2recipe.csail.mit.edu/)
- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV documentation](https://docs.opencv.org/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)


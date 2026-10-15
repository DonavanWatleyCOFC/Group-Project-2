# Software Design Document

## Project Information

**Project title:** Fridge-to-Recipe: An AI-Based Food Recognition and Recipe Recommendation System  
**Team members:** Adwoa A, Owen D, Donavan W, Carsyn W  
**Date:** 06OCT26

## 1. Statement of Purpose

The purpose of this project is to develop a mobile application that identifies food items in a photograph of a refrigerator and recommends recipes that can be prepared using those ingredients.

Many people, particularly college students, have limited time, money, ingredients, and cooking experience. They may have food available but not know what meals can be made with it. This can lead to unnecessary grocery purchases and food waste.

The application will allow a user to photograph the contents of a refrigerator or upload an existing image. A computer vision model will detect recognizable food items, and the user will be allowed to confirm or correct the detected ingredient list. The system will then compare the confirmed ingredients with a collection of recipes.

The system will provide two main categories of recommendations:

- **Make Now:** recipes that can be prepared with the available ingredients and pantry items.
- **Missing 1–2 Items:** recipes that require only one or two additional major ingredients.

Users may also provide dietary restrictions, food allergies, preferred cuisines, and pantry items such as seasonings, oils, sauces, and condiments. These preferences will be used to filter and rank recommendations.

The project is useful because it combines computer vision and recommendation technology to help users plan meals, reduce food waste, save money, and make better use of food they already own.

## 2. Data to Be Used for Training

### Grocery Store Dataset

Source: [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset)

The Grocery Store Dataset contains 5,125 natural smartphone images of grocery products from 81 fine-grained classes organized into 42 broader classes. It includes fruits, vegetables, and carton items such as milk, juice, and yogurt.

This dataset is useful for learning the appearance of individual grocery items. However, it primarily supports classification of individual items rather than detection of several objects in a crowded refrigerator. It should therefore be supplemented with a custom object-detection dataset.

### Custom Refrigerator Image Dataset

The team will collect photographs of actual refrigerators containing a limited set of common foods. Candidate classes include apples, bananas, bell peppers, broccoli, carrots, cheese, eggs, lettuce, milk, onions, oranges, tomatoes, and yogurt.

The MVP should support approximately 10–20 ingredient classes. Each visible item will be annotated with a class label and bounding box. Photographs should cover different refrigerators, camera angles, distances, lighting conditions, arrangements, and levels of occlusion. A reasonable initial target is 50–100 labeled examples per class, subject to available time and resources.

### RecipeNLG

Source: [RecipeNLG project](https://recipenlg.cs.put.poznan.pl/)  
Repository: [Official RecipeNLG repository](https://github.com/Glorf/recipenlg)  
Paper: [RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation](https://aclanthology.org/2020.inlg-1.4/)

RecipeNLG contains more than 2.2 million cooking recipes with fields such as titles, ingredient lists, cooking directions, source information, and extracted food entities. The complete dataset is large, so the project should use a manageable subset containing recipes relevant to the ingredients supported by the vision model.

The recipe dataset will initially support retrieval, filtering, and ranking. Training a large recipe-generation model is outside the MVP scope.

### Alternative Recipe Dataset: Recipe1M

Source: [Recipe1M project](https://pic2recipe.csail.mit.edu/)

Recipe1M contains 1,029,720 structured recipes and 887,706 associated images. Recipe records include titles, ingredient lists, and preparation instructions. It is an alternative if its format or access is more suitable for the team's implementation.

### Expected Preprocessing and Cleaning

Image preprocessing may include:

- Removing unusable, blurry, duplicate, or incorrectly labeled images.
- Resizing images to a consistent input size.
- Converting annotations to the selected model's format.
- Separating images into training, validation, and testing sets.
- Applying appropriate augmentation, such as cropping and brightness changes.
- Checking class balance and collecting more examples for weak classes.
- Keeping photographs from the same refrigerator session in one split to prevent data leakage.

Recipe preprocessing may include:

- Removing duplicate or incomplete recipes.
- Converting ingredient names to lowercase.
- Separating ingredient names from quantities and units for matching.
- Standardizing aliases such as `bell pepper` and `red bell pepper`.
- Normalizing singular and plural forms.
- Separating optional ingredients from required ingredients when possible.
- Identifying common pantry staples such as salt, pepper, oil, and water.
- Adding dietary and cuisine tags when the data supports them.
- Filtering recipes that conflict with declared allergies or restrictions.

## 3. AI Tools and Packages

### Programming Languages

- **Python:** AI inference, training, preprocessing, recommendation logic, and backend services.
- **JavaScript/TypeScript or Dart:** mobile development, depending on whether the team selects React Native or Flutter.

### Frameworks and Libraries

- **PyTorch:** deep-learning framework for training or fine-tuning the vision model.
- **Ultralytics YOLO:** transfer-learning-based detection of multiple food items in a refrigerator image.
- **OpenCV:** image loading, transformation, preprocessing, and visualization.
- **Pillow:** basic image handling.
- **pandas:** recipe-data loading, cleaning, filtering, and organization.
- **NumPy:** numerical operations and array processing.
- **scikit-learn:** data splitting, evaluation metrics, and possible recommendation baselines.
- **spaCy or NLTK:** optional ingredient text normalization.
- **Matplotlib or Seaborn:** training curves, confusion matrices, and evaluation charts.

The initial recommendation engine should use deterministic filtering and scoring. Each recipe can be scored using ingredient coverage, number of missing ingredients, dietary compatibility, allergen exclusions, and cuisine preference. TF-IDF, cosine similarity, or sentence embeddings may be evaluated later if simple matching is insufficient.

### Supporting Tools

- Jupyter Notebook or Google Colab for exploration, training, and experiments.
- GitHub for version control and collaboration.
- Roboflow or CVAT for bounding-box annotation.
- Visual Studio Code for development.
- React Native or Flutter for a cross-platform mobile client.
- FastAPI or Flask for a Python API; only one should be selected.

## 4. Process and Workflow Overview

1. **Define classes:** Choose approximately 10–20 common refrigerator ingredients for the MVP.
2. **Collect and clean images:** Combine appropriate public images with custom refrigerator photos, label visible items, remove unusable samples, and prevent data leakage.
3. **Prepare recipes:** Select a manageable recipe subset and normalize its ingredients.
4. **Split the image data:** Use training, validation, and held-out testing subsets, such as 70%, 15%, and 15%.
5. **Train the vision model:** Fine-tune a pretrained YOLO detector rather than training from scratch.
6. **Validate the model:** Monitor validation metrics, select the best checkpoint, and evaluate it once on the held-out test data.
7. **Confirm ingredients:** Show detections to the user and allow additions, removals, and corrections.
8. **Collect preferences:** Accept pantry items, allergies, dietary restrictions, and cuisine preferences.
9. **Filter and rank recipes:** Remove incompatible recipes, calculate matched and missing ingredients, and rank compatible recipes.
10. **Present results:** Show the annotated image, confirmed ingredients, and recipe groups with transparent match information.

### High-Level Flow

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
Allergen/dietary filtering
        |
        v
Ingredient matching and recipe ranking
        |
        v
"Make Now" and "Missing 1–2 Items" results
```

## 5. Expected Output

The application will accept a refrigerator image and return a list of recognized food items. After the user confirms the list and enters preferences or restrictions, the system will return ranked recipe recommendations.

An example detection result might include:

- Eggs — 94% confidence
- Milk — 89% confidence
- Tomato — 87% confidence
- Bell pepper — 81% confidence
- Cheese — 78% confidence

An example recommendation result might include:

### Make Now

**Vegetable and Cheese Omelet**

- Matched: eggs, milk, tomato, bell pepper, cheese
- Missing: none

### Missing 1–2 Items

**Vegetable Breakfast Burrito**

- Matched: eggs, tomato, bell pepper, cheese
- Missing: tortillas

### Measures of Success

The computer vision component may be evaluated using:

- Precision and recall.
- Mean Average Precision (mAP).
- Per-class performance and a confusion matrix.
- Percentage of supported ingredients correctly detected.
- Average number of user corrections per image.

The recommendation component may be evaluated using:

- Accuracy of matched- and missing-ingredient lists.
- Ingredient-coverage percentage.
- Percentage of results that satisfy selected dietary restrictions.
- Percentage of results that exclude declared allergens.
- User ratings of relevance and practicality.

Usability may be evaluated with college students using task completion, correction effort, ease-of-use ratings, clarity of missing-item information, and likelihood of reuse.

No recognition or filtering result should be described as an allergen-safety guarantee. Users must be instructed to verify product labels and original recipe sources.

## 6. Final Envisioned System of Deployment

The final product is envisioned as a mobile application for Android and iOS. Users will use a phone camera to photograph refrigerator contents or select an existing image.

The client will send the image to the computer vision component, display the detected ingredients for confirmation, collect preferences, and present recipe recommendations. A cross-platform framework such as React Native or Flutter may allow one codebase to support both platforms. Python may provide AI inference, recommendation logic, and backend processing through FastAPI or Flask.

For the class demonstration, the mobile application and AI services may run locally or over the same network. A later version could host the model, recipe data, and backend in a cloud environment. The final framework and hosting choices remain to be selected by the team.

## 7. Prototype: Minimum Viable Product

The MVP is a working mobile prototype that accepts one refrigerator image, detects a limited set of common ingredients, allows corrections, and recommends recipes from the confirmed list.

The MVP will include:

- Camera capture or upload of one refrigerator image.
- Detection of approximately 10–20 supported food classes.
- Bounding boxes, labels, and confidence scores.
- Addition, removal, and correction of detected ingredients.
- Manual entry of basic pantry items.
- A small set of dietary and allergy filters.
- **Make Now** recommendations.
- Recommendations missing no more than one or two major ingredients.
- Matched and missing ingredient details for each recipe.

The MVP will be tested with refrigerator images excluded from training. Team members and a small group of college students will review detections and judge the usefulness of the recommendations. The final demonstration will cover the complete workflow from image capture through recommendation.

## 8. References

- [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset)
- [A Hierarchical Grocery Store Image Dataset with Visual and Semantic Labels](https://arxiv.org/abs/1901.00711)
- [RecipeNLG project](https://recipenlg.cs.put.poznan.pl/)
- [RecipeNLG paper](https://aclanthology.org/2020.inlg-1.4/)
- [Recipe1M project and paper](https://pic2recipe.csail.mit.edu/)
- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [OpenCV documentation](https://docs.opencv.org/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)


# Development Iteration Guidelines

## Purpose

Develop the Fridge-to-Recipe mobile application in testable increments. Complete the core behavior of each iteration before expanding scope. If time becomes limited, prioritize Iterations 1–3 because they form the essential MVP path.

## Iteration 1: Dataset Preparation and Ingredient Recognition

### Goal

Establish the image-data pipeline and determine whether selected foods can be recognized in realistic refrigerator photographs.

### Tasks

- Select approximately 10–20 common ingredient classes.
- Review and subset the Grocery Store Dataset.
- Collect and label representative refrigerator photographs.
- Split data without leaking near-duplicate photo sessions across subsets.
- Fine-tune a pretrained object detector.
- Evaluate the model on held-out images.

### Deliverables

- Versioned class list, annotations, and documented splits.
- Reproducible training configuration and saved checkpoint.
- Baseline precision, recall, mAP, and per-class results.
- Example images containing predicted boxes and confidence scores.

### Completion Criteria

The model processes a new refrigerator image and returns structured detections for supported ingredients. Weaknesses and unsupported foods are documented.

## Iteration 2: Recipe Data and Recommendation Engine

### Goal

Convert a confirmed ingredient list into relevant and explainable recommendations.

### Tasks

- Select and clean a manageable recipe subset.
- Normalize ingredient names and define aliases and pantry staples.
- Implement matched ingredients, missing ingredients, and coverage calculations.
- Filter incompatible recipes before ranking.
- Produce **Make Now** and **Missing 1–2 Items** groups.
- Test exact matches, partial matches, aliases, restrictions, and allergens.

### Deliverables

- Reproducible recipe-preparation process.
- Ingredient-normalization rules.
- Deterministic recommendation module and automated tests.

### Completion Criteria

A manually supplied ingredient list produces correctly grouped recipes with accurate matched- and missing-ingredient explanations.

## Iteration 3: Initial Mobile Application and Integration

### Goal

Join image recognition and recipe matching in a usable mobile workflow.

### Tasks

- Select and document the mobile framework.
- Add camera and photo-library input.
- Connect image input to the vision component.
- Display detections and confidence values.
- Allow ingredients to be added, removed, or corrected.
- Submit confirmed ingredients to the recommender.
- Display grouped results and handle loading, empty, invalid-image, and service-error states.

### Deliverables

- Runnable mobile prototype.
- Documented client/backend API contract.
- End-to-end demonstration data and integration tests where practical.

### Completion Criteria

A user can take or upload a photo, correct detections, and receive understandable recommendations without manually interacting with backend code.

## Iteration 4: Preferences, Allergies, and Dietary Restrictions

### Goal

Personalize results and add responsible restriction handling.

### Tasks

- Add pantry items, dietary preferences, allergen exclusions, and cuisine preferences.
- Apply restrictions before recipe ranking.
- Add a visible warning that the system cannot guarantee allergen safety.
- Test conflicting and combined preferences.

### Deliverables

- Preference-entry interface.
- Updated filtering and scoring.
- Safety warning, documented limitations, and automated filter tests.

### Completion Criteria

Recommendations honor supported filters, and the interface clearly instructs users to verify ingredients and product labels.

## Iteration 5: Evaluation and Improvement

### Goal

Use measured results and student feedback to improve accuracy, relevance, and usability.

### Tasks

- Evaluate held-out refrigerator images and categorize errors.
- Add representative data for weak classes and retrain when justified.
- Verify missing-ingredient calculations on labeled examples.
- Conduct usability sessions with college students.
- Measure correction effort, task completion, relevance, and clarity.
- Fix the highest-impact issues before adding optional features.

### Deliverables

- Before-and-after model and recommendation metrics.
- Anonymized usability findings.
- Prioritized issues and implemented fixes.

### Completion Criteria

The team can demonstrate measured improvement or justify the retained configuration. The main workflow has no unresolved critical usability or safety defects.

## Iteration 6: Final Integration and Demonstration

### Goal

Prepare a stable, documented application and evidence-based presentation.

### Tasks

- Integrate the final model and recommendation logic.
- Verify the available Android and/or iOS targets.
- Test the complete workflow and failure states.
- Prepare repeatable demonstration images and backup results.
- Record final model, recommendation, and usability findings.
- Document installation, setup, datasets, model files, and limitations.

### Deliverables

- Final mobile prototype and model configuration.
- Setup and demonstration instructions.
- Evaluation summary and presentation screenshots.
- Known-limitations and future-work list.

### Completion Criteria

Another team member can follow the documentation and run the demonstration from image input through detection review, preference filtering, and both recommendation groups.

## Suggested Schedule

| Time | Iteration | Primary outcome |
| --- | --- | --- |
| Weeks 1–2 | Iteration 1 | Baseline ingredient detector |
| Weeks 3–4 | Iteration 2 | Tested recipe recommender |
| Weeks 5–6 | Iteration 3 | End-to-end mobile MVP |
| Week 7 | Iteration 4 | Preferences and restrictions |
| Week 8 | Iteration 5 | Evaluation and improvements |
| Week 9 | Iteration 6 | Final integration and presentation |

## Scope Control

If the schedule slips, reduce the number of supported food classes and recipe categories before removing ingredient confirmation, missing-item explanations, testing, or safety warnings. User accounts, saved favorites, delivery integration, expiration tracking, and cloud scaling are stretch goals.

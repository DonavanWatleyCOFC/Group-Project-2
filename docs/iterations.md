# Development Iteration Guidelines

## Purpose

These iterations deliver the Fridge-to-Recipe mobile application in testable increments. Complete the core behavior of an iteration before expanding its scope. If time becomes limited, prioritize Iterations 1–3 because they form the MVP's essential path.

## Iteration 1: Dataset Preparation and Ingredient Recognition

### Goal

Establish the data pipeline and determine whether the selected foods can be recognized in realistic refrigerator images.

### Tasks

- Select approximately 10–20 common ingredient classes.
- Review and subset the Grocery Store Dataset.
- Collect representative refrigerator photographs.
- Label each supported item with a class and bounding box.
- Divide data into training, validation, and test sets without photo-session leakage.
- Fine-tune a pretrained object-detection model.
- Evaluate it on images not used during training.

### Deliverables

- Versioned class list and annotation format.
- Documented dataset splits.
- Reproducible training configuration.
- Saved model checkpoint.
- Baseline precision, recall, mAP, and per-class results.
- Sample images containing predicted boxes, labels, and confidence scores.

### Completion Criteria

The model can process a new refrigerator image and return structured detections for supported ingredients. Known weaknesses and unsupported foods are documented.

## Iteration 2: Recipe Data and Recommendation Engine

### Goal

Convert a confirmed ingredient list into relevant, explainable recipe recommendations.

### Tasks

- Select a manageable recipe-data subset.
- Clean and standardize ingredient names.
- Define ingredient aliases and pantry staples.
- Implement dietary and allergen metadata where supported.
- Calculate matched ingredients, missing ingredients, and coverage.
- Filter incompatible recipes before ranking.
- Produce **Make Now** and **Missing 1–2 Items** groups.

### Deliverables

- Clean recipe subset or reproducible preparation script.
- Ingredient-normalization rules.
- Deterministic recommendation module.
- Automated tests for exact matches, partial matches, aliases, restrictions, and allergens.

### Completion Criteria

Given a manually supplied ingredient list, the system returns correctly grouped and ranked recipes with accurate matched- and missing-ingredient explanations.

## Iteration 3: Initial Mobile Application and Integration

### Goal

Join image recognition and recipe matching in a usable mobile workflow.

### Tasks

- Select and document the mobile framework.
- Create camera and photo-library input.
- Connect image input to the vision component.
- Display detections and confidence values.
- Allow ingredients to be added, removed, or corrected.
- Submit the confirmed list to the recommendation component.
- Display grouped recipe results and match details.
- Add loading, empty, invalid-image, and service-error states.

### Deliverables

- Installable or runnable mobile prototype.
- Documented API contract between client and backend.
- End-to-end demonstration data.
- Integration tests for the main workflow where practical.

### Completion Criteria

A user can take or upload a photo, correct the detected list, and receive understandable recipe recommendations without manually interacting with backend code.

## Iteration 4: Preferences, Allergies, and Dietary Restrictions

### Goal

Personalize results and add responsible restriction handling.

### Tasks

- Add pantry items and condiments.
- Add dietary preferences and restrictions.
- Add declared allergen exclusions.
- Add cuisine preferences when recipe metadata permits.
- Apply restrictions before ranking.
- Add a visible warning that the system cannot guarantee allergen safety.
- Create tests for conflicting and combined preferences.

### Deliverables

- Preference-entry interface.
- Updated recommendation filters and scoring.
- Safety warning and documented limitations.
- Automated filtering tests.

### Completion Criteria

Recommendations consistently honor supported filters, and the UI clearly communicates that users must verify ingredients and product labels.

## Iteration 5: Evaluation and Improvement

### Goal

Use model results and student feedback to improve accuracy, relevance, and ease of use.

### Tasks

- Evaluate new, held-out refrigerator images.
- Record false positives, false negatives, and frequent confusions.
- Add representative data for weak classes and retrain when justified.
- Verify missing-ingredient calculations on a labeled recipe test set.
- Conduct usability sessions with college students.
- Measure correction effort, task completion, recommendation relevance, and clarity.
- Fix the highest-impact issues before adding optional features.

### Deliverables

- Before-and-after model metrics.
- Recommendation evaluation results.
- Anonymized usability findings.
- Prioritized issue list and implemented fixes.

### Completion Criteria

The team can show measured improvement or explain why the final configuration was retained. The core workflow succeeds on representative inputs with no unresolved critical usability or safety defect.

## Iteration 6: Final Integration and Demonstration

### Goal

Prepare a stable, documented application and evidence-based final presentation.

### Tasks

- Integrate the selected final model and recommendation logic.
- Verify behavior on representative Android and/or iOS targets available to the team.
- Test the complete workflow and failure states.
- Prepare repeatable demonstration images and backup results.
- Record final precision, recall, mAP, recommendation, and usability findings.
- Document installation, setup, datasets, model files, and known limitations.

### Deliverables

- Final mobile prototype.
- Final model and configuration.
- Setup and demonstration instructions.
- Evaluation summary and presentation-ready screenshots.
- Known-limitations and future-work list.

### Completion Criteria

Another team member can follow the documentation and run the demonstration. The application supports image input, detection review, preference filtering, and both recommendation groups from beginning to end.

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

If the schedule slips, reduce the number of supported food classes and recipe categories before removing ingredient confirmation, transparent missing-item explanations, testing, or safety warnings. User accounts, saved favorites, shopping delivery, expiration tracking, and cloud scaling are stretch goals rather than MVP requirements.

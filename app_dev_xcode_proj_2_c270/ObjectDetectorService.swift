// xcode: set sdk=iOS

//
//  ObjectDetectorService.swift
//  app_dev_xcode_proj_2_c270
//
import CoreML
import Vision
import UIKit
import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

// 1. Data models
struct DetectedObject: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}

struct RecipeItem: Identifiable, Codable {
    var id: String { title }
    let title: String
    let matched: [String]
    let missing: [String]
    let coverage: Double
    let ingredients: [String]
    let directions: [String]
}

struct RecipeResponse: Codable {
    let count_make_now: Int
    let count_missing: Int
    let make_now: [RecipeItem]
    let missing_items: [RecipeItem]
}

// 2. Detection service (Owen's Vision CoreML Engine)
@Observable
class ObjectDetectorService {
    var detections: [DetectedObject] = []
    private var visionModel: VNCoreMLModel?

    init() {
        setupModel()
    }

    private func setupModel() {
        do {
            let coreMLModel = try best(configuration: MLModelConfiguration()).model
            self.visionModel = try VNCoreMLModel(for: coreMLModel)
        } catch {
            print("Failed to load model: \(error.localizedDescription)")
        }
    }

    func detect(image: UIImage, completion: (([String]) -> Void)? = nil) {
        guard let visionModel = visionModel else { return }
        guard let cgImage = image.cgImage else { return }

        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            if let error = error {
                print("Detection error: \(error.localizedDescription)")
                return
            }

            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }

            let parsedObjects: [DetectedObject] = results.compactMap { observation in
                guard let topLabel = observation.labels.first,
                      topLabel.confidence > 0.15 else { return nil }
                return DetectedObject(
                    label: topLabel.identifier,
                    confidence: topLabel.confidence,
                    boundingBox: observation.boundingBox
                )
            }

            DispatchQueue.main.async {
                self?.detections = parsedObjects
                let uniqueLabels = Array(Set(parsedObjects.map { $0.label.lowercased() }))
                completion?(uniqueLabels)
            }
        }

        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                print("Detection request error: \(error.localizedDescription)")
            }
        }
    }
}

// 3. Full Featured Native iOS View
struct ContentView: View {
    @State private var detector = ObjectDetectorService()
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var isFileImporterPresented = false

    // Interactive Ingredients & Kitchen Inventory
    @State private var activeIngredients: [String] = []
    @State private var detectedIngredients: Set<String> = []  // tracks AI-detected items
    @State private var newIngredientText = ""

    // Preferences & Dietary Filters
    @State private var selectedDiet = "No Restrictions"
    @State private var selectedCuisine = "All Cuisines"
    @State private var selectedAllergens: Set<String> = []

    // Recipe Results
    @State private var makeNowRecipes: [RecipeItem] = []
    @State private var missingRecipes: [RecipeItem] = []
    @State private var isLoading = false
    @State private var latestRecipeRequestID = UUID()
    @State private var selectedTab = 0 // 0 = Ready, 1 = Missing

    let dietOptions = ["No Restrictions", "Vegetarian", "Vegan", "Quick Meals"]
    let cuisineOptions = ["All Cuisines", "Mexican", "Italian", "Asian", "Mediterranean", "American"]
    let allergenList = ["Nuts", "Dairy", "Gluten", "Eggs", "Seafood"]

    // Common fridge/pantry items for quick-add suggestions
    let quickAddSuggestions = [
        "eggs", "milk", "butter", "cheese", "yogurt",
        "chicken", "beef", "onion", "garlic", "tomato",
        "potato", "carrot", "broccoli", "spinach", "rice",
        "pasta", "olive oil", "lemon", "salt", "pepper"
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Upload / Camera Card Preview
                    ZStack {
                        if let selectedImage {
                            Image(uiImage: selectedImage)
                                .resizable()
                                .scaledToFit()
                                .overlay {
                                    GeometryReader { geometry in
                                        ForEach(detector.detections) { item in
                                            let rect = convertVisionRect(item.boundingBox, to: geometry.size)
                                            ZStack(alignment: .topLeading) {
                                                Rectangle()
                                                    .stroke(Color.green, lineWidth: 2.5)
                                                    .frame(width: rect.width, height: rect.height)
                                                Text("\(item.label.capitalized) \(Int(item.confidence * 100))%")
                                                    .font(.system(size: 10, weight: .bold))
                                                    .padding(.horizontal, 6)
                                                    .padding(.vertical, 3)
                                                    .background(Color.green)
                                                    .foregroundColor(.black)
                                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                                            }
                                            .offset(x: rect.minX, y: rect.minY)
                                        }
                                    }
                                }
                                .clipShape(RoundedRectangle(cornerRadius: 18))
                                .shadow(color: .black.opacity(0.12), radius: 10, y: 5)
                        } else {
                            VStack(spacing: 12) {
                                Image(systemName: "camera.viewfinder")
                                    .font(.system(size: 48))
                                    .foregroundColor(.orange)
                                Text("Upload Fridge Photo")
                                    .font(.headline)
                                Text("Pick a photo from your gallery or files to scan food items")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 20)
                            }
                            .frame(maxWidth: .infinity)
                            .frame(height: 200)
                            .background(Color(uiColor: .secondarySystemBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 20))
                            .overlay(
                                RoundedRectangle(cornerRadius: 20)
                                    .stroke(Color.orange.opacity(0.4), style: StrokeStyle(lineWidth: 2, dash: [6]))
                            )
                        }
                    }

                    // Photos Picker Action Button
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        HStack {
                            Image(systemName: "photo.on.rectangle.angled")
                            Text(selectedImage == nil ? "Select Fridge Photo" : "Scan Another Photo")
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Color.orange)
                        .foregroundColor(.white)
                        .font(.headline)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                    }

                    // SECTION 1: Kitchen Inventory (Verify & Add/Remove)
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("🧺 Kitchen Inventory (\(activeIngredients.count))")
                                    .font(.headline)
                                Text("✅ AI detected · ✏️ Added by you — tap ✕ to remove")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            if !activeIngredients.isEmpty {
                                Button("Clear All") {
                                    activeIngredients.removeAll()
                                    detectedIngredients.removeAll()
                                }
                                .font(.caption.bold())
                                .foregroundColor(.red)
                            }
                        }

                        // Input bar to add extra food items / pantry items
                        HStack {
                            TextField("Add pantry item (e.g. olive oil, garlic)", text: $newIngredientText)
                                .textFieldStyle(.roundedBorder)
                                .autocorrectionDisabled()
                                .onSubmit { addManualIngredient() }

                            Button(action: addManualIngredient) {
                                Image(systemName: "plus.circle.fill")
                                    .font(.title3)
                                    .foregroundColor(.orange)
                            }
                            .disabled(newIngredientText.trimmingCharacters(in: .whitespaces).isEmpty)
                        }

                        // Interactive Chips — green = AI detected, blue = manually added
                        if activeIngredients.isEmpty {
                            Text("Scan a photo or tap a suggestion below to get started!")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        } else {
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 8) {
                                    ForEach(activeIngredients, id: \.self) { item in
                                        let isDetected = detectedIngredients.contains(item)
                                        HStack(spacing: 4) {
                                            Text(isDetected ? "✅" : "✏️")
                                                .font(.caption2)
                                            Text(item.capitalized)
                                                .font(.caption.bold())
                                            Button(action: { removeIngredient(item) }) {
                                                Image(systemName: "xmark.circle.fill")
                                                    .font(.caption)
                                            }
                                        }
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 6)
                                        .background(isDetected ? Color.green.opacity(0.15) : Color.blue.opacity(0.12))
                                        .foregroundColor(isDetected ? .green : .blue)
                                        .clipShape(Capsule())
                                    }
                                }
                            }
                        }

                        // Quick-add suggestion chips
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Quick Add:")
                                .font(.caption.bold())
                                .foregroundColor(.secondary)
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 6) {
                                    ForEach(quickAddSuggestions.filter { !activeIngredients.contains($0) }, id: \.self) { suggestion in
                                        Button(action: {
                                            activeIngredients.append(suggestion)
                                        }) {
                                            HStack(spacing: 3) {
                                                Image(systemName: "plus")
                                                    .font(.caption2.bold())
                                                Text(suggestion.capitalized)
                                                    .font(.caption.bold())
                                            }
                                            .padding(.horizontal, 10)
                                            .padding(.vertical, 5)
                                            .background(Color.orange.opacity(0.12))
                                            .foregroundColor(.orange)
                                            .clipShape(Capsule())
                                            .overlay(Capsule().stroke(Color.orange.opacity(0.3), lineWidth: 1))
                                        }
                                    }
                                }
                            }
                        }
                    }
                    .padding()
                    .background(Color(uiColor: .secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    // SECTION 2: Preferences, Cuisine & Allergens
                    VStack(alignment: .leading, spacing: 12) {
                        Text("🥗 Dietary, Cuisine & Allergens")
                            .font(.headline)

                        // Lifestyle & Cuisine Menus
                        HStack(spacing: 12) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Diet").font(.caption.bold()).foregroundColor(.secondary)
                                Picker("Diet", selection: $selectedDiet) {
                                    ForEach(dietOptions, id: \.self) { Text($0) }
                                }
                                .pickerStyle(.menu)
                                .padding(8)
                                .background(Color(uiColor: .systemBackground))
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                            }

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Cuisine").font(.caption.bold()).foregroundColor(.secondary)
                                Picker("Cuisine", selection: $selectedCuisine) {
                                    ForEach(cuisineOptions, id: \.self) { Text($0) }
                                }
                                .pickerStyle(.menu)
                                .padding(8)
                                .background(Color(uiColor: .systemBackground))
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        }

                        // Allergen Avoidance Badges
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Exclude Allergens:").font(.caption.bold()).foregroundColor(.secondary)
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 8) {
                                    ForEach(allergenList, id: \.self) { allergen in
                                        let isSelected = selectedAllergens.contains(allergen)
                                        Button(action: { toggleAllergen(allergen) }) {
                                            Text(allergen)
                                                .font(.caption.bold())
                                                .padding(.horizontal, 10)
                                                .padding(.vertical, 6)
                                                .background(isSelected ? Color.red.opacity(0.2) : Color(uiColor: .systemBackground))
                                                .foregroundColor(isSelected ? .red : .primary)
                                                .overlay(
                                                    Capsule().stroke(isSelected ? Color.red : Color.gray.opacity(0.3), lineWidth: 1)
                                                )
                                                .clipShape(Capsule())
                                        }
                                    }
                                }
                            }
                        }
                    }
                    .padding()
                    .background(Color(uiColor: .secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .onChange(of: selectedDiet) { _, _ in refreshRecipes() }
                    .onChange(of: selectedCuisine) { _, _ in refreshRecipes() }

                    // SECTION 3: Recipe Results
                    if isLoading {
                        VStack(spacing: 8) {
                            ProgressView()
                            Text("Cooking up recipe matches...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 20)
                    } else if !makeNowRecipes.isEmpty || !missingRecipes.isEmpty {
                        VStack(spacing: 16) {
                            Picker("Category", selection: $selectedTab) {
                                Text("🟢 Ready to Cook (\(makeNowRecipes.count))").tag(0)
                                Text("🛒 Need 1-2 Items (\(missingRecipes.count))").tag(1)
                            }
                            .pickerStyle(.segmented)

                            if selectedTab == 0 {
                                if makeNowRecipes.isEmpty {
                                    Text("No recipes where you have 100% of the items.")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                        .padding()
                                } else {
                                    ForEach(makeNowRecipes) { recipe in
                                        RecipeCardView(recipe: recipe, isReady: true)
                                    }
                                }
                            } else {
                                if missingRecipes.isEmpty {
                                    Text("No near-match recipes found.")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                        .padding()
                                } else {
                                    ForEach(missingRecipes) { recipe in
                                        RecipeCardView(recipe: recipe, isReady: false)
                                    }
                                }
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Fridge2Fork 🍳")
        }
        .onChange(of: activeIngredients) { _, _ in refreshRecipes() }
        .onChange(of: selectedItem) { _, newItem in
            Task {
                if let data = try? await newItem?.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {
                    selectedImage = uiImage
                    isLoading = true
                    detector.detect(image: uiImage) { detectedLabels in
                        // Add newly detected labels to active inventory, track as AI-detected
                        var didAddIngredient = false
                        for label in detectedLabels {
                            if !activeIngredients.contains(label) {
                                activeIngredients.append(label)
                                detectedIngredients.insert(label)
                                didAddIngredient = true
                            }
                        }
                        if !didAddIngredient {
                            refreshRecipes()
                        }
                    }
                }
            }
        }
    }

    private func addManualIngredient() {
        let trimmed = newIngredientText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !trimmed.isEmpty && !activeIngredients.contains(trimmed) {
            activeIngredients.append(trimmed)
            newIngredientText = ""
        }
    }

    private func removeIngredient(_ item: String) {
        activeIngredients.removeAll { $0 == item }
        detectedIngredients.remove(item)
    }

    private func toggleAllergen(_ allergen: String) {
        if selectedAllergens.contains(allergen) {
            selectedAllergens.remove(allergen)
        } else {
            selectedAllergens.insert(allergen)
        }
        refreshRecipes()
    }

    private func refreshRecipes() {
        let requestID = UUID()
        latestRecipeRequestID = requestID

        guard !activeIngredients.isEmpty else {
            makeNowRecipes = []
            missingRecipes = []
            isLoading = false
            return
        }

        isLoading = true
        guard let url = URL(string: "http://127.0.0.1:8000/recipes") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "ingredients": activeIngredients,
            "diet_filter": selectedDiet,
            "cuisine_filter": selectedCuisine,
            "exclude_allergens": Array(selectedAllergens)
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTask(with: request) { data, _, error in
            DispatchQueue.main.async {
                guard self.latestRecipeRequestID == requestID else { return }
                self.isLoading = false
                guard let data = data, error == nil else { return }
                if let response = try? JSONDecoder().decode(RecipeResponse.self, from: data) {
                    self.makeNowRecipes = response.make_now
                    self.missingRecipes = response.missing_items
                }
            }
        }.resume()
    }

    private func convertVisionRect(_ boundingBox: CGRect, to viewSize: CGSize) -> CGRect {
        let width = boundingBox.width * viewSize.width
        let height = boundingBox.height * viewSize.height
        let x = boundingBox.minX * viewSize.width
        let y = (1 - boundingBox.maxY) * viewSize.height
        return CGRect(x: x, y: y, width: width, height: height)
    }
}

// 4. Native SwiftUI Recipe Card
struct RecipeCardView: View {
    let recipe: RecipeItem
    let isReady: Bool
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(recipe.title)
                    .font(.headline)
                    .foregroundColor(.primary)
                Spacer()
                Text(isReady ? "100% Match" : "Need \(recipe.missing.count)")
                    .font(.caption.bold())
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(isReady ? Color.green.opacity(0.15) : Color.orange.opacity(0.15))
                    .foregroundColor(isReady ? .green : .orange)
                    .clipShape(Capsule())
            }

            if !isReady {
                Text("Need to buy: \(recipe.missing.joined(separator: ", "))")
                    .font(.caption.bold())
                    .foregroundColor(.red)
            }

            DisclosureGroup(isExpanded: $isExpanded) {
                VStack(alignment: .leading, spacing: 10) {
                    Divider()

                    Text("Ingredients:")
                        .font(.subheadline.bold())

                    ForEach(recipe.ingredients, id: \.self) { ing in
                        Text("• \(ing)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Text("Instructions:")
                        .font(.subheadline.bold())
                        .padding(.top, 4)

                    ForEach(Array(recipe.directions.enumerated()), id: \.offset) { idx, step in
                        Text("\(idx + 1). \(step)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.bottom, 2)
                    }
                }
                .padding(.top, 4)
            } label: {
                Text(isExpanded ? "Hide Instructions" : "View Cooking Directions")
                    .font(.subheadline.bold())
                    .foregroundColor(.orange)
            }
        }
        .padding()
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

#Preview {
    ContentView()
}

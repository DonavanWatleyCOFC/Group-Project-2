// xcode: set sdk=iOS

//
//  ObjectDetectorService.swift
//  app_dev_xcode_proj_2_c270
//
//  Created by Owen Duckwall on 11/16/26.
//
import CoreML
import Vision
import UIKit
import SwiftUI
import PhotosUI

// 1. Data model required by ObjectDetectorService
struct DetectedObject: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}

// 2. Your detection service
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

    func detect(image: UIImage) {
        guard let visionModel = visionModel,
              let ciImage = CIImage(image: image) else { return }

        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            if let error = error {
                print("Detection error: \(error.localizedDescription)")
                return
            }

            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                return
            }

            let parsedObjects: [DetectedObject] = results.compactMap { observation in
                guard let topLabel = observation.labels.first,
                      topLabel.confidence > 0.5 else {
                    return nil
                }

                return DetectedObject(
                    label: topLabel.identifier,
                    confidence: topLabel.confidence,
                    boundingBox: observation.boundingBox
                )
            }

            DispatchQueue.main.async {
                self?.detections = parsedObjects
            }
        }

        request.imageCropAndScaleOption = .scaleFill

        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage)
            try? handler.perform([request])
        }
    }
}

// 3. The UI View that Xcode previews
struct ContentView: View {
    @State private var detector = ObjectDetectorService()
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?

    var body: some View {
        VStack(spacing: 20) {
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
                                        .stroke(Color.green, lineWidth: 2)
                                        .frame(width: rect.width, height: rect.height)

                                    Text("\(item.label) \(Int(item.confidence * 100))%")
                                        .font(.caption2.bold())
                                        .padding(.horizontal, 4)
                                        .padding(.vertical, 2)
                                        .background(Color.green)
                                        .foregroundStyle(.black)
                                }
                                .offset(x: rect.minX, y: rect.minY)
                            }
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                ContentUnavailableView(
                    "No Photo Selected",
                    systemImage: "photo.badge.plus",
                    description: Text("Tap below to test your model")
                )
            }

            PhotosPicker("Select Photo", selection: $selectedItem, matching: .images)
                .buttonStyle(.borderedProminent)
        }
        .padding()
        .onChange(of: selectedItem) { _, newItem in
            Task {
                if let data = try? await newItem?.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {
                    selectedImage = uiImage
                    detector.detect(image: uiImage)
                }
            }
        }
    }

    private func convertVisionRect(_ boundingBox: CGRect, to viewSize: CGSize) -> CGRect {
        let width = boundingBox.width * viewSize.width
        let height = boundingBox.height * viewSize.height
        let x = boundingBox.minX * viewSize.width
        let y = (1 - boundingBox.maxY) * viewSize.height
        return CGRect(x: x, y: y, width: width, height: height)
    }
}

// 4. Triggers the Xcode Preview Canvas
#Preview {
    ContentView()
}

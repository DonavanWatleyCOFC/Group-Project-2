//
//  ObjectDetectorService.swift
//  app_dev_xcode_proj_2_c270
//
//  Created by Owen Duckwall on 11/16/26.
//
import CoreML
import Vision
import UIKit

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

            // 1. Cast results to VNRecognizedObjectObservation
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                return
            }

            // 2. Map and filter detections (e.g., confidence > 50%)
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

            // 3. Publish to UI on main thread
            DispatchQueue.main.async {
                self?.detections = parsedObjects
            }
        }

        // .scaleFill is usually best for object detectors trained on square inputs (e.g., YOLO)
        request.imageCropAndScaleOption = .scaleFill

        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage)
            try? handler.perform([request])
        }
    }
}

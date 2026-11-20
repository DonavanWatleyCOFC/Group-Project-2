// xcode: set sdk=iOS

//
//  VisualBoxing.swift
//  app_dev_xcode_proj_2_c270
//
//  Created by Owen Duckwall on 11/16/26.
//
import SwiftUI

struct DetectionOverlayView: View {
    let image: UIImage
    let detections: [DetectedObject]

    var body: some View {
        Image(uiImage: image)
            .resizable()
            .scaledToFit()
            .overlay {
                GeometryReader { geometry in
                    ForEach(detections) { item in
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
    }

    /// Converts normalized bottom-left Vision coordinates to top-left SwiftUI view coordinates
    private func convertVisionRect(_ boundingBox: CGRect, to viewSize: CGSize) -> CGRect {
        let width = boundingBox.width * viewSize.width
        let height = boundingBox.height * viewSize.height
        let x = boundingBox.minX * viewSize.width
        // Flip Y-axis from bottom-left origin to top-left origin
        let y = (1 - boundingBox.maxY) * viewSize.height

        return CGRect(x: x, y: y, width: width, height: height)
    }
}

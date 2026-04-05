import Foundation
import SwiftUI

enum PipelineStage: String, Codable {
    case pending, rfdiffusion, mpnn, rf2, complete, failed
}

enum QualityTier {
    case high, medium, low, unknown

    var color: Color {
        switch self {
        case .high: .green
        case .medium: .yellow
        case .low: .red
        case .unknown: .secondary
        }
    }
}

struct NanobodyDesign: Identifiable {
    let id: Int
    var backbonePDB: URL?
    var sequencePDB: URL?
    var validatedPDB: URL?
    var plddt: Double?
    var pae: Double?
    var ipae: Double?
    var pBind: Double?
    var mpnnScore: Double?
    var cdrRMSD: Double?
    var totalTime: Double?
    var rfdiffusionTime: Double?
    var mpnnTime: Double?
    var rf2Time: Double?
    var stage: PipelineStage = .pending
    var error: String?

    // Backbone geometry
    var caCaBond: Double?
    var radiusOfGyration: Double?
    var proteinLength: Int?

    // Diffusion trajectory (one PDB per denoising step)
    var trajectoryPDBs: [URL] = []

    // RF2 intermediate all-atom PDB (updated each recycle)
    var rf2IntermediatePDB: URL?

    var qualityTier: QualityTier {
        guard let plddt else { return .unknown }
        if plddt > 0.85 { return .high }
        if plddt > 0.80 { return .medium }
        return .low
    }

    /// Best available PDB for visualization.
    var displayPDB: URL? {
        validatedPDB ?? sequencePDB ?? backbonePDB
    }
}

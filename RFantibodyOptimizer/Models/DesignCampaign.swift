import Foundation
import Observation

enum CampaignStatus: String {
    case idle, initializing, running, completed, failed, cancelled
}

@MainActor
@Observable
class DesignCampaign {
    var config = DesignConfig()
    var designs: [NanobodyDesign] = []
    var status: CampaignStatus = .idle
    var currentDesignIndex: Int = 0
    var currentStage: PipelineStage = .pending
    var currentStep: Int = 0
    var totalSteps: Int = 0
    var rf2CurrentRecycle: Int = 0
    var rf2TotalRecycles: Int = 0
    var rf2CaRMSD: Double?
    var startTime: Date?
    var endTime: Date?
    var selectedDesignID: Int?
    var errorMessage: String?
    var outputDir: URL?
    var pythonLog: String = ""

    // MARK: - Computed Properties

    var elapsedTime: TimeInterval {
        guard let start = startTime else { return 0 }
        let end = endTime ?? Date()
        return end.timeIntervalSince(start)
    }

    var elapsedTimeFormatted: String {
        let t = Int(elapsedTime)
        let m = t / 60
        let s = t % 60
        return String(format: "%d:%02d", m, s)
    }

    var completedDesigns: [NanobodyDesign] {
        designs.filter { $0.stage == .complete }
    }

    var estimatedTimeRemaining: TimeInterval? {
        guard completedDesigns.count > 0 else { return nil }
        let avgTime = completedDesigns.compactMap(\.totalTime).reduce(0, +)
            / Double(completedDesigns.count)
        let remaining = config.numDesigns - completedDesigns.count
        return avgTime * Double(remaining)
    }

    var etaFormatted: String? {
        guard let eta = estimatedTimeRemaining else { return nil }
        let m = Int(eta) / 60
        let s = Int(eta) % 60
        return String(format: "~%d:%02d remaining", m, s)
    }

    // MARK: - Aggregate Statistics

    var averagePLDDT: Double? {
        let vals = completedDesigns.compactMap(\.plddt)
        guard !vals.isEmpty else { return nil }
        return vals.reduce(0, +) / Double(vals.count)
    }

    var averagePBind: Double? {
        let vals = completedDesigns.compactMap(\.pBind)
        guard !vals.isEmpty else { return nil }
        return vals.reduce(0, +) / Double(vals.count)
    }

    var highQualityCount: Int {
        completedDesigns.filter { ($0.plddt ?? 0) > 0.85 }.count
    }

    var highQualityFraction: Double? {
        guard !completedDesigns.isEmpty else { return nil }
        return Double(highQualityCount) / Double(completedDesigns.count)
    }

    // MARK: - Lifecycle

    func reset() {
        designs = []
        status = .idle
        currentDesignIndex = 0
        currentStage = .pending
        currentStep = 0
        totalSteps = 0
        rf2CurrentRecycle = 0
        rf2TotalRecycles = 0
        rf2CaRMSD = nil
        startTime = nil
        endTime = nil
        selectedDesignID = nil
        errorMessage = nil
        pythonLog = ""
    }
}

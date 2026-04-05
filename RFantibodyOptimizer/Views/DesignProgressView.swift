import SwiftUI

struct DesignProgressView: View {
    @Environment(DesignCampaign.self) private var campaign
    @Environment(PipelineRunner.self) private var runner

    var body: some View {
        VStack(spacing: 0) {
            // Progress bars — compact, no scroll
            VStack(spacing: 12) {
                // Campaign progress
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Design \(campaign.currentDesignIndex + 1) of \(campaign.config.numDesigns)")
                            .font(.subheadline)

                        Spacer()

                        // Pause / Resume
                        if runner.isPaused {
                            Button("Resume") {
                                runner.resume()
                            }
                            .buttonStyle(.bordered)
                            .tint(.blue)
                            .controlSize(.small)

                            Text("PAUSED")
                                .font(.caption.bold())
                                .foregroundStyle(.orange)
                        } else {
                            Button {
                                runner.pause()
                            } label: {
                                Label("Pause", systemImage: "pause.fill")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                        if let eta = campaign.etaFormatted {
                            Text(eta)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Text(campaign.elapsedTimeFormatted)
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    ProgressView(
                        value: Double(campaign.completedDesigns.count),
                        total: Double(max(1, campaign.config.numDesigns))
                    )
                }

                // Stage progress
                if campaign.status == .initializing {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text("Loading models...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                } else if campaign.status == .running {
                    HStack {
                        StageLabel(stage: campaign.currentStage)
                        Spacer()
                        stageDetail
                    }

                    if campaign.currentStage == .rfdiffusion && campaign.totalSteps > 0 {
                        ProgressView(
                            value: Double(campaign.currentStep),
                            total: Double(campaign.totalSteps)
                        )
                    } else if campaign.currentStage == .rf2 && campaign.rf2TotalRecycles > 0 {
                        ProgressView(
                            value: Double(campaign.rf2CurrentRecycle),
                            total: Double(campaign.rf2TotalRecycles)
                        )
                    }
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(.bar)

            Divider()

            // Full results table — fills remaining space
            ResultsTable()
        }
    }

    @ViewBuilder
    private var stageDetail: some View {
        switch campaign.currentStage {
        case .rfdiffusion:
            Text("Step \(campaign.currentStep)/\(campaign.totalSteps)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        case .mpnn:
            Text("Designing sequence...")
                .font(.caption)
                .foregroundStyle(.secondary)
        case .rf2:
            HStack(spacing: 8) {
                if campaign.rf2TotalRecycles > 0 {
                    Text("Recycle \(campaign.rf2CurrentRecycle)/\(campaign.rf2TotalRecycles)")
                        .font(.caption.monospacedDigit())
                    if let rmsd = campaign.rf2CaRMSD {
                        Text("RMSD \(rmsd, specifier: "%.2f")Å")
                            .font(.caption.monospacedDigit())
                    }
                } else {
                    Text("Validating...")
                        .font(.caption)
                }
            }
            .foregroundStyle(.secondary)
        default:
            EmptyView()
        }
    }
}

struct StageLabel: View {
    let stage: PipelineStage

    var body: some View {
        Label(stage.displayName, systemImage: stage.icon)
            .font(.subheadline.bold())
            .foregroundStyle(stage.color)
    }
}

extension PipelineStage {
    var displayName: String {
        switch self {
        case .pending: "Pending"
        case .rfdiffusion: "RFdiffusion"
        case .mpnn: "ProteinMPNN"
        case .rf2: "RF2 Validation"
        case .complete: "Complete"
        case .failed: "Failed"
        }
    }

    var icon: String {
        switch self {
        case .pending: "clock"
        case .rfdiffusion: "wand.and.stars"
        case .mpnn: "character.textbox"
        case .rf2: "checkmark.shield"
        case .complete: "checkmark.circle.fill"
        case .failed: "xmark.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .pending: .secondary
        case .rfdiffusion: .blue
        case .mpnn: .purple
        case .rf2: .orange
        case .complete: .green
        case .failed: .red
        }
    }
}

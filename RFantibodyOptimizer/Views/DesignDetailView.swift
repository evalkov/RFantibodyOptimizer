import SwiftUI

struct DesignDetailView: View {
    @Environment(DesignCampaign.self) private var campaign
    let design: NanobodyDesign
    @State private var colorScheme: ProteinColorScheme = .cdr
    @State private var trajectoryFrame: Int = 0
    @State private var showTrajectory = false
    private var hasTrajectory: Bool { !design.trajectoryPDBs.isEmpty }

    var body: some View {
        VStack(spacing: 0) {
            // 3D Protein Viewer
            ProteinViewer(
                pdbURL: activeURL,
                colorScheme: colorScheme,
                preserveCamera: showTrajectory,
                diffusionMode: showTrajectory
            )
            .frame(minHeight: 400)

            // Trajectory player bar
            if hasTrajectory && showTrajectory {
                Divider()
                TrajectoryPlayer(
                    frames: design.trajectoryPDBs,
                    currentFrame: $trajectoryFrame
                )
                .padding(.horizontal)
                .padding(.vertical, 6)
                .background(.bar)
            }

            Divider()

            // Controls + Metrics
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Viewer controls
                    HStack {
                        Picker("Color", selection: $colorScheme) {
                            ForEach(ProteinColorScheme.allCases) { scheme in
                                Text(scheme.rawValue).tag(scheme)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 300)

                        Spacer()

                        if hasTrajectory {
                            Toggle("Diffusion", isOn: $showTrajectory)
                                .onChange(of: showTrajectory) {
                                    if showTrajectory {
                                        trajectoryFrame = design.trajectoryPDBs.count - 1
                                    }
                                }
                        }

                    }

                    Divider()

                    // Metrics grid
                    Text("Design #\(design.id + 1)")
                        .font(.headline)

                    Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 8) {
                        // Structure quality
                        GridRow {
                            MetricLabel("pLDDT")
                            MetricValue(design.plddt, format: "%.4f",
                                        color: design.plddt.map { MetricColor.plddt($0) })
                            MetricLabel("PAE")
                            MetricValue(design.pae, format: "%.2f A",
                                        color: design.pae.map { MetricColor.pae($0) })
                        }

                        GridRow {
                            MetricLabel("iPAE")
                            MetricValue(design.ipae, format: "%.2f A",
                                        color: design.ipae.map { MetricColor.ipae($0) })
                            MetricLabel(campaign.config.validator == .protenix ? "ipTM" : "P(bind)")
                            MetricValue(design.pBind, format: "%.4f",
                                        color: design.pBind.map { MetricColor.pBind($0) })
                        }

                        GridRow {
                            MetricLabel("MPNN Score")
                            MetricValue(design.mpnnScore, format: "%.4f",
                                        color: design.mpnnScore.map { MetricColor.mpnn($0) })
                            MetricLabel("CDR RMSD")
                            MetricValue(design.cdrRMSD, format: "%.2f A",
                                        color: design.cdrRMSD.map { MetricColor.rmsd($0) })
                        }

                        // Geometry
                        GridRow {
                            MetricLabel("CA-CA Bond")
                            MetricValue(design.caCaBond, format: "%.3f A")
                            MetricLabel("Rg")
                            MetricValue(design.radiusOfGyration, format: "%.1f A")
                        }

                        // Timing
                        GridRow {
                            MetricLabel("Length")
                            MetricValue(design.proteinLength.map { Double($0) }, format: "%.0f res")
                            MetricLabel("Total Time")
                            MetricValue(design.totalTime, format: "%.1f s")
                        }
                    }
                    .font(.body.monospacedDigit())

                    // Stage timing breakdown
                    if let rd = design.rfdiffusionTime,
                       let mp = design.mpnnTime,
                       let rf = design.rf2Time {
                        Divider()
                        HStack(spacing: 16) {
                            TimingBar(label: "RFdiff", time: rd, color: .blue)
                            TimingBar(label: "MPNN", time: mp, color: .purple)
                            TimingBar(label: "RF2", time: rf, color: .orange)
                        }
                    }

                    // Error
                    if let error = design.error {
                        Divider()
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                            .font(.caption)
                    }
                }
                .padding()
            }
            .frame(maxHeight: 300)
        }
    }

    private var activeURL: URL? {
        if showTrajectory, trajectoryFrame < design.trajectoryPDBs.count {
            return design.trajectoryPDBs[trajectoryFrame]
        }
        return design.sequencePDB ?? design.backbonePDB
    }
}

// MARK: - Helpers

struct MetricLabel: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        Text(text)
            .foregroundStyle(.secondary)
            .font(.caption)
    }
}

struct MetricValue: View {
    let value: Double?
    let format: String
    var color: Color? = nil

    init(_ value: Double?, format: String, color: Color? = nil) {
        self.value = value
        self.format = format
        self.color = color
    }

    var body: some View {
        if let v = value {
            Text(String(format: format, v))
                .foregroundStyle(color ?? .primary)
        } else {
            Text("-")
                .foregroundStyle(.tertiary)
        }
    }
}

struct TimingBar: View {
    let label: String
    let time: Double
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(String(format: "%.1fs", time))
                .font(.caption.monospacedDigit().bold())
                .foregroundStyle(color)
        }
    }
}

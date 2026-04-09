import SwiftUI
import UniformTypeIdentifiers

struct ResultsTable: View {
    @Environment(DesignCampaign.self) private var campaign
    @State private var sortOrder = [KeyPathComparator(\SortableDesign.plddt, order: .reverse)]
    @State private var showLog = false

    var body: some View {
        @Bindable var campaign = campaign

        VStack(spacing: 0) {
            // Status bar
            HStack {
                if campaign.status == .completed {
                    Label("Campaign Complete", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else if campaign.status == .failed {
                    Label("Failed", systemImage: "xmark.circle.fill")
                        .foregroundStyle(.red)
                } else if campaign.status == .cancelled {
                    Label("Cancelled", systemImage: "stop.circle.fill")
                        .foregroundStyle(.orange)
                }

                Spacer()

                if !campaign.completedDesigns.isEmpty {
                    Text("\(campaign.completedDesigns.count) designs")
                        .foregroundStyle(.secondary)
                    if let avg = campaign.averagePLDDT {
                        Text("| Avg pLDDT: \(avg, specifier: "%.3f")")
                            .foregroundStyle(.secondary)
                    }
                    if let frac = campaign.highQualityFraction {
                        Text("| High quality: \(frac * 100, specifier: "%.0f")%")
                            .foregroundStyle(.secondary)
                    }
                    Text("| \(campaign.elapsedTimeFormatted)")
                        .foregroundStyle(.secondary)
                }

                if !campaign.completedDesigns.isEmpty {
                    Button {
                        exportTSV(campaign: campaign)
                    } label: {
                        Image(systemName: "tablecells")
                    }
                    .buttonStyle(.borderless)
                    .help("Export statistics as TSV")

                    Button {
                        exportArchive(campaign: campaign)
                    } label: {
                        Image(systemName: "archivebox")
                    }
                    .buttonStyle(.borderless)
                    .help("Export all designs as .tgz")
                }

                Button {
                    showLog.toggle()
                } label: {
                    Image(systemName: "terminal")
                }
                .buttonStyle(.borderless)
                .help("Show Python log")
            }
            .font(.caption)
            .padding(.horizontal)
            .padding(.vertical, 6)
            .background(.bar)

            Divider()

            // Error message
            if let err = campaign.errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                    Text(err)
                        .font(.caption)
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(.red.opacity(0.1))
                Divider()
            }

            // Show ALL designs (not just complete) so user can see status
            let allDesigns = campaign.designs
                .map(SortableDesign.init)
                .sorted(using: sortOrder)

            if allDesigns.isEmpty {
                ContentUnavailableView(
                    "No Designs",
                    systemImage: "tray",
                    description: Text("The pipeline did not produce any designs. Check the log for errors.")
                )
            } else {
                Table(allDesigns, selection: $campaign.selectedDesignID, sortOrder: $sortOrder) {
                    TableColumn("#", value: \.id) { d in
                        Text("\(d.id + 1)")
                    }
                    .width(30)

                    TableColumn("Stage") { d in
                        StageCell(stage: d.design.stage)
                    }
                    .width(55)

                    TableColumn("pLDDT", value: \.plddt) { d in
                        MetricCell(value: d.design.plddt, format: "%.3f",
                                   color: d.design.plddt.map { MetricColor.plddt($0) })
                    }
                    .width(65)

                    TableColumn("PAE", value: \.pae) { d in
                        MetricCell(value: d.design.pae, format: "%.1f",
                                   color: d.design.pae.map { MetricColor.pae($0) })
                    }
                    .width(50)

                    TableColumn("iPAE", value: \.ipae) { d in
                        MetricCell(value: d.design.ipae, format: "%.1f",
                                   color: d.design.ipae.map { MetricColor.ipae($0) })
                    }
                    .width(50)

                    TableColumn(campaign.config.validator == .protenix ? "ipTM" : "P(bind)",
                               value: \.pBind) { d in
                        MetricCell(value: d.design.pBind, format: "%.3f",
                                   color: d.design.pBind.map { MetricColor.pBind($0) })
                    }
                    .width(65)

                    TableColumn("MPNN", value: \.mpnnScore) { d in
                        MetricCell(value: d.design.mpnnScore, format: "%.3f",
                                   color: d.design.mpnnScore.map { MetricColor.mpnn($0) })
                    }
                    .width(60)

                    TableColumn("CDR RMSD", value: \.cdrRMSD) { d in
                        MetricCell(value: d.design.cdrRMSD, format: "%.1f",
                                   color: d.design.cdrRMSD.map { MetricColor.rmsd($0) })
                    }
                    .width(70)

                    TableColumn("Time", value: \.totalTime) { d in
                        Text(d.design.totalTime.map { String(format: "%.0fs", $0) } ?? "-")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    .width(50)
                }
                .tableStyle(.inset(alternatesRowBackgrounds: true))
            }

            // Summary stats bar
            if campaign.completedDesigns.count >= 3 {
                Divider()
                StatsSummaryBar(designs: campaign.completedDesigns,
                               validator: campaign.config.validator)
            }
        }
        .sheet(isPresented: $showLog) {
            LogView(campaign: campaign)
        }
    }
}

// MARK: - Stage Cell

struct StageCell: View {
    let stage: PipelineStage

    var body: some View {
        Text(stage.shortLabel)
            .font(.caption2.bold())
            .foregroundStyle(stage.color)
    }
}

extension PipelineStage {
    var shortLabel: String {
        switch self {
        case .pending: "..."
        case .rfdiffusion: "RFdiff"
        case .mpnn: "MPNN"
        case .rf2: "RF2"
        case .complete: "Done"
        case .failed: "Fail"
        }
    }
}

// MARK: - Log View (debug)

struct LogView: View {
    let campaign: DesignCampaign
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Python Log")
                    .font(.headline)
                Spacer()
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }
            .padding()

            Divider()

            ScrollView {
                Text(logText)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
        }
        .frame(minWidth: 600, minHeight: 400)
    }

    private var logText: String {
        var lines: [String] = []

        // Show any error
        if let err = campaign.errorMessage {
            lines.append("=== ERROR ===")
            lines.append(err)
            lines.append("")
        }

        // Show design summary
        lines.append("=== DESIGNS ===")
        if campaign.designs.isEmpty {
            lines.append("(no designs were created — pipeline may have failed during initialization)")
        }
        for design in campaign.designs {
            var s = "Design \(design.id): stage=\(design.stage.rawValue)"
            if let p = design.plddt { s += " pLDDT=\(String(format: "%.3f", p))" }
            if let b = design.pBind { s += " P(bind)=\(String(format: "%.3f", b))" }
            if let t = design.totalTime { s += " time=\(String(format: "%.1fs", t))" }
            if let e = design.error { s += " error=\(e)" }
            lines.append(s)
        }

        // Python stdout/stderr
        if !campaign.pythonLog.isEmpty {
            lines.append("")
            lines.append("=== PYTHON LOG ===")
            lines.append(campaign.pythonLog)
        }

        return lines.joined(separator: "\n")
    }
}

// MARK: - Export Functions

@MainActor private func exportTSV(campaign: DesignCampaign) {
    let panel = NSSavePanel()
    panel.allowedContentTypes = [.tabSeparatedText]
    panel.nameFieldStringValue = "designs.tsv"
    panel.title = "Export Statistics"
    guard panel.runModal() == .OK, let url = panel.url else { return }

    var lines: [String] = []
    lines.append(["Design", "Stage", "pLDDT", "PAE", "iPAE", "P(bind)",
                   "MPNN_Score", "CDR_RMSD", "Time_s"].joined(separator: "\t"))

    for d in campaign.designs where d.stage == .complete {
        let row: [String] = [
            "\(d.id + 1)",
            d.stage.rawValue,
            d.plddt.map { String(format: "%.4f", $0) } ?? "",
            d.pae.map { String(format: "%.2f", $0) } ?? "",
            d.ipae.map { String(format: "%.2f", $0) } ?? "",
            d.pBind.map { String(format: "%.4f", $0) } ?? "",
            d.mpnnScore.map { String(format: "%.4f", $0) } ?? "",
            d.cdrRMSD.map { String(format: "%.2f", $0) } ?? "",
            d.totalTime.map { String(format: "%.1f", $0) } ?? "",
        ]
        lines.append(row.joined(separator: "\t"))
    }

    try? lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
}

@MainActor private func exportArchive(campaign: DesignCampaign) {
    let panel = NSSavePanel()
    panel.allowedContentTypes = [.init(filenameExtension: "tgz")!]
    panel.nameFieldStringValue = "designs.tgz"
    panel.title = "Export All Designs"
    guard panel.runModal() == .OK, let destURL = panel.url else { return }

    // Collect all PDB files into a temp staging directory
    let fm = FileManager.default
    let staging = fm.temporaryDirectory.appending(path: "RFantibodyExport-\(UUID().uuidString)")
    try? fm.createDirectory(at: staging, withIntermediateDirectories: true)

    for d in campaign.designs where d.stage == .complete {
        let prefix = "design_\(d.id + 1)"
        if let url = d.backbonePDB {
            try? fm.copyItem(at: url, to: staging.appending(path: "\(prefix)_backbone.pdb"))
        }
        if let url = d.sequencePDB {
            try? fm.copyItem(at: url, to: staging.appending(path: "\(prefix)_sequence.pdb"))
        }
        if let url = d.validatedPDB {
            try? fm.copyItem(at: url, to: staging.appending(path: "\(prefix)_validated.pdb"))
        }
    }

    // Create tar.gz
    let proc = Process()
    proc.executableURL = URL(filePath: "/usr/bin/tar")
    proc.arguments = ["-czf", destURL.path(), "-C", staging.path(), "."]
    try? proc.run()
    proc.waitUntilExit()

    // Clean up staging
    try? fm.removeItem(at: staging)
}

// MARK: - Sortable wrapper (handles nil for Table sorting)

struct SortableDesign: Identifiable {
    let design: NanobodyDesign
    var id: Int { design.id }
    var plddt: Double { design.plddt ?? -1 }
    var pae: Double { design.pae ?? 999 }
    var ipae: Double { design.ipae ?? 999 }
    var pBind: Double { design.pBind ?? -1 }
    var mpnnScore: Double { design.mpnnScore ?? 999 }
    var cdrRMSD: Double { design.cdrRMSD ?? 999 }
    var totalTime: Double { design.totalTime ?? 999 }
}

// MARK: - Metric Coloring

/// Thresholds for metric quality coloring.
enum MetricColor {
    /// Higher is better (pLDDT, P(bind))
    static func higherBetter(_ v: Double, good: Double, moderate: Double) -> Color {
        v > good ? .green : v > moderate ? .yellow : .red
    }
    /// Lower is better (PAE, iPAE, RMSD, MPNN)
    static func lowerBetter(_ v: Double, good: Double, moderate: Double) -> Color {
        v < good ? .green : v < moderate ? .yellow : .red
    }

    static func plddt(_ v: Double) -> Color { higherBetter(v, good: 0.85, moderate: 0.80) }
    static func pae(_ v: Double) -> Color { lowerBetter(v, good: 10, moderate: 20) }
    static func ipae(_ v: Double) -> Color { lowerBetter(v, good: 10, moderate: 15) }
    static func pBind(_ v: Double) -> Color { higherBetter(v, good: 0.7, moderate: 0.5) }
    static func mpnn(_ v: Double) -> Color { lowerBetter(v, good: 1.5, moderate: 2.5) }
    static func rmsd(_ v: Double) -> Color { lowerBetter(v, good: 2.5, moderate: 5.0) }
}

// MARK: - Metric Cell

struct MetricCell: View {
    let value: Double?
    let format: String
    var color: Color? = nil

    var body: some View {
        if let v = value {
            Text(String(format: format, v))
                .monospacedDigit()
                .foregroundStyle(color ?? .primary)
        } else {
            Text("--")
                .foregroundStyle(.quaternary)
        }
    }
}

// MARK: - Summary Stats Bar

struct StatsSummaryBar: View {
    let designs: [NanobodyDesign]
    var validator: Validator = .rf2

    var body: some View {
        HStack(spacing: 24) {
            StatItem(label: "pLDDT", values: designs.compactMap(\.plddt))
            StatItem(label: validator == .protenix ? "ipTM" : "P(bind)",
                     values: designs.compactMap(\.pBind))
            StatItem(label: "CDR RMSD", values: designs.compactMap(\.cdrRMSD))
            StatItem(label: "MPNN", values: designs.compactMap(\.mpnnScore))

            Spacer()

            let high = designs.filter { ($0.plddt ?? 0) > 0.85 }.count
            Text("High quality: \(high)/\(designs.count)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.bar)
    }
}

struct StatItem: View {
    let label: String
    let values: [Double]

    var body: some View {
        if !values.isEmpty {
            let mean = values.reduce(0, +) / Double(values.count)
            let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
                / Double(max(1, values.count - 1))
            let sd = variance.squareRoot()

            VStack(alignment: .leading, spacing: 1) {
                Text(label)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                Text("\(mean, specifier: "%.2f") +/- \(sd, specifier: "%.2f")")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }
}

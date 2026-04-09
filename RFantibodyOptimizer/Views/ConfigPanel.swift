import SwiftUI

struct ConfigPanel: View {
    @Environment(DesignCampaign.self) private var campaign
    @Environment(PipelineRunner.self) private var runner
    @State private var showAdvanced = false

    var body: some View {
        @Bindable var campaign = campaign

        Form {
            Section("Input Structures") {
                FilePickerRow(
                    label: "Target PDB",
                    selection: $campaign.config.targetPDB,
                    allowedExtensions: ["pdb"]
                )
                FilePickerRow(
                    label: "Framework PDB",
                    selection: $campaign.config.frameworkPDB,
                    allowedExtensions: ["pdb"]
                )
            }

            Section("CDR Loops") {
                LoopRow(name: "H1", config: $campaign.config.loopH1)
                LoopRow(name: "H2", config: $campaign.config.loopH2)
                LoopRow(name: "H3", config: $campaign.config.loopH3)
            }

            Section("Diffusion") {
                Stepper("Designs: \(campaign.config.numDesigns)",
                        value: $campaign.config.numDesigns, in: 1...200)

                Picker("Mode", selection: $campaign.config.mode) {
                    ForEach(DesignMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }

                HStack {
                    Text("Timesteps: \(campaign.config.diffusionT)")
                    Spacer()
                    Slider(
                        value: Binding(
                            get: { Double(campaign.config.diffusionT) },
                            set: { campaign.config.diffusionT = Int($0) }
                        ),
                        in: 15...50,
                        step: 5
                    )
                    .frame(width: 120)
                }

                Picker("Template Scheme", selection: $campaign.config.tScheme) {
                    ForEach(TScheme.allCases) { scheme in
                        Text(scheme.label).tag(scheme)
                    }
                }

                if campaign.config.tScheme != .singleT {
                    Text(campaign.config.tScheme.help)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Hotspot Residues") {
                TextField("e.g. A100,A103,B50", text: $campaign.config.hotspotRes)
                    .textFieldStyle(.roundedBorder)

                Text("Target residues to guide binding. Leave empty for unbiased design.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            Section("MPNN") {
                Toggle("Enable MPNN", isOn: Binding(
                    get: { !campaign.config.skipMPNN },
                    set: { campaign.config.skipMPNN = !$0 }
                ))

                if !campaign.config.skipMPNN {
                    HStack {
                        Text("Temperature: \(campaign.config.mpnnTemp, specifier: "%.2f")")
                        Spacer()
                        Slider(value: $campaign.config.mpnnTemp, in: 0.01...0.5)
                            .frame(width: 120)
                    }

                    Stepper("Sequences: \(campaign.config.mpnnSeqs)",
                            value: $campaign.config.mpnnSeqs, in: 1...8)
                }
            }

            Section("RF2 Validation") {
                Toggle("Enable RF2", isOn: Binding(
                    get: { !campaign.config.skipRF2 },
                    set: { campaign.config.skipRF2 = !$0 }
                ))

                if !campaign.config.skipRF2 {
                    Stepper("Max recycles: \(campaign.config.rf2Recycles)",
                            value: $campaign.config.rf2Recycles, in: 1...20)

                    HStack {
                        Text("Threshold: \(campaign.config.rf2Threshold, specifier: "%.1f") A")
                        Spacer()
                        Slider(value: $campaign.config.rf2Threshold, in: 0.1...2.0, step: 0.1)
                            .frame(width: 120)
                    }
                }
            }

            DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                HStack {
                    Text("CA noise scale: \(campaign.config.noiseScaleCA, specifier: "%.2f")")
                    Spacer()
                    Slider(value: $campaign.config.noiseScaleCA, in: 0.1...2.0, step: 0.1)
                        .frame(width: 120)
                }

                HStack {
                    Text("Frame noise scale: \(campaign.config.noiseScaleFrame, specifier: "%.2f")")
                    Spacer()
                    Slider(value: $campaign.config.noiseScaleFrame, in: 0.1...2.0, step: 0.1)
                        .frame(width: 120)
                }

                Stepper("Seed: \(campaign.config.seed)",
                        value: $campaign.config.seed, in: 0...99999)

                Divider()

                Toggle("Step cache", isOn: $campaign.config.cacheEnabled)

                if campaign.config.cacheEnabled {
                    HStack {
                        Text("Cache threshold: \(campaign.config.cacheThreshold, specifier: "%.2f")")
                        Spacer()
                        Slider(value: $campaign.config.cacheThreshold, in: 0.01...0.50, step: 0.01)
                            .frame(width: 120)
                    }

                    Stepper("Cache warmup: \(campaign.config.cacheWarmup)",
                            value: $campaign.config.cacheWarmup, in: 1...10)
                }

                Divider()

                Picker("Validator", selection: $campaign.config.validator) {
                    ForEach(Validator.allCases) { v in
                        Text(v.label).tag(v)
                    }
                }
                .help(campaign.config.validator.help)
            }

            Section {
                if campaign.status == .running || campaign.status == .initializing {
                    Button("Stop Campaign", role: .destructive) {
                        runner.stop(campaign: campaign)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .frame(maxWidth: .infinity)
                } else {
                    Button("Start Design Campaign") {
                        runner.start(campaign: campaign)
                    }
                    .buttonStyle(.borderedProminent)
                    .frame(maxWidth: .infinity)
                    .disabled(!campaign.config.isValid)
                }

                if let eta = formattedETA {
                    Text(eta)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                // Show loop summary
                let loops = campaign.config.designLoopsFormatted
                if !loops.isEmpty {
                    Text("Loops: \(loops.joined(separator: ", "))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if campaign.status == .failed, let err = campaign.errorMessage {
                Section("Error") {
                    Text(err)
                        .foregroundStyle(.red)
                        .font(.caption)
                }
            }
        }
        .formStyle(.grouped)
    }

    private var formattedETA: String? {
        guard campaign.config.isValid else { return nil }
        let t = campaign.config.estimatedTotalTime
        let m = Int(t) / 60
        return "Estimated: ~\(m) min for \(campaign.config.numDesigns) designs"
    }
}

// MARK: - Loop Row

struct LoopRow: View {
    let name: String
    @Binding var config: LoopConfig

    private var rangeLabel: String {
        let lo = config.effectiveMin
        let hi = config.effectiveMax
        if lo == hi {
            if lo == config.nativeLength { return "\(lo) (native)" }
            return "\(lo)"
        }
        return "\(lo)–\(hi)"
    }

    var body: some View {
        Toggle(isOn: $config.enabled) {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(name).fontWeight(.medium)
                    Spacer()
                    Text(rangeLabel)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }

                if config.enabled {
                    RangeSliderView(
                        lo: Binding(
                            get: { Double(config.effectiveMin) },
                            set: { config.minLength = Int($0) }
                        ),
                        hi: Binding(
                            get: { Double(config.effectiveMax) },
                            set: { config.maxLength = Int($0) }
                        ),
                        range: 1...20,
                        step: 1
                    )
                }
            }
        }
    }
}

// MARK: - Range Slider

struct RangeSliderView: View {
    @Binding var lo: Double
    @Binding var hi: Double
    let range: ClosedRange<Double>
    let step: Double

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let span = range.upperBound - range.lowerBound
            let loFrac = (lo - range.lowerBound) / span
            let hiFrac = (hi - range.lowerBound) / span

            ZStack(alignment: .leading) {
                // Track
                Capsule()
                    .fill(Color.secondary.opacity(0.2))
                    .frame(height: 4)

                // Filled range
                Capsule()
                    .fill(Color.accentColor)
                    .frame(width: max(0, CGFloat(hiFrac - loFrac) * w), height: 4)
                    .offset(x: CGFloat(loFrac) * w)

                // Low thumb
                Circle()
                    .fill(Color.white)
                    .shadow(radius: 1)
                    .frame(width: 16, height: 16)
                    .offset(x: CGFloat(loFrac) * w - 8)
                    .gesture(
                        DragGesture()
                            .onChanged { v in
                                let frac = max(0, min(1, v.location.x / w))
                                let val = (range.lowerBound + frac * span).rounded()
                                let snapped = (val / step).rounded() * step
                                lo = min(snapped, hi)
                                lo = max(lo, range.lowerBound)
                            }
                    )

                // High thumb
                Circle()
                    .fill(Color.white)
                    .shadow(radius: 1)
                    .frame(width: 16, height: 16)
                    .offset(x: CGFloat(hiFrac) * w - 8)
                    .gesture(
                        DragGesture()
                            .onChanged { v in
                                let frac = max(0, min(1, v.location.x / w))
                                let val = (range.lowerBound + frac * span).rounded()
                                let snapped = (val / step).rounded() * step
                                hi = max(snapped, lo)
                                hi = min(hi, range.upperBound)
                            }
                    )
            }
        }
        .frame(height: 20)
    }
}

// MARK: - File Picker Row

struct FilePickerRow: View {
    let label: String
    @Binding var selection: URL?
    let allowedExtensions: [String]

    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(label)
                if let url = selection {
                    Text(url.lastPathComponent)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                } else {
                    Text("Not selected")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
            Spacer()
            Button("Choose...") {
                let panel = NSOpenPanel()
                panel.allowedContentTypes = allowedExtensions.compactMap {
                    .init(filenameExtension: $0)
                }
                panel.canChooseDirectories = false
                panel.allowsMultipleSelection = false
                if panel.runModal() == .OK {
                    selection = panel.url
                }
            }
            .buttonStyle(.bordered)
        }
    }
}

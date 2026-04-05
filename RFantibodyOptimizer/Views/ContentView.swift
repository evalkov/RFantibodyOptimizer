import SwiftUI

struct ContentView: View {
    @Binding var darkMode: Bool
    @Environment(DesignCampaign.self) private var campaign

    /// Live PDB for the design currently being processed.
    /// During RF2 keep showing the MPNN/diffusion PDB (same coordinate frame as target).
    private var livePreviewURL: URL? {
        let idx = campaign.currentDesignIndex
        guard idx < campaign.designs.count else { return nil }
        let design = campaign.designs[idx]
        // After MPNN, keep showing sequencePDB (also during RF2 — same frame as target)
        if let seqPDB = design.sequencePDB,
           (campaign.currentStage == .mpnn || campaign.currentStage == .rf2) {
            return seqPDB
        }
        // During diffusion, use latest trajectory frame
        return design.trajectoryPDBs.last
    }


    private var stageOverlayText: String {
        switch campaign.currentStage {
        case .rfdiffusion:
            return "Diffusion \(campaign.currentStep)/\(campaign.totalSteps)"
        case .mpnn:
            return "Designing sequence..."
        case .rf2:
            if campaign.rf2TotalRecycles > 0 {
                let rmsd = campaign.rf2CaRMSD.map { String(format: " RMSD %.2fÅ", $0) } ?? ""
                return "RF2 recycle \(campaign.rf2CurrentRecycle)/\(campaign.rf2TotalRecycles)\(rmsd)"
            }
            return "Validating structure..."
        default:
            return ""
        }
    }

    var body: some View {
        @Bindable var campaign = campaign

        NavigationSplitView {
            ConfigPanel()
                .navigationSplitViewColumnWidth(min: 280, ideal: 320, max: 380)
        } content: {
            Group {
                switch campaign.status {
                case .idle:
                    WelcomeView()
                case .initializing, .running:
                    DesignProgressView()
                case .completed, .failed, .cancelled:
                    ResultsTable()
                }
            }
            .navigationSplitViewColumnWidth(min: 500, ideal: 700)
        } detail: {
            if let selectedID = campaign.selectedDesignID,
               let design = campaign.designs.first(where: { $0.id == selectedID }),
               design.sequencePDB != nil || design.backbonePDB != nil {
                DesignDetailView(design: design)
            } else if (campaign.status == .running || campaign.status == .initializing),
                      livePreviewURL != nil {
                // Live preview during pipeline run
                VStack(spacing: 0) {
                    ProteinViewer(
                        pdbURL: livePreviewURL,
                        colorScheme: .cdr,
                        preserveCamera: true,
                        diffusionMode: campaign.currentStage == .rfdiffusion
                    )
                    .overlay(alignment: .topLeading) {
                        Text(stageOverlayText)
                            .font(.caption.monospacedDigit().bold())
                            .padding(6)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
                            .padding(8)
                    }
                }
            } else {
                ContentUnavailableView(
                    "No Design Selected",
                    systemImage: "atom",
                    description: Text("Select a design from the results table to view its 3D structure and metrics.")
                )
            }
        }
        .navigationTitle("RFantibodyOptimizer")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    darkMode.toggle()
                } label: {
                    Image(systemName: darkMode ? "sun.max.fill" : "moon.fill")
                }
                .help(darkMode ? "Switch to Light Mode" : "Switch to Dark Mode")
            }
        }
    }
}

struct WelcomeView: View {
    var body: some View {
        ContentUnavailableView {
            Label("Nanobody Designer", systemImage: "wand.and.stars")
        } description: {
            Text("Configure your design campaign in the sidebar and press Start.")
        }
    }
}

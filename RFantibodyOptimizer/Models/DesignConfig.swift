import Foundation

enum DesignMode: String, Codable, CaseIterable, Identifiable {
    case full, fast, draft

    var id: String { rawValue }

    var label: String {
        switch self {
        case .full: "Full (best quality)"
        case .fast: "Fast (balanced)"
        case .draft: "Draft (screening)"
        }
    }

    var estimatedSecondsPerStep: Double {
        switch self {
        case .full: 2.1
        case .fast: 1.9
        case .draft: 1.3
        }
    }
}

enum TScheme: String, Codable, CaseIterable, Identifiable {
    case singleT = "single_T"
    case fixedDock = "fixed_dock"
    case noT = "no_T"
    case singleTCorrectSelfcond = "single_T_correct_selfcond"
    case selfcondEmb = "selfcond_emb"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .singleT: "Single T (default)"
        case .fixedDock: "Fixed Dock (CDR only)"
        case .noT: "No Template"
        case .singleTCorrectSelfcond: "Self-Cond Corrected"
        case .selfcondEmb: "Self-Cond Embedding"
        }
    }

    var help: String {
        switch self {
        case .singleT: "Framework templated; loops and target masked"
        case .fixedDock: "Only CDR loops diffuse; framework and target fixed"
        case .noT: "No template info; design from scratch"
        case .singleTCorrectSelfcond: "Self-conditioning with drift correction"
        case .selfcondEmb: "Dedicated self-conditioning embedding track"
        }
    }
}

enum Validator: String, Codable, CaseIterable, Identifiable {
    case rf2, protenix

    var id: String { rawValue }

    var label: String {
        switch self {
        case .rf2: "RF2"
        case .protenix: "Protenix-Mini"
        }
    }

    var help: String {
        switch self {
        case .rf2: "RoseTTAFold2 — P(bind), iPAE, pLDDT"
        case .protenix: "Protenix-Mini-Flow — ipTM, PAE, pLDDT (faster)"
        }
    }
}

/// Per-loop configuration: enabled + length range.
struct LoopConfig: Codable, Hashable {
    var enabled: Bool = true
    /// Min loop length. 0 = use native length.
    var minLength: Int = 0
    /// Max loop length. 0 = use native length.
    var maxLength: Int = 0
    /// Native length from the framework PDB (for display/defaults).
    var nativeLength: Int

    /// Format for design_loops config: "H1:", "H1:9", or "H1:5-10"
    func formatted(name: String) -> String {
        let lo = effectiveMin
        let hi = effectiveMax
        if lo == hi && lo == nativeLength {
            return "\(name):"
        }
        if lo == hi {
            return "\(name):\(lo)"
        }
        return "\(name):\(lo)-\(hi)"
    }

    var effectiveMin: Int { minLength == 0 ? nativeLength : minLength }
    var effectiveMax: Int { maxLength == 0 ? nativeLength : maxLength }

    // Legacy compat
    var length: Int { effectiveMin }
    var effectiveLength: Int { effectiveMin }
}

struct DesignConfig: Codable {
    var targetPDB: URL?
    var frameworkPDB: URL? = Bundle.main.url(forResource: "h-NbBCII10", withExtension: "pdb")
    var numDesigns: Int = 10
    var mode: DesignMode = .draft
    var diffusionT: Int = 15
    var seed: Int = 0

    // Loop configuration (defaults = NbBCII10 native lengths)
    var loopH1 = LoopConfig(minLength: 9, maxLength: 11, nativeLength: 10)
    var loopH2 = LoopConfig(minLength: 5, maxLength: 7, nativeLength: 6)
    var loopH3 = LoopConfig(minLength: 13, maxLength: 18, nativeLength: 16)

    // Template scheme
    var tScheme: TScheme = .singleT

    // Hotspot residues (comma-separated, e.g. "A100,A103,B50")
    var hotspotRes: String = ""

    // Noise scales (advanced)
    var noiseScaleCA: Double = 1.0
    var noiseScaleFrame: Double = 1.0

    // Adaptive step cache (advanced)
    var cacheEnabled: Bool = true
    var cacheThreshold: Double = 0.15
    var cacheWarmup: Int = 3

    // Structure validator (advanced)
    var validator: Validator = .rf2

    // MPNN
    var mpnnTemp: Double = 0.1
    var mpnnSeqs: Int = 1
    var skipMPNN: Bool = false

    // RF2
    var rf2Recycles: Int = 10
    var rf2Threshold: Double = 0.5
    var skipRF2: Bool = false

    // MARK: - Computed

    var enabledLoops: [String] {
        var loops: [String] = []
        if loopH1.enabled { loops.append("H1") }
        if loopH2.enabled { loops.append("H2") }
        if loopH3.enabled { loops.append("H3") }
        return loops
    }

    /// Design loops in config format: ["H1:", "H2:5-10", "H3:12"]
    var designLoopsFormatted: [String] {
        var result: [String] = []
        if loopH1.enabled { result.append(loopH1.formatted(name: "H1")) }
        if loopH2.enabled { result.append(loopH2.formatted(name: "H2")) }
        if loopH3.enabled { result.append(loopH3.formatted(name: "H3")) }
        return result
    }

    var isValid: Bool {
        targetPDB != nil && frameworkPDB != nil && !enabledLoops.isEmpty
    }

    var estimatedTotalTime: Double {
        let perDesign = mode.estimatedSecondsPerStep * Double(diffusionT)
            + (skipMPNN ? 0 : 1.0)
            + (skipRF2 ? 0 : 35.0)
        return perDesign * Double(numDesigns)
    }

    /// Serialize to JSON for passing to design_service.py via stdin.
    func toJSON(outputDir: URL) -> Data {
        var dict: [String: Any] = [
            "target_pdb": targetPDB?.path(percentEncoded: false) ?? "",
            "framework_pdb": frameworkPDB?.path(percentEncoded: false) ?? "",
            "design_loops": designLoopsFormatted,
            "num_designs": numDesigns,
            "mode": mode.rawValue,
            "diffusion_T": diffusionT,
            "t_scheme": tScheme.rawValue,
            "noise_scale_ca": noiseScaleCA,
            "noise_scale_frame": noiseScaleFrame,
            "mpnn_temp": mpnnTemp,
            "mpnn_seqs": mpnnSeqs,
            "rf2_recycles": rf2Recycles,
            "rf2_threshold": rf2Threshold,
            "output_dir": outputDir.path(percentEncoded: false),
            "seed": seed,
            "skip_mpnn": skipMPNN,
            "skip_rf2": skipRF2,
            "cache_enabled": cacheEnabled,
            "cache_threshold": cacheThreshold,
            "cache_warmup": cacheWarmup,
            "validator": validator.rawValue,
        ]
        if !hotspotRes.trimmingCharacters(in: .whitespaces).isEmpty {
            dict["hotspot_res"] = hotspotRes.trimmingCharacters(in: .whitespaces)
        }
        return try! JSONSerialization.data(
            withJSONObject: dict, options: [.prettyPrinted, .sortedKeys])
    }
}

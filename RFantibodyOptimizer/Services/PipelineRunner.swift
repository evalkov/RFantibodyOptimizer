import Foundation
import Observation

@Observable
@MainActor
class PipelineRunner {
    private var process: Process?
    private var stderrPipe: Pipe?
    private var buffer = Data()

    var isRunning: Bool { process?.isRunning ?? false }
    var isPaused: Bool = false

    func pause() {
        guard let proc = process, proc.isRunning, !isPaused else { return }
        proc.suspend()
        isPaused = true
    }

    func resume() {
        guard let proc = process, isPaused else { return }
        proc.resume()
        isPaused = false
    }

    /// Locate the pipeline root — either inside the app bundle (distributed)
    /// or the repo working tree (development).
    static var defaultRepoRoot: URL {
        // 1. Bundled: Contents/Resources/ contains scripts/, src/, models/, etc.
        if let resources = Bundle.main.resourceURL,
           FileManager.default.fileExists(atPath: resources.appending(path: "scripts/design_service.py").path()) {
            return resources
        }
        // 2. Dev: walk up from the app bundle looking for the repo
        let bundlePath = Bundle.main.bundlePath
        var dir = URL(filePath: bundlePath).deletingLastPathComponent()
        for _ in 0..<8 {
            if FileManager.default.fileExists(atPath: dir.appending(path: "scripts/design_service.py").path()) {
                return dir
            }
            dir = dir.deletingLastPathComponent()
        }
        // 3. Dev: derive repo root from this source file's compile-time path
        let sourceFileURL = URL(filePath: #filePath)
        let repoCandidate = sourceFileURL
            .deletingLastPathComponent()  // Services/
            .deletingLastPathComponent()  // RFantibodyOptimizer/
            .deletingLastPathComponent()  // repo root
        if FileManager.default.fileExists(atPath: repoCandidate.appending(path: "scripts/design_service.py").path()) {
            return repoCandidate
        }
        // 4. Last resort
        return repoCandidate
    }

    static var defaultPythonURL: URL {
        defaultRepoRoot.appending(path: "python/bin/python3")
    }

    /// Fall back to venv layout used during development.
    static var defaultPythonURLResolved: URL {
        let bundled = defaultRepoRoot.appending(path: "python/bin/python3")
        if FileManager.default.fileExists(atPath: bundled.path()) {
            return bundled
        }
        return defaultRepoRoot.appending(path: "pilot_mps/.venv/bin/python")
    }

    static var defaultScriptPath: String {
        defaultRepoRoot.appending(path: "scripts/design_service.py").path()
    }

    func start(campaign: DesignCampaign) {
        isPaused = false
        campaign.reset()
        campaign.status = .initializing
        campaign.startTime = Date()

        // Create output directory
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let outputDir = FileManager.default.temporaryDirectory
            .appending(path: "RFantibodyOptimizer/\(timestamp)")
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        campaign.outputDir = outputDir

        // Pre-create design slots
        campaign.designs = (0..<campaign.config.numDesigns).map {
            NanobodyDesign(id: $0)
        }

        // Build process
        let root = Self.defaultRepoRoot
        let proc = Process()
        proc.executableURL = Self.defaultPythonURLResolved
        proc.arguments = [Self.defaultScriptPath]
        proc.currentDirectoryURL = root
        proc.environment = [
            "PYTHONPATH": root.appending(path: "src").path()
                + ":" + root.appending(path: "include/SE3Transformer").path(),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": NSHomeDirectory(),
        ]

        // Stdin: send config JSON
        let stdinPipe = Pipe()
        proc.standardInput = stdinPipe

        // Stdout: capture for debug log
        let stdoutPipe = Pipe()
        proc.standardOutput = stdoutPipe
        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                campaign.pythonLog += text
            }
        }

        // Stderr: JSON events
        let errPipe = Pipe()
        proc.standardError = errPipe
        self.stderrPipe = errPipe
        self.buffer = Data()

        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            Task { @MainActor in
                self?.handleStderrData(data, campaign: campaign)
            }
        }

        // Termination handler — drain remaining pipe data before finalizing
        proc.terminationHandler = { [weak self] _ in
            let exitCode = proc.terminationStatus

            // Stop handlers and read any remaining data synchronously
            // (process has exited so write end is closed — reads return immediately)
            errPipe.fileHandleForReading.readabilityHandler = nil
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            let stderrRemaining = errPipe.fileHandleForReading.readDataToEndOfFile()
            let stdoutRemaining = stdoutPipe.fileHandleForReading.readDataToEndOfFile()

            Task { @MainActor [weak self] in
                // Process any remaining stdout
                if !stdoutRemaining.isEmpty,
                   let text = String(data: stdoutRemaining, encoding: .utf8) {
                    campaign.pythonLog += text
                }

                // Process any remaining stderr events
                if let self, !stderrRemaining.isEmpty {
                    self.handleStderrData(stderrRemaining, campaign: campaign)
                }

                // Flush partial line left in buffer
                if let self, !self.buffer.isEmpty,
                   let line = String(data: self.buffer, encoding: .utf8),
                   !line.isEmpty {
                    self.parseEvent(line, campaign: campaign)
                    self.buffer = Data()
                }

                // Handle termination status
                if exitCode != 0 && campaign.status != .cancelled {
                    campaign.status = .failed
                    campaign.errorMessage = campaign.errorMessage
                        ?? "Python process exited with code \(exitCode)"
                }
                campaign.endTime = Date()
                self?.process = nil
            }
        }

        self.process = proc

        do {
            try proc.run()

            // Send config
            let configData = campaign.config.toJSON(outputDir: outputDir)
            stdinPipe.fileHandleForWriting.write(configData)
            stdinPipe.fileHandleForWriting.closeFile()
        } catch {
            campaign.status = .failed
            campaign.errorMessage = "Failed to start Python process: \(error.localizedDescription)"
            campaign.endTime = Date()
        }
    }

    func stop(campaign: DesignCampaign) {
        guard let proc = process, proc.isRunning else { return }
        if isPaused { proc.resume() } // must resume before terminate
        isPaused = false
        campaign.status = .cancelled
        proc.terminate()
    }

    // MARK: - Event Parsing

    private func handleStderrData(_ data: Data, campaign: DesignCampaign) {
        buffer.append(data)

        // Split on newlines and process complete lines
        let newline = Data("\n".utf8)
        while let range = buffer.range(of: newline) {
            let lineData = buffer.subdata(in: buffer.startIndex..<range.lowerBound)
            buffer.removeSubrange(buffer.startIndex...range.lowerBound)

            guard let line = String(data: lineData, encoding: .utf8),
                  !line.isEmpty else { continue }
            parseEvent(line, campaign: campaign)
        }
    }

    private func parseEvent(_ line: String, campaign: DesignCampaign) {
        guard let data = line.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let event = json["event"] as? String else {
            // Not JSON — append to log
            campaign.pythonLog += "[stderr] " + line + "\n"
            return
        }

        switch event {
        case "pipeline_start":
            campaign.status = .running
            campaign.totalSteps = json["T"] as? Int ?? 0

        case "init_complete":
            campaign.status = .running

        case "design_start":
            let idx = json["design_idx"] as? Int ?? 0
            campaign.currentDesignIndex = idx
            campaign.currentStep = 0
            if idx < campaign.designs.count {
                campaign.designs[idx].stage = .rfdiffusion
            }

        case "stage_start":
            let stage = json["stage"] as? String ?? ""
            campaign.currentStage = PipelineStage(rawValue: stage) ?? .pending
            let idx = json["design_idx"] as? Int ?? 0
            if idx < campaign.designs.count {
                campaign.designs[idx].stage = PipelineStage(rawValue: stage) ?? .pending
            }

        case "step_progress":
            let stage = json["stage"] as? String ?? ""
            if stage == "rf2" {
                campaign.rf2CurrentRecycle = json["step"] as? Int ?? 0
                campaign.rf2TotalRecycles = json["total"] as? Int ?? campaign.rf2TotalRecycles
                campaign.rf2CaRMSD = json["ca_rmsd"] as? Double
                // Store intermediate all-atom PDB for live side chain visualization
                let stepIdx = json["design_idx"] as? Int ?? campaign.currentDesignIndex
                if stepIdx < campaign.designs.count, let path = urlFromJSON(json["pdb_path"]) {
                    campaign.designs[stepIdx].rf2IntermediatePDB = path
                }
            } else {
                campaign.currentStep = json["step"] as? Int ?? 0
                campaign.totalSteps = json["total"] as? Int ?? campaign.totalSteps
                // Store trajectory PDB for diffusion visualization
                let stepIdx = json["design_idx"] as? Int ?? campaign.currentDesignIndex
                if stepIdx < campaign.designs.count, let path = urlFromJSON(json["pdb_path"]) {
                    campaign.designs[stepIdx].trajectoryPDBs.append(path)
                }
            }

        case "stage_complete":
            let idx = json["design_idx"] as? Int ?? 0
            let stage = json["stage"] as? String ?? ""
            guard idx < campaign.designs.count else { break }

            switch stage {
            case "rfdiffusion":
                campaign.designs[idx].backbonePDB = urlFromJSON(json["pdb_path"])
                campaign.designs[idx].rfdiffusionTime = json["time_s"] as? Double
                campaign.designs[idx].proteinLength = json["L"] as? Int
                campaign.designs[idx].caCaBond = json["ca_ca_bond"] as? Double
                campaign.designs[idx].radiusOfGyration = json["rg"] as? Double

            case "mpnn":
                campaign.designs[idx].sequencePDB = urlFromJSON(json["pdb_path"])
                campaign.designs[idx].mpnnScore = json["score"] as? Double
                campaign.designs[idx].mpnnTime = json["time_s"] as? Double

            case "rf2":
                campaign.designs[idx].validatedPDB = urlFromJSON(json["pdb_path"])
                campaign.designs[idx].plddt = json["plddt"] as? Double
                campaign.designs[idx].pae = json["pae"] as? Double
                campaign.designs[idx].ipae = json["ipae"] as? Double
                campaign.designs[idx].pBind = json["p_bind"] as? Double
                campaign.designs[idx].cdrRMSD = json["cdr_rmsd"] as? Double
                campaign.designs[idx].rf2Time = json["time_s"] as? Double
                if let err = json["error"] as? String {
                    campaign.designs[idx].error = err
                }

            default: break
            }

        case "design_complete":
            let idx = json["design_idx"] as? Int ?? 0
            guard idx < campaign.designs.count else { break }
            campaign.designs[idx].stage = .complete
            campaign.designs[idx].totalTime = json["total_time_s"] as? Double

        case "pipeline_complete":
            campaign.status = .completed
            campaign.endTime = Date()

        case "cancelled":
            campaign.status = .cancelled
            campaign.endTime = Date()

        case "error":
            campaign.errorMessage = json["message"] as? String
            // Don't set failed status here — wait for process termination

        default:
            break
        }
    }

    private func urlFromJSON(_ value: Any?) -> URL? {
        guard let path = value as? String, !path.isEmpty else { return nil }
        return URL(filePath: path)
    }
}

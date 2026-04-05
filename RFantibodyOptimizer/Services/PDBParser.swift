import Foundation

struct PDBData {
    let rawText: String
    let chains: [String]
    let residueCount: Int
    let bFactors: [Double]

    static func from(url: URL) throws -> PDBData {
        let text = try String(contentsOf: url, encoding: .utf8)
        var chains = Set<String>()
        var bFactors: [Double] = []
        var residueCount = 0
        var lastResID = ""

        for line in text.components(separatedBy: .newlines) {
            guard line.hasPrefix("ATOM") && line.count >= 60 else { continue }

            let chainID = String(line[line.index(line.startIndex, offsetBy: 21)])
            chains.insert(chainID)

            // Track unique residues
            let resID = String(line[line.index(line.startIndex, offsetBy: 22)..<line.index(line.startIndex, offsetBy: 27)]).trimmingCharacters(in: .whitespaces)
            let fullResID = "\(chainID)_\(resID)"
            if fullResID != lastResID {
                lastResID = fullResID
                residueCount += 1

                // B-factor (columns 61-66)
                if line.count >= 66 {
                    let bStr = String(line[line.index(line.startIndex, offsetBy: 60)..<line.index(line.startIndex, offsetBy: 66)]).trimmingCharacters(in: .whitespaces)
                    if let b = Double(bStr) {
                        bFactors.append(b)
                    }
                }
            }
        }

        return PDBData(
            rawText: text,
            chains: chains.sorted(),
            residueCount: residueCount,
            bFactors: bFactors
        )
    }
}

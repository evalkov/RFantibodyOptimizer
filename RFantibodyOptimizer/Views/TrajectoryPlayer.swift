import SwiftUI

struct TrajectoryPlayer: View {
    let frames: [URL]
    @Binding var currentFrame: Int
    @State private var isPlaying = false

    private var maxFrame: Int { max(0, frames.count - 1) }

    var body: some View {
        HStack(spacing: 8) {
            // Step back
            Button {
                isPlaying = false
                currentFrame = max(0, currentFrame - 1)
            } label: {
                Image(systemName: "backward.frame.fill")
            }
            .buttonStyle(.borderless)
            .disabled(currentFrame == 0)

            // Play / Pause
            Button {
                isPlaying.toggle()
            } label: {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 16)
            }
            .buttonStyle(.borderless)

            // Step forward
            Button {
                isPlaying = false
                currentFrame = min(maxFrame, currentFrame + 1)
            } label: {
                Image(systemName: "forward.frame.fill")
            }
            .buttonStyle(.borderless)
            .disabled(currentFrame >= maxFrame)

            // Timeline slider
            Slider(
                value: Binding(
                    get: { Double(currentFrame) },
                    set: {
                        currentFrame = Int($0)
                        isPlaying = false
                    }
                ),
                in: 0...Double(maxFrame),
                step: 1
            )

            // Frame label
            Text("Step \(currentFrame)/\(maxFrame)")
                .monospacedDigit()
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 70, alignment: .trailing)
        }
        .task(id: isPlaying) {
            guard isPlaying else { return }
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(400))
                guard !Task.isCancelled else { break }
                if currentFrame < maxFrame {
                    currentFrame += 1
                } else {
                    currentFrame = 0 // loop
                }
            }
        }
    }
}

import SwiftUI

@main
struct RFantibodyOptimizerApp: App {
    @State private var campaign = DesignCampaign()
    @State private var runner = PipelineRunner()
    @AppStorage("darkMode") private var darkMode = false

    var body: some Scene {
        WindowGroup {
            ContentView(darkMode: $darkMode)
                .environment(campaign)
                .environment(runner)
                .preferredColorScheme(darkMode ? .dark : .light)
        }
        .defaultSize(width: 1400, height: 900)
    }
}

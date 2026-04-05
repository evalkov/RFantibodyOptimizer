import SwiftUI
import WebKit

enum ProteinColorScheme: String, CaseIterable, Identifiable {
    case chain = "Chain"
    case plddt = "pLDDT"
    case cdr = "CDR Loops"

    var id: String { rawValue }
}

struct ProteinViewer: NSViewRepresentable {
    let pdbURL: URL?
    let colorScheme: ProteinColorScheme
    var preserveCamera: Bool = false
    /// Show CDR residues as spheres instead of cartoon (for diffusion trajectory).
    var diffusionMode: Bool = false

    @Environment(\.colorScheme) private var appColorScheme

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")

        let handler = context.coordinator
        config.userContentController.add(handler, name: "ready")

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = handler
        context.coordinator.webView = webView

        let html = Self.viewerHTML
        webView.loadHTMLString(html, baseURL: nil)

        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        guard context.coordinator.isReady else {
            context.coordinator.pendingPDB = pdbURL
            context.coordinator.pendingColorScheme = colorScheme
            return
        }

        loadPDB(into: webView)
    }

    private func loadPDB(into webView: WKWebView) {
        guard let url = pdbURL else {
            webView.evaluateJavaScript("viewer.removeAllModels(); viewer.render();")
            return
        }

        guard let pdbString = try? String(contentsOf: url, encoding: .utf8) else { return }

        let escaped = pdbString
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'", with: "\\'")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "")

        let bgColor = appColorScheme == .dark ? "#1E1E1E" : "#FFFFFF"
        let styleJS = diffusionMode ? diffusionStyleJS() : colorSchemeJS(colorScheme)
        let zoomJS = preserveCamera ? "" : "viewer.zoomTo();"

        let js = """
        viewer.setBackgroundColor('\(bgColor)');
        document.body.style.background = '\(bgColor)';
        viewer.removeAllModels();
        viewer.addModel('\(escaped)', 'pdb');
        \(styleJS)
        \(zoomJS)
        viewer.render();
        """

        webView.evaluateJavaScript(js)
    }

    /// Diffusion trajectory style: CDR as spheres, framework/target as cartoon.
    private func diffusionStyleJS() -> String {
        """
        viewer.setStyle({}, {cartoon: {color: '#D1D5DB'}});
        viewer.setStyle({chain: 'T'}, {cartoon: {color: '#10B981'}});
        viewer.setStyle({b: 0}, {sphere: {color: '#991B1B', radius: 0.5}});
        """
    }

    private func colorSchemeJS(_ scheme: ProteinColorScheme) -> String {
        switch scheme {
        case .chain:
            return """
            viewer.setStyle({}, {cartoon: {color: '#AAAAAA'}});
            viewer.setStyle({chain: 'H'}, {cartoon: {color: '#3B82F6'}});
            viewer.setStyle({chain: 'L'}, {cartoon: {color: '#8B5CF6'}});
            viewer.setStyle({chain: 'T'}, {cartoon: {color: '#10B981'}});
            """
        case .plddt:
            return """
            viewer.setStyle({}, {cartoon: {
                colorfunc: function(atom) {
                    var b = atom.b;
                    if (b > 0.9 || b > 90) return '#0053D6';
                    if (b > 0.7 || b > 70) return '#65CBF3';
                    if (b > 0.5 || b > 50) return '#FFDB13';
                    return '#FF7D45';
                }
            }});
            """
        case .cdr:
            return """
            viewer.setStyle({}, {cartoon: {
                colorfunc: function(atom) {
                    if (atom.chain === 'T') return '#10B981';
                    if (atom.b < 0.5) return '#991B1B';
                    return '#D1D5DB';
                }
            }});
            """
        }
    }

    // MARK: - Coordinator

    class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        var webView: WKWebView?
        var isReady = false
        var pendingPDB: URL?
        var pendingColorScheme: ProteinColorScheme?

        func userContentController(_ userContentController: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            if message.name == "ready" {
                isReady = true
            }
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        }
    }

    // MARK: - Embedded HTML

    static let viewerHTML = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; }
        body { background: #FFFFFF; overflow: hidden; }
        #viewer { width: 100vw; height: 100vh; position: relative; }
    </style>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    </head>
    <body>
    <div id="viewer"></div>
    <script>
        var viewer = $3Dmol.createViewer("viewer", {
            backgroundColor: "#FFFFFF",
            antialias: true,
        });
        viewer.render();

        try {
            window.webkit.messageHandlers.ready.postMessage("ready");
        } catch(e) {}
    </script>
    </body>
    </html>
    """
}

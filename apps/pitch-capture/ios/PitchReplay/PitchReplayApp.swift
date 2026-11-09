import AVKit
import SwiftUI
import UIKit
import UniformTypeIdentifiers

@main
@MainActor
struct PitchReplayApp: App {
    @StateObject private var model = ReplayModel()

    var body: some Scene {
        WindowGroup { ReplayView(model: model).preferredColorScheme(.dark) }
    }
}

@MainActor
final class ReplayModel: ObservableObject {
    let camera = Camera()
    @Published var hand = ThrowHand.right
    @Published var sessions: [RecordedSession] = []
    @Published var livePoints: [Joint: Point2D] = [:]
    @Published var aspect = 16.0 / 9
    @Published var cameraReady = false
    @Published var opening = false
    @Published var recording = false
    @Published var busy = false
    @Published var status = ""

    init() {
        camera.onPose = { [weak self] points, aspect in
            guard let self, self.cameraReady else { return }
            self.livePoints = points
            self.aspect = aspect
        }
        do {
            let archive = try Archive.read()
            sessions = archive.sessions
            if archive.unreadable > 0 { status = "읽을 수 없는 기록 \(archive.unreadable)개가 있습니다." }
        } catch { status = error.localizedDescription }
    }

    var entries: [PitchSelection] {
        sessions.flatMap { session in session.pitches.map { PitchSelection(session: session, pitch: $0) } }
    }

    func openCamera() {
        opening = true
        camera.open { [weak self] result in
            guard let self else { return }
            self.opening = false
            switch result {
            case .success: self.cameraReady = true; self.status = "후면 카메라가 켜졌습니다."
            case .failure(let error): self.status = error.localizedDescription
            }
        }
    }

    func closeCamera() {
        camera.close()
        cameraReady = false
        livePoints = [:]
    }

    func start() {
        guard cameraReady, !busy, !recording else { return }
        do {
            let url = try Archive.folder().appendingPathComponent(UUID().uuidString + ".mov")
            let recordingHand = hand
            recording = true
            status = "녹화 중 · 최대 90초"
            camera.record(url) { [weak self] result in
                guard let self else { return }
                self.recording = false
                self.closeCamera()
                switch result {
                case .success(let url): self.ingest(url, hand: recordingHand, copy: false)
                case .failure(let error): self.status = error.localizedDescription
                }
            }
        } catch { status = error.localizedDescription }
    }

    func importVideo(_ result: Result<URL, Error>) {
        guard !recording, !busy else { return }
        switch result {
        case .success(let url): closeCamera(); ingest(url, hand: hand, copy: true)
        case .failure(let error): status = error.localizedDescription
        }
    }

    private func ingest(_ url: URL, hand: ThrowHand, copy: Bool) {
        busy = true
        status = "녹화 파일에서 관절과 투구 구간을 찾는 중입니다."
        let task = Task.detached(priority: .userInitiated) {
            let local: URL
            if copy {
                let security = url.startAccessingSecurityScopedResource()
                defer { if security { url.stopAccessingSecurityScopedResource() } }
                let original = AVURLAsset(url: url)
                _ = try await Archive.videoDuration(original)
                let tracks = try await original.loadTracks(withMediaType: .video)
                guard !tracks.isEmpty else { throw ReplayError.video }
                local = try Archive.folder().appendingPathComponent(UUID().uuidString + "." +
                    (url.pathExtension.isEmpty ? "mp4" : url.pathExtension))
                try FileManager.default.copyItem(at: url, to: local)
            } else { local = url }
            let session = try await Archive.analyze(local, hand: hand)
            try Archive.write(session)
            return session
        }
        Task {
            defer { busy = false }
            do {
                let session = try await task.value
                sessions.insert(session, at: 0)
                let tracked = session.poses.filter { !$0.points.isEmpty }.count
                status = "영상 저장 · 투구 후보 \(session.pitches.count)개 · 관절 관측 \(tracked)프레임"
            } catch { status = "영상 분석 실패: \(error.localizedDescription)" }
        }
    }

    func video(_ session: RecordedSession) -> URL? {
        try? Archive.folder().appendingPathComponent(session.videoName)
    }

    func mark(_ selection: PitchSelection, event: PitchEvent, time: Double) {
        guard let index = sessions.firstIndex(where: { $0.id == selection.session.id }),
              let pitchIndex = sessions[index].pitches.firstIndex(where: { $0.id == selection.id }) else { return }
        var next = sessions[index]
        guard next.pitches[pitchIndex].mark(event, at: time) else {
            status = "투구 구간 안에서 착지를 릴리스보다 앞에 표시하세요."
            return
        }
        save(next, at: index)
    }

    func addManual(_ id: UUID, start: Double, end: Double) -> UUID? {
        guard let index = sessions.firstIndex(where: { $0.id == id }),
              start >= 0, end <= sessions[index].duration, end - start >= 0.25 else {
            status = "같은 영상에서 시작과 끝을 0.25초 이상 벌려 선택하세요."
            return nil
        }
        var next = sessions[index]
        let pitch = Pitch(start: start, end: end, automatic: false)
        next.pitches.append(pitch)
        return save(next, at: index) ? pitch.id : nil
    }

    @discardableResult private func save(_ session: RecordedSession, at index: Int) -> Bool {
        do {
            try Archive.write(session)
            sessions[index] = session
            status = "표시를 저장했습니다."
            return true
        } catch { status = error.localizedDescription; return false }
    }
}

private struct PoseLines: View {
    let aspect: Double
    let layers: [([Joint: Point2D], Color)]

    var body: some View {
        Canvas { context, size in
            let width = min(size.width, size.height * aspect)
            let height = width / aspect
            let left = (size.width - width) / 2
            let top = (size.height - height) / 2
            func position(_ p: Point2D) -> CGPoint {
                CGPoint(x: left + width * p.x, y: top + height * p.y)
            }
            for (points, color) in layers {
                var lines = Path()
                for (first, second) in Joint.bones {
                    if let a = points[first], let b = points[second] {
                        lines.move(to: position(a))
                        lines.addLine(to: position(b))
                    }
                }
                context.stroke(lines, with: .color(color), style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                for point in points.values {
                    let p = position(point)
                    context.fill(Path(ellipseIn: CGRect(x: p.x - 2.4, y: p.y - 2.4, width: 4.8, height: 4.8)),
                                 with: .color(color))
                }
            }
        }.allowsHitTesting(false).accessibilityHidden(true)
    }
}

private final class CameraSurface: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

private struct CameraFeed: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> CameraSurface { CameraSurface() }
    func updateUIView(_ view: CameraSurface, context: Context) {
        if view.preview.session !== session { view.preview.session = session }
        view.preview.videoGravity = .resizeAspect
        if let link = view.preview.connection, link.isVideoOrientationSupported {
            link.videoOrientation = .landscapeRight
        }
    }
}

private final class VideoSurface: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }
    var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
}

private struct MovieFrame: UIViewRepresentable {
    let url: URL
    let time: Double
    let onReady: (URL, Double) -> Void

    final class Coordinator {
        var url: URL?
        var time: Double?
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeUIView(context: Context) -> VideoSurface {
        let view = VideoSurface()
        view.backgroundColor = .black
        view.playerLayer.videoGravity = .resizeAspect
        view.playerLayer.player = AVPlayer()
        return view
    }
    func updateUIView(_ view: VideoSurface, context: Context) {
        guard let player = view.playerLayer.player else { return }
        if context.coordinator.url != url {
            player.replaceCurrentItem(with: AVPlayerItem(url: url))
            context.coordinator.url = url
            context.coordinator.time = nil
        }
        if context.coordinator.time != time {
            context.coordinator.time = time
            player.currentItem?.cancelPendingSeeks()
            let coordinator = context.coordinator
            player.seek(to: CMTime(seconds: time, preferredTimescale: 60000), toleranceBefore: .zero, toleranceAfter: .zero) { finished in
                DispatchQueue.main.async {
                    if finished && coordinator.url == url && coordinator.time == time { onReady(url, time) }
                }
            }
        }
    }
}

@MainActor
private struct ReplayView: View {
    @ObservedObject var model: ReplayModel
    @State private var screen = 0
    @State private var importing = false
    @State private var reviewID: UUID?
    @State private var time = 0.0
    @State private var manualSession: UUID?
    @State private var manualStart: Double?
    @State private var aID: UUID?
    @State private var bID: UUID?
    @State private var sameView = false
    @State private var phase = 0.0
    @State private var shownReview: Double?
    @State private var shownCompare: Double?

    private let mint = Color(red: 0.34, green: 0.92, blue: 0.76)
    private let amber = Color(red: 1, green: 0.68, blue: 0.34)
    private let panel = Color(red: 0.075, green: 0.10, blue: 0.13)

    var body: some View {
        GeometryReader { geometry in
            let stacked = geometry.size.width < 680
            Group {
                if stacked {
                    ScrollView { VStack(spacing: 18) { rail; stage.frame(minHeight: 280) } }
                } else {
                    HStack(alignment: .top, spacing: 18) {
                        ScrollView { rail }.frame(width: 245)
                        stage
                    }
                }
            }.padding(16)
        }
        .background(Color(red: 0.045, green: 0.064, blue: 0.085).ignoresSafeArea())
        .tint(mint)
        .fileImporter(isPresented: $importing, allowedContentTypes: [.movie], onCompletion: model.importVideo)
        .onChange(of: screen) { next in
            if next != 0 { model.closeCamera() }
            shownReview = nil
            shownCompare = nil
        }
        .onChange(of: aID) { _ in sameView = false }
        .onChange(of: bID) { _ in sameView = false }
    }

    private var rail: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Pitch Replay").font(.system(size: 28, weight: .semibold))
            Picker("화면", selection: $screen) {
                Text("촬영").tag(0); Text("복기").tag(1); Text("비교").tag(2)
            }.pickerStyle(.segmented).disabled(model.recording || model.opening || model.busy)

            switch screen {
            case 0: captureControls
            case 1: reviewControls
            default: compareControls
            }
            if model.busy { ProgressView().accessibilityLabel("분석 중") }
            if !model.status.isEmpty { Text(model.status).font(.footnote).foregroundStyle(.secondary) }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private var captureControls: some View {
        Group {
            Picker("투구 손", selection: $model.hand) {
                ForEach(ThrowHand.allCases, id: \.self) { Text($0.title).tag($0) }
            }.pickerStyle(.segmented).disabled(model.recording || model.busy)
            if model.recording {
                Button("녹화 종료") { model.camera.stop() }.buttonStyle(.borderedProminent)
            } else if model.cameraReady {
                Button("녹화 시작") { model.start() }.buttonStyle(.borderedProminent).disabled(model.busy)
            } else {
                Button("후면 카메라 켜기") { model.openCamera() }
                    .buttonStyle(.borderedProminent).disabled(model.busy || model.opening)
            }
            Button("기존 영상 가져오기") { importing = true }
                .buttonStyle(.bordered).disabled(model.recording || model.busy || model.opening)
            Text("휴대폰을 가로로 고정하고 전신이 들어오게 촬영하세요. 90초까지 저장합니다.")
                .font(.footnote).foregroundStyle(.secondary)
        }
    }

    private func clipName(_ item: PitchSelection) -> String {
        let n = (item.session.pitches.firstIndex { $0.id == item.id } ?? 0) + 1
        return item.session.created.formatted(.dateTime.hour().minute().second()) + " · \(n)구"
    }

    private var reviewItem: PitchSelection? { model.entries.first { $0.id == reviewID } }
    private var reviewSession: RecordedSession? {
        reviewItem?.session ?? model.sessions.first { $0.id == manualSession }
    }

    private var reviewControls: some View {
        Group {
            Picker("복기할 투구", selection: $reviewID) {
                Text("투구 선택").tag(nil as UUID?)
                ForEach(model.entries) { Text(clipName($0)).tag(Optional($0.id)) }
            }
            .onChange(of: reviewID) { _ in
                if let pitch = reviewItem?.pitch { time = pitch.start }
                else { time = 0 }
                shownReview = nil
                manualStart = nil
            }
            if let item = reviewItem {
                HStack {
                    Button("착지 확인") { model.mark(item, event: .landing, time: time) }
                    Button("릴리스 확인") { model.mark(item, event: .release, time: time) }
                }.buttonStyle(.bordered).disabled(shownReview != time)
                Text("착지 \(eventText(item.pitch.landing)) · 릴리스 \(eventText(item.pitch.release))")
                    .font(.caption).monospacedDigit()
            } else {
                Picker("투구 구간 추가", selection: $manualSession) {
                    Text("원본 선택").tag(nil as UUID?)
                    ForEach(model.sessions) { session in
                        Text(session.created.formatted(.dateTime.hour().minute().second())).tag(Optional(session.id))
                    }
                }.onChange(of: manualSession) { _ in time = 0; manualStart = nil; shownReview = nil }
                if let session = reviewSession {
                    HStack {
                        Button("구간 시작") { manualStart = time }
                        Button("구간 끝 저장") {
                            if let start = manualStart, let id = model.addManual(session.id, start: start, end: time) {
                                reviewID = id
                            }
                        }.disabled(manualStart == nil)
                    }.buttonStyle(.bordered).disabled(shownReview != time)
                    if let start = manualStart { Text("시작 \(eventText(start))").font(.caption) }
                }
            }
        }
    }

    private func eventText(_ time: Double?) -> String {
        time.map { String(format: "%.3f초", $0) } ?? "미확인"
    }

    private var compareControls: some View {
        Group {
            Picker("A · 기준 투구", selection: $aID) {
                Text("선택").tag(nil as UUID?)
                ForEach(model.entries) { Text(clipName($0)).tag(Optional($0.id)) }
            }.foregroundStyle(mint)
            Picker("B · 비교 투구", selection: $bID) {
                Text("선택").tag(nil as UUID?)
                ForEach(model.entries) { Text(clipName($0)).tag(Optional($0.id)) }
            }.foregroundStyle(amber)
            Toggle("같은 선수와 고정된 카메라 시야", isOn: $sameView).font(.footnote)
            Text("착지에서 릴리스까지의 진행률을 맞춥니다. 화면에서 보이는 2D 차이입니다.")
                .font(.footnote).foregroundStyle(.secondary)
        }
    }

    private var stage: some View {
        VStack(spacing: 8) {
            switch screen {
            case 0:
                if model.cameraReady {
                    ZStack {
                        CameraFeed(session: model.camera.session)
                        PoseLines(aspect: model.aspect, layers: [(model.livePoints, mint)])
                    }
                    Text(model.livePoints.isEmpty ? "관절 관측 없음" : "관절 \(model.livePoints.count)/12개")
                        .font(.caption.monospacedDigit())
                } else { placeholder("카메라를 켜거나 촬영한 영상을 가져오세요.") }
            case 1:
                if let session = reviewSession, let url = model.video(session) {
                    let range = (reviewItem?.pitch.start ?? 0)...(reviewItem?.pitch.end ?? session.duration)
                    ZStack {
                        MovieFrame(url: url, time: time) { _, displayed in shownReview = displayed }
                        if shownReview == time {
                            PoseLines(aspect: session.aspect, layers: [(session.nearestPose(at: time)?.points ?? [:], mint)])
                        }
                    }
                    HStack {
                        Button("이전") { time = session.frameTime(from: time, moving: -1, in: range) }
                        Slider(value: $time, in: range, onEditingChanged: { editing in
                            if !editing { time = session.frameTime(from: time, moving: 0, in: range) }
                        }).accessibilityLabel("영상 시각")
                        Button("다음") { time = session.frameTime(from: time, moving: 1, in: range) }
                        Text(eventText(time)).monospacedDigit()
                    }.buttonStyle(.bordered)
                    if session.frameTimes.isEmpty {
                        Text("프레임 시각을 읽지 못해 이전·다음은 약 1/30초씩 이동합니다.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                } else { placeholder("저장한 투구나 원본을 고르세요.") }
            default: comparison
            }
        }
        .padding(12).frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(panel, in: RoundedRectangle(cornerRadius: 12))
    }

    private var comparison: some View {
        Group {
            if let a = model.entries.first(where: { $0.id == aID }),
               let b = model.entries.first(where: { $0.id == bID }),
               PitchSelection.canCompare(a, b, sameView: sameView),
               let url = model.video(a.session) {
                let ta = a.pitch.time(at: phase)
                let tb = b.pitch.time(at: phase)
                let pa = a.session.nearestPose(at: ta)?.points ?? [:]
                let pb = b.session.nearestPose(at: tb)?.points ?? [:]
                ZStack {
                    MovieFrame(url: url, time: ta) { _, displayed in shownCompare = displayed }
                    if shownCompare == ta { PoseLines(aspect: a.session.aspect, layers: [(pa, mint), (pb, amber)]) }
                }
                HStack {
                    Text("착지").font(.caption)
                    Slider(value: $phase).accessibilityLabel("착지에서 릴리스까지")
                    Text("릴리스").font(.caption)
                }
                Text("A \(eventText(ta)) · 관절 \(pa.count)/12개     B \(eventText(tb)) · 관절 \(pb.count)/12개")
                    .font(.caption.monospacedDigit())
            } else {
                placeholder("같은 시야에서 관절이 보이는 서로 다른 두 투구를 고르세요. 투구 손, 영상 비율, 착지·릴리스 표시를 확인하세요.")
            }
        }
    }

    private func placeholder(_ text: String) -> some View {
        Text(text).foregroundStyle(.secondary).multilineTextAlignment(.center)
            .frame(maxWidth: .infinity, maxHeight: .infinity).padding()
    }
}

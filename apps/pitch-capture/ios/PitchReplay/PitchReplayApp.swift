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
    @Published var highSpeed = false
    @Published private(set) var markedModeReady = false
    @Published private(set) var liveFPS: Double?
    @Published private(set) var choosingBallPath = false
    @Published private(set) var firstBallPoint: Point2D?
    @Published private(set) var ballCorridor: BallCorridor?
    @Published private(set) var liveBall = BallReadout.waiting
    @Published var scaleMeters = ""
    private var ballRevision = 0

    init() {
        camera.onPose = { [weak self] points, aspect in
            guard let self, self.cameraReady else { return }
            self.livePoints = points
            if abs(self.aspect - aspect) > 0.01 {
                self.aspect = aspect
                if self.ballCorridor != nil { self.chooseBallPath() }
            }
        }
        camera.onBall = { [weak self] readout, revision in
            guard let self, self.cameraReady, revision == self.ballRevision else { return }
            self.liveBall = readout
        }
        camera.onRate = { [weak self] rate in
            guard let self, self.cameraReady else { return }
            self.liveFPS = rate
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
        let requestedHighSpeed = highSpeed
        camera.open(highSpeed: requestedHighSpeed) { [weak self] result in
            guard let self else { return }
            self.opening = false
            switch result {
            case .success(let markedMode):
                self.cameraReady = true
                self.markedModeReady = markedMode
                self.status = requestedHighSpeed && !markedMode
                    ? "이 카메라 구성에서는 240 fps 표식 추적을 사용할 수 없습니다."
                    : "후면 카메라가 켜졌습니다."
            case .failure(let error): self.status = error.localizedDescription
            }
        }
    }

    func closeCamera() {
        camera.close()
        cameraReady = false
        livePoints = [:]
        liveFPS = nil
        markedModeReady = false
        choosingBallPath = false
        firstBallPoint = nil
        scaleMeters = ""
        installBallCorridor(nil)
    }

    func chooseBallPath() {
        choosingBallPath = true
        firstBallPoint = nil
        scaleMeters = ""
        installBallCorridor(nil)
        status = "공이 날아가는 방향으로 영상에서 시작점과 끝점을 고르세요."
    }

    func placeBallPoint(_ point: Point2D) {
        guard choosingBallPath else { return }
        guard let start = firstBallPoint else {
            firstBallPoint = point
            status = "이제 공이 향하는 쪽에 끝점을 고르세요."
            return
        }
        guard let corridor = BallCorridor(start: start, end: point, aspect: aspect, meters: nil) else {
            status = "시작점에서 충분히 떨어진 끝점을 고르세요. 카메라는 투구 옆에 두세요."
            return
        }
        choosingBallPath = false
        firstBallPoint = nil
        installBallCorridor(corridor)
        status = "공이 추적 구간에 들어오면 궤적을 표시합니다."
    }

    func applyBallScale() {
        guard let corridor = ballCorridor else { return }
        let text = scaleMeters.trimmingCharacters(in: .whitespacesAndNewlines)
        let meters = Double(text.replacingOccurrences(of: ",", with: "."))
        if !text.isEmpty {
            guard let meters, meters.isFinite, (0.01...100).contains(meters) else {
                status = "기준 거리를 0.01~100 m 사이 숫자로 입력하세요."
                return
            }
        }
        guard let calibrated = BallCorridor(start: corridor.start, end: corridor.end,
                                            aspect: aspect, meters: meters) else { return }
        installBallCorridor(calibrated)
        status = meters == nil ? "거리 보정을 해제했습니다." : "화면 거리 보정을 적용했습니다."
    }

    private func installBallCorridor(_ corridor: BallCorridor?) {
        ballRevision += 1
        ballCorridor = corridor
        liveBall = .waiting
        camera.setBallCorridor(corridor, revision: ballRevision)
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

private struct BallOverlay: View {
    let aspect: Double
    let corridor: BallCorridor?
    let first: Point2D?
    let readout: BallReadout
    let color: Color

    var body: some View {
        Canvas { context, size in
            let rect = imageRect(in: size, aspect: aspect)
            func position(_ p: Point2D) -> CGPoint {
                CGPoint(x: rect.minX + p.x * rect.width, y: rect.minY + p.y * rect.height)
            }
            func dot(_ p: CGPoint, radius: CGFloat, color: Color) {
                let rect = CGRect(x: p.x - radius, y: p.y - radius, width: radius * 2, height: radius * 2)
                context.fill(Path(ellipseIn: rect), with: .color(color))
            }
            if let corridor {
                let a = position(corridor.start)
                let b = position(corridor.end)
                let axis = corridor.direction
                let pad = BallCorridor.halfWidth * rect.height
                let offset = CGPoint(x: -axis.y * pad, y: axis.x * pad)
                var outline = Path()
                outline.move(to: CGPoint(x: a.x + offset.x, y: a.y + offset.y))
                outline.addLine(to: CGPoint(x: b.x + offset.x, y: b.y + offset.y))
                outline.addLine(to: CGPoint(x: b.x - offset.x, y: b.y - offset.y))
                outline.addLine(to: CGPoint(x: a.x - offset.x, y: a.y - offset.y))
                outline.closeSubpath()
                var clipped = context
                clipped.clip(to: Path(rect))
                clipped.stroke(outline, with: .color(.white.opacity(0.65)), style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
                dot(a, radius: 3, color: .white)
                dot(b, radius: 3, color: .white)
            } else if let first {
                dot(position(first), radius: 4, color: .white)
            }
            guard let start = readout.path.first else { return }
            let trace = color.opacity(readout.active ? 1 : 0.70)
            var line = Path()
            line.move(to: position(start))
            for point in readout.path.dropFirst() { line.addLine(to: position(point)) }
            context.stroke(line, with: .color(.black.opacity(0.70)), style: StrokeStyle(lineWidth: 4, lineCap: .round))
            context.stroke(line, with: .color(trace), style: StrokeStyle(lineWidth: 2, lineCap: .round))
            for point in readout.path { dot(position(point), radius: 2.5, color: trace) }
        }.allowsHitTesting(false).accessibilityHidden(true)
    }
}

private func imageRect(in size: CGSize, aspect: Double) -> CGRect {
    let width = min(size.width, size.height * aspect)
    let height = width / aspect
    return CGRect(x: (size.width - width) / 2, y: (size.height - height) / 2, width: width, height: height)
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
        if let link = view.preview.connection, link.isVideoOrientationSupported, link.videoOrientation != .landscapeRight {
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
            if model.cameraReady {
                Button(model.ballCorridor == nil ? "투구 경로 지정" : "투구 경로 다시 지정") { model.chooseBallPath() }
                    .buttonStyle(.bordered).disabled(model.choosingBallPath)
                if model.choosingBallPath {
                    Text(model.firstBallPoint == nil ? "1. 영상에서 공이 들어올 지점을 탭하세요." : "2. 공이 나갈 지점을 탭하세요.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                if model.ballCorridor != nil {
                    HStack(spacing: 8) {
                        TextField("기준 거리 (m)", text: $model.scaleMeters)
                            .textFieldStyle(.roundedBorder).keyboardType(.decimalPad)
                            .accessibilityLabel("두 지점 사이의 실제 거리, 미터")
                        Button("적용") { model.applyBallScale() }.buttonStyle(.bordered)
                    }
                    Text("공이 지나는 평면에서 잰 두 지점 사이 거리를 입력하세요. 거리 없이도 궤적과 휨(px)을 볼 수 있습니다.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if !model.recording {
                    Button("카메라 닫기") { model.closeCamera() }.buttonStyle(.bordered)
                }
            } else {
                Toggle("표식 회전 모드", isOn: $model.highSpeed)
                    .font(.footnote).disabled(model.busy || model.opening)
                if model.highSpeed {
                    Text("240 fps 지원 기기에서 밝은 곳에 고정하세요. 흰 공에 파란 점 하나를 찍고 회전축을 카메라 방향에 가깝게 맞추세요.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Button("기존 영상 가져오기") { importing = true }
                .buttonStyle(.bordered).disabled(model.recording || model.busy || model.opening)
            Text("휴대폰을 투구 방향 옆에 가로로 고정하세요. 전신과 공이 지나는 구간을 영상에 담으세요.")
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
                    livePreview
                    liveReadout
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

    private var livePreview: some View {
        GeometryReader { geometry in
            ZStack {
                CameraFeed(session: model.camera.session)
                PoseLines(aspect: model.aspect, layers: [(model.livePoints, mint)])
                BallOverlay(aspect: model.aspect, corridor: model.ballCorridor, first: model.firstBallPoint,
                            readout: model.liveBall, color: amber)
                if model.choosingBallPath {
                    Rectangle().fill(.clear).contentShape(Rectangle())
                        .gesture(SpatialTapGesture().onEnded { event in
                            let rect = imageRect(in: geometry.size, aspect: model.aspect)
                            guard rect.contains(event.location), !rect.isEmpty else { return }
                            model.placeBallPoint(Point2D(x: (event.location.x - rect.minX) / rect.width,
                                                        y: (event.location.y - rect.minY) / rect.height))
                        })
                        .accessibilityLabel(model.firstBallPoint == nil ? "투구 경로 시작점 선택" : "투구 경로 끝점 선택")
                }
            }
        }.frame(minHeight: 160)
    }

    private var liveReadout: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text("관절 \(model.livePoints.count)/12")
                if let fps = model.liveFPS { Text(String(format: "%.0f fps", fps)) }
                if model.ballCorridor != nil {
                    let ball = model.liveBall
                    Text(ball.path.isEmpty ? "공 대기" : "\(ball.active ? "공 추적 중" : "마지막 궤적") · \(ball.path.count)프레임")
                        .foregroundStyle(amber)
                }
            }.font(.caption.monospacedDigit())
            if model.ballCorridor != nil {
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 16) { ballValues }
                    VStack(alignment: .leading, spacing: 4) { ballValues }
                }.font(.caption.monospacedDigit())
                Text(ballNote)
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private var ballValues: some View {
        Group {
            Text(model.liveBall.speedKPH.map { String(format: "화면 속도 %.1f km/h", $0) } ?? "화면 속도 —")
            Text(model.liveBall.movementCM.map { String(format: "화면 휨 %+.1f cm", $0) }
                 ?? model.liveBall.movementPX.map { String(format: "화면 휨 %+.1f px", $0) } ?? "화면 휨 —")
            Text(model.liveBall.rpm.map { String(format: "표식 회전 약 %.0f RPM", $0) } ?? "회전수 —")
        }.fixedSize(horizontal: true, vertical: false)
    }

    private var ballNote: String {
        let scale = model.ballCorridor?.meters == nil
            ? "거리 보정 후 속도와 휨(cm) 표시"
            : "처음 3프레임 평균 속도"
        let spin: String
        if model.liveBall.rpm != nil { spin = "카메라를 향한 축 기준 표식 추정" }
        else if model.markedModeReady { spin = "회전수는 선명한 파란 점과 연속 고속 프레임이 필요" }
        else { spin = "회전수는 카메라를 닫고 표식 회전 모드로 다시 시작" }
        return "\(scale) · 휨은 양 끝점을 이은 선에서 가장 먼 편차 (+는 아래) · \(spin)"
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

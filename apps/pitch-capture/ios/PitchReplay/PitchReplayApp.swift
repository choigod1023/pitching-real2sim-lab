import AVKit
import SwiftUI
import UIKit
import UniformTypeIdentifiers

@main
@MainActor
struct PitchReplayApp: App {
    @StateObject private var model = ReplayModel()

    var body: some Scene {
        WindowGroup { ReplayView(model: model) }
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
    private enum Destination: Hashable {
        case capture, review, compare
    }

    @ObservedObject var model: ReplayModel
    @Environment(\.dynamicTypeSize) private var typeSize
    @State private var screen = Destination.capture
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
    private var captureLocked: Bool { model.recording || model.opening || model.busy }

    var body: some View {
        TabView(selection: tabSelection) {
            page("촬영", stage: { stage(.capture) }, options: { captureControls })
                .tabItem { Label("촬영", systemImage: "video") }
                .tag(Destination.capture)
            page("복기", stage: { stage(.review) }, options: { reviewControls })
                .tabItem { Label("복기", systemImage: "list.bullet.rectangle") }
                .tag(Destination.review)
            page("비교", stage: { stage(.compare) }, options: { compareControls })
                .tabItem { Label("비교", systemImage: "square.split.2x1") }
                .tag(Destination.compare)
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.movie], onCompletion: model.importVideo)
        .onChange(of: screen) { next in
            if next != .capture { model.closeCamera() }
        }
        .onChange(of: reviewID) { _ in
            time = reviewItem?.pitch.start ?? 0
            shownReview = nil
            manualStart = nil
        }
        .onChange(of: manualSession) { _ in
            time = 0
            shownReview = nil
            manualStart = nil
        }
        .onChange(of: aID) { _ in sameView = false }
        .onChange(of: bID) { _ in sameView = false }
    }

    private var tabSelection: Binding<Destination> {
        Binding(get: { screen }, set: { destination in
            if !captureLocked { screen = destination }
        })
    }

    private func page<Stage: View, Options: View>(
        _ title: String, @ViewBuilder stage: () -> Stage, @ViewBuilder options: () -> Options
    ) -> some View {
        let preview = stage()
        let rows = options()
        return NavigationStack {
            GeometryReader { geometry in
                if geometry.size.width >= 620 && !typeSize.isAccessibilitySize {
                    HStack(spacing: 0) {
                        preview.frame(maxWidth: .infinity, maxHeight: .infinity).padding(12)
                        Divider()
                        Form { rows; notice }
                            .frame(width: min(340, max(252, geometry.size.width * 0.35)))
                    }
                } else {
                    Form {
                        Section {
                            preview.frame(minHeight: 180)
                                .listRowInsets(EdgeInsets(top: 8, leading: 8, bottom: 8, trailing: 8))
                        }
                        rows
                        notice
                    }
                }
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    @ViewBuilder private var notice: some View {
        if model.busy || !model.status.isEmpty {
            Section {
                if model.busy { ProgressView("영상 분석 중") }
                if !model.status.isEmpty { Text(model.status).font(.footnote).foregroundStyle(.secondary) }
            }
        }
    }

    private var captureControls: some View {
        Group {
            Section {
                Picker("투구 손", selection: $model.hand) {
                    ForEach(ThrowHand.allCases, id: \.self) { Text($0.title).tag($0) }
                }.disabled(captureLocked)
                if model.cameraReady {
                    Button("카메라 닫기") { model.closeCamera() }.disabled(model.recording)
                } else {
                    Toggle("표식 회전 모드", isOn: $model.highSpeed).disabled(captureLocked)
                }
            } header: {
                Text("카메라")
            } footer: {
                if model.highSpeed && !model.cameraReady {
                    Text("240 fps 지원 기기를 밝은 곳에 고정하세요. 파란 표식이 보이도록 회전축을 카메라 방향에 가깝게 맞추세요.")
                }
            }
            if model.cameraReady {
                Section {
                    Button(model.ballCorridor == nil ? "투구 경로 지정" : "투구 경로 다시 지정") {
                        model.chooseBallPath()
                    }.disabled(model.choosingBallPath)
                    if model.choosingBallPath {
                        Text(model.firstBallPoint == nil ? "영상에서 공이 들어올 지점을 탭하세요." : "공이 나갈 지점을 탭하세요.")
                            .foregroundStyle(.secondary)
                    }
                    if model.ballCorridor != nil {
                        TextField("기준 거리 (m)", text: $model.scaleMeters)
                            .keyboardType(.decimalPad)
                            .accessibilityLabel("두 지점 사이의 실제 거리, 미터")
                        Button("거리 적용") { model.applyBallScale() }
                    }
                } header: {
                    Text("공 추적")
                } footer: {
                    if model.ballCorridor != nil {
                        Text(ballNote)
                    }
                }
            }
            Section("영상") {
                Button { importing = true } label: {
                    Label("영상 가져오기", systemImage: "square.and.arrow.down")
                }.disabled(captureLocked)
            }
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
            Section("영상") {
                Picker("복기할 투구", selection: $reviewID) {
                    Text("투구 선택").tag(nil as UUID?)
                    ForEach(model.entries) { Text(clipName($0)).tag(Optional($0.id)) }
                }
                .pickerStyle(.navigationLink)
            }
            if let item = reviewItem {
                Section("투구 구간") {
                    Button { model.mark(item, event: .landing, time: time) } label: {
                        LabeledContent("착지 확인", value: eventText(item.pitch.landing))
                    }.disabled(shownReview != time)
                    Button { model.mark(item, event: .release, time: time) } label: {
                        LabeledContent("릴리스 확인", value: eventText(item.pitch.release))
                    }.disabled(shownReview != time)
                }
            } else {
                Section("투구 구간 추가") {
                    Picker("원본 영상", selection: $manualSession) {
                        Text("원본 선택").tag(nil as UUID?)
                        ForEach(model.sessions) { session in
                            Text(session.created.formatted(.dateTime.hour().minute().second())).tag(Optional(session.id))
                        }
                    }.pickerStyle(.navigationLink)
                    if let session = reviewSession {
                        Button("구간 시작") { manualStart = time }
                            .disabled(shownReview != time)
                        Button("구간 끝 저장") {
                            if let start = manualStart, let id = model.addManual(session.id, start: start, end: time) {
                                reviewID = id
                            }
                        }.disabled(manualStart == nil || shownReview != time)
                        if let start = manualStart { LabeledContent("시작", value: eventText(start)) }
                    }
                }
            }
        }
    }

    private func eventText(_ time: Double?) -> String {
        time.map { String(format: "%.3f초", $0) } ?? "미확인"
    }

    private var compareControls: some View {
        Section {
            Picker("A · 기준 투구", selection: $aID) {
                Text("선택").tag(nil as UUID?)
                ForEach(model.entries) { Text(clipName($0)).tag(Optional($0.id)) }
            }.pickerStyle(.navigationLink)
            Picker("B · 비교 투구", selection: $bID) {
                Text("선택").tag(nil as UUID?)
                ForEach(model.entries) { Text(clipName($0)).tag(Optional($0.id)) }
            }.pickerStyle(.navigationLink)
            Toggle("같은 선수와 고정된 카메라 시야", isOn: $sameView)
        } header: {
            Text("비교할 투구")
        } footer: {
            Text("착지에서 릴리스까지의 진행률을 맞춥니다. 화면에서 보이는 2D 차이입니다.")
        }
    }

    @ViewBuilder private func stage(_ destination: Destination) -> some View {
        switch destination {
        case .capture: captureStage
        case .review: reviewStage
        case .compare: comparison
        }
    }

    private var captureStage: some View {
        VStack(spacing: 8) {
            if model.cameraReady {
                livePreview
                    .background(.black)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                liveReadout
                Button {
                    if model.recording { model.camera.stop() } else { model.start() }
                } label: {
                    Label {
                        Text(model.recording ? "녹화 종료" : "녹화 시작").foregroundStyle(.primary)
                    } icon: {
                        Image(systemName: model.recording ? "stop.circle.fill" : "record.circle")
                            .foregroundStyle(.red)
                    }.frame(minHeight: 34)
                }
                .buttonStyle(.bordered)
                .disabled(model.busy)
            } else {
                placeholder("카메라 준비", symbol: "video", detail: "기기를 투구 방향 옆에 가로로 고정하세요. 전신과 공이 지나는 구간을 영상에 담으세요.")
                Button { model.openCamera() } label: {
                    Label("후면 카메라 켜기", systemImage: "video")
                        .frame(minHeight: 34)
                }.buttonStyle(.borderedProminent)
                    .disabled(captureLocked)
            }
        }
    }

    @ViewBuilder private var reviewStage: some View {
        if let session = reviewSession, let url = model.video(session) {
            let range = (reviewItem?.pitch.start ?? 0)...(reviewItem?.pitch.end ?? session.duration)
            VStack(spacing: 4) {
                ZStack {
                    MovieFrame(url: url, time: time) { _, displayed in shownReview = displayed }
                    if shownReview == time {
                        PoseLines(aspect: session.aspect, layers: [(session.nearestPose(at: time)?.points ?? [:], mint)])
                    }
                }
                .background(.black)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                HStack(spacing: 8) {
                    Button {
                        time = session.frameTime(from: time, moving: -1, in: range)
                    } label: {
                        Image(systemName: "backward.end").frame(minWidth: 44, minHeight: 44)
                    }.accessibilityLabel("이전 프레임")
                    Slider(value: $time, in: range, onEditingChanged: { editing in
                        if !editing { time = session.frameTime(from: time, moving: 0, in: range) }
                    }).accessibilityLabel("영상 시각")
                    Button {
                        time = session.frameTime(from: time, moving: 1, in: range)
                    } label: {
                        Image(systemName: "forward.end").frame(minWidth: 44, minHeight: 44)
                    }.accessibilityLabel("다음 프레임")
                }.buttonStyle(.borderless)
                Text(eventText(time)).font(.footnote.monospacedDigit())
                if session.frameTimes.isEmpty {
                    Text("프레임 시각이 없어 약 1/30초씩 이동합니다.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
        } else {
            placeholder("복기할 영상 선택", symbol: "film", detail: "저장한 투구 또는 원본을 고르세요.")
        }
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
        }.frame(minHeight: 100)
    }

    private var liveReadout: some View {
        VStack(alignment: .leading, spacing: 4) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) { trackingStatus }
                VStack(alignment: .leading, spacing: 4) { trackingStatus }
            }.font(.footnote.monospacedDigit()).foregroundStyle(.secondary)
            if model.ballCorridor != nil {
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 16) { ballValues }.fixedSize(horizontal: true, vertical: false)
                    VStack(alignment: .leading, spacing: 4) { ballValues }
                }.font(.footnote.monospacedDigit())
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private var trackingStatus: some View {
        Group {
            Text("관절 \(model.livePoints.count)/12")
            if let fps = model.liveFPS { Text(String(format: "%.0f fps", fps)) }
            if model.ballCorridor != nil {
                let ball = model.liveBall
                Text(ball.path.isEmpty ? "공 대기" : "\(ball.active ? "공 추적 중" : "마지막 궤적") · \(ball.path.count)프레임")
            }
        }
    }

    private var ballValues: some View {
        Group {
            Text(model.liveBall.speedKPH.map { String(format: "화면 속도 %.1f km/h", $0) } ?? "화면 속도 —")
            Text(model.liveBall.movementCM.map { String(format: "화면 휨 %+.1f cm", $0) }
                 ?? model.liveBall.movementPX.map { String(format: "화면 휨 %+.1f px", $0) } ?? "화면 휨 —")
            Text(model.liveBall.rpm.map { String(format: "표식 회전 약 %.0f RPM", $0) } ?? "회전수 —")
        }
    }

    private var ballNote: String {
        let scale = model.ballCorridor?.meters == nil
            ? "속도를 보려면 공이 지나는 평면에서 두 지점 사이의 실제 거리를 입력하세요."
            : "처음 3프레임 평균 속도"
        let spin: String
        if model.liveBall.rpm != nil { spin = "카메라를 향한 축 기준 표식 추정" }
        else if model.markedModeReady { spin = "회전수는 선명한 파란 점과 연속 고속 프레임이 필요" }
        else { spin = "회전수는 카메라를 닫고 표식 회전 모드로 다시 시작" }
        return "\(scale) · 휨은 양 끝점을 이은 선에서 가장 먼 편차 (+는 아래) · \(spin)"
    }

    @ViewBuilder private var comparison: some View {
        if let a = model.entries.first(where: { $0.id == aID }),
           let b = model.entries.first(where: { $0.id == bID }),
           PitchSelection.canCompare(a, b, sameView: sameView),
           let url = model.video(a.session) {
            let ta = a.pitch.time(at: phase)
            let tb = b.pitch.time(at: phase)
            let pa = a.session.nearestPose(at: ta)?.points ?? [:]
            let pb = b.session.nearestPose(at: tb)?.points ?? [:]
            VStack(spacing: 8) {
                ZStack {
                    MovieFrame(url: url, time: ta) { _, displayed in shownCompare = displayed }
                    if shownCompare == ta { PoseLines(aspect: a.session.aspect, layers: [(pa, mint), (pb, amber)]) }
                }.background(.black)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                HStack {
                    Text("착지").font(.caption)
                    Slider(value: $phase).accessibilityLabel("착지에서 릴리스까지")
                    Text("릴리스").font(.caption)
                }
                Text("A \(eventText(ta)) · 관절 \(pa.count)/12개     B \(eventText(tb)) · 관절 \(pb.count)/12개")
                    .font(.caption.monospacedDigit())
            }
        } else {
            placeholder("비교할 투구 선택", symbol: "square.split.2x1", detail: "같은 시야의 서로 다른 두 투구를 고르세요. 각 투구의 착지와 릴리스가 필요합니다.")
        }
    }

    private func placeholder(_ title: String, symbol: String, detail: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: symbol).font(.title).foregroundStyle(.secondary).accessibilityHidden(true)
            Text(title).font(.headline)
            Text(detail).font(.footnote).foregroundStyle(.secondary)
        }
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(12)
    }
}

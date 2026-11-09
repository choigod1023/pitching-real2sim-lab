import AVFoundation
import CryptoKit
import CoreVideo
import Foundation
import ImageIO
import Vision

enum ReplayError: LocalizedError {
    case camera, video, duration
    var errorDescription: String? {
        switch self {
        case .camera: return "후면 카메라를 열지 못했습니다."
        case .video: return "영상의 시간이나 화면 크기를 읽지 못했습니다."
        case .duration: return "영상은 0초보다 길고 90초 이하여야 합니다."
        }
    }
}

enum PoseReader {
    private static let keys: [(Joint, VNHumanBodyPoseObservation.JointName)] = [
        (.leftShoulder, .leftShoulder), (.rightShoulder, .rightShoulder),
        (.leftElbow, .leftElbow), (.rightElbow, .rightElbow),
        (.leftWrist, .leftWrist), (.rightWrist, .rightWrist),
        (.leftHip, .leftHip), (.rightHip, .rightHip),
        (.leftKnee, .leftKnee), (.rightKnee, .rightKnee),
        (.leftAnkle, .leftAnkle), (.rightAnkle, .rightAnkle),
    ]

    static func points(_ buffer: CVPixelBuffer, orientation: CGImagePropertyOrientation = .up) -> [Joint: Point2D] {
        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: orientation, options: [:])
        guard (try? handler.perform([request])) != nil, let result = request.results?.first,
              let points = try? result.recognizedPoints(.all) else { return [:] }
        return Dictionary(uniqueKeysWithValues: keys.compactMap { joint, key in
            guard let p = points[key], p.confidence >= 0.55 else { return nil }
            let x = Double(p.location.x)
            let y = 1 - Double(p.location.y)
            guard (0...1).contains(x), (0...1).contains(y) else { return nil }
            return (joint, Point2D(x: x, y: y))
        })
    }
}

final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureFileOutputRecordingDelegate {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "pitch.camera")
    private let videoQueue = DispatchQueue(label: "pitch.video", qos: .userInitiated)
    private let poseQueue = DispatchQueue(label: "pitch.pose", qos: .userInitiated)
    private let video = AVCaptureVideoDataOutput()
    private let movie = AVCaptureMovieFileOutput()
    private var device: AVCaptureDevice?
    private var configured = false
    private var lastSample = -Double.infinity
    private var readingPose = false
    private var tracker: LiveBallTracker?
    private var trackingRevision = 0
    private var publishedBall = BallReadout.waiting
    private var lastBallPublish = -Double.infinity
    private var markedMode = false
    private var rateStart: Double?
    private var rateCount = 0
    private var finished: ((Result<URL, Error>) -> Void)?
    private var stopRequested = false
    var onPose: (([Joint: Point2D], Double) -> Void)?
    var onBall: ((BallReadout, Int) -> Void)?
    var onRate: ((Double) -> Void)?

    func open(highSpeed: Bool, _ completion: @escaping (Result<Bool, Error>) -> Void) {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] allowed in
            guard let self else { return }
            guard allowed else {
                DispatchQueue.main.async { completion(.failure(ReplayError.camera)) }
                return
            }
            self.queue.async {
                let result = Result {
                    try self.configure()
                    let markedMode = self.configureFormat(highSpeed: highSpeed)
                    self.videoQueue.sync {
                        self.markedMode = markedMode
                        self.rateStart = nil
                        self.rateCount = 0
                        self.lastSample = -.infinity
                    }
                    if !self.session.isRunning { self.session.startRunning() }
                    guard self.session.isRunning else { throw ReplayError.camera }
                    return markedMode
                }
                DispatchQueue.main.async { completion(result) }
            }
        }
    }

    private func configure() throws {
        guard !configured else { return }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        else { throw ReplayError.camera }
        let input = try AVCaptureDeviceInput(device: device)
        video.alwaysDiscardsLateVideoFrames = true
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        guard session.canAddInput(input) else { throw ReplayError.camera }
        session.addInput(input)
        session.sessionPreset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high
        guard session.canAddOutput(video) else { session.removeInput(input); throw ReplayError.camera }
        session.addOutput(video)
        guard session.canAddOutput(movie) else {
            session.removeOutput(video); session.removeInput(input)
            throw ReplayError.camera
        }
        session.addOutput(movie)
        for output in [video as AVCaptureOutput, movie] {
            if let link = output.connection(with: .video), link.isVideoOrientationSupported {
                link.videoOrientation = .landscapeRight
            }
            if let link = output.connection(with: .video), link.isVideoStabilizationSupported {
                link.preferredVideoStabilizationMode = .off
            }
        }
        movie.maxRecordedDuration = CMTime(seconds: 90, preferredTimescale: 1000)
        video.setSampleBufferDelegate(self, queue: videoQueue)
        self.device = device
        configured = true
    }

    private func configureFormat(highSpeed: Bool) -> Bool {
        guard let device else { return false }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        let normalPreset: AVCaptureSession.Preset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high
        session.sessionPreset = normalPreset
        var enabled = false
        if highSpeed {
            let formats = device.formats.compactMap { format -> (AVCaptureDevice.Format, CMTime, Int32)? in
                let size = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                guard size.width >= 1280, size.height >= 720,
                      format.unsupportedCaptureOutputClasses.allSatisfy({
                          ObjectIdentifier($0) != ObjectIdentifier(AVCaptureMovieFileOutput.self)
                              && ObjectIdentifier($0) != ObjectIdentifier(AVCaptureVideoDataOutput.self)
                      }),
                      let range = format.videoSupportedFrameRateRanges.first(where: { (239...241).contains($0.maxFrameRate) })
                else { return nil }
                return (format, range.minFrameDuration, size.width * size.height)
            }
            if let (format, duration, _) = formats.min(by: { $0.2 < $1.2 }),
               (try? device.lockForConfiguration()) != nil {
                device.activeFormat = format
                if #available(iOS 18.0, *), device.isAutoVideoFrameRateEnabled {
                    device.isAutoVideoFrameRateEnabled = false
                }
                device.activeVideoMinFrameDuration = duration
                device.activeVideoMaxFrameDuration = duration
                device.unlockForConfiguration()
                enabled = true
            }
        }
        let supported = video.availableVideoPixelFormatTypes
        let nv12 = [kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
        let native = CMFormatDescriptionGetMediaSubType(device.activeFormat.formatDescription)
        let yuv = nv12.contains(native) && supported.contains(native) ? native : nv12.first { supported.contains($0) }
        if let yuv { video.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: yuv] }
        else { video.videoSettings = nil }
        return enabled && yuv != nil
    }

    func setBallCorridor(_ corridor: BallCorridor?, revision: Int) {
        videoQueue.async {
            self.trackingRevision = revision
            self.tracker = corridor.map { LiveBallTracker(corridor: $0) }
            self.publishedBall = .waiting
            self.lastBallPublish = -.infinity
        }
    }

    func close() {
        queue.async {
            if self.movie.isRecording { self.movie.stopRecording() }
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    func record(_ url: URL, finished: @escaping (Result<URL, Error>) -> Void) {
        queue.async {
            guard self.session.isRunning, self.finished == nil else {
                DispatchQueue.main.async { finished(.failure(ReplayError.camera)) }
                return
            }
            self.finished = finished
            self.stopRequested = false
            self.movie.startRecording(to: url, recordingDelegate: self)
        }
    }

    func stop() {
        queue.async {
            guard self.finished != nil else { return }
            self.stopRequested = true
            if self.movie.isRecording { self.movie.stopRecording() }
        }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo url: URL, from connections: [AVCaptureConnection]) {
        queue.async { if self.stopRequested && self.movie.isRecording { self.movie.stopRecording() } }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        guard time.isFinite, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            tracker?.breakContinuity()
            publishBall()
            return
        }
        measureRate(time)
        if let tracker {
            processBall(buffer, time: time, tracker: tracker)
            publishBall(time: time)
        }
        guard !readingPose, time < lastSample || time - lastSample >= 0.048 else { return }
        lastSample = time
        readingPose = true
        poseQueue.async {
            let points = PoseReader.points(buffer)
            let aspect = Double(CVPixelBufferGetWidth(buffer)) / Double(CVPixelBufferGetHeight(buffer))
            DispatchQueue.main.async { [weak self] in self?.onPose?(points, aspect) }
            self.videoQueue.async { self.readingPose = false }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        tracker?.breakContinuity()
        publishBall()
    }

    private func measureRate(_ time: Double) {
        guard let start = rateStart, time > start else {
            rateStart = time
            rateCount = 0
            return
        }
        rateCount += 1
        if time - start >= 0.5 {
            let rate = Double(rateCount) / (time - start)
            rateStart = time
            rateCount = 0
            DispatchQueue.main.async { [weak self] in self?.onRate?(rate) }
        }
    }

    private func processBall(_ buffer: CVPixelBuffer, time: Double, tracker: LiveBallTracker) {
        let format = CVPixelBufferGetPixelFormatType(buffer)
        guard format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
              CVPixelBufferGetPlaneCount(buffer) == 2,
              CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else {
            tracker.breakContinuity()
            return
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let luma = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else {
            tracker.breakContinuity()
            return
        }
        var marker: LiveBallTracker.MarkerReader?
        if markedMode, let chroma = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) {
            let width = CVPixelBufferGetWidthOfPlane(buffer, 1)
            let height = CVPixelBufferGetHeightOfPlane(buffer, 1)
            let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
            marker = { x, y, radius in
                BallMarker.blueDot(chroma: chroma.assumingMemoryBound(to: UInt8.self),
                                   width: width, height: height, stride: stride, x: x, y: y, radius: radius)
            }
        }
        tracker.process(luma: luma.assumingMemoryBound(to: UInt8.self),
                        width: CVPixelBufferGetWidthOfPlane(buffer, 0),
                        height: CVPixelBufferGetHeightOfPlane(buffer, 0),
                        stride: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0), time: time, marker: marker)
    }

    private func publishBall(time: Double? = nil) {
        guard let readout = tracker?.readout, readout != publishedBall else { return }
        if let time {
            if readout.active == publishedBall.active, (readout.rpm == nil) == (publishedBall.rpm == nil),
               time >= lastBallPublish, time - lastBallPublish < 1.0 / 30 { return }
            lastBallPublish = time
        }
        publishedBall = readout
        let revision = trackingRevision
        DispatchQueue.main.async { [weak self] in self?.onBall?(readout, revision) }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo url: URL,
                    from connections: [AVCaptureConnection], error: Error?) {
        let valid = (error as NSError?)?.userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool
        queue.async {
            let callback = self.finished
            self.finished = nil
            let result: Result<URL, Error>
            if let error, valid != true { result = .failure(error) }
            else { result = .success(url) }
            DispatchQueue.main.async { callback?(result) }
        }
    }
}

enum Archive {
    static func videoDuration(_ asset: AVURLAsset) async throws -> Double {
        let seconds = try await asset.load(.duration).seconds
        guard seconds.isFinite, seconds > 0, seconds <= 90.25 else { throw ReplayError.duration }
        return seconds
    }

    static func folder() throws -> URL {
        let root = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                               appropriateFor: nil, create: true)
            .appendingPathComponent("PitchReplay", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    static func read() throws -> (sessions: [RecordedSession], unreadable: Int) {
        let root = try folder()
        let urls = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
        let sessions = urls.compactMap { url -> RecordedSession? in
            guard let data = try? Data(contentsOf: url),
                  let session = try? JSONDecoder().decode(RecordedSession.self, from: data),
                  FileManager.default.fileExists(atPath: root.appendingPathComponent(session.videoName).path)
            else { return nil }
            return session
        }
        return (sessions.sorted { $0.created > $1.created }, urls.count - sessions.count)
    }

    static func write(_ session: RecordedSession) throws {
        let url = try folder().appendingPathComponent(session.id.uuidString + ".json")
        try JSONEncoder().encode(session).write(to: url, options: .atomic)
    }

    static func analyze(_ url: URL, hand: ThrowHand) async throws -> RecordedSession {
        let asset = AVURLAsset(url: url)
        let duration = try await videoDuration(asset)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw ReplayError.video }
        let size = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let oriented = CGRect(origin: .zero, size: size).applying(transform).size
        guard oriented.width > 0, oriented.height > 0 else { throw ReplayError.video }
        let aspect = Double(oriented.width / oriented.height)
        let orientation: CGImagePropertyOrientation
        switch (Int(transform.a.rounded()), Int(transform.b.rounded()), Int(transform.c.rounded()), Int(transform.d.rounded())) {
        case (0, 1, -1, 0): orientation = .right
        case (0, -1, 1, 0): orientation = .left
        case (-1, 0, 0, -1): orientation = .down
        case (-1, 0, 0, 1): orientation = .upMirrored
        case (1, 0, 0, -1): orientation = .downMirrored
        case (0, 1, 1, 0): orientation = .leftMirrored
        case (0, -1, -1, 0): orientation = .rightMirrored
        default: orientation = .up
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        guard reader.canAdd(output) else { throw ReplayError.video }
        reader.add(output)
        var samples: [PoseSample] = []
        var frameTimes: [Double] = []
        var lastTime = -Double.infinity
        if reader.startReading() {
            while let buffer = output.copyNextSampleBuffer() {
                autoreleasepool {
                    let time = CMSampleBufferGetPresentationTimeStamp(buffer).seconds
                    guard time.isFinite, time >= 0, time <= duration else { return }
                    frameTimes.append(time)
                    guard time - lastTime >= 0.048, let image = CMSampleBufferGetImageBuffer(buffer) else { return }
                    lastTime = time
                    samples.append(PoseSample(time: time, points: PoseReader.points(image, orientation: orientation)))
                }
            }
        }
        if reader.status != .completed { samples = []; frameTimes = [] }
        frameTimes = Array(Set(frameTimes)).sorted()
        samples.sort { $0.time < $1.time }
        let pitches = PitchDetector.detect(samples, hand: hand, aspect: aspect, duration: duration)
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty { hash.update(data: data) }
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        return RecordedSession(id: UUID(), videoName: url.lastPathComponent, fingerprint: digest,
                               created: Date(), hand: hand, aspect: aspect, duration: duration,
                               frameTimes: frameTimes, poses: samples, pitches: pitches)
    }
}

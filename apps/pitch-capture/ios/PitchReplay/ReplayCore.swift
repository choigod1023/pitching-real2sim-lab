import Foundation

enum ThrowHand: String, Codable, CaseIterable {
    case right, left
    var title: String { self == .right ? "오른손" : "왼손" }
}

enum Joint: String, Codable {
    case leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist
    case leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle

    static let bones: [(Joint, Joint)] = [
        (.leftShoulder, .rightShoulder), (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
        (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
        (.leftShoulder, .leftHip), (.rightShoulder, .rightHip), (.leftHip, .rightHip),
        (.leftHip, .leftKnee), (.leftKnee, .leftAnkle), (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
    ]
}

struct Point2D: Codable, Equatable {
    let x: Double
    let y: Double
    func distance(to point: Point2D, aspect: Double) -> Double {
        hypot((point.x - x) * aspect, point.y - y)
    }
}

struct PoseSample: Codable {
    let time: Double
    let points: [Joint: Point2D]
    var canDraw: Bool {
        Joint.bones.contains { points[$0.0] != nil && points[$0.1] != nil }
    }
}

enum PitchEvent: Equatable { case landing, release }

struct Pitch: Codable, Identifiable {
    let id: UUID
    let start: Double
    let end: Double
    let automatic: Bool
    var landing: Double?
    var release: Double?

    init(start: Double, end: Double, automatic: Bool) {
        id = UUID()
        self.start = start
        self.end = end
        self.automatic = automatic
    }

    mutating func mark(_ event: PitchEvent, at time: Double) -> Bool {
        guard time.isFinite, (start...end).contains(time) else { return false }
        let newLanding = event == .landing ? time : landing
        let newRelease = event == .release ? time : release
        if let a = newLanding, let b = newRelease, a >= b { return false }
        landing = newLanding
        release = newRelease
        return true
    }

    var isConfirmed: Bool {
        guard let a = landing, let b = release else { return false }
        return start <= a && a < b && b <= end
    }

    func time(at phase: Double) -> Double {
        guard let a = landing, let b = release else { return start }
        return a + min(1, max(0, phase)) * (b - a)
    }
}

struct RecordedSession: Codable, Identifiable {
    let id: UUID
    let videoName: String
    let fingerprint: String
    let created: Date
    let hand: ThrowHand
    let aspect: Double
    let duration: Double
    let frameTimes: [Double]
    let poses: [PoseSample]
    var pitches: [Pitch]

    func nearestPose(at time: Double, limit: Double = 0.115) -> PoseSample? {
        guard time.isFinite, !poses.isEmpty else { return nil }
        let next = Self.lowerBound(poses, at: time, value: { $0.time })
        let nearby = [next - 1, next].filter { poses.indices.contains($0) }
        guard let index = nearby.min(by: { abs(poses[$0].time - time) < abs(poses[$1].time - time) }),
              abs(poses[index].time - time) <= limit else { return nil }
        return poses[index]
    }

    func frameTime(from time: Double, moving offset: Int, in range: ClosedRange<Double>) -> Double {
        guard !frameTimes.isEmpty else {
            return min(range.upperBound, max(range.lowerBound, time + Double(offset) / 30))
        }
        let next = Self.lowerBound(frameTimes, at: time, value: { $0 })
        let near = min(next, frameTimes.count - 1)
        let closest = near > 0 && abs(frameTimes[near - 1] - time) < abs(frameTimes[near] - time)
            ? near - 1 : near
        let index = min(frameTimes.count - 1, max(0, closest + offset))
        return min(range.upperBound, max(range.lowerBound, frameTimes[index]))
    }

    private static func lowerBound<T>(_ values: [T], at time: Double, value: (T) -> Double) -> Int {
        var low = 0
        var high = values.count
        while low < high {
            let mid = (low + high) / 2
            if value(values[mid]) < time { low = mid + 1 } else { high = mid }
        }
        return low
    }
}

struct PitchSelection: Identifiable {
    let session: RecordedSession
    let pitch: Pitch
    var id: UUID { pitch.id }

    static func canCompare(_ a: Self, _ b: Self, sameView: Bool) -> Bool {
        let sameSource = a.session.id == b.session.id || a.session.fingerprint == b.session.fingerprint
        let overlap = max(a.pitch.start, b.pitch.start) < min(a.pitch.end, b.pitch.end)
        guard sameView && a.id != b.id && !(sameSource && overlap) &&
            a.pitch.isConfirmed && b.pitch.isConfirmed &&
            a.session.hand == b.session.hand && a.session.aspect > 0 && b.session.aspect > 0 &&
            abs(a.session.aspect / b.session.aspect - 1) <= 0.025 else { return false }
        return [a, b].allSatisfy { take in
            let observations: [Double?] = (0...10).map { fraction in
                take.session.nearestPose(at: take.pitch.time(at: Double(fraction) / 10))
                    .flatMap { $0.canDraw ? $0.time : nil }
            }
            let times = observations.compactMap { $0 }
            return observations[0] != nil && observations[10] != nil && times.count >= 7 && Set(times).count >= 2
        }
    }
}

enum PitchDetector {
    static func detect(_ samples: [PoseSample], hand: ThrowHand, aspect: Double, duration: Double) -> [Pitch] {
        let indices: (Joint, Joint, Joint) = hand == .left
            ? (.leftShoulder, .leftWrist, .leftHip) : (.rightShoulder, .rightWrist, .rightHip)
        var previous: PoseSample?
        var burst: (time: Double, speed: Double, count: Int, last: Double)?
        var peaks: [Double] = []
        func flush() {
            if let value = burst, value.count >= 2 { peaks.append(value.time) }
            burst = nil
        }
        for sample in samples {
            let prior = previous
            previous = sample
            if let value = burst, sample.time - value.last > 0.25 { flush() }
            guard let prior, let wrist = sample.points[indices.1], let old = prior.points[indices.1],
                  let shoulder = sample.points[indices.0], let hip = sample.points[indices.2] else { continue }
            let dt = sample.time - prior.time
            let torso = shoulder.distance(to: hip, aspect: aspect)
            guard (0.012...0.20).contains(dt), torso > 0.035,
                  wrist.y <= shoulder.y + 0.75 * torso,
                  sample.time - (peaks.last ?? -.infinity) >= 1.7 else { continue }
            let speed = wrist.distance(to: old, aspect: aspect) / torso / dt
            guard (3.4...42).contains(speed) else { continue }
            var value = burst ?? (sample.time, speed, 0, sample.time)
            value.count += 1
            value.last = sample.time
            if speed > value.speed { value.speed = speed; value.time = sample.time }
            burst = value
        }
        flush()
        return peaks.indices.compactMap { index in
            let peak = peaks[index]
            let lower = index == 0 ? 0 : (peaks[index - 1] + peak) / 2
            let upper = index == peaks.count - 1 ? duration : (peak + peaks[index + 1]) / 2
            let start = max(0, peak - 1.12, lower)
            let end = min(duration, peak + 0.78, upper)
            return end - start >= 0.25 ? Pitch(start: start, end: end, automatic: true) : nil
        }
    }
}

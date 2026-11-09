import Foundation

@main
struct ReplayCoreChecks {
    static func main() {
        let sample = { (time: Double, x: Double) in
            PoseSample(time: time, points: [
                .rightShoulder: Point2D(x: 0.4, y: 0.3),
                .rightHip: Point2D(x: 0.4, y: 0.5),
                .rightWrist: Point2D(x: x, y: 0.32),
            ])
        }
        let data = [sample(0.7, 0.3), sample(0.75, 0.35), sample(0.8, 0.43),
                    PoseSample(time: 0.85, points: [:]), sample(1.2, 0.4)]
        let result = PitchDetector.detect(data, hand: .right, aspect: 16 / 9, duration: 2)
        precondition(result.count == 1 && result[0].automatic)
        precondition(PitchDetector.detect(data, hand: .left, aspect: 16 / 9, duration: 2).isEmpty)
        precondition(PitchDetector.detect([data[0], data[1]], hand: .right, aspect: 16 / 9, duration: 2).isEmpty)
        var pitch = result[0]
        precondition(!pitch.mark(.release, at: 9))
        precondition(pitch.mark(.landing, at: 0.5))
        precondition(!pitch.mark(.release, at: 0.4))
        precondition(pitch.mark(.release, at: 1))
        precondition(pitch.isConfirmed && pitch.time(at: 0.5) == 0.75)
        let session = RecordedSession(id: UUID(), videoName: "test.mov", fingerprint: "test", created: Date(),
                                      hand: .right, aspect: 16 / 9, duration: 2,
                                      frameTimes: [0.7, 0.74, 0.8, 1.2], poses: data, pitches: [pitch])
        precondition(session.nearestPose(at: 0.85)?.points.isEmpty == true)
        precondition(session.nearestPose(at: 1) == nil)
        precondition(session.frameTime(from: 0.73, moving: 1, in: 0...2) == 0.8)
        let tracked = (0...10).map { sample(0.5 + Double($0) * 0.05, 0.4) }
        let trackedSession = RecordedSession(id: session.id, videoName: session.videoName, fingerprint: session.fingerprint,
                                             created: session.created, hand: .right, aspect: 16 / 9, duration: 2,
                                             frameTimes: [], poses: tracked, pitches: [pitch])
        let first = PitchSelection(session: trackedSession, pitch: pitch)
        var duplicate = Pitch(start: 0.4, end: 1.1, automatic: false)
        precondition(duplicate.mark(.landing, at: 0.5) && duplicate.mark(.release, at: 1))
        precondition(!PitchSelection.canCompare(first, .init(session: session, pitch: duplicate), sameView: true))
        let independent = RecordedSession(id: UUID(), videoName: "other.mov", fingerprint: "other", created: Date(),
                                           hand: .right, aspect: 16 / 9, duration: 2,
                                           frameTimes: [], poses: tracked, pitches: [duplicate])
        let other = PitchSelection(session: independent, pitch: duplicate)
        precondition(!PitchSelection.canCompare(.init(session: session, pitch: pitch), other, sameView: true))
        precondition(!PitchSelection.canCompare(first, other, sameView: false))
        precondition(PitchSelection.canCompare(first, other, sameView: true))
        print("ReplayCore: detection, missing observation, event order, stepping and A/B checks passed")
    }
}

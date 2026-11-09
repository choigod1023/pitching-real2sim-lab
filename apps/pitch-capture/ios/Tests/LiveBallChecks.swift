import Foundation

@main
struct LiveBallChecks {
    static func main() {
        let width = 1280, height = 720, stride = width + 16
        let start = Point2D(x: 0.10, y: 0.5), end = Point2D(x: 0.90, y: 0.5)
        let calibrated = BallCorridor(start: start, end: end, aspect: Double(width) / Double(height), meters: 5.12)!
        let uncalibrated = BallCorridor(start: start, end: end, aspect: calibrated.aspect, meters: nil)!
        let tracker = LiveBallTracker(corridor: calibrated)
        let pixelsOnly = LiveBallTracker(corridor: uncalibrated)
        let reverse = LiveBallTracker(corridor: calibrated)
        let leftward = LiveBallTracker(corridor: BallCorridor(start: end, end: start, aspect: calibrated.aspect, meters: nil)!)
        let slow = LiveBallTracker(corridor: calibrated)

        func send(_ tracker: LiveBallTracker, x: Int?, y: Int = 360, time: Double, phase: Double? = nil) {
            var luma = [UInt8](repeating: 54, count: stride * height)
            var chroma = [UInt8](repeating: 128, count: stride * (height / 2))
            if let x {
                for row in (y - 14)...(y + 14) {
                    for column in (x - 14)...(x + 14) {
                        if (column - x) * (column - x) + (row - y) * (row - y) <= 14 * 14 {
                            luma[row * stride + column] = 218
                        }
                    }
                }
                if let phase {
                    let dotX = Double(x) + 0.5 + 8 * cos(phase)
                    let dotY = Double(y) + 0.5 + 8 * sin(phase)
                    for row in (y / 2 - 7)...(y / 2 + 7) {
                        for column in (x / 2 - 7)...(x / 2 + 7) {
                            if hypot(Double(2 * column + 1) - dotX, Double(2 * row + 1) - dotY) <= 3.5 {
                                chroma[row * stride + 2 * column] = 198
                                chroma[row * stride + 2 * column + 1] = 95
                            }
                        }
                    }
                }
            }
            luma.withUnsafeBufferPointer { luma in
                chroma.withUnsafeBufferPointer { chroma in
                    tracker.process(luma: luma.baseAddress!, width: width, height: height, stride: stride, time: time) {
                        BallMarker.blueDot(chroma: chroma.baseAddress!, width: width / 2, height: height / 2,
                                           stride: stride, x: $0, y: $1, radius: $2)
                    }
                }
            }
        }

        for input in [tracker, pixelsOnly, reverse, leftward, slow] { send(input, x: nil, time: 0) }
        for index in 1...13 {
            let phase = 2 * Double.pi * Double(index) / 8  // 30 rotations per second at 240 fps.
            let time = Double(index) / 240
            let x = 180 + 31 * index
            let y = 360 + min(index, 14 - index)
            send(tracker, x: x, y: y, time: time, phase: phase)
            send(pixelsOnly, x: x, y: y, time: time)
            send(reverse, x: 1100 - 31 * index, y: y, time: time, phase: phase)
            send(leftward, x: 1100 - 31 * index, y: y, time: time)
            send(slow, x: x, y: y, time: Double(index) / 60, phase: phase)
            if index <= 2 { precondition(tracker.readout.path.isEmpty) }
            if index == 4 { precondition(tracker.readout.movementCM == nil) }
        }
        let complete = tracker.readout
        precondition(complete.active && complete.path.count == 13)
        precondition(abs((complete.speedKPH ?? 0) - 133.92) < 1)
        precondition((2...4).contains(complete.movementCM ?? 0))
        precondition(abs((complete.rpm ?? 0) - 1800) < 80)
        precondition(pixelsOnly.readout.path.count == 13)
        precondition(pixelsOnly.readout.speedKPH == nil && pixelsOnly.readout.movementCM == nil && pixelsOnly.readout.rpm == nil)
        precondition((5...7).contains(pixelsOnly.readout.movementPX ?? 0))
        precondition(leftward.readout.path.count == 13 && (5...7).contains(leftward.readout.movementPX ?? 0))
        precondition(reverse.readout.path.isEmpty)
        precondition(slow.readout.path.count == 13 && slow.readout.rpm == nil)

        send(tracker, x: nil, time: 14.0 / 240)
        precondition(tracker.readout == complete.ended())
        for index in 15...17 { send(tracker, x: 180 + 31 * index, time: Double(index) / 240) }
        precondition(tracker.readout.active && tracker.readout.path.count == 3 && tracker.readout.rpm == nil)
        send(tracker, x: 180 + 31 * 20, time: 20.0 / 240)
        precondition(!tracker.readout.active && tracker.readout.path.count == 3)

        print("LiveBall: timed luma path, projected speed and movement, blue-dot RPM, calibration, reverse and gap checks passed")
    }
}

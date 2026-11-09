import Foundation

struct BallCorridor {
    static let halfWidth = 0.08 // Fraction of the image height.

    let start: Point2D
    let end: Point2D
    let aspect: Double
    let meters: Double?

    init?(start: Point2D, end: Point2D, aspect: Double, meters: Double?) {
        guard aspect.isFinite, aspect > 0,
              [start, end].allSatisfy({ (0...1).contains($0.x) && (0...1).contains($0.y) }),
              abs(end.x - start.x) * aspect >= abs(end.y - start.y),
              start.distance(to: end, aspect: aspect) >= 0.18 else { return nil }
        self.start = start
        self.end = end
        self.aspect = aspect
        self.meters = meters.flatMap { $0.isFinite && $0 > 0 ? $0 : nil }
    }

    var length: Double { start.distance(to: end, aspect: aspect) }

    var direction: Point2D {
        Point2D(x: (end.x - start.x) * aspect / length, y: (end.y - start.y) / length)
    }
}

struct BallMarker {
    let phase: Double
    let radiusFraction: Double

    static func blueDot(chroma: UnsafePointer<UInt8>, width: Int, height: Int, stride: Int,
                        x: Double, y: Double, radius: Double) -> BallMarker? {
        let x0 = max(0, Int((x - radius) / 2))
        let x1 = min(width, Int((x + radius) / 2) + 1)
        let y0 = max(0, Int((y - radius) / 2))
        let y1 = min(height, Int((y + radius) / 2) + 1)
        guard x0 < x1, y0 < y1, stride >= width * 2 else { return nil }
        var count = 0.0
        var sumX = 0.0, sumY = 0.0, squared = 0.0
        for row in y0..<y1 {
            for column in x0..<x1 {
                let dx = (Double(2 * column + 1) - x) / radius
                let dy = (Double(2 * row + 1) - y) / radius
                let distance = dx * dx + dy * dy
                guard distance <= 0.90 * 0.90 else { continue }
                let index = row * stride + column * 2
                let blue = Int(chroma[index]), red = Int(chroma[index + 1])
                guard blue >= 152, blue - red >= 25 else { continue }
                count += 1
                sumX += dx
                sumY += dy
                squared += distance
            }
        }
        guard count >= 3, 4 * count <= 0.20 * Double.pi * radius * radius else { return nil }
        let cx = sumX / count, cy = sumY / count
        let radial = hypot(cx, cy)
        guard (0.30...0.80).contains(radial), squared / count - radial * radial <= 0.05 else { return nil }
        return BallMarker(phase: atan2(cy, cx), radiusFraction: radial)
    }
}

struct BallObservation {
    let center: Point2D
    let time: Double
    let diameter: Double // Fraction of image height.
    let marker: BallMarker?
}

struct BallReadout: Equatable {
    let path: [Point2D]
    let speedKPH: Double?
    let movementPX: Double?
    let movementCM: Double?
    let rpm: Double?
    let active: Bool

    static let waiting = BallReadout(path: [], speedKPH: nil, movementPX: nil, movementCM: nil, rpm: nil, active: false)

    func ended() -> BallReadout {
        BallReadout(path: path, speedKPH: speedKPH, movementPX: movementPX, movementCM: movementCM,
                    rpm: rpm, active: false)
    }

    private init(path: [Point2D], speedKPH: Double?, movementPX: Double?, movementCM: Double?, rpm: Double?, active: Bool) {
        self.path = path
        self.speedKPH = speedKPH
        self.movementPX = movementPX
        self.movementCM = movementCM
        self.rpm = rpm
        self.active = active
    }

    init(observations: [BallObservation], corridor: BallCorridor, imageHeight: Int, priorRPM: Double?) {
        let centers = observations.map(\.center)
        path = centers
        active = true
        let departure = Self.chordDeparture(centers, aspect: corridor.aspect)
        movementPX = departure.map { $0 * Double(imageHeight) }
        if let meters = corridor.meters {
            let initial = observations.prefix(3)
            let dt = initial[2].time - initial[0].time
            let travel = centers[0].distance(to: centers[1], aspect: corridor.aspect)
                + centers[1].distance(to: centers[2], aspect: corridor.aspect)
            speedKPH = 3.6 * meters * travel / (corridor.length * dt)
            movementCM = departure.map { 100 * meters * $0 / corridor.length }
        } else {
            speedKPH = nil
            movementCM = nil
        }
        rpm = Self.markedSpin(observations) ?? priorRPM
    }

    private static func chordDeparture(_ path: [Point2D], aspect: Double) -> Double? {
        guard path.count >= 5, let start = path.first, let end = path.last else { return nil }
        let dx = (end.x - start.x) * aspect
        let dy = end.y - start.y
        let length = hypot(dx, dy)
        guard length > 0 else { return nil }
        // Orient the perpendicular toward image-down for either pitching direction.
        let sign = dx < 0 ? -1.0 : 1.0
        return path.dropFirst().dropLast().map { point in
            sign * (dx * (point.y - start.y) - dy * (point.x - start.x) * aspect) / length
        }.max(by: { abs($0) < abs($1) })
    }

    private static func markedSpin(_ observations: [BallObservation]) -> Double? {
        // A missing dot ends the optical phase segment. Never bridge an occlusion.
        let start = observations.lastIndex { $0.marker == nil }.map { $0 + 1 } ?? 0
        let samples = observations[start...]
        guard samples.count >= 8, let first = samples.first, let last = samples.last,
              let firstMarker = first.marker else { return nil }
        let times = samples.map { $0.time - first.time }
        var phase = [0.0]
        var previous = firstMarker.phase
        var direction = 0.0
        for (before, sample) in zip(samples, samples.dropFirst()) {
            guard let marker = sample.marker, abs(marker.radiusFraction - firstMarker.radiusFraction) <= 0.15,
                  (0.003...0.005).contains(sample.time - before.time) else { return nil }
            let delta = atan2(sin(marker.phase - previous), cos(marker.phase - previous))
            if direction == 0 { direction = delta.sign == .minus ? -1 : 1 }
            guard (0.05...(Double.pi * 0.60)).contains(delta * direction) else { return nil }
            phase.append(phase[phase.count - 1] + delta)
            previous = marker.phase
        }
        guard abs(phase[phase.count - 1]) >= 2 * Double.pi else { return nil }
        let rate = phase[phase.count - 1] / (last.time - first.time)
        let maxError = zip(times, phase).map { time, angle in abs(angle - rate * time) }.max() ?? .infinity
        let estimate = abs(rate) * 60 / (2 * Double.pi)
        return maxError <= 0.24 && estimate <= 3600 ? estimate : nil
    }
}

final class LiveBallTracker {
    typealias MarkerReader = (_ x: Double, _ y: Double, _ radius: Double) -> BallMarker?

    private struct Region {
        let x: Int
        let y: Int
        let width: Int
        let rows: [Range<Int>]
    }

    private struct Track {
        var observations: [BallObservation]
        var error = 0.0
    }

    private let corridor: BallCorridor
    private var region: Region?
    private var imageWidth = 0
    private var imageHeight = 0
    private var previous: [UInt8] = []
    private var changed: [UInt8] = []
    private var component: [Int] = []
    private var lastTime: Double?
    private var interval: Double?
    private var tracks: [Track] = []
    private var confirmed: Track?
    private(set) var readout = BallReadout.waiting

    init(corridor: BallCorridor) { self.corridor = corridor }

    func breakContinuity() {
        lastTime = nil
        interval = nil
        clearTracks()
    }

    private func clearTracks() {
        tracks.removeAll(keepingCapacity: true)
        confirmed = nil
        readout = readout.ended()
    }

    func process(luma: UnsafePointer<UInt8>, width: Int, height: Int, stride: Int,
                 time: Double, marker: MarkerReader? = nil) {
        guard time.isFinite, width > 0, height > 0, stride >= width,
              abs(Double(width) / Double(height) - corridor.aspect) <= 0.01 else {
            breakContinuity()
            return
        }
        if width != imageWidth || height != imageHeight {
            imageWidth = width
            imageHeight = height
            region = makeRegion(width: width, height: height)
            let size = region.map { $0.width * $0.rows.count } ?? 0
            previous = [UInt8](repeating: 0, count: size)
            changed = previous
            breakContinuity()
        }
        guard let region else { return }
        var hasBaseline = false
        if let lastTime {
            let delta = time - lastTime
            hasBaseline = delta > 0 && delta <= 0.12 && (interval.map { delta <= $0 * 1.65 } ?? true)
            if hasBaseline { interval = interval.map { 0.8 * $0 + 0.2 * delta } ?? delta }
            else { breakContinuity() }
        }
        lastTime = time
        var changedCount = 0
        var total = 0
        for (row, range) in region.rows.enumerated() {
            let source = luma.advanced(by: (region.y + row) * stride + region.x)
            let base = row * region.width
            total += range.count
            for x in range {
                let value = source[x]
                let index = base + x
                let moving = hasBaseline && value >= 145 && Int(value) - Int(previous[index]) >= 28
                previous[index] = value
                changed[index] = moving ? 1 : 0
                if moving { changedCount += 1 }
            }
        }
        guard hasBaseline, changedCount <= max(64, total / 12) else {
            clearTracks()
            return
        }
        updateTracks(candidates(in: region, time: time, marker: marker))
    }

    private func makeRegion(width: Int, height: Int) -> Region? {
        let pad = BallCorridor.halfWidth * Double(height)
        let x0 = max(0, Int(floor(min(corridor.start.x, corridor.end.x) * Double(width) - pad)))
        let x1 = min(width, Int(ceil(max(corridor.start.x, corridor.end.x) * Double(width) + pad)))
        let y0 = max(0, Int(floor(min(corridor.start.y, corridor.end.y) * Double(height) - pad)))
        let y1 = min(height, Int(ceil(max(corridor.start.y, corridor.end.y) * Double(height) + pad)))
        guard x1 > x0, y1 > y0 else { return nil }
        let axis = corridor.direction
        let length = corridor.length
        let rows = (y0..<y1).map { y -> Range<Int> in
            let dy = (Double(y) + 0.5) / Double(height) - corridor.start.y
            var left = x1
            var right = x1
            for x in x0..<x1 {
                let dx = ((Double(x) + 0.5) / Double(width) - corridor.start.x) * corridor.aspect
                let along = dx * axis.x + dy * axis.y
                if (0...length).contains(along) && abs(dx * axis.y - dy * axis.x) <= BallCorridor.halfWidth {
                    left = min(left, x)
                    right = x + 1
                }
            }
            return (left - x0)..<(right - x0)
        }
        return Region(x: x0, y: y0, width: x1 - x0, rows: rows)
    }

    private func candidates(in region: Region, time: Double, marker: MarkerReader?) -> [BallObservation] {
        var candidates: [BallObservation] = []
        let maxSize = max(6, Double(imageHeight) * 0.085)
        let rowWidth = region.width
        for (row, range) in region.rows.enumerated() {
            for column in range {
                let index = row * rowWidth + column
                guard changed[index] == 1 else { continue }
                component.removeAll(keepingCapacity: true)
                component.append(index)
                changed[index] = 0
                var i = 0
                var x0 = column, x1 = column, y0 = row, y1 = row
                var sumX = 0, sumY = 0
                while i < component.count {
                    let p = component[i]
                    i += 1
                    let x = p % rowWidth, y = p / rowWidth
                    x0 = min(x0, x); x1 = max(x1, x)
                    y0 = min(y0, y); y1 = max(y1, y)
                    sumX += x; sumY += y
                    if x > 0 { visit(p - 1) }
                    if x + 1 < rowWidth { visit(p + 1) }
                    if y > 0 { visit(p - rowWidth) }
                    if y + 1 < region.rows.count { visit(p + rowWidth) }
                }
                let w = Double(x1 - x0 + 1), h = Double(y1 - y0 + 1)
                let count = Double(component.count)
                guard count >= 4, max(w, h) <= maxSize, max(w, h) <= 3.5 * min(w, h),
                      count / (w * h) >= 0.30 else { continue }
                let x = Double(region.x) + Double(sumX) / count + 0.5
                let y = Double(region.y) + Double(sumY) / count + 0.5
                let dot = min(w, h) >= 20 && max(w, h) / min(w, h) <= 1.3
                    ? marker?(x, y, (w + h) / 4) : nil
                candidates.append(BallObservation(
                    center: Point2D(x: x / Double(imageWidth), y: y / Double(imageHeight)),
                    time: time, diameter: sqrt(w * h) / Double(imageHeight), marker: dot))
                // A crowded corridor does not identify a ball reliably.
                if candidates.count > 8 { return [] }
            }
        }
        return candidates
    }

    private func visit(_ index: Int) {
        if changed[index] == 1 {
            changed[index] = 0
            component.append(index)
        }
    }

    private func updateTracks(_ candidates: [BallObservation]) {
        if let track = confirmed {
            if track.observations.count < 160, let next = extend(track, with: candidates) {
                publish(next)
                return
            }
            clearTracks()
        }
        let extensions = tracks.compactMap { extend($0, with: candidates) }
        let viable = extensions.filter { $0.observations.count >= 3 }.sorted { $0.error < $1.error }
        if let best = viable.first, viable.count == 1 || viable[1].error - best.error >= 0.16 {
            tracks.removeAll(keepingCapacity: true)
            publish(best)
            return
        }
        tracks = Array(extensions.filter { $0.observations.count == 2 }.sorted { $0.error < $1.error }.prefix(8))
        tracks += candidates.map { Track(observations: [$0]) }
    }

    private func publish(_ track: Track) {
        let priorRPM = confirmed != nil ? readout.rpm : nil
        confirmed = track
        readout = BallReadout(observations: track.observations, corridor: corridor,
                              imageHeight: imageHeight, priorRPM: priorRPM)
    }

    private func extend(_ track: Track, with candidates: [BallObservation]) -> Track? {
        guard let last = track.observations.last else { return nil }
        let direction = corridor.direction
        let matches = candidates.compactMap { candidate -> (BallObservation, Double)? in
            let dt = candidate.time - last.time
            guard dt > 0, (0.45...2.2).contains(candidate.diameter / last.diameter) else { return nil }
            let dx = (candidate.center.x - last.center.x) * corridor.aspect
            let dy = candidate.center.y - last.center.y
            let forward = dx * direction.x + dy * direction.y
            let cross = abs(dy * direction.x - dx * direction.y)
            let rate = forward / dt
            guard forward >= 1.5 / Double(imageHeight), (0.20...12).contains(rate),
                  cross <= forward * 0.6 + 1.5 / Double(imageHeight) else { return nil }
            var error = cross / max(forward, 0.004)
            if track.observations.count >= 2 {
                let before = track.observations[track.observations.count - 2]
                let priorStep = Point2D(x: (last.center.x - before.center.x) * corridor.aspect,
                                        y: last.center.y - before.center.y)
                let ratio = dt / (last.time - before.time)
                let predicted = hypot(priorStep.x, priorStep.y) * ratio
                let residual = hypot(dx - priorStep.x * ratio, dy - priorStep.y * ratio)
                let uncertainty = 2.5 / Double(imageHeight)
                guard residual <= 0.45 * predicted + uncertainty else { return nil }
                error = residual / max(predicted, uncertainty)
            }
            return (candidate, error)
        }.sorted { $0.1 < $1.1 }
        guard let best = matches.first, matches.count == 1 || matches[1].1 - best.1 >= 0.12 else { return nil }
        var result = track
        result.observations.append(best.0)
        result.error += best.1
        return result
    }
}

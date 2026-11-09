package lab.pitch.replay

import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot

internal data class BallPoint(val x: Float, val y: Float) {
    operator fun minus(other: BallPoint) = BallPoint(x - other.x, y - other.y)
    val length get() = hypot(x, y)
}

internal data class BallSetup(val start: BallPoint, val end: BallPoint, val meters: Double?)

internal class BallCorridor(val setup: BallSetup) {
    private val axis = setup.end - setup.start
    val length = axis.length
    private val unitX = axis.x / length
    private val unitY = axis.y / length
    val halfWidth = length * 0.12f

    fun progress(point: BallPoint): Float {
        val delta = point - setup.start
        return delta.x * unitX + delta.y * unitY
    }

    fun lateral(point: BallPoint): Float {
        val delta = point - setup.start
        return delta.y * unitX - delta.x * unitY
    }

    fun contains(point: BallPoint) = progress(point) in 0f..length && abs(lateral(point)) <= halfWidth
}

internal data class BallReadout(
    val points: List<BallPoint>,
    val speedKmh: Double?,
    val deviationPx: Double?,
    val deviationCm: Double?,
    val active: Boolean
)

/** A new bright component is an optical candidate, not a semantic ball identification. */
internal class BrightMotion(private val corridor: BallCorridor, width: Int, height: Int) {
    private val start = corridor.setup.start
    private val end = corridor.setup.end
    private val left = floor(minOf(start.x, end.x) - corridor.halfWidth).toInt().coerceIn(0, width)
    private val right = ceil(maxOf(start.x, end.x) + corridor.halfWidth).toInt().coerceIn(left, width)
    private val top = floor(minOf(start.y, end.y) - corridor.halfWidth).toInt().coerceIn(0, height)
    private val bottom = ceil(maxOf(start.y, end.y) + corridor.halfWidth).toInt().coerceIn(top, height)
    private val columns = right - left
    private val rows = bottom - top
    private val maxSide = (maxOf(width, height) * 0.055f).toInt().coerceIn(8, 64)
    private var previous = ByteArray(columns * rows)
    private var current = ByteArray(previous.size)
    private val changed = BooleanArray(previous.size)
    private val component = IntArray(previous.size)
    private var primed = false

    fun candidates(luma: ByteBuffer, rowStride: Int, pixelStride: Int): List<BallPoint> {
        if (columns < 3 || rows < 3) return emptyList()
        val buffer = luma.duplicate()
        val base = buffer.position()
        var changedCount = 0
        for (y in 0 until rows) {
            val source = base + (top + y) * rowStride + left * pixelStride
            val dest = y * columns
            if (pixelStride == 1) {
                buffer.position(source)
                buffer.get(current, dest, columns)
            } else {
                for (x in 0 until columns) current[dest + x] = buffer[source + x * pixelStride]
            }
            if (!primed) continue
            for (x in 0 until columns) {
                val index = dest + x
                val value = current[index].toInt() and 255
                val before = previous[index].toInt() and 255
                changed[index] = value >= 144 && value - before >= 24
                if (changed[index]) changedCount++
            }
        }
        val swap = previous
        previous = current
        current = swap
        if (!primed) {
            primed = true
            return emptyList()
        }
        if (changedCount < 4 || changedCount > changed.size * 0.08) return emptyList()

        val found = mutableListOf<BallPoint>()
        for (seed in changed.indices) {
            if (!changed[seed]) continue
            val center = componentCenter(seed) ?: continue
            if (corridor.contains(center)) found += center
            if (found.size > MAX_CANDIDATES) return emptyList()
        }
        return found
    }

    private fun componentCenter(seed: Int): BallPoint? {
        var head = 0
        var count = 1
        var minX = columns
        var maxX = -1
        var minY = rows
        var maxY = -1
        var sumX = 0
        var sumY = 0
        component[0] = seed
        changed[seed] = false
        while (head < count) {
            val index = component[head++]
            val x = index % columns
            val y = index / columns
            minX = minOf(minX, x)
            maxX = maxOf(maxX, x)
            minY = minOf(minY, y)
            maxY = maxOf(maxY, y)
            sumX += x
            sumY += y
            for (nextY in maxOf(0, y - 1)..minOf(rows - 1, y + 1)) {
                for (nextX in maxOf(0, x - 1)..minOf(columns - 1, x + 1)) {
                    val next = nextY * columns + nextX
                    if (!changed[next]) continue
                    changed[next] = false
                    component[count++] = next
                }
            }
        }
        val wide = maxX - minX + 1
        val tall = maxY - minY + 1
        val longSide = maxOf(wide, tall)
        val shortSide = minOf(wide, tall)
        if (count < 4 || shortSide < 2 || longSide > maxSide || longSide > 4 * shortSide ||
            count < wide * tall * 0.28 || minX == 0 || minY == 0 || maxX == columns - 1 || maxY == rows - 1
        ) return null
        return BallPoint(left + sumX.toFloat() / count, top + sumY.toFloat() / count)
    }
}

private const val MAX_CANDIDATES = 18
private const val MAX_SEEDS = 28

/** Accepts only timestamped observations in consecutive analysis frames. */
internal class LiveBallTracker(
    private val corridor: BallCorridor,
    private val viewDown: BallPoint = BallPoint(0f, 1f)
) {
    private data class Sample(val point: BallPoint, val ns: Long)
    private var previousNs: Long? = null
    private var frameInterval = 0.0
    private var interruptedInterval = 0.0
    private var starts = emptyList<Sample>()
    private var pairs = emptyList<Pair<Sample, Sample>>()
    private var active = emptyList<Sample>()
    private var latest: BallReadout? = null
    private val metersPerPixel = corridor.setup.meters?.div(corridor.length)

    fun accept(ns: Long, candidates: List<BallPoint>): BallReadout? {
        val before = previousNs
        previousNs = ns
        val dt = if (before == null) 0.0 else (ns - before) * 1e-9
        val continuous = dt > 0 && dt <= 0.080 &&
            (frameInterval == 0.0 || dt <= 0.002 + frameInterval * 1.60)
        when {
            continuous -> {
                frameInterval = if (frameInterval == 0.0) dt else 0.9 * frameInterval + 0.1 * dt
                interruptedInterval = 0.0
            }
            dt <= 0.0 || dt > 0.080 -> {
                frameInterval = 0.0
                interruptedInterval = 0.0
            }
            interruptedInterval > 0 && abs(dt / interruptedInterval - 1.0) <= 0.12 -> {
                frameInterval = dt
                interruptedInterval = 0.0
            }
            else -> interruptedInterval = dt
        }

        val current = candidates.filter(corridor::contains).map { Sample(it, ns) }
        val next = if (!continuous) emptyList() else if (active.isNotEmpty()) {
            val continuation = current.singleOrNull { follows(active[active.lastIndex - 1], active.last(), it) }
            if (continuation == null) emptyList() else active + continuation
        } else {
            val triples = pairs.flatMap { (a, b) ->
                current.filter { follows(a, b, it) }.map { listOf(a, b, it) }
            }
            triples.singleOrNull().orEmpty()
        }
        pairs = if (continuous && next.isEmpty() && active.isEmpty()) {
            val seeds = starts.flatMap { a -> current.filter { movesForward(a, it) }.map { a to it } }
            if (seeds.size <= MAX_SEEDS) seeds else emptyList()
        } else emptyList()
        starts = if (before == null || dt > 0.0) {
            current.filter { corridor.progress(it.point) <= corridor.length * 0.32f }
        } else emptyList()
        active = next
        latest = when {
            active.isNotEmpty() -> readout(active)
            latest?.active == true -> latest?.copy(active = false)
            else -> latest
        }
        return latest
    }

    private fun movesForward(a: Sample, b: Sample): Boolean {
        val forward = corridor.progress(b.point) - corridor.progress(a.point)
        val sideways = abs(corridor.lateral(b.point) - corridor.lateral(a.point))
        val seconds = (b.ns - a.ns) * 1e-9
        val pixelsPerSecond = forward / seconds
        val inRange = metersPerPixel?.let { pixelsPerSecond * it in 7.0..65.0 }
            ?: (pixelsPerSecond / corridor.length in 0.65..16.0)
        return forward >= 2.5f && sideways <= 2f + forward * 0.45f && inRange
    }

    private fun follows(a: Sample, b: Sample, c: Sample): Boolean {
        if (!movesForward(b, c)) return false
        val timeRatio = (c.ns - b.ns).toDouble() / (b.ns - a.ns)
        val velocityStep = b.point - a.point
        val predicted = BallPoint(
            b.point.x + velocityStep.x * timeRatio.toFloat(),
            b.point.y + velocityStep.y * timeRatio.toFloat()
        )
        return (c.point - predicted).length <= maxOf(3.0, velocityStep.length * timeRatio * 0.28)
    }

    private fun readout(samples: List<Sample>): BallReadout {
        val first = samples.first()
        val last = samples.last()
        val delta = last.point - first.point
        val seconds = (last.ns - first.ns) * 1e-9
        val speed = metersPerPixel?.let { delta.length * it / seconds * 3.6 }
        val deviation = if (samples.size >= 5) {
            val sign = if (delta.x * viewDown.y - delta.y * viewDown.x < 0f) -1 else 1
            samples.subList(1, samples.lastIndex).map { sample ->
                val offset = sample.point - first.point
                sign * (delta.x * offset.y - delta.y * offset.x).toDouble() / delta.length
            }.maxByOrNull { abs(it) }
        } else null
        return BallReadout(
            samples.map { it.point }, speed, deviation,
            metersPerPixel?.let { scale -> deviation?.times(scale * 100) }, active = true
        )
    }
}

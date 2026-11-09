package lab.pitch.replay

import java.util.UUID
import kotlin.math.abs
import kotlin.math.hypot

internal const val MAX_VIDEO_MS = 90_000L
internal const val MIN_PITCH_US = 250_000L
internal val jointIds = intArrayOf(11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
internal val bones = listOf(
    11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16, 11 to 23, 12 to 24, 23 to 24,
    23 to 25, 25 to 27, 27 to 29, 29 to 31, 24 to 26, 26 to 28, 28 to 30, 30 to 32
)

internal enum class Hand(val label: String, val shoulder: Int, val wrist: Int, val hip: Int) {
    RIGHT("오른손", 12, 16, 24),
    LEFT("왼손", 11, 15, 23)
}

internal data class Joint(val x: Float, val y: Float)
internal data class PoseFrame(val timeUs: Long, val joints: Map<Int, Joint>) {
    val canDraw: Boolean
        get() = ((11 in joints && 23 in joints) || (12 in joints && 24 in joints)) &&
            bones.count { (a, b) -> a in joints && b in joints } >= 4
}

internal fun observedJoint(x: Float, y: Float, likelihood: Float): Joint? =
    if (x in 0f..1f && y in 0f..1f && likelihood in .55f..1f) Joint(x, y) else null

internal data class Pitch(
    val id: String = UUID.randomUUID().toString(),
    val startUs: Long,
    val endUs: Long,
    val peakUs: Long? = null,
    val landingUs: Long? = null,
    val releaseUs: Long? = null
) {
    val confirmed: Boolean
        get() = landingUs != null && releaseUs != null &&
            landingUs in startUs..endUs && releaseUs in startUs..endUs && landingUs < releaseUs
}

internal data class Session(
    val id: String,
    val createdAtMs: Long,
    val fingerprint: String,
    val hand: Hand,
    val width: Int,
    val height: Int,
    val durationUs: Long,
    val frames: List<PoseFrame>,
    val pitches: List<Pitch>,
    val analysisError: String? = null
) {
    val aspect: Float get() = width.toFloat() / height
}

internal data class Take(val session: Session, val pitch: Pitch)

internal fun pitchCandidates(frames: List<PoseFrame>, hand: Hand, aspect: Float, durationUs: Long): List<Pitch> {
    data class Burst(var peakUs: Long, var peakSpeed: Double, var lastUs: Long, var count: Int)

    fun distance(a: Joint, b: Joint) = hypot(((a.x - b.x) * aspect).toDouble(), (a.y - b.y).toDouble())

    val peaks = mutableListOf<Long>()
    var previous: PoseFrame? = null
    var burst: Burst? = null

    fun flush() {
        burst?.takeIf { it.count >= 2 }?.let { peaks.add(it.peakUs) }
        burst = null
    }

    for (frame in frames) {
        val time = frame.timeUs
        val last = previous
        previous = frame
        if (burst?.let { time - it.lastUs > 250_000 } == true) flush()
        if (last == null || time - last.timeUs !in 12_000L..200_000L) continue
        if (peaks.lastOrNull()?.let { time - it < 1_700_000 } == true) continue
        val shoulder = frame.joints[hand.shoulder] ?: continue
        val wrist = frame.joints[hand.wrist] ?: continue
        val hip = frame.joints[hand.hip] ?: continue
        val previousWrist = last.joints[hand.wrist] ?: continue
        val torso = distance(shoulder, hip)
        if (torso <= .035 || wrist.y > shoulder.y + .75 * torso) continue
        val wristMotion = distance(wrist, previousWrist) / torso * 1_000_000 / (time - last.timeUs)
        if (wristMotion !in 3.4..42.0) continue
        val active = burst
        if (active == null) {
            burst = Burst(time, wristMotion, time, 1)
        } else {
            active.lastUs = time
            active.count++
            if (wristMotion > active.peakSpeed) {
                active.peakUs = time
                active.peakSpeed = wristMotion
            }
        }
    }
    flush()
    return peaks.filter { it in 0..durationUs }.let { validPeaks ->
        validPeaks.mapIndexedNotNull { index, peak ->
            val start = maxOf(0L, peak - 1_120_000, validPeaks.getOrNull(index - 1)?.let { (it + peak) / 2 } ?: 0L)
            val end = minOf(durationUs, peak + 780_000, validPeaks.getOrNull(index + 1)?.let { (it + peak) / 2 } ?: durationUs)
            if (end - start >= MIN_PITCH_US) Pitch(startUs = start, endUs = end, peakUs = peak) else null
        }
    }
}

internal fun nearestPose(frames: List<PoseFrame>, timeUs: Long): PoseFrame? {
    if (timeUs < 0 || frames.isEmpty()) return null
    val index = frames.binarySearchBy(timeUs) { it.timeUs }
    if (index >= 0) return frames[index].takeIf { it.canDraw }
    val insertion = -index - 1
    val before = frames.getOrNull(insertion - 1)
    val after = frames.getOrNull(insertion)
    val closest = listOfNotNull(before, after).minBy { abs(it.timeUs - timeUs) }
    return closest.takeIf { abs(it.timeUs - timeUs) <= 90_000 && it.canDraw }
}

internal enum class Event { LANDING, RELEASE }

internal fun markEvent(pitch: Pitch, event: Event, timeUs: Long): Pitch {
    require(timeUs in pitch.startUs..pitch.endUs) { "현재 시점이 투구 구간 밖입니다." }
    val updated = when (event) {
        Event.LANDING -> pitch.copy(landingUs = timeUs)
        Event.RELEASE -> pitch.copy(releaseUs = timeUs)
    }
    require(updated.landingUs == null || updated.releaseUs == null || updated.confirmed) {
        "착지는 릴리스보다 앞이어야 합니다."
    }
    return updated
}

internal fun manualPitch(startUs: Long, endUs: Long, durationUs: Long): Pitch {
    require(startUs >= 0 && endUs <= durationUs && endUs - startUs >= MIN_PITCH_US) {
        "영상 안에서 시작과 끝을 0.25초 이상 떨어지게 지정하세요."
    }
    return Pitch(startUs = startUs, endUs = endUs)
}

internal fun phaseTime(pitch: Pitch, fraction: Double): Long {
    check(pitch.confirmed)
    val landing = checkNotNull(pitch.landingUs)
    val release = checkNotNull(pitch.releaseUs)
    return landing + ((release - landing) * fraction.coerceIn(0.0, 1.0)).toLong()
}

internal fun comparisonError(a: Take?, b: Take?, sameView: Boolean): String? {
    if (a == null || b == null || a.pitch.id == b.pitch.id) return "서로 다른 두 투구를 고르세요."
    if (a.session.fingerprint == b.session.fingerprint &&
        maxOf(a.pitch.startUs, b.pitch.startUs) < minOf(a.pitch.endUs, b.pitch.endUs)
    ) return "같은 원본에서 겹치는 구간입니다."
    if (!a.pitch.confirmed || !b.pitch.confirmed) return "두 투구의 착지와 릴리스를 확인하세요."
    if (a.session.hand != b.session.hand || abs(a.session.aspect / b.session.aspect - 1) > .025) {
        return "투구 손과 영상의 가로세로 비율이 같아야 합니다."
    }
    if (!sameView) return "같은 선수와 고정된 시야인지 확인하세요."
    if (!listOf(a, b).all { take ->
            val observed = (0..10).map { nearestPose(take.session.frames, phaseTime(take.pitch, it / 10.0)) != null }
            observed.first() && observed.last() && observed.count { it } >= 7
        }
    ) return "착지부터 릴리스까지 관절이 충분히 보이는 두 영상을 고르세요."
    return null
}
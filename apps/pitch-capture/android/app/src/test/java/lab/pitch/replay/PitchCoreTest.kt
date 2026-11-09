package lab.pitch.replay

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class PitchCoreTest {
    private val visible = mapOf(
        11 to Joint(.40f, .30f), 12 to Joint(.50f, .30f),
        13 to Joint(.38f, .42f), 14 to Joint(.55f, .42f),
        15 to Joint(.35f, .54f), 16 to Joint(.58f, .54f),
        23 to Joint(.40f, .52f), 24 to Joint(.50f, .52f)
    )

    @Test fun rejectsUnreliableJointsAndDoesNotDrawAcrossAnUnobservedFrame() {
        assertNull(observedJoint(.4f, .6f, .549f))
        assertNull(observedJoint(Float.NaN, .6f, .9f))
        assertNull(observedJoint(.4f, 1.01f, .9f))
        assertNotNull(observedJoint(.4f, .6f, .55f))
        val frames = listOf(
            PoseFrame(0, visible),
            PoseFrame(50_000, emptyMap()),
            PoseFrame(100_000, visible)
        )
        assertNull(nearestPose(frames, 49_000))
        assertNull(nearestPose(frames, 50_000))
        assertEquals(100_000L, nearestPose(frames, 110_000)?.timeUs)
        assertNull(nearestPose(frames, 191_000))
    }

    @Test fun detectsBurstsOnlyOnTheSelectedThrowingSide() {
        val frames = (0..125).map { index ->
            val wristX = when (index) {
                20, 100 -> .43f
                21, 101 -> .49f
                22, 102 -> .56f
                in 23..99, in 103..125 -> .56f
                else -> .38f
            }
            PoseFrame(index * 33_333L, visible + (16 to Joint(wristX, .3f)))
        }
        val candidates = pitchCandidates(frames, Hand.RIGHT, 1.5f, 4_300_000)
        assertEquals(2, candidates.size)
        assertTrue(candidates[0].endUs < candidates[1].startUs)
        assertTrue(candidates.all { it.startUs <= it.peakUs!! && it.peakUs <= it.endUs && !it.confirmed })
        assertTrue(pitchCandidates(frames, Hand.LEFT, 1.5f, 4_300_000).isEmpty())
    }

    @Test fun requiresEventsInThePitchAndInTheCorrectOrder() {
        val pitch = manualPitch(100_000, 650_000, 1_000_000)
        val landing = markEvent(pitch, Event.LANDING, 300_000)
        assertFalse(landing.confirmed)
        assertThrows(IllegalArgumentException::class.java) { markEvent(landing, Event.RELEASE, 300_000) }
        assertThrows(IllegalArgumentException::class.java) { markEvent(landing, Event.RELEASE, 660_000) }
        assertThrows(IllegalArgumentException::class.java) { manualPitch(500_000, 699_999, 1_000_000) }
        val confirmed = markEvent(landing, Event.RELEASE, 500_000)
        assertTrue(confirmed.confirmed)
        assertEquals(300_000L, phaseTime(confirmed, -.1))
        assertEquals(350_000L, phaseTime(confirmed, .25))
        assertEquals(500_000L, phaseTime(confirmed, 1.1))
    }

    @Test fun comparisonRequiresDifferentConfirmedTakesAndEnoughObservedFrames() {
        val aPitch = Pitch("a", 0, 500_000, landingUs = 100_000, releaseUs = 400_000)
        val bPitch = Pitch("b", 0, 750_000, landingUs = 150_000, releaseUs = 600_000)
        val frames = (0..25).map { PoseFrame(it * 30_000L, visible) }
        val a = Take(Session("one", 0, "hash", Hand.RIGHT, 1600, 900, 900_000, frames, listOf(aPitch)), aPitch)
        val b = Take(a.session.copy(id = "two", fingerprint = "hash2", pitches = listOf(bPitch)), bPitch)
        assertNull(comparisonError(a, b, true))
        assertNotNull(comparisonError(a, b, false))
        assertNotNull(comparisonError(a, a, true))
        assertNotNull(comparisonError(a, b.copy(session = b.session.copy(fingerprint = "hash")), true))
        assertNotNull(comparisonError(a, b.copy(session = b.session.copy(hand = Hand.LEFT)), true))
        assertNotNull(comparisonError(a, b.copy(session = b.session.copy(width = 1440)), true))
        assertNotNull(comparisonError(a, b.copy(pitch = bPitch.copy(releaseUs = null)), true))
        val obscured = frames.map { if (it.timeUs >= 500_000) it.copy(joints = emptyMap()) else it }
        assertNotNull(comparisonError(a, b.copy(session = b.session.copy(frames = obscured)), true))
    }
}
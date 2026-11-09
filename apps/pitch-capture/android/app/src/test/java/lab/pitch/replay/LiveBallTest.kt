package lab.pitch.replay

import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LiveBallTest {
    private val step = 33_333_333L
    private val start = BallPoint(32f, 50f)
    private val end = BallPoint(232f, 50f)

    @Test
    fun brightMotionRejectsStaticObjectsAndSceneWideBrightnessChanges() {
        val width = 256
        val height = 100
        val stride = 272
        val buffer = ByteBuffer.allocate(4 + stride * height)
        val detector = BrightMotion(BallCorridor(BallSetup(start, end, null)), width, height)
        buffer.position(4)
        fun frame(ballX: Int, background: Int): List<BallPoint> {
            for (y in 0 until height) for (x in 0 until width) {
                val ball = x in ballX - 2..ballX + 2 && y in 48..52
                val static = x in 150..154 && y in 52..56
                buffer.put(4 + y * stride + x, (if (ball || static) 230 else background).toByte())
            }
            return detector.candidates(buffer, stride, 1)
        }

        assertTrue(frame(40, 52).isEmpty())
        assertEquals(listOf(BallPoint(64f, 50f)), frame(64, 52))
        assertTrue(frame(64, 52).isEmpty())
        assertTrue(frame(88, 200).isEmpty())
        assertEquals(4, buffer.position())
    }

    @Test
    fun reportsProjectedSpeedAndDeviationOnlyAfterEnoughObservations() {
        val tracker = LiveBallTracker(BallCorridor(BallSetup(start, end, 10.0)))
        fun observe(frame: Int, x: Float, y: Float) =
            tracker.accept(frame * step, listOf(BallPoint(x, y)))

        assertNull(observe(0, 40f, 50f))
        assertNull(observe(1, 64f, 50f))
        val first = checkNotNull(observe(2, 88f, 53f))
        assertTrue(first.active)
        assertEquals(3, first.points.size)
        assertEquals(129.8529, checkNotNull(first.speedKmh), 0.001)
        assertNull(first.deviationCm)
        assertNull(observe(3, 112f, 51f)?.deviationCm)
        val measured = checkNotNull(observe(4, 136f, 50f))
        assertEquals(3.0, checkNotNull(measured.deviationPx), 0.001)
        assertEquals(15.0, checkNotNull(measured.deviationCm), 0.001)
        assertEquals(129.6, checkNotNull(measured.speedKmh), 0.001)
    }

    @Test
    fun uncalibratedChordDepartureUsesScreenDownForEitherThrowDirection() {
        val corridor = BallCorridor(BallSetup(end, start, null))
        for (screenY in listOf(1f, -1f)) {
            val tracker = LiveBallTracker(corridor, BallPoint(0f, screenY))
            var latest: BallReadout? = null
            for (index in 0..4) {
                val point = BallPoint(224f - 24 * index, if (index == 2) 53f else 50f)
                latest = tracker.accept(index * step, listOf(point))
            }
            assertNull(latest?.speedKmh)
            assertNull(latest?.deviationCm)
            assertEquals(3.0 * screenY, checkNotNull(latest?.deviationPx), 0.001)
        }
    }

    @Test
    fun reverseMotionAndAmbiguousTracksAreWithheld() {
        val reverse = LiveBallTracker(BallCorridor(BallSetup(start, end, null)))
        for (index in 0..4) {
            assertNull(reverse.accept(index * step, listOf(BallPoint(130f - index * 20f, 50f))))
        }

        val ambiguous = LiveBallTracker(BallCorridor(BallSetup(start, end, null)))
        for (index in 0..4) {
            val x = 42f + index * 18f
            assertNull(ambiguous.accept(index * step, listOf(BallPoint(x, 43f), BallPoint(x, 57f))))
        }
    }

    @Test
    fun droppedCallbacksAndUnobservedCentersBreakWithoutInterpolation() {
        val tracker = LiveBallTracker(BallCorridor(BallSetup(start, end, null)))
        fun observe(index: Int, x: Float) = tracker.accept(index * step, listOf(BallPoint(x, 50f)))

        observe(0, 40f)
        observe(1, 60f)
        val accepted = checkNotNull(observe(2, 80f))
        assertNull(accepted.speedKmh)
        assertNull(accepted.deviationCm)
        assertFalse(checkNotNull(observe(4, 120f)).active)
        assertEquals(accepted.points, observe(5, 140f)?.points)

        observe(6, 40f)
        observe(7, 60f)
        val second = observe(8, 80f)
        assertNotNull(second)
        assertTrue(checkNotNull(second).active)
        assertFalse(checkNotNull(tracker.accept(9 * step, emptyList())).active)
        assertEquals(second.points, observe(10, 120f)?.points)
    }
}

package lab.pitch.replay

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.media.MediaPlayer
import android.util.Log
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.FrameLayout
import java.io.File
import kotlin.math.roundToInt

internal const val MINT = 0xFF8CE4C3.toInt()
internal const val ORANGE = 0xFFF5AB70.toInt()

internal class PoseOverlay(context: Context) : View(context) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var layers = emptyList<Pair<PoseFrame, Int>>()
    private val lineWidth = 2.5f * resources.displayMetrics.density

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    fun show(vararg poses: Pair<PoseFrame?, Int>) {
        layers = poses.mapNotNull { (pose, color) -> pose?.takeIf { it.canDraw }?.let { it to color } }
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        paint.strokeCap = Paint.Cap.ROUND
        for ((pose, color) in layers) {
            for ((a, b) in bones) {
                val start = pose.joints[a] ?: continue
                val end = pose.joints[b] ?: continue
                paint.color = 0xB8000000.toInt()
                paint.strokeWidth = lineWidth * 2
                canvas.drawLine(start.x * width, start.y * height, end.x * width, end.y * height, paint)
                paint.color = color
                paint.strokeWidth = lineWidth
                canvas.drawLine(start.x * width, start.y * height, end.x * width, end.y * height, paint)
            }
            paint.color = color
            for ((_, point) in pose.joints) {
                canvas.drawCircle(point.x * width, point.y * height, lineWidth * 1.3f, paint)
            }
        }
    }
}

internal class ReplayView(
    context: Context,
    source: File,
    private val aspect: Float,
    private val startUs: Long,
    private val endUs: Long,
    private val timeChanged: (Long?, Boolean) -> Unit,
    private val failed: (String) -> Unit
) : FrameLayout(context) {
    private val texture = TextureView(context)
    val overlay = PoseOverlay(context)
    private var surface: Surface? = null
    private var player: MediaPlayer? = null
    private var prepared = false
    private var requestedUs: Long? = null
    private var inFlightUs: Long? = null
    private var closed = false
    private var resumeUs = startUs

    var positionUs: Long? = null
        private set

    private val tick = object : Runnable {
        override fun run() {
            val media = player ?: return
            if (closed || !prepared || inFlightUs != null || !media.isPlaying) return
            val time = media.currentPosition.toLong() * 1_000
            if (time >= endUs) {
                media.pause()
                seek(endUs)
            } else {
                positionUs = time.coerceIn(startUs, endUs)
                timeChanged(positionUs, true)
                postDelayed(this, 32)
            }
        }
    }

    init {
        setBackgroundColor(Color.BLACK)
        addView(texture, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT, Gravity.CENTER))
        addView(overlay, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT, Gravity.CENTER))
        texture.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, width: Int, height: Int) {
                if (closed) return
                try {
                    surface = Surface(st)
                    val previous = player
                    if (previous != null) {
                        previous.setSurface(surface)
                        if (prepared && inFlightUs == null) seek(resumeUs)
                    } else {
                        prepare(source)
                    }
                } catch (error: Exception) {
                    fail(error)
                }
            }

            override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
                pause()
                runCatching { player?.setSurface(null) }
                surface?.release()
                surface = null
                return true
            }

            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, width: Int, height: Int) = Unit
            override fun onSurfaceTextureUpdated(st: SurfaceTexture) = Unit
        }
    }

    override fun onSizeChanged(width: Int, height: Int, oldWidth: Int, oldHeight: Int) {
        super.onSizeChanged(width, height, oldWidth, oldHeight)
        if (width == 0 || height == 0) return
        val fitWidth = minOf(width, (height * aspect).roundToInt())
        val fitHeight = minOf(height, (width / aspect).roundToInt())
        texture.layoutParams = LayoutParams(fitWidth, fitHeight, Gravity.CENTER)
        overlay.layoutParams = LayoutParams(fitWidth, fitHeight, Gravity.CENTER)
    }

    private fun prepare(source: File) {
        val media = MediaPlayer()
        player = media
        media.setDataSource(source.absolutePath)
        media.setSurface(surface)
        media.setVolume(0f, 0f)
        media.setOnPreparedListener {
            if (closed) return@setOnPreparedListener
            prepared = true
            seek(requestedUs ?: resumeUs)
        }
        media.setOnSeekCompleteListener {
            if (closed) return@setOnSeekCompleteListener
            val target = inFlightUs ?: return@setOnSeekCompleteListener
            inFlightUs = null
            val pending = requestedUs
            if (pending != null && pending != target) {
                beginSeek(pending)
            } else {
                requestedUs = null
                positionUs = (media.currentPosition.toLong() * 1_000).coerceIn(startUs, endUs)
                resumeUs = checkNotNull(positionUs)
                timeChanged(positionUs, false)
            }
        }
        media.setOnCompletionListener {
            if (!closed) {
                removeCallbacks(tick)
                positionUs = (media.currentPosition.toLong() * 1_000).coerceIn(startUs, endUs)
                resumeUs = checkNotNull(positionUs)
                timeChanged(positionUs, false)
            }
        }
        media.setOnErrorListener { _, what, extra ->
            fail(IllegalStateException("MediaPlayer error $what/$extra"))
            true
        }
        media.prepareAsync()
    }

    fun seek(timeUs: Long) {
        if (closed) return
        pause()
        val target = timeUs.coerceIn(startUs, endUs)
        requestedUs = target
        positionUs = null
        overlay.show()
        timeChanged(null, false)
        if (prepared && inFlightUs == null) beginSeek(target)
    }

    private fun beginSeek(target: Long) {
        try {
            inFlightUs = target
            player?.seekTo(target / 1_000, MediaPlayer.SEEK_CLOSEST)
        } catch (error: Exception) {
            fail(error)
        }
    }

    fun toggle() {
        val media = player ?: return
        val position = positionUs ?: return
        if (!prepared || inFlightUs != null) return
        try {
            if (media.isPlaying) {
                pause()
            } else if (endUs - position <= 1_000) {
                seek(startUs)
            } else {
                media.start()
                timeChanged(position, true)
                post(tick)
            }
        } catch (error: Exception) {
            fail(error)
        }
    }

    fun pause() {
        removeCallbacks(tick)
        val media = player ?: return
        if (!prepared || inFlightUs != null) return
        if (media.isPlaying) {
            media.pause()
            positionUs = (media.currentPosition.toLong() * 1_000).coerceIn(startUs, endUs)
            resumeUs = checkNotNull(positionUs)
            timeChanged(positionUs, false)
        }
    }

    fun release() {
        closed = true
        removeCallbacks(tick)
        player?.release()
        player = null
        surface?.release()
        surface = null
    }

    private fun fail(error: Exception) {
        if (closed) return
        Log.w("PitchReplay", "Source video could not play", error)
        release()
        positionUs = null
        timeChanged(null, false)
        failed("이 기기에서 원본 영상을 재생하지 못했습니다.")
    }
}
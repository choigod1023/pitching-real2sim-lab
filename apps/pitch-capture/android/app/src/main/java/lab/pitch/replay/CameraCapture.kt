package lab.pitch.replay

import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recording
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.camera.view.video.AudioConfig
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import java.io.File
import java.io.Closeable
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

internal class CameraCapture(
    private val activity: ComponentActivity,
    private val preview: PreviewView,
    private val poseChanged: (PoseFrame?) -> Unit,
    private val ballChanged: (BallReadout?) -> Unit,
    private val geometryChanged: () -> Unit,
    private val readyChanged: (Boolean) -> Unit,
    private val failed: (String) -> Unit
) : Closeable {
    private val controller = LifecycleCameraController(activity)
    private val detector = streamPoseDetector()
    private val main = ContextCompat.getMainExecutor(activity)
    private val analyzer = Executors.newSingleThreadExecutor()
    @Volatile private var closed = false
    @Volatile private var streaming = false
    @Volatile private var viewGeometry: ViewGeometry? = null
    @Volatile private var analysisFootprint: RectF? = null
    @Volatile private var ballSetup: BallSetup? = null
    private var tracking: Tracking? = null
    private val poseInFlight = AtomicBoolean()
    private var lastPoseNs: Long? = null
    private var recording: Recording? = null

    private data class ViewGeometry(val sensorToView: Matrix, val width: Int, val height: Int)
    private data class Tracking(
        val setup: BallSetup,
        val bufferToView: Matrix,
        val width: Int,
        val height: Int,
        val detector: BrightMotion,
        val tracker: LiveBallTracker,
        var sent: BallReadout? = null
    )

    fun open() {
        try {
            preview.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            preview.scaleType = PreviewView.ScaleType.FIT_CENTER
            preview.previewStreamState.observe(activity) {
                if (!closed) {
                    streaming = it == PreviewView.StreamState.STREAMING
                    readyChanged(streaming)
                    if (!streaming) {
                        analysisFootprint = null
                        setBallSetup(null)
                        geometryChanged()
                        poseChanged(null)
                    }
                }
            }
            controller.cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            controller.setEnabledUseCases(CameraController.IMAGE_ANALYSIS or CameraController.VIDEO_CAPTURE)
            controller.videoCaptureQualitySelector = QualitySelector.fromOrderedList(
                listOf(Quality.HD, Quality.SD), FallbackStrategy.lowerQualityOrHigherThan(Quality.SD)
            )
            controller.isPinchToZoomEnabled = false
            controller.imageAnalysisBackpressureStrategy = ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
            controller.setImageAnalysisAnalyzer(analyzer, object : ImageAnalysis.Analyzer {
                override fun getTargetCoordinateSystem() = ImageAnalysis.COORDINATE_SYSTEM_VIEW_REFERENCED

                override fun updateTransform(matrix: Matrix?) {
                    val geometry = matrix?.let { ViewGeometry(Matrix(it), preview.width, preview.height) }
                    val previous = viewGeometry
                    if (geometry == previous) return
                    viewGeometry = geometry
                    analysisFootprint = null
                    if (previous != null) {
                        setBallSetup(null)
                        geometryChanged()
                        poseChanged(null)
                    }
                }

                override fun analyze(image: ImageProxy) = analyzeFrame(image)
            })
            controller.bindToLifecycle(activity)
            preview.controller = controller
            controller.initializationFuture.addListener({
                if (!closed) runCatching { controller.initializationFuture.get() }
                    .onFailure { reportError(it) }
            }, main)
        } catch (error: Exception) {
            reportError(error)
        }
    }

    fun canMark(point: BallPoint) = analysisFootprint?.contains(point.x, point.y) == true

    fun setBallSetup(setup: BallSetup?) {
        ballSetup = setup
        ballChanged(null)
    }

    private fun analyzeFrame(image: ImageProxy) {
        try {
            val geometry = viewGeometry ?: return
            if (closed || !streaming || geometry.width <= 0 || geometry.height <= 0 ||
                image.format != ImageFormat.YUV_420_888 || image.planes.size != 3
            ) return
            val bufferToSensor = Matrix()
            if (!image.imageInfo.sensorToBufferTransformMatrix.invert(bufferToSensor)) return
            val bufferToView = Matrix().apply { setConcat(geometry.sensorToView, bufferToSensor) }
            val footprint = RectF(image.cropRect).apply { bufferToView.mapRect(this) }
            if (viewGeometry === geometry && streaming) analysisFootprint = footprint
            val timestamp = image.imageInfo.timestamp
            try {
                analyzeBall(image, geometry, bufferToView, timestamp)
            } catch (error: Exception) {
                Log.w("PitchReplay", "Live ball frame could not be read", error)
                tracking = null
                main.execute { if (!closed && viewGeometry === geometry) ballChanged(null) }
            }
            analyzePose(image, geometry, bufferToView, timestamp)
        } catch (error: Exception) {
            Log.w("PitchReplay", "Live analysis frame could not be read", error)
        } finally {
            image.close()
        }
    }

    private fun analyzeBall(image: ImageProxy, geometry: ViewGeometry, bufferToView: Matrix, timestamp: Long) {
        val setup = ballSetup
        if (setup == null) {
            tracking = null
            return
        }
        val current = tracking?.takeIf {
            it.setup === setup && it.width == image.width && it.height == image.height && it.bufferToView == bufferToView
        } ?: startTracking(setup, image, bufferToView) ?: return
        tracking = current
        val y = image.planes[0]
        val visible = RectF(image.cropRect)
        val candidates = current.detector.candidates(y.buffer, y.rowStride, y.pixelStride)
            .filter { visible.contains(it.x, it.y) }
        val readout = current.tracker.accept(timestamp, candidates)
        if (readout === current.sent) return
        current.sent = readout
        val mapped = readout?.copy(points = readout.points.map { bufferToView.map(it) })
        main.execute {
            if (!closed && streaming && ballSetup === setup && viewGeometry === geometry) ballChanged(mapped)
        }
    }

    private fun startTracking(setup: BallSetup, image: ImageProxy, bufferToView: Matrix): Tracking? {
        val viewToBuffer = Matrix()
        if (!bufferToView.invert(viewToBuffer)) return null
        val start = viewToBuffer.map(setup.start)
        val end = viewToBuffer.map(setup.end)
        val crop = RectF(image.cropRect)
        if (!crop.contains(start.x, start.y) || !crop.contains(end.x, end.y) || (end - start).length < 24f) return null
        val corridor = BallCorridor(BallSetup(start, end, setup.meters))
        val viewDown = viewToBuffer.map(BallPoint(0f, 1f)) - viewToBuffer.map(BallPoint(0f, 0f))
        return Tracking(
            setup, bufferToView, image.width, image.height,
            BrightMotion(corridor, image.width, image.height), LiveBallTracker(corridor, viewDown)
        )
    }

    private fun analyzePose(image: ImageProxy, geometry: ViewGeometry, bufferToView: Matrix, timestamp: Long) {
        val last = lastPoseNs
        if (last != null && timestamp >= last && timestamp - last < 83_000_000) return
        if (!poseInFlight.compareAndSet(false, true)) return
        lastPoseNs = timestamp
        try {
            val rotation = image.imageInfo.rotationDegrees
            val (y, u, v) = image.planes.map { YuvPlane(it.buffer, it.rowStride, it.pixelStride) }
            // ML Kit keeps the copied buffer. CameraX can deliver the next luma frame immediately.
            val frame = packNv21(Rect(0, 0, image.width, image.height), y, u, v)
            val input = InputImage.fromByteArray(
                frame.bytes, frame.width, frame.height, rotation, InputImage.IMAGE_FORMAT_NV21
            )
            val poseToBuffer = Matrix().apply {
                setRotate(-rotation.toFloat())
                postTranslate(
                    if (rotation == 180 || rotation == 270) frame.width.toFloat() else 0f,
                    if (rotation == 90 || rotation == 180) frame.height.toFloat() else 0f
                )
                postScale(frame.scale.toFloat(), frame.scale.toFloat())
            }
            val poseToView = Matrix().apply { setConcat(bufferToView, poseToBuffer) }
            detector.process(input).addOnCompleteListener(main) { result ->
                poseInFlight.set(false)
                if (closed || !streaming || viewGeometry !== geometry) return@addOnCompleteListener
                val pose = if (result.isSuccessful) result.result else null
                val joints = buildMap {
                    if (pose != null) for (id in jointIds) {
                        val landmark = pose.getPoseLandmark(id) ?: continue
                        val point = poseToView.map(BallPoint(landmark.position.x, landmark.position.y))
                        observedJoint(point.x / geometry.width, point.y / geometry.height, landmark.inFrameLikelihood)
                            ?.let { put(id, it) }
                    }
                }
                poseChanged(PoseFrame(timestamp / 1_000, joints).takeIf { it.canDraw })
            }
        } catch (error: Exception) {
            poseInFlight.set(false)
            throw error
        }
    }

    fun record(file: File, progress: (Long) -> Unit, finished: (Boolean) -> Unit) {
        val output = FileOutputOptions.Builder(file).setDurationLimitMillis(MAX_VIDEO_MS).build()
        recording = controller.startRecording(output, AudioConfig.AUDIO_DISABLED, main) { event ->
            when (event) {
                is VideoRecordEvent.Start, is VideoRecordEvent.Status ->
                    progress(event.recordingStats.recordedDurationNanos / 1_000_000)
                is VideoRecordEvent.Finalize -> {
                    recording = null
                    val valid = !event.hasError() || event.error == VideoRecordEvent.Finalize.ERROR_DURATION_LIMIT_REACHED ||
                        event.error == VideoRecordEvent.Finalize.ERROR_SOURCE_INACTIVE
                    if (event.hasError() && !valid) {
                        Log.w("PitchReplay", "Video recording failed with error ${event.error}", event.cause)
                    }
                    finished(valid && file.length() > 0)
                }
            }
        }
    }

    fun stop() {
        recording?.stop()
    }

    override fun close() {
        if (closed) return
        closed = true
        stop()
        preview.previewStreamState.removeObservers(activity)
        controller.clearImageAnalysisAnalyzer()
        preview.controller = null
        controller.unbind()
        detector.close()
        analyzer.shutdown()
    }

    private fun reportError(error: Throwable) {
        Log.w("PitchReplay", "Camera preview could not start", error)
        readyChanged(false)
        failed("후면 카메라를 열지 못했습니다. 저장한 영상은 가져올 수 있습니다.")
    }
}

private fun Matrix.map(point: BallPoint): BallPoint {
    val coordinates = floatArrayOf(point.x, point.y)
    mapPoints(coordinates)
    return BallPoint(coordinates[0], coordinates[1])
}

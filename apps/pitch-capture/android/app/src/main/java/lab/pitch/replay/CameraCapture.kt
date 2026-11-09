package lab.pitch.replay

import android.util.Log
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.mlkit.vision.MlKitAnalyzer
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
import java.io.File
import java.io.Closeable
import java.util.concurrent.Executors

internal class CameraCapture(
    private val activity: ComponentActivity,
    private val preview: PreviewView,
    private val poseChanged: (PoseFrame?) -> Unit,
    private val readyChanged: (Boolean) -> Unit,
    private val failed: (String) -> Unit
) : Closeable {
    private val controller = LifecycleCameraController(activity)
    private val detector = streamPoseDetector()
    private val main = ContextCompat.getMainExecutor(activity)
    private val analyzer = Executors.newSingleThreadExecutor()
    private var closed = false
    private var recording: Recording? = null

    fun open() {
        try {
            preview.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            preview.scaleType = PreviewView.ScaleType.FIT_CENTER
            preview.previewStreamState.observe(activity) {
                if (!closed) {
                    readyChanged(it == PreviewView.StreamState.STREAMING)
                    if (it != PreviewView.StreamState.STREAMING) poseChanged(null)
                }
            }
            controller.cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            controller.setEnabledUseCases(CameraController.IMAGE_ANALYSIS or CameraController.VIDEO_CAPTURE)
            controller.videoCaptureQualitySelector = QualitySelector.fromOrderedList(
                listOf(Quality.HD, Quality.SD), FallbackStrategy.lowerQualityOrHigherThan(Quality.SD)
            )
            controller.isPinchToZoomEnabled = false
            controller.setImageAnalysisAnalyzer(
                analyzer, MlKitAnalyzer(
                    listOf(detector), ImageAnalysis.COORDINATE_SYSTEM_VIEW_REFERENCED, main
                ) { result ->
                    if (!closed) {
                        val pose = result.getValue(detector)
                        val width = preview.width.toFloat()
                        val height = preview.height.toFloat()
                        val joints = if (pose == null || width <= 0 || height <= 0) emptyMap() else buildMap {
                            for (id in jointIds) {
                                val landmark = pose.getPoseLandmark(id) ?: continue
                                val point = landmark.position
                                observedJoint(point.x / width, point.y / height, landmark.inFrameLikelihood)
                                    ?.let { put(id, it) }
                            }
                        }
                        poseChanged(PoseFrame(0, joints).takeIf { it.canDraw })
                    }
                }
            )
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
package lab.pitch.replay

import android.graphics.ImageFormat
import android.graphics.Rect
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit

internal fun streamPoseDetector(): PoseDetector = PoseDetection.getClient(
    PoseDetectorOptions.Builder().setDetectorMode(PoseDetectorOptions.STREAM_MODE).build()
)

internal data class VideoInfo(val width: Int, val height: Int, val rotation: Int, val durationUs: Long)

internal fun videoInfo(file: File): VideoInfo {
    val metadata = MediaMetadataRetriever()
    try {
        metadata.setDataSource(file.absolutePath)
        val width = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
        val height = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
        val rotation = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
        val durationMs = metadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        require(width > 0 && height > 0 && durationMs in 1..(MAX_VIDEO_MS + 250) &&
            rotation in listOf(0, 90, 180, 270)
        ) { "길이가 90초 이하인 재생 가능한 영상을 선택하세요." }
        return if (rotation == 90 || rotation == 270) {
            VideoInfo(height, width, rotation, durationMs * 1_000)
        } else {
            VideoInfo(width, height, rotation, durationMs * 1_000)
        }
    } finally {
        metadata.release()
    }
}

internal fun analyseVideo(file: File, info: VideoInfo, progress: (Int) -> Unit): List<PoseFrame> {
    val extractor = MediaExtractor()
    try {
        extractor.setDataSource(file.absolutePath)
        val index = (0 until extractor.trackCount).firstOrNull {
            extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
        } ?: error("영상 트랙을 찾을 수 없습니다.")
        extractor.selectTrack(index)
        val format = extractor.getTrackFormat(index)
        val mime = checkNotNull(format.getString(MediaFormat.KEY_MIME))
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        // The buffer is decoded unrotated. ML Kit receives the container rotation once.
        format.setInteger(MediaFormat.KEY_ROTATION, 0)
        val codec = MediaCodec.createDecoderByType(mime)
        try {
            codec.configure(format, null, null, 0)
            codec.start()
            val detector = streamPoseDetector()
            try {
                return decodeFrames(extractor, codec, detector, info, progress)
            } finally {
                detector.close()
            }
        } finally {
            codec.release()
        }
    } finally {
        extractor.release()
    }
}

private fun decodeFrames(
    extractor: MediaExtractor,
    codec: MediaCodec,
    detector: PoseDetector,
    video: VideoInfo,
    progress: (Int) -> Unit
): List<PoseFrame> {
    val frames = mutableListOf<PoseFrame>()
    val bufferInfo = MediaCodec.BufferInfo()
    var endedInput = false
    var endedOutput = false
    var lastPercent = -1
    var lastActivityMs = SystemClock.elapsedRealtime()
    while (!endedOutput) {
        if (!endedInput) {
            val slot = codec.dequeueInputBuffer(5_000)
            if (slot >= 0) {
                val buffer = checkNotNull(codec.getInputBuffer(slot))
                buffer.clear()
                val size = extractor.readSampleData(buffer, 0)
                endedInput = size < 0
                if (endedInput) {
                    codec.queueInputBuffer(slot, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                } else {
                    codec.queueInputBuffer(slot, 0, size, extractor.sampleTime, 0)
                    extractor.advance()
                }
                lastActivityMs = SystemClock.elapsedRealtime()
            }
        }
        val slot = codec.dequeueOutputBuffer(bufferInfo, 10_000)
        if (slot >= 0) {
            try {
                val pts = bufferInfo.presentationTimeUs
                endedOutput = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                val imageFrame = bufferInfo.size > 0 && bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0
                if (imageFrame && pts in 0..video.durationUs &&
                    (frames.lastOrNull()?.let { pts - it.timeUs >= 33_000 } ?: true)
                ) {
                    val image = checkNotNull(codec.getOutputImage(slot)) { "디코더가 영상 프레임을 내보내지 못했습니다." }
                    val packed = try {
                        packNv21(image)
                    } finally {
                        image.close()
                    }
                    frames += PoseFrame(pts, detectJoints(detector, packed, video.rotation))
                    val percent = (pts * 100 / video.durationUs).toInt()
                    if (percent >= lastPercent + 2) {
                        lastPercent = percent
                        progress(percent)
                    }
                }
            } finally {
                codec.releaseOutputBuffer(slot, false)
            }
            lastActivityMs = SystemClock.elapsedRealtime()
        }
        check(SystemClock.elapsedRealtime() - lastActivityMs < 10_000) { "영상 디코딩 시간이 초과됐습니다." }
    }
    check(frames.isNotEmpty()) { "분석할 영상 프레임이 없습니다." }
    return frames
}

internal data class Nv21(val bytes: ByteArray, val width: Int, val height: Int, val scale: Int)
internal data class YuvPlane(val buffer: ByteBuffer, val rowStride: Int, val pixelStride: Int)

private fun packNv21(image: Image): Nv21 {
    require(image.format == ImageFormat.YUV_420_888 && image.planes.size == 3) {
        "기기에서 이 영상의 픽셀 형식을 읽을 수 없습니다."
    }
    val (y, u, v) = image.planes.map { YuvPlane(it.buffer, it.rowStride, it.pixelStride) }
    return packNv21(image.cropRect, y, u, v)
}

internal fun packNv21(crop: Rect, y: YuvPlane, u: YuvPlane, v: YuvPlane): Nv21 {
    val scale = (maxOf(crop.width(), crop.height()) + 1279) / 1280
    require(scale > 0)
    val width = (crop.width() / scale) and -2
    val height = (crop.height() / scale) and -2
    require(width > 0 && height > 0)
    val bytes = ByteArray(width * height * 3 / 2)
    val luma = y.buffer.duplicate()
    val blue = u.buffer.duplicate()
    val red = v.buffer.duplicate()
    val lumaStart = luma.position() + crop.top * y.rowStride + crop.left * y.pixelStride
    val blueStart = blue.position() + (crop.top / 2) * u.rowStride + (crop.left / 2) * u.pixelStride
    val redStart = red.position() + (crop.top / 2) * v.rowStride + (crop.left / 2) * v.pixelStride
    for (row in 0 until height) {
        val source = lumaStart + row * scale * y.rowStride
        val target = row * width
        if (scale == 1 && y.pixelStride == 1) {
            luma.position(source)
            luma.get(bytes, target, width)
        } else {
            for (col in 0 until width) bytes[target + col] = luma[source + col * scale * y.pixelStride]
        }
    }
    for (row in 0 until height / 2) {
        val blueRow = blueStart + row * scale * u.rowStride
        val redRow = redStart + row * scale * v.rowStride
        val target = width * height + row * width
        for (col in 0 until width / 2) {
            bytes[target + col * 2] = red[redRow + col * scale * v.pixelStride]
            bytes[target + col * 2 + 1] = blue[blueRow + col * scale * u.pixelStride]
        }
    }
    return Nv21(bytes, width, height, scale)
}

private fun detectJoints(detector: PoseDetector, frame: Nv21, rotation: Int): Map<Int, Joint> {
    val input = InputImage.fromByteArray(frame.bytes, frame.width, frame.height, rotation, InputImage.IMAGE_FORMAT_NV21)
    val pose = Tasks.await(detector.process(input), 10, TimeUnit.SECONDS)
    val swap = rotation == 90 || rotation == 270
    val width = if (swap) frame.height else frame.width
    val height = if (swap) frame.width else frame.height
    return buildMap {
        for (id in jointIds) {
            val landmark = pose.getPoseLandmark(id) ?: continue
            val point = landmark.position
            observedJoint(point.x / width, point.y / height, landmark.inFrameLikelihood)?.let { put(id, it) }
        }
    }
}

internal fun captureAnalysis(file: File, video: VideoInfo, progress: (Int) -> Unit): Pair<List<PoseFrame>, String?> =
    try {
        analyseVideo(file, video, progress) to null
    } catch (error: Exception) {
        Log.w("PitchReplay", "Recorded pose analysis failed", error)
        emptyList<PoseFrame>() to "기기에서 관절을 분석하지 못했습니다. 원본을 열어 구간을 확인할 수 있습니다."
    }

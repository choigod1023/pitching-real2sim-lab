package lab.pitch.replay

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.util.AtomicFile
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.UUID

internal class Archive(context: Context) {
    private val root = File(context.filesDir, "pitch-replay")

    fun newSessionId() = UUID.randomUUID().toString()

    fun source(id: String): File = File(File(root, id), "source.mp4")

    fun prepare(id: String): File = source(id).also {
        check(it.parentFile?.let { parent -> parent.isDirectory || parent.mkdirs() } == true) {
            "영상 저장 공간을 만들지 못했습니다."
        }
    }

    fun copySource(resolver: ContentResolver, uri: Uri, destination: File) {
        val input = resolver.openInputStream(uri) ?: error("선택한 영상을 열 수 없습니다.")
        input.use {
            destination.outputStream().use { output ->
                var total = 0L
                val buffer = ByteArray(128 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    total += read
                    require(total <= 512L * 1024 * 1024) { "512MB 이하의 영상을 선택하세요." }
                    output.write(buffer, 0, read)
                }
            }
        }
    }

    fun scan(): List<Session> =
        root.listFiles().orEmpty().mapNotNull { folder ->
            val metadataFile = metadata(folder.name)
            val baseFile = metadataFile.baseFile
            if (!folder.isDirectory || !File(folder, "source.mp4").isFile ||
                (!baseFile.isFile && !File(folder, "session.json.bak").isFile)
            ) return@mapNotNull null
            runCatching { decode(folder.name, metadataFile.readFully().toString(Charsets.UTF_8)) }
                .onFailure { Log.w("PitchReplay", "Could not read session metadata", it) }.getOrNull()
        }.sortedByDescending { it.createdAtMs }

    fun save(session: Session) {
        val target = metadata(session.id)
        val stream = target.startWrite()
        try {
            stream.write(encode(session).toString().toByteArray(Charsets.UTF_8))
            target.finishWrite(stream)
        } catch (error: Exception) {
            target.failWrite(stream)
            throw error
        }
    }

    fun fingerprint(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(128 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    private fun metadata(id: String) = AtomicFile(File(source(id).parentFile, "session.json"))

    private fun encode(session: Session) = JSONObject().apply {
        put("schema", 1)
        put("createdAtMs", session.createdAtMs)
        put("fingerprint", session.fingerprint)
        put("hand", session.hand.name)
        put("width", session.width)
        put("height", session.height)
        put("durationUs", session.durationUs)
        put("analysisError", session.analysisError)
        put("pitches", JSONArray().apply {
            for (pitch in session.pitches) put(JSONObject().apply {
                put("id", pitch.id)
                put("startUs", pitch.startUs)
                put("endUs", pitch.endUs)
                put("peakUs", pitch.peakUs)
                put("landingUs", pitch.landingUs)
                put("releaseUs", pitch.releaseUs)
            })
        })
        put("frames", JSONArray().apply {
            for (frame in session.frames) put(JSONArray().apply {
                put(frame.timeUs)
                for (id in jointIds) {
                    val joint = frame.joints[id]
                    put(joint?.let { JSONArray().put(it.x.toDouble()).put(it.y.toDouble()) } ?: JSONObject.NULL)
                }
            })
        })
    }

    private fun decode(id: String, text: String): Session {
        val data = JSONObject(text)
        require(data.getInt("schema") == 1)
        val duration = data.getLong("durationUs")
        val width = data.getInt("width")
        val height = data.getInt("height")
        require(width > 0 && height > 0 && duration in 1..(MAX_VIDEO_MS + 250) * 1_000)
        val serializedFrames = data.getJSONArray("frames")
        val frames = buildList<PoseFrame> {
            for (index in 0 until serializedFrames.length()) {
                val frame = serializedFrames.getJSONArray(index)
                val time = frame.getLong(0)
                if (time !in 0..duration || lastOrNull()?.let { time <= it.timeUs } == true) continue
                add(PoseFrame(time, buildMap {
                    for ((slot, jointId) in jointIds.withIndex()) {
                        val point = frame.optJSONArray(slot + 1) ?: continue
                        observedJoint(point.getDouble(0).toFloat(), point.getDouble(1).toFloat(), 1f)?.let {
                            put(jointId, it)
                        }
                    }
                }))
            }
        }
        val serializedPitches = data.getJSONArray("pitches")
        val pitches = buildList {
            for (index in 0 until serializedPitches.length()) {
                val pitch = serializedPitches.getJSONObject(index)
                val start = pitch.getLong("startUs")
                val end = pitch.getLong("endUs")
                if (start !in 0..duration || end !in 0..duration || end - start < MIN_PITCH_US) continue
                val decoded = Pitch(
                    id = pitch.getString("id"), startUs = start, endUs = end,
                    peakUs = pitch.numberOrNull("peakUs"), landingUs = pitch.numberOrNull("landingUs"),
                    releaseUs = pitch.numberOrNull("releaseUs")
                )
                val invalidEvent = listOfNotNull(decoded.landingUs, decoded.releaseUs).any { it !in start..end }
                val invalidOrder = decoded.landingUs != null && decoded.releaseUs != null && !decoded.confirmed
                add(if (invalidEvent || invalidOrder) decoded.copy(landingUs = null, releaseUs = null) else decoded)
            }
        }
        return Session(
            id, data.getLong("createdAtMs"), data.getString("fingerprint"), Hand.valueOf(data.getString("hand")),
            width, height, duration, frames, pitches, data.optString("analysisError").takeIf { it.isNotBlank() }
        )
    }

    private fun JSONObject.numberOrNull(key: String) = if (isNull(key)) null else getLong(key)
}
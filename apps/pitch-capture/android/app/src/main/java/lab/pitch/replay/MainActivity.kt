package lab.pitch.replay

import android.Manifest
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.net.Uri
import android.os.Bundle
import android.text.format.DateFormat
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

private const val BACKGROUND = 0xFF101A18.toInt()
private const val PANEL = 0xFF23332E.toInt()
private const val INK = 0xFFF1F5F2.toInt()
private const val MUTED = 0xFFADC0B7.toInt()
private val analysisWorker = Executors.newSingleThreadExecutor()

class MainActivity : ComponentActivity() {
    private enum class Page(val label: String) { CAPTURE("촬영"), LIBRARY("보관함"), COMPARE("겹쳐보기") }

    private lateinit var archive: Archive
    private lateinit var root: LinearLayout
    private lateinit var workspace: LinearLayout
    private lateinit var navButtons: List<Button>
    private var page = Page.CAPTURE
    private var camera: CameraCapture? = null
    private var replay: ReplayView? = null
    private var cameraRequested = false
    private var recording = false
    private var stopping = false
    private var busy: String? = null
    private var saving = false
    private var hand = Hand.RIGHT
    private var sessions = emptyList<Session>()
    private var selectedSessionId: String? = null
    private var selectedPitchId: String? = null
    private var manualStartUs: Long? = null
    private var aId: String? = null
    private var bId: String? = null
    private var sameView = false
    private var notice = ""
    private var statusText: TextView? = null

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        cameraRequested = granted
        notice = if (granted) "" else "카메라 권한을 허용하면 촬영할 수 있습니다."
        render()
    }

    private val openDocument = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) addVideo(uri = uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        archive = Archive(applicationContext)
        root = column()
        root.setBackgroundColor(BACKGROUND)
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val safe = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(safe.left + dp(12), safe.top + dp(4), safe.right + dp(12), safe.bottom + dp(4))
            insets
        }
        setContentView(root)
        render()
        analysisWorker.execute {
            val saved = archive.scan()
            runOnUiThread {
                if (!isDestroyed) {
                    sessions = saved
                    if (page != Page.CAPTURE && !recording && busy == null) render()
                }
            }
        }
    }

    override fun onStop() {
        replay?.pause()
        if (recording && !stopping) {
            stopping = true
            camera?.stop()
        }
        super.onStop()
    }

    override fun onDestroy() {
        camera?.close()
        replay?.release()
        super.onDestroy()
    }

    private fun go(target: Page) {
        if (page == target || recording || busy != null || saving) return
        page = target
        notice = ""
        render()
    }

    private fun render() {
        camera?.close()
        camera = null
        replay?.release()
        replay = null
        statusText = null
        root.removeAllViews()

        val header = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        header.addView(label("Pitch Replay", 22, bold = true), LinearLayout.LayoutParams(0, -2, 1f))
        navButtons = Page.entries.map { target ->
            button(target.label, primary = target == page) { go(target) }.also {
                it.isEnabled = !recording && busy == null && !saving
                header.addView(it, LinearLayout.LayoutParams(-2, dp(48)).apply { marginStart = dp(4) })
            }
        }
        root.addView(header, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(8) })
        workspace = LinearLayout(this)
        root.addView(workspace, LinearLayout.LayoutParams(-1, 0, 1f))
        when (page) {
            Page.CAPTURE -> renderCapture()
            Page.LIBRARY -> renderLibrary()
            Page.COMPARE -> renderCompare()
        }
    }

    private fun renderCapture() {
        val stage = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        val preview = PreviewView(this)
        val skeleton = PoseOverlay(this)
        val guide = stageLabel("후면 카메라 · 가로 촬영")
        stage.addView(preview, FrameLayout.LayoutParams(-1, -1))
        stage.addView(skeleton, FrameLayout.LayoutParams(-1, -1))
        stage.addView(guide, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.START).apply {
            topMargin = dp(12)
            marginStart = dp(12)
        })
        val center = stageLabel("카메라를 켜면 여기서 구도를 확인할 수 있습니다.").apply { gravity = Gravity.CENTER }
        stage.addView(center, FrameLayout.LayoutParams(-2, -2, Gravity.CENTER))
        val controls = screen(stage)
        addHeading(controls, "투구 촬영", "휴대폰을 고정하고 투수 한 명의 전신이 화면에 들어오게 맞추세요.")
        var ready = false
        val live = label("카메라 대기", 14, color = MUTED)
        status(controls)
        val handButton = addAction(controls, "투구 손 · ${hand.label}") {}
        handButton.setOnClickListener {
            AlertDialog.Builder(this).setTitle("투구 손")
                .setSingleChoiceItems(Hand.entries.map { it.label }.toTypedArray(), hand.ordinal) { dialog, index ->
                    hand = Hand.entries[index]
                    handButton.text = "투구 손 · ${hand.label}"
                    dialog.dismiss()
                }.setNegativeButton("닫기", null).show()
        }
        val primary = button("카메라 켜기", primary = true) {}
        addControl(controls, primary)
        val import = addAction(controls, "영상 가져오기") { openDocument.launch(arrayOf("video/*")) }
        addControl(controls, live)

        fun updateButtons() {
            primary.text = when {
                stopping -> "녹화 마무리 중"
                recording -> "녹화 끝내기"
                cameraRequested -> "녹화 시작"
                else -> "카메라 켜기"
            }
            primary.isEnabled = busy == null && !stopping && (recording || !cameraRequested || ready)
            handButton.isEnabled = !recording && busy == null
            import.isEnabled = !recording && busy == null
            updateNavigation()
        }

        primary.setOnClickListener {
            when {
                recording -> {
                    stopping = true
                    camera?.stop()
                    updateButtons()
                }
                cameraRequested && ready -> {
                    val id = archive.newSessionId()
                    val chosenHand = hand
                    try {
                        val file = archive.prepare(id)
                        recording = true
                        guide.text = "녹화 중 · 00초 / 90초"
                        setNotice("녹화를 끝내면 원본 영상에서 투구 구간을 찾습니다.")
                        updateButtons()
                        checkNotNull(camera).record(file, progress = { elapsed ->
                            if (!isDestroyed) guide.text = "녹화 중 · %02d초 / 90초".format(elapsed / 1_000)
                        }) { valid ->
                            recording = false
                            stopping = false
                            if (valid) {
                                addVideo(id = id, chosenHand = chosenHand)
                            } else {
                                discardDraft(file)
                                if (!isDestroyed) {
                                    guide.text = "후면 카메라 · 가로 촬영"
                                    setNotice("녹화를 저장하지 못했습니다. 다시 촬영해 주세요.")
                                    updateButtons()
                                }
                            }
                        }
                    } catch (error: Exception) {
                        Log.w("PitchReplay", "Could not start recording", error)
                        discardDraft(archive.source(id))
                        recording = false
                        guide.text = "후면 카메라 · 가로 촬영"
                        setNotice("녹화를 시작하지 못했습니다. 카메라와 저장 공간을 확인하세요.")
                        updateButtons()
                    }
                }
                !cameraRequested -> {
                    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                        cameraRequested = true
                        render()
                    } else {
                        cameraPermission.launch(Manifest.permission.CAMERA)
                    }
                }
            }
        }
        updateButtons()

        if (cameraRequested && busy == null) {
            center.text = "후면 카메라를 여는 중"
            camera = CameraCapture(this, preview, poseChanged = {
                skeleton.show(it to MINT)
                live.text = if (it == null) "전신이 보이도록 구도를 맞추세요." else "관절이 화면에 잡혔습니다."
            }, readyChanged = {
                ready = it
                center.visibility = if (it) View.GONE else View.VISIBLE
                updateButtons()
            }, failed = {
                center.text = "후면 카메라를 열지 못했습니다."
                setNotice(it)
            }).also { it.open() }
        }
    }

    private fun renderLibrary() {
        val session = sessions.firstOrNull { it.id == selectedSessionId } ?: sessions.firstOrNull()
        selectedSessionId = session?.id
        if (session == null) {
            val controls = screen(emptyStage("저장한 투구 영상이 없습니다."))
            addHeading(controls, "보관함", "촬영하거나 가져온 영상이 여기에 모입니다.")
            addAction(controls, "촬영 화면 열기", primary = true) { go(Page.CAPTURE) }
            status(controls)
            return
        }
        val pitch = session.pitches.firstOrNull { it.id == selectedPitchId }
        val start = pitch?.startUs ?: 0
        val end = pitch?.endUs ?: session.durationUs
        var playing = false
        val controls = column()
        val time = label("영상을 여는 중", 15, bold = true)
        val seekBar = slider(1_000, "영상 위치") { progress ->
            replay?.seek(start + (end - start) * progress / 1_000)
        }
        val play = button("재생") { replay?.toggle() }
        val marks = label("", 14, color = MUTED)
        val viewer = ReplayView(this, archive.source(session.id), session.aspect, start, end, timeChanged = { current, running ->
            playing = running
            time.text = current?.let { "${formatTime(it)} / ${formatTime(end)}" } ?: "프레임을 찾는 중"
            play.text = if (running) "멈춤" else "재생"
            play.isEnabled = current != null
            if (current != null) seekBar.progress = ((current - start) * 1_000 / (end - start)).toInt()
            replay?.overlay?.show(current?.let { nearestPose(session.frames, it) } to MINT)
        }, failed = ::setNotice)
        replay = viewer
        screen(viewer, controls)
        addHeading(controls, if (pitch == null) "원본 영상" else pitchTitle(session, pitch), sessionTitle(session))
        addAction(controls, "영상 선택") { selectSession() }
        if (session.pitches.isNotEmpty()) addAction(controls, "투구 구간 선택") { selectPitch(session) }
        addControl(controls, time)
        addControl(controls, seekBar)
        val rewind = button("−0.033초") { replay?.positionUs?.let { replay?.seek(it - 33_000) } }
        val advance = button("+0.033초") { replay?.positionUs?.let { replay?.seek(it + 33_000) } }
        addButtonRow(controls, rewind, play, advance)
        addControl(controls, marks)
        status(controls)
        if (pitch == null) {
            marks.text = manualStartUs?.let { "시작 ${formatTime(it)}" } ?: "누락된 투구의 시작과 끝을 지정하세요."
            addButtonRow(controls,
                button("시작 지정") {
                    currentPausedTime(playing)?.let {
                        manualStartUs = it
                        marks.text = "시작 ${formatTime(it)}"
                    }
                },
                button("끝 지정", primary = true) {
                    val first = manualStartUs
                    val last = currentPausedTime(playing)
                    if (first == null) setNotice("투구 시작 위치를 먼저 지정하세요.")
                    if (first != null && last != null) {
                        runCatching { manualPitch(first, last, session.durationUs) }
                            .onSuccess { created ->
                                val currentSession = sessions.firstOrNull { it.id == session.id } ?: return@onSuccess
                                saveUpdated(currentSession.copy(pitches = currentSession.pitches + created)) {
                                    selectedPitchId = created.id
                                    manualStartUs = null
                                    render()
                                }
                            }.onFailure { setNotice(it.message.orEmpty()) }
                    }
                }
            )
        } else {
            fun showMarks(value: Pitch) {
                marks.text = "착지 ${value.landingUs?.let(::formatTime) ?: "미확인"} · 릴리스 ${value.releaseUs?.let(::formatTime) ?: "미확인"}"
            }
            showMarks(pitch)
            addButtonRow(controls,
                button("착지 확인") { recordEvent(session.id, pitch.id, Event.LANDING, playing, ::showMarks) },
                button("릴리스 확인", primary = true) { recordEvent(session.id, pitch.id, Event.RELEASE, playing, ::showMarks) }
            )
            addButtonRow(controls,
                button("A로 선택") { setTake(true, pitch.id) },
                button("B로 선택") { setTake(false, pitch.id) }
            )
            addAction(controls, "누락된 투구 추가") {
                manualStartUs = null
                selectedPitchId = null
                render()
            }
        }
        session.analysisError?.let { addControl(controls, label(it, 14, color = MUTED)) }
    }

    private fun renderCompare() {
        val a = take(aId)
        val b = take(bId)
        val problem = comparisonError(a, b, sameView)
        val controls = column()
        val phaseLabel = label("착지 0% · 릴리스 100%", 14, color = MUTED)
        val frameStatus = label("", 14, color = MUTED)
        val phase = slider(100, "착지부터 릴리스까지의 진행률") { percent ->
            if (a != null && problem == null) replay?.seek(phaseTime(a.pitch, percent / 100.0))
        }
        phase.isEnabled = problem == null
        val stage = if (a == null) {
            emptyStage("기준 영상 A를 선택하세요.")
        } else {
            val first = a.pitch.landingUs?.takeIf { problem == null } ?: a.pitch.startUs
            val last = a.pitch.releaseUs?.takeIf { problem == null } ?: a.pitch.endUs
            ReplayView(this, archive.source(a.session.id), a.session.aspect, first, last, timeChanged = { current, _ ->
                if (current != null && b != null && problem == null) {
                    val fraction = (current - first).toDouble() / (last - first)
                    phase.progress = (fraction * 100).toInt().coerceIn(0, 100)
                    val aFrame = nearestPose(a.session.frames, current)
                    val bFrame = nearestPose(b.session.frames, phaseTime(b.pitch, fraction))
                    val both = aFrame != null && bFrame != null
                    replay?.overlay?.show(aFrame?.takeIf { both } to MINT, bFrame?.takeIf { both } to ORANGE)
                    frameStatus.text = if (both) "" else "이 시점에는 두 관절선을 함께 표시할 수 없습니다."
                    phaseLabel.text = "${phase.progress}% · A ${formatTime(current)} · B ${formatTime(phaseTime(b.pitch, fraction))}"
                } else if (problem == null) {
                    frameStatus.text = "프레임을 찾는 중"
                }
            }, failed = ::setNotice).also { replay = it }
        }
        screen(stage, controls)
        addHeading(controls, "두 투구 겹쳐보기", "A 영상 위에 같은 진행률의 관절선을 표시합니다.")
        addAction(controls, "A · ${a?.let { pitchTitle(it.session, it.pitch) } ?: "기준 투구 선택"}") { chooseTake(true) }
            .setTextColor(MINT)
        addAction(controls, "B · ${b?.let { pitchTitle(it.session, it.pitch) } ?: "비교 투구 선택"}") { chooseTake(false) }
            .setTextColor(ORANGE)
        val check = CheckBox(this).apply {
            text = "같은 선수와 고정된 시야를 두 원본에서 확인했습니다."
            setTextColor(INK)
            buttonTintList = ColorStateList.valueOf(MINT)
            minHeight = dp(48)
            isChecked = sameView
            isEnabled = a != null && b != null
            setOnCheckedChangeListener { _, checked ->
                sameView = checked
                render()
            }
        }
        addControl(controls, check)
        addControl(controls, label(problem ?: "민트색 A · 주황색 B", 14, color = if (problem == null) MUTED else ORANGE))
        addControl(controls, phase)
        addControl(controls, phaseLabel)
        addControl(controls, frameStatus)
        status(controls)
    }

    private fun recordEvent(sessionId: String, pitchId: String, event: Event, playing: Boolean, updated: (Pitch) -> Unit) {
        val time = currentPausedTime(playing) ?: return
        val session = sessions.firstOrNull { it.id == sessionId } ?: return
        val pitch = session.pitches.firstOrNull { it.id == pitchId } ?: return
        runCatching { markEvent(pitch, event, time) }.onSuccess { value ->
            saveUpdated(session.copy(pitches = session.pitches.map { if (it.id == value.id) value else it })) {
                updated(value)
                setNotice("표시를 저장했습니다.")
            }
        }.onFailure { setNotice(it.message.orEmpty()) }
    }

    private fun currentPausedTime(playing: Boolean): Long? {
        val current = replay?.positionUs
        if (playing || current == null) {
            setNotice("영상이 멈추고 프레임이 표시된 뒤 위치를 지정하세요.")
            return null
        }
        return current
    }

    private fun saveUpdated(session: Session, afterSave: () -> Unit) {
        if (saving) return
        saving = true
        updateNavigation()
        analysisWorker.execute {
            val result = runCatching { archive.save(session) }
            runOnUiThread {
                if (!isDestroyed) {
                    saving = false
                    result.onSuccess {
                        sessions = sessions.map { if (it.id == session.id) session else it }
                        afterSave()
                    }.onFailure {
                        Log.w("PitchReplay", "Could not update session", it)
                        setNotice("표시를 저장하지 못했습니다. 저장 공간을 확인하세요.")
                    }
                    updateNavigation()
                }
            }
        }
    }

    private fun addVideo(uri: Uri? = null, id: String = archive.newSessionId(), chosenHand: Hand = hand) {
        if (busy != null) return
        val startedAt = System.currentTimeMillis()
        busy = if (uri == null) "녹화 영상 분석 준비 중" else "원본 영상을 복사하는 중"
        if (!isDestroyed) render()
        val resolver = applicationContext.contentResolver
        analysisWorker.execute {
            val file = archive.source(id)
            val result = runCatching {
                if (uri != null) archive.copySource(resolver, uri, archive.prepare(id))
                val video = videoInfo(file)
                require(uri == null || video.durationUs <= MAX_VIDEO_MS * 1_000) { "90초 이하의 영상을 선택하세요." }
                val fingerprint = archive.fingerprint(file)
                val (frames, error) = captureAnalysis(file, video) { percent ->
                    runOnUiThread {
                        if (!isDestroyed) {
                            busy = "원본 영상에서 관절 분석 중 · $percent%"
                            statusText?.text = busy
                        }
                    }
                }
                val pitches = pitchCandidates(frames, chosenHand, video.width.toFloat() / video.height, video.durationUs)
                Session(id, startedAt, fingerprint, chosenHand, video.width, video.height, video.durationUs, frames, pitches, error)
                    .also { archive.save(it) }
            }.onFailure {
                Log.w("PitchReplay", "Could not archive video", it)
                discardDraft(file)
            }
            runOnUiThread {
                if (!isDestroyed) {
                    busy = null
                    result.onSuccess { session ->
                        sessions = listOf(session) + sessions.filter { it.id != session.id }
                        selectedSessionId = id
                        selectedPitchId = session.pitches.firstOrNull()?.id
                        manualStartUs = null
                        page = Page.LIBRARY
                        notice = if (session.pitches.isEmpty()) "자동 후보가 없습니다. 원본에서 투구 구간을 지정하세요."
                            else "투구 후보 ${session.pitches.size}개를 저장했습니다. 착지와 릴리스를 확인하세요."
                    }.onFailure {
                        notice = if (it is IllegalArgumentException && !it.message.isNullOrBlank()) it.message.orEmpty()
                        else "영상을 보관하지 못했습니다. 파일 형식과 저장 공간을 확인하세요."
                    }
                    render()
                }
            }
        }
    }

    private fun discardDraft(file: File) {
        file.delete()
        file.parentFile?.delete()
    }

    private fun selectSession() {
        if (saving) return
        AlertDialog.Builder(this).setTitle("원본 영상")
            .setItems(sessions.map(::sessionTitle).toTypedArray()) { _, index ->
                val session = sessions[index]
                selectedSessionId = session.id
                selectedPitchId = session.pitches.firstOrNull()?.id
                manualStartUs = null
                notice = ""
                render()
            }.setNegativeButton("닫기", null).show()
    }

    private fun selectPitch(session: Session) {
        if (saving) return
        AlertDialog.Builder(this).setTitle("투구 구간")
            .setItems(session.pitches.map { pitchTitle(session, it) + " · " + formatTime(it.startUs) }.toTypedArray()) { _, index ->
                selectedPitchId = session.pitches[index].id
                notice = ""
                render()
            }.setNegativeButton("닫기", null).show()
    }

    private fun chooseTake(first: Boolean) {
        val takes = sessions.flatMap { session -> session.pitches.map { Take(session, it) } }
        if (takes.isEmpty()) {
            setNotice("보관함에서 투구 구간을 먼저 저장하세요.")
            return
        }
        AlertDialog.Builder(this).setTitle(if (first) "기준 투구 A" else "비교 투구 B")
            .setItems(takes.map { pitchTitle(it.session, it.pitch) + " · " + sessionTitle(it.session) }.toTypedArray()) { _, index ->
                setTake(first, takes[index].pitch.id)
                render()
            }.setNegativeButton("닫기", null).show()
    }

    private fun setTake(first: Boolean, pitchId: String) {
        if (first) aId = pitchId else bId = pitchId
        sameView = false
        setNotice(if (first) "기준 투구 A로 선택했습니다." else "비교 투구 B로 선택했습니다.")
    }

    private fun take(id: String?): Take? {
        for (session in sessions) session.pitches.firstOrNull { it.id == id }?.let { return Take(session, it) }
        return null
    }

    private fun pitchTitle(session: Session, pitch: Pitch): String = "투구 ${session.pitches.indexOf(pitch) + 1}"

    private fun sessionTitle(session: Session): String {
        val date = Date(session.createdAtMs)
        return "${DateFormat.getMediumDateFormat(this).format(date)} ${DateFormat.getTimeFormat(this).format(date)} · ${session.hand.label}"
    }

    private fun formatTime(timeUs: Long) = String.format(Locale.ROOT, "%.3f초", timeUs / 1_000_000.0)

    private fun updateNavigation() {
        if (::navButtons.isInitialized) navButtons.forEach { it.isEnabled = !recording && busy == null && !saving }
    }

    private fun setNotice(message: String) {
        notice = message
        statusText?.text = message
    }

    private fun status(controls: LinearLayout) = label(busy ?: notice, 14, color = ORANGE).also {
        statusText = it
        it.accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        addControl(controls, it)
    }

    private fun screen(stage: View, controls: LinearLayout = column()): LinearLayout {
        workspace.addView(stage, LinearLayout.LayoutParams(0, -1, 1.55f))
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            addView(controls, ViewGroup.LayoutParams(-1, -2))
        }
        workspace.addView(scroll, LinearLayout.LayoutParams(0, -1, 1f).apply { marginStart = dp(12) })
        return controls
    }

    private fun emptyStage(message: String) = FrameLayout(this).apply {
        setBackgroundColor(Color.BLACK)
        addView(stageLabel(message), FrameLayout.LayoutParams(-2, -2, Gravity.CENTER))
    }

    private fun stageLabel(message: String) = label(message, 14).apply {
        setPadding(dp(8), dp(4), dp(8), dp(4))
        setBackgroundColor(0xC0101A18.toInt())
    }

    private fun addHeading(parent: LinearLayout, title: String, detail: String) {
        addControl(parent, label(title, 21, bold = true))
        addControl(parent, label(detail, 14, color = MUTED))
    }

    private fun addControl(parent: LinearLayout, view: View) {
        parent.addView(view, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(4) })
    }

    private fun addAction(parent: LinearLayout, title: String, primary: Boolean = false, run: () -> Unit) =
        button(title, primary, run).also { addControl(parent, it) }

    private fun addButtonRow(parent: LinearLayout, vararg buttons: Button) {
        val row = LinearLayout(this)
        for ((index, view) in buttons.withIndex()) {
            row.addView(view, LinearLayout.LayoutParams(0, -2, 1f).apply { if (index > 0) marginStart = dp(4) })
        }
        addControl(parent, row)
    }

    private fun slider(limit: Int, description: String, changed: (Int) -> Unit) = SeekBar(this).apply {
        max = limit
        minHeight = dp(48)
        contentDescription = description
        progressTintList = ColorStateList.valueOf(MINT)
        thumbTintList = ColorStateList.valueOf(MINT)
        setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(bar: SeekBar, progress: Int, fromUser: Boolean) {
                if (fromUser) changed(progress)
            }

            override fun onStartTrackingTouch(bar: SeekBar) = Unit
            override fun onStopTrackingTouch(bar: SeekBar) = Unit
        })
    }

    private fun column() = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

    private fun label(value: String, size: Int, bold: Boolean = false, color: Int = INK) = TextView(this).apply {
        text = value
        textSize = size.toFloat()
        setTextColor(color)
        if (bold) typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }

    private fun button(title: String, primary: Boolean = false, run: () -> Unit) = Button(this).apply {
        text = title
        textSize = 14f
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        isAllCaps = false
        minHeight = dp(48)
        minimumHeight = dp(48)
        minWidth = 0
        minimumWidth = 0
        setPadding(dp(8), 0, dp(8), 0)
        stateListAnimator = null
        setTextColor(ColorStateList(
            arrayOf(intArrayOf(android.R.attr.state_enabled), intArrayOf()),
            intArrayOf(if (primary) BACKGROUND else INK, if (primary) 0xFF465A50.toInt() else MUTED)
        ))
        background = RippleDrawable(ColorStateList.valueOf(0x224B6259), GradientDrawable().apply {
            setColor(if (primary) MINT else PANEL)
            cornerRadius = dp(8).toFloat()
        }, null)
        setOnClickListener { run() }
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density + .5f).toInt()
}
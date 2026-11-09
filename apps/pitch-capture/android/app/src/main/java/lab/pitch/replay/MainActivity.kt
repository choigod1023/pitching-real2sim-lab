package lab.pitch.replay

import android.Manifest
import android.content.DialogInterface
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.text.format.DateFormat
import android.util.Log
import android.view.Gravity
import android.view.Menu
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.R as MaterialR
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.google.android.material.button.MaterialButton
import com.google.android.material.checkbox.MaterialCheckBox
import com.google.android.material.color.MaterialColors
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.navigation.NavigationBarView
import com.google.android.material.navigationrail.NavigationRailView
import com.google.android.material.slider.LabelFormatter
import com.google.android.material.slider.Slider
import com.google.android.material.slider.TickVisibilityMode
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import com.google.android.material.textview.MaterialTextView
import java.io.File
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

private val analysisWorker = Executors.newSingleThreadExecutor()

class MainActivity : AppCompatActivity() {
    private enum class Page(val label: String, val icon: Int) {
        CAPTURE("촬영", R.drawable.ic_capture),
        LIBRARY("보관함", R.drawable.ic_library),
        COMPARE("겹쳐보기", R.drawable.ic_compare),
    }

    private lateinit var archive: Archive
    private lateinit var root: LinearLayout
    private lateinit var workspace: LinearLayout
    private lateinit var navigation: NavigationBarView
    private val ink get() = themeColor(MaterialR.attr.colorOnSurface)
    private val muted get() = themeColor(MaterialR.attr.colorOnSurfaceVariant)
    private val surface get() = themeColor(MaterialR.attr.colorSurface)
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
        root.setBackgroundColor(surface)
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val safe = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(safe.left, safe.top, safe.right, safe.bottom)
            WindowInsetsCompat.CONSUMED
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

        val window = resources.configuration
        val useRail = window.screenWidthDp >= 600 ||
            (window.screenWidthDp >= 480 && window.screenWidthDp > window.screenHeightDp)
        val sideBySide = window.screenWidthDp >= 600 || useRail && window.screenHeightDp < 480
        val toolbar = MaterialToolbar(this).apply {
            title = "Pitch Replay"
            minimumHeight = dp(56)
            setTitleTextColor(ink)
            setContentInsetsRelative(dp(16), dp(16))
        }
        root.addView(toolbar, LinearLayout.LayoutParams(-1, -2))

        navigation = if (useRail) {
            NavigationRailView(this).apply {
                setMenuGravity(Gravity.CENTER_VERTICAL)
                setCollapsedItemMinimumHeight(dp(64))
            }
        } else {
            BottomNavigationView(this)
        }
        navigation.setBackgroundColor(surface)
        navigation.labelVisibilityMode = NavigationBarView.LABEL_VISIBILITY_LABELED
        Page.entries.forEach { target ->
            navigation.menu.add(Menu.NONE, target.ordinal + 1, target.ordinal, target.label).setIcon(target.icon)
        }
        navigation.selectedItemId = page.ordinal + 1
        navigation.setOnItemSelectedListener { item ->
            go(Page.entries[item.itemId - 1])
            true
        }
        updateNavigation()

        val body = LinearLayout(this)
        root.addView(body, LinearLayout.LayoutParams(-1, 0, 1f))
        if (useRail) body.addView(navigation, LinearLayout.LayoutParams(-2, -1))
        workspace = LinearLayout(this).apply {
            orientation = if (sideBySide) LinearLayout.HORIZONTAL else LinearLayout.VERTICAL
            setPaddingRelative(dp(if (useRail) 4 else 16), dp(4), dp(16), dp(12))
        }
        body.addView(workspace, LinearLayout.LayoutParams(0, -1, 1f))
        if (!useRail) root.addView(navigation, LinearLayout.LayoutParams(-1, -2))
        when (page) {
            Page.CAPTURE -> renderCapture()
            Page.LIBRARY -> renderLibrary()
            Page.COMPARE -> renderCompare()
        }
    }

    private fun renderCapture() {
        val stage = FrameLayout(this)
        val preview = PreviewView(this)
        val skeleton = PoseOverlay(this)
        val flight = BallOverlay(this)
        val guide = stageLabel("후면 카메라 · 가로 촬영")
        stage.addView(preview, FrameLayout.LayoutParams(-1, -1))
        stage.addView(skeleton, FrameLayout.LayoutParams(-1, -1))
        stage.addView(flight, FrameLayout.LayoutParams(-1, -1))
        stage.addView(guide, FrameLayout.LayoutParams(-2, -2, Gravity.TOP or Gravity.START).apply {
            topMargin = dp(12)
            marginStart = dp(12)
        })
        val center = stageLabel("카메라를 켜고 구도를 확인하세요.").apply { gravity = Gravity.CENTER }
        stage.addView(center, FrameLayout.LayoutParams(-2, -2, Gravity.CENTER))
        val live = stageLabel("").apply { visibility = View.GONE }
        stage.addView(live, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.START).apply {
            bottomMargin = dp(12)
            marginStart = dp(12)
        })
        val controls = screen(stage)
        addHeading(controls, "투구 촬영", "휴대폰을 고정하고 투수 전신과 공의 경로를 담으세요.")
        var ready = false
        var picking = false
        var ballStart: BallPoint? = null
        var ballEnd: BallPoint? = null
        var distance: Double? = null
        var trace: BallReadout? = null
        val primary = button("카메라 켜기", primary = true) {}
        val calibrate = button("공 추적 설정") {}
        addButtonRow(controls, primary, calibrate)
        val ballStatus = label("", 14, color = muted)
        addControl(controls, ballStatus)
        status(controls)
        val optical = column().apply {
            setPadding(dp(12), dp(4), dp(12), dp(4))
            background = rounded(themeColor(MaterialR.attr.colorSurfaceContainerLow), 12)
        }
        val speedValue = addMetric(optical, "화면 평면 평균 구속")
        val movementValue = addMetric(optical, "화면 휨")
        addMetric(optical, "회전수 (RPM)").text = "측정 불가"
        addControl(controls, optical)
        val handButton = addAction(controls, "투구 손 · ${hand.label}") {}
        handButton.setOnClickListener {
            MaterialAlertDialogBuilder(this).setTitle("투구 손")
                .setSingleChoiceItems(Hand.entries.map { it.label }.toTypedArray(), hand.ordinal) { dialog, index ->
                    hand = Hand.entries[index]
                    handButton.text = "투구 손 · ${hand.label}"
                    dialog.dismiss()
                }.setNegativeButton("닫기", null).show()
        }
        val import = addAction(controls, "영상 가져오기") { openDocument.launch(arrayOf("video/*")) }

        fun updateBall() {
            flight.showSetup(ballStart, ballEnd)
            flight.showReadout(trace)
            calibrate.text = when {
                picking -> "설정 취소"
                ballEnd != null -> "공 추적 재설정"
                else -> "공 추적 설정"
            }
            ballStatus.text = when {
                picking && ballStart == null -> "공이 날아가는 구간의 시작점을 화면에서 누르세요."
                picking -> "공이 날아가는 방향으로 떨어진 끝점을 누르세요."
                ballEnd == null -> "공 추적 설정에서 공이 지나갈 구간을 지정하세요."
                trace != null -> "${if (trace?.active == true) "추적 중" else "최근 궤적"} · 관측 ${trace?.points?.size}점"
                else -> "공 추적 대기 · 손이 화면 구간 밖에 오게 맞추세요."
            }
            val speed = trace?.speedKmh?.let { String.format(Locale.KOREA, "%.1f km/h", it) }
            val movement = trace?.deviationCm?.let { String.format(Locale.KOREA, "%+.1f cm", it) }
                ?: trace?.deviationPx?.let { String.format(Locale.KOREA, "%+.1f px", it) }
            speedValue.text = speed ?: if (distance == null) "거리 보정 필요" else "관측 대기"
            movementValue.text = movement ?: "관측 대기"
        }

        fun clearBall() {
            picking = false
            ballStart = null
            ballEnd = null
            distance = null
            trace = null
            updateBall()
        }

        fun updateButtons() {
            guide.setTextColor(if (recording) themeColor(MaterialR.attr.colorOnErrorContainer) else Color.WHITE)
            guide.background = rounded(
                if (recording) themeColor(MaterialR.attr.colorErrorContainer) else 0xEB141B18.toInt(),
                8,
            )
            primary.text = when {
                stopping -> "녹화 마무리 중"
                recording -> "녹화 끝내기"
                cameraRequested -> "녹화 시작"
                else -> "카메라 켜기"
            }
            primary.isEnabled = busy == null && !stopping && (recording || !cameraRequested || ready)
            calibrate.isEnabled = ready && !recording && busy == null
            handButton.isEnabled = !recording && busy == null
            import.isEnabled = !recording && busy == null
            updateNavigation()
        }

        calibrate.setOnClickListener {
            val startPicking = !picking
            clearBall()
            picking = startPicking
            camera?.setBallSetup(null)
            updateBall()
        }

        fun markBall(point: BallPoint) {
            if (camera?.canMark(point) != true) {
                ballStatus.text = "영상이 보이는 영역 안에서 점을 선택하세요."
                return
            }
            val start = ballStart
            if (start == null) {
                ballStart = point
                updateBall()
                return
            }
            if ((point - start).length < dp(56)) {
                ballStatus.text = "시작점에서 더 떨어진 끝점을 선택하세요."
                return
            }
            ballEnd = point
            picking = false
            camera?.setBallSetup(BallSetup(start, point, null))
            updateBall()
            askDistance { meters ->
                if (ready && ballStart == start && ballEnd == point) {
                    distance = meters
                    camera?.setBallSetup(BallSetup(start, point, meters))
                    updateBall()
                }
            }
        }

        preview.setOnTouchListener { _, event ->
            if (!picking) false
            else {
                if (event.actionMasked == MotionEvent.ACTION_UP) {
                    preview.performClick()
                    markBall(BallPoint(event.x, event.y))
                }
                true
            }
        }

        primary.setOnClickListener {
            when {
                recording -> {
                    stopping = true
                    camera?.stop()
                    updateButtons()
                }
                cameraRequested && ready -> {
                    if (picking) {
                        clearBall()
                        camera?.setBallSetup(null)
                    }
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
        updateBall()
        updateButtons()

        if (cameraRequested && busy == null) {
            center.text = "후면 카메라를 여는 중"
            camera = CameraCapture(this, preview, poseChanged = {
                skeleton.show(it to MINT)
                live.text = if (it == null) "전신이 보이도록 구도를 맞추세요." else "관절이 화면에 잡혔습니다."
            }, ballChanged = {
                trace = it
                updateBall()
            }, geometryChanged = {
                clearBall()
            }, readyChanged = {
                ready = it
                center.visibility = if (it) View.GONE else View.VISIBLE
                live.visibility = if (it) View.VISIBLE else View.GONE
                updateButtons()
            }, failed = {
                center.text = "후면 카메라를 열지 못했습니다."
                setNotice(it)
            }).also { it.open() }
        }
    }

    private fun askDistance(applyDistance: (Double) -> Unit) {
        val distanceField = TextInputLayout(this, null, MaterialR.attr.textInputOutlinedStyle).apply {
            hint = "두 점 사이 거리 (m)"
        }
        val input = TextInputEditText(distanceField.context).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setSingleLine()
        }
        distanceField.addView(input, LinearLayout.LayoutParams(-1, -2))
        val field = column().apply {
            setPadding(dp(24), 0, dp(24), 0)
            addView(distanceField, LinearLayout.LayoutParams(-1, -2))
        }
        val dialog = MaterialAlertDialogBuilder(this)
            .setTitle("거리 보정")
            .setMessage("공이 지나는 평면에서 선택한 두 점 사이의 실측 거리를 입력하세요.")
            .setView(field)
            .setNegativeButton("궤적만 보기", null)
            .setPositiveButton("적용", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(DialogInterface.BUTTON_POSITIVE).setOnClickListener {
                val distance = input.text.toString().trim().replace(',', '.').toDoubleOrNull()
                if (distance == null || distance !in 0.3..35.0) {
                    distanceField.error = "0.3~35 m 사이의 거리를 입력하세요."
                } else {
                    applyDistance(distance)
                    dialog.dismiss()
                }
            }
        }
        dialog.show()
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
        val time = label("영상을 여는 중", 16, bold = true).apply { fontFeatureSettings = "tnum" }
        val seekBar = slider(1_000, "영상 위치") { progress ->
            replay?.seek(start + (end - start) * progress / 1_000)
        }
        val play = button("재생", primary = true) { replay?.toggle() }
        val marks = label("", 14, color = muted)
        val viewer = ReplayView(this, archive.source(session.id), session.aspect, start, end, timeChanged = { current, running ->
            playing = running
            time.text = current?.let { "${formatTime(it)} / ${formatTime(end)}" } ?: "프레임을 찾는 중"
            play.text = if (running) "멈춤" else "재생"
            play.isEnabled = current != null
            if (current != null) seekBar.value = ((current - start) * 1_000 / (end - start)).coerceIn(0L, 1_000L).toFloat()
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
                button("끝 지정") {
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
                button("릴리스 확인") { recordEvent(session.id, pitch.id, Event.RELEASE, playing, ::showMarks) }
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
        session.analysisError?.let { addControl(controls, label(it, 14, color = muted)) }
    }

    private fun renderCompare() {
        val a = take(aId)
        val b = take(bId)
        val problem = comparisonError(a, b, sameView)
        val controls = column()
        val phaseLabel = label("착지 0% · 릴리스 100%", 14, color = muted).apply { fontFeatureSettings = "tnum" }
        val frameStatus = label("", 14, color = muted)
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
                    val percent = (fraction * 100).toInt().coerceIn(0, 100)
                    phase.value = percent.toFloat()
                    val aFrame = nearestPose(a.session.frames, current)
                    val bFrame = nearestPose(b.session.frames, phaseTime(b.pitch, fraction))
                    val both = aFrame != null && bFrame != null
                    replay?.overlay?.show(aFrame?.takeIf { both } to MINT, bFrame?.takeIf { both } to ORANGE)
                    frameStatus.text = if (both) "" else "이 시점에는 두 관절선을 함께 표시할 수 없습니다."
                    phaseLabel.text = "${percent}% · A ${formatTime(current)} · B ${formatTime(phaseTime(b.pitch, fraction))}"
                } else if (problem == null) {
                    frameStatus.text = "프레임을 찾는 중"
                }
            }, failed = ::setNotice).also { replay = it }
        }
        screen(stage, controls)
        addHeading(controls, "두 투구 겹쳐보기", "A 영상 위에 같은 진행률의 관절선을 표시합니다.")
        addAction(controls, "A · ${a?.let { pitchTitle(it.session, it.pitch) } ?: "기준 투구 선택"}") { chooseTake(true) }
        addAction(controls, "B · ${b?.let { pitchTitle(it.session, it.pitch) } ?: "비교 투구 선택"}") { chooseTake(false) }
        val check = MaterialCheckBox(this).apply {
            text = "같은 선수와 고정된 시야를 두 원본에서 확인했습니다."
            minHeight = dp(48)
            isChecked = sameView
            isEnabled = a != null && b != null
            setOnCheckedChangeListener { _, checked ->
                sameView = checked
                render()
            }
        }
        addControl(controls, check)
        addControl(controls, label(problem ?: "민트색 A · 주황색 B", 14, color = muted))
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
                            updateStatus()
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
        MaterialAlertDialogBuilder(this).setTitle("원본 영상")
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
        MaterialAlertDialogBuilder(this).setTitle("투구 구간")
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
        MaterialAlertDialogBuilder(this).setTitle(if (first) "기준 투구 A" else "비교 투구 B")
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
        val enabled = !recording && busy == null && !saving
        val menu = navigation.menu
        for (index in 0 until menu.size()) menu.getItem(index).isEnabled = enabled
    }

    private fun setNotice(message: String) {
        notice = message
        updateStatus()
    }

    private fun updateStatus() {
        val message = busy ?: notice
        statusText?.apply {
            text = message
            visibility = if (message.isBlank()) View.GONE else View.VISIBLE
        }
    }

    private fun status(controls: LinearLayout) = label("", 14).also {
        statusText = it
        it.accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        it.setPadding(dp(12), dp(8), dp(12), dp(8))
        it.background = rounded(themeColor(MaterialR.attr.colorSurfaceContainerHigh), 8)
        addControl(controls, it)
        updateStatus()
    }

    private fun screen(stage: View, controls: LinearLayout = column()): LinearLayout {
        stage.background = rounded(Color.BLACK, 12)
        stage.clipToOutline = true
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            isVerticalScrollBarEnabled = false
            addView(controls, ViewGroup.LayoutParams(-1, -2))
        }
        val window = resources.configuration
        if (workspace.orientation == LinearLayout.HORIZONTAL) {
            val controlsWidth = (window.screenWidthDp / 3).coerceIn(192, 360)
            workspace.addView(stage, LinearLayout.LayoutParams(0, -1, 1f))
            workspace.addView(scroll, LinearLayout.LayoutParams(dp(controlsWidth), -1).apply { marginStart = dp(16) })
        } else {
            val stageHeight = minOf((window.screenWidthDp - 32) * 9 / 16, window.screenHeightDp / 3)
            workspace.addView(stage, LinearLayout.LayoutParams(-1, dp(stageHeight)))
            workspace.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f).apply { topMargin = dp(12) })
        }
        return controls
    }

    private fun emptyStage(message: String) = FrameLayout(this).apply {
        addView(stageLabel(message).apply { gravity = Gravity.CENTER },
            FrameLayout.LayoutParams(-2, -2, Gravity.CENTER))
    }

    private fun stageLabel(message: String) = label(message, 14, color = Color.WHITE).apply {
        setPadding(dp(8), dp(4), dp(8), dp(4))
        background = rounded(0xEB141B18.toInt(), 8)
        fontFeatureSettings = "tnum"
    }

    private fun addHeading(parent: LinearLayout, title: String, detail: String) {
        addControl(parent, label(title, 22, bold = true).apply {
            ViewCompat.setAccessibilityHeading(this, true)
        }, gap = 4)
        addControl(parent, label(detail, 14, color = muted), gap = 12)
    }

    private fun addControl(parent: LinearLayout, view: View, gap: Int = 8) {
        parent.addView(view, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(gap) })
    }

    private fun addMetric(parent: LinearLayout, title: String): TextView {
        val metric = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(4), 0, dp(4))
        }
        metric.addView(label(title, 14, color = muted), LinearLayout.LayoutParams(0, -2, 1f))
        val value = label("", 14, bold = true).apply {
            gravity = Gravity.END
            fontFeatureSettings = "tnum"
        }
        metric.addView(value, LinearLayout.LayoutParams(0, -2, 1f).apply { marginStart = dp(8) })
        parent.addView(metric, LinearLayout.LayoutParams(-1, -2))
        return value
    }

    private fun addAction(parent: LinearLayout, title: String, primary: Boolean = false, run: () -> Unit) =
        button(title, primary, run).also { addControl(parent, it) }

    private fun addButtonRow(parent: LinearLayout, vararg buttons: Button) {
        if (resources.configuration.fontScale >= 1.4f) {
            buttons.forEach { addControl(parent, it) }
            return
        }
        val row = LinearLayout(this)
        for ((index, view) in buttons.withIndex()) {
            row.addView(view, LinearLayout.LayoutParams(0, -2, 1f).apply { if (index > 0) marginStart = dp(4) })
        }
        addControl(parent, row)
    }

    private fun slider(limit: Int, description: String, changed: (Int) -> Unit) = Slider(this).apply {
        valueTo = limit.toFloat()
        stepSize = 1f
        tickVisibilityMode = TickVisibilityMode.TICK_VISIBILITY_HIDDEN
        labelBehavior = LabelFormatter.LABEL_GONE
        minimumHeight = dp(48)
        contentDescription = description
        addOnChangeListener { _, value, fromUser -> if (fromUser) changed(value.toInt()) }
    }

    private fun column() = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

    private fun label(value: String, size: Int, bold: Boolean = false, color: Int = ink) = MaterialTextView(this).apply {
        text = value
        textSize = size.toFloat()
        setTextColor(color)
        if (bold) typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }

    private fun button(title: String, primary: Boolean = false, run: () -> Unit) = MaterialButton(
        this, null, if (primary) MaterialR.attr.materialButtonStyle else MaterialR.attr.materialButtonOutlinedStyle,
    ).apply {
        text = title
        isAllCaps = false
        minHeight = dp(48)
        minimumHeight = dp(48)
        minWidth = dp(48)
        minimumWidth = dp(48)
        setPaddingRelative(dp(8), paddingTop, dp(8), paddingBottom)
        setOnClickListener { run() }
    }

    private fun rounded(color: Int, radius: Int) = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dp(radius).toFloat()
    }

    private fun themeColor(attribute: Int) = MaterialColors.getColor(this, attribute, "Pitch Replay")

    private fun dp(value: Int) = (value * resources.displayMetrics.density + .5f).toInt()
}

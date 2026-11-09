package lab.pitch.replay

import android.content.Context
import android.graphics.Canvas
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.view.View

internal class BallOverlay(context: Context) : View(context) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val density = resources.displayMetrics.density
    private val dash = DashPathEffect(floatArrayOf(4f * density, 4f * density), 0f)
    private var start: BallPoint? = null
    private var end: BallPoint? = null
    private var readout: BallReadout? = null

    init {
        importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
    }

    fun showSetup(start: BallPoint?, end: BallPoint?) {
        this.start = start
        this.end = end
        invalidate()
    }

    fun showReadout(value: BallReadout?) {
        readout = value
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val first = start
        val last = end
        if (first != null) {
            paint.color = 0xC9F1F5F2.toInt()
            paint.strokeWidth = density
            if (last != null) {
                val axis = last - first
                val sideX = -axis.y * 0.12f
                val sideY = axis.x * 0.12f
                paint.color = 0x77F1F5F2.toInt()
                canvas.drawLine(first.x + sideX, first.y + sideY, last.x + sideX, last.y + sideY, paint)
                canvas.drawLine(first.x - sideX, first.y - sideY, last.x - sideX, last.y - sideY, paint)
                paint.color = 0xC9F1F5F2.toInt()
                paint.pathEffect = dash
                canvas.drawLine(first.x, first.y, last.x, last.y, paint)
                paint.pathEffect = null
                val x = axis.x / axis.length * 8f * density
                val y = axis.y / axis.length * 8f * density
                canvas.drawLine(last.x, last.y, last.x - x - y / 2, last.y - y + x / 2, paint)
                canvas.drawLine(last.x, last.y, last.x - x + y / 2, last.y - y - x / 2, paint)
            }
            canvas.drawCircle(first.x, first.y, 3f * density, paint)
        }
        val trace = readout ?: return
        paint.strokeCap = Paint.Cap.ROUND
        for ((index, point) in trace.points.withIndex()) {
            if (index > 0) {
                val before = trace.points[index - 1]
                paint.color = 0xCC000000.toInt()
                paint.strokeWidth = 5f * density
                canvas.drawLine(before.x, before.y, point.x, point.y, paint)
                paint.color = ORANGE
                paint.strokeWidth = 2f * density
                canvas.drawLine(before.x, before.y, point.x, point.y, paint)
            }
            paint.color = ORANGE
            canvas.drawCircle(point.x, point.y, (if (index == trace.points.lastIndex) 4f else 2f) * density, paint)
        }
    }
}

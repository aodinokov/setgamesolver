/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.content.res.TypedArray
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList
import kotlin.math.max
import java.io.InputStream

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var showDetections: Boolean = false
    private var results: List<Detected> = LinkedList<Detected>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var groupBoxPaintMap = HashMap<Int, Paint>()
    private var thumbnailsBitmap: Bitmap? = null

    private var scaleFactor: Float = 1f

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear(showDetectionVal: Boolean) {
        showDetections = showDetectionVal

        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE

        // init colors for groups
        groupBoxPaintMap.clear()
        val colors: TypedArray = resources.obtainTypedArray(R.array.groupColors)
        for (i in 0 until colors.length()) {
            var p = Paint()
            p.color = colors.getColor(i, 0)
            p.strokeWidth = boxPaint.strokeWidth
            p.style = boxPaint.style
            groupBoxPaintMap[i] = p
        }

        // init thumbnail Bitmap
        if (thumbnailsBitmap == null) {
            val inputStream = resources.assets.open("setgame-cards.png")
            thumbnailsBitmap = BitmapFactory.decodeStream(inputStream)
        }

    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        if (!showDetections)
            return

        for (result in results) {
            val boundingBox = result.getBoundingBox()

            val top = boundingBox.top * scaleFactor
            val bottom = boundingBox.bottom * scaleFactor
            val left = boundingBox.left * scaleFactor
            val right = boundingBox.right * scaleFactor

            // Draw bounding box around detected objects
            val drawableRect = RectF(left, top, right, bottom)
            if (result is Grouppable) {
                for(groupId in result.getGroupIds()) {
                    assert(groupBoxPaintMap.size > 0)
                    val pb = groupBoxPaintMap.get(groupId%groupBoxPaintMap.size)!!
                    canvas.drawRect(drawableRect, pb)
                    // make all groups visible
                    drawableRect.left -= pb.strokeWidth
                    drawableRect.top -= pb.strokeWidth
                    drawableRect.bottom += pb.strokeWidth
                    drawableRect.right += pb.strokeWidth
                }
            }else{
                canvas.drawRect(drawableRect, boxPaint)
            }

            // Create text to display alongside detected objects
            var shiftX = 0
            var label = result.getCategories()[0].label + " "
            val crd = cardFromString(result.getCategories()[0].label)
            if (crd != null && thumbnailsBitmap != null) {
                // don't need label - we'll draw a picture instead
                label = ""
                shiftX = thumbnailsBitmap!!.width/9 // we have put 9 cards in the row

                val idx = (((crd.fill.code-1)*3+(crd.shape.code-1))*3 + (crd.color.code-1))*3 + (crd.count-1)
                assert(idx >=0 && idx < 81)

                val column = idx % 9
                val row = idx / 9

                val src = Rect(
                        thumbnailsBitmap!!.width/9 *column,
                        thumbnailsBitmap!!.height/9*row,
                        thumbnailsBitmap!!.width/9 *(column+1),
                        thumbnailsBitmap!!.height/9*(row+1))
                val dst = RectF(
                        left,
                        top,
                        left + thumbnailsBitmap!!.width/9,
                        top + thumbnailsBitmap!!.height/9)

                canvas.drawBitmap(thumbnailsBitmap!!,
                        src,
                        dst,
                        textBackgroundPaint
                )
            }

            val drawableText = label + String.format("%.2f", result.getCategories()[0].score)

            // Draw rect behind display text
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left + shiftX,
                top,
                left + shiftX + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text for detected object
            canvas.drawText(drawableText, left+ shiftX, top + bounds.height(), textPaint)
        }
    }

    fun setResults(
      detectionResults: MutableList<Detected>,
      imageHeight: Int,
      imageWidth: Int,
    ) {
        results = detectionResults

        // PreviewView is in FILL_START mode. So we need to scale up the bounding box to match with
        // the size that the captured images will be displayed.
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}

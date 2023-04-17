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
import org.tensorflow.lite.task.vision.detector.Detection
import kotlin.math.absoluteValue

open class DetectedCard(name: String, bounds: RectF) {
    public var detectedName = name
    public var bounds = bounds
}

class ViewCard(name: String, bounds: RectF): DetectedCard(name, bounds) {
    public var overriddenName: String? = null

    // groups defines the color it will be shown
    public var groups = HashSet<Int>()

    fun name(): String{
        if (overriddenName != null)
            return overriddenName.toString()
        return detectedName
    }

    fun updateAttmept(newDetectedName: String, newBounds: RectF): Boolean {
        // check if new bounds are in intersect with the arg
        if ((bounds.centerX() - newBounds.centerX()).absoluteValue < bounds.width()/2 &&
            (bounds.centerY() - newBounds.centerY()).absoluteValue < bounds.height()/2) {

                detectedName = newDetectedName
                bounds = newBounds
                return true
        }
        return false
    }
}

class ViewData() {
    private var nonOverlappingSolutionMode = false
    public var cards = LinkedList<ViewCard>()
    public var resultingGroupsCount = 0

    fun reset() {
        // clean the list
        cards = LinkedList<ViewCard>()
    }
    fun setNonOverlappingSolutionMode(mode: Boolean) {
        nonOverlappingSolutionMode = mode
        //recalc groups
        findSets()
    }
    fun updateList(newList: List<DetectedCard>) {
        // try to track the same cards
        var prevCards = cards
        cards = LinkedList<ViewCard>()

        outer@for (newCard in newList) {
            for (card in prevCards) {
                if (card.updateAttmept(newCard.detectedName, newCard.bounds)) {
                    prevCards.remove(card)
                    cards.add(card)
                    break@outer
                }
            }
            //new card didn't find any matching card - add a new one
            cards.add(ViewCard(newCard.detectedName, newCard.bounds))
        }
    }
    fun findSets(): Boolean {
        var vCardsByName = HashMap<Card,ViewCard>()
        var inSet = HashSet<Card>()
        // store all cards to set
        for (vCard in cards) {
            var card = cardFromString(vCard.name())
            if (card == null)
                return false

            inSet.add(card)
            vCardsByName.put(card, vCard)
        }

        var solutions = findAllSolutions(inSet)

        // clean groups
        for (vCard in cards) {
            vCard.groups.clear()
        }

        // mode 1
        if (!nonOverlappingSolutionMode) {
            // TODO: how to keep the same group from scan to scan?
            var groupId = 0
            for (g in solutions) {
                for (c in g.cards) {
                    //find corresponding vCard
                    var vCard = vCardsByName.get(c)
                    assert(vCard != null)
                    if (vCard != null) {
                        vCard.groups.add(groupId)
                    }
                }
                // next id
                groupId++
            }
            resultingGroupsCount = groupId
            return true
        }
        // mode 2
        var groupId = 0
        for (ss in findAllNonOverlappingSolutions(solutions)) {
            // for each solutionset
            for (s in ss) {
                for (c in s.cards) {
                    //find corresponding vCard
                    var vCard = vCardsByName.get(c)
                    assert(vCard != null)
                    if (vCard != null) {
                        vCard.groups.add(groupId)
                    }
                }
            }
            // next id
            groupId++
        }
        resultingGroupsCount = groupId
        return true
    }
}

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    // TODO: switch newResults when ready
    private var results: List<Detection> = LinkedList<Detection>()
    private var newResults: ViewData = ViewData()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var scaleFactor: Float = 1f

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
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
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        for (result in results) {
            val boundingBox = result.boundingBox

            val top = boundingBox.top * scaleFactor
            val bottom = boundingBox.bottom * scaleFactor
            val left = boundingBox.left * scaleFactor
            val right = boundingBox.right * scaleFactor

            // Draw bounding box around detected objects
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text to display alongside detected objects
            val drawableText =
                result.categories[0].label + " " +
                        String.format("%.2f", result.categories[0].score)

            // Draw rect behind display text
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text for detected object
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }

    fun setResults(
      detectionResults: MutableList<Detection>,
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

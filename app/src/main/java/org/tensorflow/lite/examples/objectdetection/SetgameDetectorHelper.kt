/*
 * Copyright 2022 Alexey Odinokov. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.util.*
import kotlin.math.absoluteValue

/*WA: it was necessary to create our own copy, because we couldn't inherit from Detection */
abstract class Detected() {
    abstract fun getBoundingBox(): RectF
    abstract fun getCategories(): MutableList<Category>
}

class  DetectedImpl(var x: Detection): Detected() {
    var bounds = x.boundingBox
    var cats = x.categories?: LinkedList<Category>()
    override fun getBoundingBox(): RectF {
        return bounds
    }
    override fun getCategories(): MutableList<Category> {
        return cats
    }
}

/* new interface to extend Detection to draw if detection belongs to several sets */
abstract class Grouppable: Detected() {
    abstract fun getGroupIds(): Set<Int>
}


class ViewCard(var x: Detected):  Grouppable(){
    public var overriddenName: String? = null
    public var detectedName = x.getCategories()[0].label
    public var bounds = x.getBoundingBox()
    var cats = x.getCategories()?: LinkedList<Category>()

    // groups defines the color it will be shown
    public var groups = HashSet<Int>()

    fun name(): String{
        if (overriddenName != null)
            return overriddenName.toString()
        return detectedName
    }

    fun updateAttmept(x: Detected): Boolean {
        val newDetectedName = x.getCategories()[0].label
        val newBounds = x.getBoundingBox()
        // check if new bounds are in intersect with the arg
        if ((bounds.centerX() - newBounds.centerX()).absoluteValue < bounds.width()/2 &&
            (bounds.centerY() - newBounds.centerY()).absoluteValue < bounds.height()/2) {

            detectedName = newDetectedName
            bounds = newBounds
            return true
        }
        return false
    }

    // Grouppable interface impl
    override fun getGroupIds(): Set<Int> {
        return groups
    }

    // Detection interface impl
    override fun getBoundingBox(): RectF {
        return bounds
    }

    override fun getCategories(): MutableList<Category> {
        return cats
    }
}

class SetgameDetectorHelper(
    var scanEnabled: Boolean = false,
    var nonOverlappingSolutionMode: Boolean = false,
    var threshold: Float = 0.5f,
    var numThreads: Int = 2,
    var maxResults: Int = 20,
    var currentDelegate: Int = 0,
    var currentModel: Int = 0,
    val context: Context,
    val objectDetectorListener: DetectorListener?
) {

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private var objectDetector: ObjectDetector? = null

    // we're keeping cards so the they can be overridden
    private var cards = LinkedList<ViewCard>()

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        objectDetector = null
    }

    fun clearCards() {
        // clean the list
        cards.clear()
    }

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    fun setupObjectDetector() {
        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    objectDetectorListener?.onError("GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName =
            when (currentModel) {
                MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
                MODEL_SETGAME -> "setgame-detect.tflite"
                MODEL_SETGAME_DBG -> "setgame-detect.tflite"
                else -> "mobilenetv1.tflite"
            }

        try {
            objectDetector =
                ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            objectDetectorListener?.onError(
                "Object detector failed to initialize. See error logs for details"
            )
            Log.e("Test", "TFLite failed to load model with error: " + e.message)
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) {
            setupObjectDetector()
        }

        if (!scanEnabled) {
            return
        }
        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        val imageProcessor =
            ImageProcessor.Builder()
                .add(Rot90Op(-imageRotation / 90))
                .build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val results = setgameResults(image, imageRotation, objectDetector?.detect(tensorImage))
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        objectDetectorListener?.onResults(
            results,
            inferenceTime,
            tensorImage.height,
            tensorImage.width)
    }

    // setgame specific functions
    fun setgameResults(
        image: Bitmap,
        imageRotation: Int,
        results: MutableList<Detection>?): MutableList<Detected> {
        var res = LinkedList<Detected>()
        if (results != null) {
            for (r in results) {
                res.add(DetectedImpl(r))
            }
        }
        if (currentModel != MODEL_SETGAME || results == null)
            return res

        // more complex calculations are needed for setgame
        // TBD: here we need to add some heuristic to validate what exactly
        // cards are located by detector - this can be done with
        // image and imageRotation

        // ...

        // res must contain ALL found cards on the Bitmap
        updateList(res)
        res.clear()
        if (findSets() != true)
            return res

        for (r in cards) {
            res.add(r)
        }

        return res
    }

    fun updateList(results: MutableList<Detected>) {
        // try to track the same cards
        var prevCards = cards
        cards = LinkedList<ViewCard>()

        outer@for (newCard in results) {
            for (card in prevCards) {
                if (card.updateAttmept(newCard)) {
                    prevCards.remove(card)
                    cards.add(card)
                    continue@outer
                }
            }
            //new card didn't find any matching card - add a new one
            cards.add(ViewCard(newCard))
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
        return true
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
            results: MutableList<Detected>?,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_SETGAME = 0
        const val MODEL_SETGAME_DBG = 1
        const val MODEL_MOBILENETV1 = 2
    }
}

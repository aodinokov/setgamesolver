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
import android.view.Surface
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.util.*
import kotlin.math.absoluteValue
import kotlin.math.max

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
    //public var detectedName = x.getCategories()[0].label
    public var classifiedName = ""
    public var classifiedScore: Float = 0.0F

    public var bounds = x.getBoundingBox()
    public var prevBounds: RectF? = null

    // keep what came from Detected
    public var cats = x.getCategories()?: LinkedList<Category>()

    // groups defines the color it will be shown
    public var groups = HashSet<Int>()

    // timers
    public var detectedTime = SystemClock.uptimeMillis()
    public var classifiedTime = SystemClock.uptimeMillis()

    fun name(): String{
        if (overriddenName != null)
            return overriddenName.toString()
        return classifiedName
    }

    fun isWithinBorders(x: Detected): Boolean {
        val newBounds = x.getBoundingBox()
        // check if new bounds are in intersect with the arg
        if ((bounds.centerX() - newBounds.centerX()).absoluteValue < bounds.width()/2 &&
            (bounds.centerY() - newBounds.centerY()).absoluteValue < bounds.height()/2) {
            return true
        }
        return false
    }

    fun updateBorders(x: Detected) {
        prevBounds = bounds
        bounds = x.getBoundingBox()
    }

    // Work with live time
    fun markReDetected() {
        detectedTime = SystemClock.uptimeMillis()
    }

    fun isOutdated():Boolean {
        // if time when it was last time re-detected is more then const (e.g. 2sec)
        return false
    }

    fun updateClassification(newName: String, newScore: Float) {
        classifiedName = newName
        classifiedScore = newScore
        markReClassified()
    }
    fun markReClassified() {
        classifiedTime = SystemClock.uptimeMillis()
    }

    fun isReClassifyCandidate(): Boolean {
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
    private var imageClassifier: ImageClassifier? = null

    // we're keeping cards so the they can be traced and overridden
    private var cards = LinkedList<ViewCard>()

    init {
        setupObjectDetector()
        setupImageClassifier()
    }

    fun clearObjectDetector() {
        objectDetector = null
        imageClassifier = null
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
                    objectDetectorListener?.onError("Detector creations: GPU is not supported on this device")
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

    private fun setupImageClassifier() {
        val optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
                .setScoreThreshold(0.1f/*threshold*/)
                .setMaxResults(maxResults)

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    objectDetectorListener?.onError("Classifier creation: GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName =
                when (currentModel) {
                    MODEL_SETGAME -> "setgame-classify.tflite"
                    else -> "setgame-classify.tflite"
                }

        try {
            imageClassifier =
                    ImageClassifier.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            objectDetectorListener?.onError(
                    "Classifier creation:Image classifier failed to initialize. See error logs for details"
            )
            Log.e("Test", "TFLite failed to load model with error: " + e.message)
        }
    }

    // Receive the device rotation (Surface.x values range from 0->3) and return EXIF orientation
    // http://jpegclub.org/exif_orientation.html
    private fun getOrientationFromRotation(rotation: Int) : ImageProcessingOptions.Orientation {
        when (rotation) {
            Surface.ROTATION_270 ->
                return ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            Surface.ROTATION_180 ->
                return ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            Surface.ROTATION_90 ->
                return ImageProcessingOptions.Orientation.TOP_LEFT
            else ->
                return ImageProcessingOptions.Orientation.RIGHT_TOP
        }
    }
    fun classifyImage(image: Bitmap, rotation: Int): List<Classifications>? {
        if (imageClassifier == null) {
            setupImageClassifier()
        }
        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        val imageProcessor =
                ImageProcessor.Builder()
                        .build()

        // Preprocess the image and convert it into a TensorImage for classification.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val imageProcessingOptions = ImageProcessingOptions.builder()
                .setOrientation(getOrientationFromRotation(rotation))
                .build()

        return imageClassifier?.classify(tensorImage, imageProcessingOptions)
    }
    fun detect(image: Bitmap, imageRotation: Int) {
        // ensure all tools are ready
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
                //.add(Rot90Op(-imageRotation / 90))
                .build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
        val scaleFactor = max(image.width * 1f / tensorImage.width,  image.height * 1f / tensorImage.height)

        val results = setgameResults(image, imageRotation, scaleFactor, objectDetector?.detect(tensorImage))
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
        scaleFactor: Float,
        results: MutableList<Detection>?): MutableList<Detected> {
        var res = LinkedList<Detected>()
        if (results != null) {
            for (r in results) {
                res.add(DetectedImpl(r))
            }
        }
        if (currentModel != MODEL_SETGAME || results == null)
            return res

        // below starts the SETGAME specific code
        updateWithNewDetections(image, imageRotation, scaleFactor, res)
        res.clear() // re-use the list to send results back

        // find sets and mark them as groups
        findSets()

        // copy our cards to the resulting list
        for (r in cards) res.add(r)
        return res
    }

    fun updateWithNewDetections(
            image: Bitmap,
            imageRotation: Int,
            scaleFactor: Float,
            results: MutableList<Detected>) {
        var reDetectedCards = LinkedList<ViewCard>()
        var newDet = LinkedList<Detected>()
        var newCards = LinkedList<ViewCard>()

        outer@for (det in results) {
            for (card in cards) {
                if (card.isWithinBorders(det)) {
                    // move to the re-detected list
                    cards.remove(card)
                    reDetectedCards.add(card)

                    // mark as re-detected
                    card.markReDetected()
                    // update new borders
                    card.updateBorders(det)

                    continue@outer
                }
            }
            // new card didn't find any matching card - add a new one
            newDet.add(det)
        }

        // try to reClassify redetectedCards if they're timed out
        // limit this to 5 cards at a time - we'll update them next detection period
        var reclassifiedCounter = 0
        for (card in reDetectedCards) {
            if (!card.isReClassifyCandidate())
                continue
            var border = card.getBoundingBox()
            var imageFrag = Bitmap.createBitmap(image,
                    (border.left*scaleFactor).toInt(),
                    (border.top*scaleFactor).toInt(),
                    ((border.right - border.left)*scaleFactor).toInt(),
                    ((border.bottom - border.top)*scaleFactor).toInt())
            var classRotation = 0
            if (imageFrag.width < imageFrag.height)
                classRotation = 1
            var res = classifyImage(imageFrag, classRotation)
            res?.let { it ->
                if (it.isNotEmpty()) {
                    val sortedCategories = it[0].categories //.sortedBy { it?.score }
                    card.updateClassification(
                            sortedCategories[0].label,
                            sortedCategories[0].score)
                    newCards.add(card)
                }
            }
            reclassifiedCounter++
            if (reclassifiedCounter > 5)
                break
        }

        // classify the newly appeared cards in newDet and add the to newCards
        for (det in newDet) {
            var border = det.getBoundingBox()
            var imageFrag = Bitmap.createBitmap(image,
                    (border.left*scaleFactor).toInt(),
                    (border.top*scaleFactor).toInt(),
                    ((border.right - border.left)*scaleFactor).toInt(),
                    ((border.bottom - border.top)*scaleFactor).toInt())
            var classRotation = 1
            if (imageFrag.width < imageFrag.height)
                classRotation = 0
            var res = classifyImage(imageFrag, classRotation)
            res?.let { it ->
                if (it.isNotEmpty()) {
                    val sortedCategories = it[0].categories//.sortedBy { it?.score }
                    if (sortedCategories.size > 0) {
                        var card = ViewCard(det)
                        card.updateClassification(
                                sortedCategories[0].label,
                                sortedCategories[0].score)
                        newCards.add(card)
                    }
                }
            }
        }
        // TODO: cards contains the list non matching cards - we need to handle them
        // try to identify their new position and redetect them and add to reDetectedCards

        // update the internal list with all we found
        //cards = reDetectedCards
        //cards.addAll(newCards)
        cards = newCards
    }
    fun findSets(): Boolean {
        var vCardsByName = HashMap<Card,ViewCard>()
        var inSet = HashSet<Card>()
        // store all cards to set
        for (vCard in cards) {
            var card = cardFromString(vCard.name())
            // add only classified cards
            if (card != null) {
                inSet.add(card)
                vCardsByName.put(card, vCard)
            }
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

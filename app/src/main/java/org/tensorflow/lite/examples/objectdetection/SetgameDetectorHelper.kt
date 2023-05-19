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
import android.graphics.Color
import android.graphics.ColorMatrix
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
import kotlin.collections.HashMap
import kotlin.math.absoluteValue
import kotlin.math.pow

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

data class BoundsTransformation(
        var deltaX: Float,
        var deltaY: Float,
        var scaleX:Float,
        var scaleY: Float)

data class CategoryStat(
        var scoreMax: Float
//        var scaleSum: Float, // probabilites sum
//        var scaleNum: Float  // number of times we added
)

class ViewCard(var x: Detected):  Grouppable(){
    public var overriddenName: String? = null

    public var classificationStat = HashMap<String, CategoryStat>()
    // corrections by adhoc algorithm
    public var colorCorrection: CardColor? = null

    public var bounds = x.getBoundingBox()
    public var prevBounds: RectF? = null

    // keep what came from Detected & classified and their timestamp
    public var detectedTime = SystemClock.uptimeMillis()
    public var detectedCategories = x.getCategories()?: LinkedList<Category>()
    public var classifiedTime = SystemClock.uptimeMillis()
    public var classifiedCategories: MutableList<Category> = LinkedList<Category>()

    // groups defines the color it will be shown
    public var groups = HashSet<Int>()

    fun name(): String{
        if (overriddenName != null)
            return overriddenName.toString()

        val cats = getAccumulatedClassifications()
        if (cats != null && cats.size > 0) {
            return cats[0].label
        }
        return ""
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

    fun getBoundsTransformation(): BoundsTransformation? {
        if (
                prevBounds == null ||
                prevBounds!!.width() == 0.0f ||
                prevBounds!!.height() == 0.0f)
            return null
        return BoundsTransformation(
               bounds.centerX() - prevBounds!!.centerX(),
                bounds.centerY() - prevBounds!!.centerY(),
                bounds.width()/prevBounds!!.width(),
                bounds.height()/prevBounds!!.height()
        )
    }

    fun applyBoundsTransformation(t: BoundsTransformation) {
        prevBounds = bounds
        val boundsTmp = RectF(
                prevBounds!!.left + t.deltaX,
                prevBounds!!.right + t.deltaX,
                prevBounds!!.top + t.deltaY,
                prevBounds!!.bottom + t.deltaY,
        )
        //now scale
        bounds = RectF(
                boundsTmp.centerX()-boundsTmp.width()*t.scaleX/2,
                boundsTmp.centerX()+boundsTmp.width()*t.scaleX/2,
                boundsTmp.centerY()-boundsTmp.height()*t.scaleX/2,
                boundsTmp.centerY()+boundsTmp.height()*t.scaleX/2,
        )
    }

    // Work with live time
    fun markReDetected(x: Detected) {
        updateBorders(x)
        detectedCategories = x.getCategories()
        detectedTime = SystemClock.uptimeMillis()
    }

    fun isDetectionOutdated():Boolean {
        // if time when it was last time re-detected or re-classified is more then const (e.g. 2sec)
        return  SystemClock.uptimeMillis() - detectedTime < 2000 ||
                SystemClock.uptimeMillis() - classifiedTime < 2000
    }

    fun updateClassification(newCategories: MutableList<Category>) {
        classifiedCategories = newCategories

        for (cat in classifiedCategories) {
            var stat = classificationStat[cat.label]
            var curScore = cat.score
            if (stat != null) {
                if (stat.scoreMax < cat.score) {
                    stat.scoreMax = cat.score
                }
                curScore = stat.scoreMax
            }else {
                classificationStat[cat.label] = CategoryStat(cat.score)
            }
        }

        classifiedTime = SystemClock.uptimeMillis()
    }

    fun updateCorrections(color: CardColor?) {
        // TODO: maybe add timer
        //if (color != null) {
            colorCorrection = color
        //}
    }

    fun getAccumulatedClassifications(): MutableList<Category>? {
        if (classifiedCategories == null && classifiedCategories.size == 0)
            return null

        var res = LinkedList<Category>()
        for (cat in classificationStat.keys){
            res.add(Category(cat, classificationStat[cat]!!.scoreMax))
        }
        res.sortByDescending { it.score }

        // update the first element with corrections
        if (colorCorrection != null && res.size > 0) {
            var crd = cardFromString(res[0].label)
            if (crd != null) {
                crd.cardColor = colorCorrection!!
                res[0] = Category(cardToString(crd),  -1.0f * res[0].score  /*mark that there was correction*/)
            }
        }

        return res
    }

    fun isReClassifyCandidate(): Boolean {
        return SystemClock.uptimeMillis() - detectedTime < 500
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
        val cats = getAccumulatedClassifications()
        if (cats != null)
            return cats!!

        return detectedCategories
    }
}

//copy to send to output
class ResultCard(val x: ViewCard):  Grouppable() {
    public var cats = x.getCategories()?: LinkedList<Category>()
    public var bounds = x.getBoundingBox()
    public var groups = HashSet<Int>()

    init {
        for (g in x.groups)
            groups.add(g)
    }
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

data class classificationWithCorrection(val classifications: MutableList<Category>, val color: CardColor?)

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

    // Mutable buffers for fun classify
    private var buffer: Bitmap = Bitmap.createBitmap(1000, 1000, Bitmap.Config.ARGB_8888)
    private var pixels = IntArray(1000 * 1000);
    fun classify(image: Bitmap, imageRotation: Int, border: RectF): classificationWithCorrection? {
        var top = border.top.toInt()
        var bottom = border.bottom.toInt()
        var left = border.left.toInt()
        var right = border.right.toInt()

        // filter by picture size
        if (right - left>= 1000 || bottom - top >= 1000)
            return null

        // rotate within image
        when (imageRotation/90) {
            Surface.ROTATION_270 -> {
                // need to test
                val newLeft = image.width -top
                val newRight = image.width -bottom
                val newTop = right
                val newBottom = left
                top = newTop
                bottom = newBottom
                left = newLeft
                right = newRight
            }

            Surface.ROTATION_180 -> {
                val newLeft = image.width - right
                val newRight = image.width - left
                val newTop = image.height - bottom
                val newBottom = image.height - top
                top = newTop
                bottom = newBottom
                left = newLeft
                right = newRight
            }

            Surface.ROTATION_90 -> {
                val newLeft = top
                val newRight = bottom
                val newTop = image.height- right
                val newBottom = image.height- left
                top = newTop
                bottom = newBottom
                left = newLeft
                right = newRight
            }
        }

        // filter by bitmap limitaion (we could ajust them though)
        // the initial rect somtimes may have negative left/top or
        // too big right/bottom
        if (left < 0) left = 0
        if (right > image.width) right = image.width
        if (top < 0 ) top = 0
        if (bottom > image.height) bottom = image.height

        // those are not changable
        val width = right - left
        val height = bottom - top

        // we want them to be vertical (this is weird - I think to trained horizontal)
        var classificationRotation = Surface.ROTATION_90
        if (width < height)
            classificationRotation = 0

        if (width <= 0 || height <=0 )
            return null

        assert(left + width <= image.width &&
                top + height <= image.height, {
                "left+width:" + (left + width).toString() +
                ", top+height" + (top + height).toString() +
                ", image: " + image.width.toString() + "x" + image.height.toString() +
                ", rot: " + imageRotation.toString() +
                ", initial rect(ltrb): "+  border.left.toInt().toString()+ "x" + border.top.toInt().toString()+ "x" + border.right.toInt().toString()+ "x" + border.bottom.toInt().toString() +
                ", left: " + left.toString()+
                ", top: "+ top.toString()+
                ", width: " + width.toString() +
                ", heigth: " + height.toString()})

        buffer.width = width
        buffer.height = height
        image.getPixels(pixels, 0,width,
                left,
                top,
                width,
                height)

        buffer.setPixels(pixels, 0,width,0,0,
                width,
                height)

        // todo - make configurable and with different capabilities
        var clr = adhocCardColorGuess(buffer, pixels)
        val res = classifyImage(buffer, classificationRotation)

        res?.let { it ->
            if (it.isNotEmpty()) {
                //compare clr
                if (clr != null && it[0].categories.size > 0){
                    val crd = cardFromString(it[0].categories[0].label)
                    if (crd != null && crd.cardColor == clr) {
                        // we don't need correction - make it null
                        clr = null
                    }
                }
                return classificationWithCorrection(it[0].categories, clr)
            }
        }
        return null
    }

    fun arePixelsSimilar(pixel1: Int, pixel2: Int, threshold: Float): Boolean {
        // Get the RGB values of the pixels.
        val pixel1Red = Color.red(pixel1)
        val pixel1Green = Color.green(pixel1)
        val pixel1Blue = Color.blue(pixel1)

        val pixel2Red = Color.red(pixel2)
        val pixel2Green = Color.green(pixel2)
        val pixel2Blue = Color.blue(pixel2)

        // Get the maximum values of the RGB channels.
        val p1Max = Math.max(Math.max(pixel1Red, pixel1Green), pixel1Blue)
        val p2Max = Math.max(Math.max(pixel2Red, pixel2Green), pixel2Blue)
        // both black
        if (p1Max == 0 && p2Max == 0) return true
        // can't compare similarity with black
        if (p1Max == 0 && p2Max != 0 || p1Max !=0 && p2Max == 0) return false

        // Calculate the normalized Euclidean distance between the pixels.
        val normalizedDistance = Math.sqrt(((pixel1Red.toDouble() / p1Max.toDouble() - pixel2Red.toDouble() / p2Max.toDouble()).pow(2) +
                (pixel1Green.toDouble() / p1Max.toDouble() - pixel2Green.toDouble() / p2Max.toDouble()).pow(2) +
                (pixel1Blue.toDouble() / p1Max.toDouble() - pixel2Blue.toDouble() / p2Max.toDouble()).pow(2)))

        // Return true if the normalized Euclidean distance is less than a threshold.
        return normalizedDistance < threshold
    }
    fun adhocCardColorGuess(buffer: Bitmap, pixels: IntArray): CardColor? {
        //constants
        var redCardPx = Color.rgb(235, 30, 45)
        var greenCardPx = Color.rgb(20, 170, 80)
        var purpleCardPx = Color.rgb(100,50,150)

        val threshold = 0.5f

        var redPixCount = 0
        var greenPixCount = 0
        var purplePixCount = 0

        var steps = 0
        if (buffer.height > buffer.width) {
            // todo - maybe better count from center in both dirs and stop when pixels are white
            for (i in 0 until buffer.height/2) {
                steps =  steps + 1
                var x = pixels[buffer.width / 2 + (buffer.height/2 + i) * buffer.width]
                if (arePixelsSimilar(redCardPx, x, threshold))redPixCount = redPixCount + 1
                if (arePixelsSimilar(greenCardPx, x, threshold))greenPixCount = greenPixCount + 1
                if (arePixelsSimilar(purpleCardPx, x, threshold))purplePixCount = purplePixCount + 1
                steps =  steps + 1
                x = pixels[buffer.width / 2 + (buffer.height/2 - i) * buffer.width]
                if (arePixelsSimilar(redCardPx, x, threshold))redPixCount = redPixCount + 1
                if (arePixelsSimilar(greenCardPx, x, threshold))greenPixCount = greenPixCount + 1
                if (arePixelsSimilar(purpleCardPx, x, threshold))purplePixCount = purplePixCount + 1
                if (Math.max(Math.max(redPixCount, greenPixCount), purplePixCount) >= 3)
                    break
            }
        }else {
            for (i in 0 until buffer.width/2) {
                steps =  steps + 1
                var x = pixels[buffer.height * buffer.width / 2 + buffer.width/2 + i]
                if (arePixelsSimilar(redCardPx, x, threshold))redPixCount = redPixCount + 1
                if (arePixelsSimilar(greenCardPx, x, threshold))greenPixCount = greenPixCount + 1
                if (arePixelsSimilar(purpleCardPx, x, threshold))purplePixCount = purplePixCount + 1
                steps =  steps + 1
                x = pixels[buffer.height * buffer.width / 2 + buffer.width/2 - i]
                if (arePixelsSimilar(redCardPx, x, threshold))redPixCount = redPixCount + 1
                if (arePixelsSimilar(greenCardPx, x, threshold))greenPixCount = greenPixCount + 1
                if (arePixelsSimilar(purpleCardPx, x, threshold))purplePixCount = purplePixCount + 1
                if (Math.max(Math.max(redPixCount, greenPixCount), purplePixCount) >= 3)
                    break
            }
        }
        val max = Math.max(Math.max(redPixCount, greenPixCount), purplePixCount)
        if (max < 3) {
            // if number is less than 5%
            return null
        }
        if (redPixCount == max) return CardColor.RED
        if (greenPixCount == max) return CardColor.GREEN
        return CardColor.PURPLE
    }

//    fun tryToFixWhiteBalance(buffer: Bitmap, pixels: IntArray) {
//
//        var vR = 0
//        var vG = 0
//        var vB = 0
//
//        if (buffer.height > buffer.width) {
//            for (i in 0 until buffer.height) {
//                val x = pixels[buffer.width / 2 + i * buffer.width]
//                if (vR + vG + vB < Color.red(x)+ Color.green(x) + Color.blue(x)) {
//                    vR = Color.red(x)
//                    vG = Color.green(x)
//                    vB = Color.blue(x)
//                }
//            }
//        }else {
//            for (i in 0 until buffer.width) {
//                val x = pixels[buffer.height * buffer.width / 2 + i ]
//                if (vR + vG + vB < Color.red(x)+ Color.green(x) + Color.blue(x)) {
//                    vR = Color.red(x)
//                    vG = Color.green(x)
//                    vB = Color.blue(x)
//                }
//            }
//        }
//        if (vR < 100 || vG < 100 || vB < 100)
//            return // won't do it since it's too dark
//
//
//        // slow.. Can't apply Matrix to bitmap
//        for (i in 0 until buffer.width * buffer.height) {
//            var R = Color.red(pixels[i])*(256+30)/vR
//            var G = Color.green(pixels[i])*(256+30)/vG
//            var B = Color.blue(pixels[i])*(256+30)/vB
//
//            if (R>255) R=255
//            if (G>255) G=255
//            if (B>255) B=255
//
//            pixels[i] = Color.rgb(R,G,B)
//        }
//    }

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

        // below starts the SETGAME specific code
        updateWithNewDetections(image, imageRotation, res)

        // find sets and mark them as groups
        findSets()

        // copy our cards to the resulting list
        res.clear() // re-use the list to send results back
        for (r in cards) res.add(ResultCard(r))
        return res
    }

    fun updateWithNewDetections(
            image: Bitmap,
            imageRotation: Int,
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

                    // mark as re-detected & udpdate all info
                    card.markReDetected(det)

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
            var res = classify(image, imageRotation, card.getBoundingBox())
            if (res != null) {
                card.updateClassification(res.classifications)
                card.updateCorrections(res.color)
            }
            reclassifiedCounter++
            if (reclassifiedCounter > 5)
                break
        }

        // classify the newly appeared cards in newDet and add the to newCards
        for (det in newDet) {
            var res = classify(image, imageRotation, det.getBoundingBox())
            if (res != null) {
                var card = ViewCard(det)
                card.updateClassification(res.classifications)
                card.updateCorrections(res.color)
                newCards.add(card)
            }
        }
        // TODO: cards contains the list non matching cards - we need to handle them
        // try to identify their new position based on the trajectory of redetected cards
        // and redetect them and add to reDetectedCards
        var t: BoundsTransformation? = null
        for (card in reDetectedCards) {
            // TBD: we're handling only move without zooming, rotating and etc. even though it's possible to try those as well later
            val xt = card.getBoundsTransformation()
            if (xt != null) {
                t = xt
                // reset scaling for now to make a more stable result
                t.scaleX = 1.0f
                t.scaleY = 1.0f
            }
        }
        if (t != null) {
            // we can try to transform and classify
            for (card in cards) {
                card.applyBoundsTransformation(t!!)
                var res = classify(image, imageRotation, card.getBoundingBox())
                if (res != null && res.classifications.size > 0){
                    card.updateClassification(res.classifications)
                    card.updateCorrections(res.color)
                    reDetectedCards.add(card)
                }else {
                    if (!card.isDetectionOutdated()){
                        reDetectedCards.add(card)
                    }
                }
            }
        }

        // update the internal list with all we found
        cards = reDetectedCards
        cards.addAll(newCards)
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


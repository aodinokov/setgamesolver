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
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import com.google.gson.Gson
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
import java.io.BufferedReader
import java.io.InputStreamReader
import java.lang.Double.max
import java.lang.Double.min
import java.util.*
import kotlin.collections.HashMap
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

data class BoundsTransformation(
        var deltaX: Float,
        var deltaY: Float,
        var scaleX:Float,
        var scaleY: Float)

//data class CategoryStat(
//        var scoreMax: Float
//)

class ViewCard(var x: Detected):  Grouppable(){
    public var overriddenName: String? = null

    public var bounds = x.getBoundingBox()
    public var prevBounds: RectF? = null

    // keep what came from Detected & classified and their timestamp
    public var detectedTime = SystemClock.uptimeMillis()
    public var detectedCategories = x.getCategories()?: LinkedList<Category>()
    public var classifiedTime = SystemClock.uptimeMillis()
    //public var classifiedCategories: MutableList<Category> = LinkedList<Category>()
    //public var classificationStat = HashMap<String, CategoryStat>()

    public var classificationMax = Array<Category?>(4, {null})

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

    fun updateClassifications(newCategories: Array<MutableList<Category>>?) {
        //assert(false) // TBD
//        if (newCategories != null && newCategories.size >SetgameDetectorHelper.SHAPE_CLASSIFIER)
//        updateClassification(newCategories[SetgameDetectorHelper.COLOR_CLASSIFIER])
        if (newCategories == null)
            return

        assert(newCategories.size == classificationMax.size)
        for (i in 0 until classificationMax.size) {
            if (newCategories[i].size == 0)
                continue
            val newCat = newCategories[i][0]
            if (classificationMax[i] == null || classificationMax[i]!!.score <= newCat.score)
                classificationMax[i] = newCat

        }

    }
//    fun updateClassification(newCategories: MutableList<Category>) {
//        classifiedCategories = newCategories
//
//        for (cat in classifiedCategories) {
//            var stat = classificationStat[cat.label]
//            var curScore = cat.score
//            if (stat != null) {
//                if (stat.scoreMax < cat.score) {
//                    stat.scoreMax = cat.score
//                }
//                curScore = stat.scoreMax
//            }else {
//                classificationStat[cat.label] = CategoryStat(cat.score)
//            }
//        }
//
//        classifiedTime = SystemClock.uptimeMillis()
//    }

//    fun getAccumulatedClassifications(): MutableList<Category>? {
//        if (classifiedCategories == null && classifiedCategories.size == 0)
//            return null
//
//        var res = LinkedList<Category>()
//        for (cat in classificationStat.keys){
//            res.add(Category(cat, classificationStat[cat]!!.scoreMax))
//        }
//        res.sortByDescending { it.score }
//
//        return res
//    }

    fun getAccumulatedClassifications(): MutableList<Category>? {
        if (classificationMax[0]!= null &&
                classificationMax[1]!= null &&
                classificationMax[2]!= null &&
                classificationMax[3]!= null) {
            var res = LinkedList<Category>()
            res.add(Category(
                    classificationMax[0]!!.label + "-" +
                            classificationMax[1]!!.label + "-" +
                            classificationMax[2]!!.label + "-" +
                            classificationMax[3]!!.label,
                    classificationMax[0]!!.score *
                            classificationMax[1]!!.score *
                            classificationMax[2]!!.score *
                            classificationMax[3]!!.score))
            return res
        }
        return null
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
    private var imageClassifiers: Array<ImageClassifier?> =Array<ImageClassifier?>(4, {null})
    private var adhocColorClassifierColormap: Array<Array<Array<Int>>>? = null

    // we're keeping cards so the they can be traced and overridden
    private var cards = LinkedList<ViewCard>()

    init {
        setupObjectDetector()
        for (i in 0 until imageClassifiers.size)
            setupImageClassifier(i)
    }

    fun clearObjectDetector() {
        objectDetector = null
        for (i in 0 until imageClassifiers.size)
            imageClassifiers[i] = null
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

    private fun setupImageClassifier(i: Int) {
        val optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
                .setScoreThreshold(0.1f)
                .setMaxResults(3)

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
                when (i) {
                    COUNT_CLASSIFIER -> "setgame-classify-count.tflite"
                    COLOR_CLASSIFIER -> "setgame-classify-color.tflite"
                    FILL_CLASSIFIER -> "setgame-classify-fill.tflite"
                    SHAPE_CLASSIFIER -> "setgame-classify-shape.tflite"
                    else -> "setgame-classify.tflite"
                }

        try {
            imageClassifiers[i] =
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
    fun classifyImage(i: Int, image: Bitmap, rotation: Int): List<Classifications>? {
        if (imageClassifiers[i] == null) {
            setupImageClassifier(i)
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

        return imageClassifiers[i]?.classify(tensorImage, imageProcessingOptions)
    }

    // Mutable buffers for fun classify
    private var buffer: Bitmap = Bitmap.createBitmap(1000, 1000, Bitmap.Config.ARGB_8888)
    private var pixels = IntArray(1000 * 1000);
    fun classify(image: Bitmap, imageRotation: Int, border: RectF): Array<MutableList<Category>>? {
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

        var r = Array<MutableList<Category>>(imageClassifiers.size, { LinkedList<Category>() })
        for (i in 0 until imageClassifiers.size) {
            if (i == COLOR_CLASSIFIER)
                continue /*skip for now*/
            val res = classifyImage(i, buffer, classificationRotation)

            res?.let { it ->
                if (it.isNotEmpty()) {
                    r[i] = it[0].categories
                }
            }
        }

        // if shape and fill are classified with good probability - good time to do adhoc
        if (    r[SHAPE_CLASSIFIER].size > 0 &&
                r[FILL_CLASSIFIER].size > 0) {
            r[COLOR_CLASSIFIER] = adhocCardColorGuess(buffer, pixels)
        }

        return r
    }

    /* R = 4, G= 2, P = 1*/
    fun getColorFlagsByPixel(pixel: Int): Int {
        //lazy init
        if (adhocColorClassifierColormap == null) {
            // Open the JSON file.
            val inputStream = context.assets.open("setgame-classify-color.json")
            // Create a buffered reader.
            val bufferedReader = BufferedReader(InputStreamReader(inputStream))
            // Read the JSON file.
            val jsonString = bufferedReader.use { it.readText() }
            // Create a Gson object.
            val gson = Gson()
            adhocColorClassifierColormap = gson.fromJson(jsonString, Array<Array<Array<Int>>>::class.java)
        }
        if (adhocColorClassifierColormap == null)
            return 0

        val r = Color.red(pixel)/256.0
        val g = Color.green(pixel)/256.0
        val b = Color.blue(pixel)/256.0

        var h = 0
        var s = 0.0
        var v = 0.0

        val mx = max(r,max(g,b))
        val mn = min(r, min(g,b))
        val df = mx - mn
        if (df != 0.0) {
            if (mx == r) {
                h = (60.0 * ((g - b) / df) + 360.0).toInt() % 360
            } else if (mx == g) {
                h = (60.0 * ((b - r) / df) + 120.0).toInt() % 360
            } else {
                h = (60.0 * ((r - g) / df) + 240.0).toInt() % 360
            }
        }
        if (mx != 0.0) {
            s = df/ mx
        }
        v = mx

        var shift = 0
        if ((h/5)%2 == 0) {
            shift = 4
        }
        return adhocColorClassifierColormap!![(v*100).toInt()][(s*100).toInt()][h/10].shl(shift) and 0x0f
    }

    fun adhocCardColorGuess(buffer: Bitmap, pixels: IntArray): LinkedList<Category> {
        var rc = 0
        var rg = 0
        var rp = 0
        var tc = 0

        if (buffer.height > buffer.width) {
            // go vertically
            for (x in buffer.width.toInt()/2 -3 until buffer.width.toInt()/2 + 3)
                for (y in 1*buffer.height.toInt()/4 until 3*buffer.height.toInt()/4) {
                    tc +=1
                    val px = pixels[y*buffer.width.toInt() + x]
                    val flags = getColorFlagsByPixel(px)
                    if (flags and 0x4 != 0)
                        rc += 1
                    if (flags and 0x2 != 0)
                        rg += 1
                    if (flags and 0x1 != 0)
                        rp += 1
                }
        }else {
            // go horizontally
            for (y in buffer.height.toInt()/2 -3 until buffer.height.toInt()/2 + 3)
                for (x in 1*buffer.width.toInt()/4 until 3*buffer.width.toInt()/4) {
                    tc +=1
                    val px = pixels[y*buffer.width.toInt() + x]
                    val flags = getColorFlagsByPixel(px)
                    if (flags and 0x4 != 0)
                        rc += 1
                    if (flags and 0x2 != 0)
                        rg += 1
                    if (flags and 0x1 != 0)
                        rp += 1
                }
        }
        if (tc == 0)
            return  LinkedList<Category>()

        var r = LinkedList<Category>()
        r.add(Category("red", rc.toFloat()/tc.toFloat()))
        r.add(Category("green", rg.toFloat()/tc.toFloat()))
        r.add(Category("purple", rp.toFloat()/tc.toFloat()))

        r.sortByDescending { it.score }

        return r
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
                card.updateClassifications(res)
            }
            reclassifiedCounter++
            if (reclassifiedCounter > 5)
                break
        }

        // classify the newly appeared cards in newDet and add the to newCards
        for (det in newDet) {
            var res = classify(image, imageRotation, det.getBoundingBox())
            if (res != null/*&& res.classifications.size > 0 */) {
                var card = ViewCard(det)
                card.updateClassifications(res)
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
                if (res != null && res[SHAPE_CLASSIFIER].size > 0){
                    card.updateClassifications(res)
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
        const val COUNT_CLASSIFIER = 0
        const val COLOR_CLASSIFIER = 1
        const val FILL_CLASSIFIER = 2
        const val SHAPE_CLASSIFIER = 3
    }
}


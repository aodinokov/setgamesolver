/*
 * Copyright 2023 Alexey Odinokov. All Rights Reserved.
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

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.content.Context
import android.util.Log
import android.view.Surface
import com.google.gson.Gson
import org.tensorflow.lite.examples.objectdetection.fragments.DelegationMode
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.Classifications
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.io.BufferedReader
import java.io.InputStreamReader
import java.lang.Double.max
import java.lang.Double.min
import java.util.*

class ClassifierHelper(
        val context: Context,
        var threshold: Float = 0.1f,
        var numThreads: Int = 2,
        var currentDelegate: DelegationMode = DelegationMode.Cpu,
        var classifierErrorListener: ClassifierErrorListener? = null
) {
    private var imageClassifiers: Array<ImageClassifier?> =Array(4) { null }

    fun clearClassifier() {
        for (i in imageClassifiers.indices)
            imageClassifiers[i] = null
    }

    private fun setupClassifier(i: Int) {
        val optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(3)

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        when (currentDelegate) {
            DelegationMode.Cpu-> {
                // Default
            }
            DelegationMode.Gpu -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    classifierErrorListener?.onClassifierError("Classifier creation: GPU is not supported on this device")
                }
            }
            DelegationMode.Nnapi -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName =
                when (i) {
                    NUMBER_CLASSIFIER -> "setgame-classify-number.tflite"
                    COLOR_CLASSIFIER -> "setgame-classify-color.tflite"
                    SHADING_CLASSIFIER -> "setgame-classify-shading.tflite"
                    SHAPE_CLASSIFIER -> "setgame-classify-shape.tflite"
                    else -> "setgame-classify.tflite"
                }

        try {
            imageClassifiers[i] =
                    ImageClassifier.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            classifierErrorListener?.onClassifierError(
                    "Classifier creation:Image classifier failed to initialize. See error logs for details"
            )
            Log.e("Test", "TFLite failed to load model with error: " + e.message)
        }
    }

    // Receive the device rotation (Surface.x values range from 0->3) and return EXIF orientation
    // http://jpegclub.org/exif_orientation.html
    private fun getOrientationFromRotation(rotation: Int) : ImageProcessingOptions.Orientation {
        return when (rotation) {
            Surface.ROTATION_270 ->
                ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            Surface.ROTATION_180 ->
                ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            Surface.ROTATION_90 ->
                ImageProcessingOptions.Orientation.TOP_LEFT
            else ->
                ImageProcessingOptions.Orientation.RIGHT_TOP
        }
    }

    private fun classifyImage(i: Int, image: Bitmap, rotation: Int): List<Classifications>? {
        if (imageClassifiers[i] == null) {
            setupClassifier(i)
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

    private var adhocColorClassifierColormap: Array<Array<Array<Int>>>? = null

    /* color per bit: R = 4, G= 2, P = 1*/
    private fun getColorFlagsByPixel(pixel: Int): Int {
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
        if (adhocColorClassifierColormap == null) {
            classifierErrorListener?.onClassifierError("Classifier creation: Couldn't initialize ad-hoc part")
            return 0
        }

        val r = Color.red(pixel)/256.0
        val g = Color.green(pixel)/256.0
        val b = Color.blue(pixel)/256.0

        var h = 0
        var s = 0.0
        // note that v == mx in hcv

        val mx = max(r,max(g,b))
        val mn = min(r, min(g,b))
        val df = mx - mn
        if (df != 0.0) {
            h = when (mx) {
                r -> {
                    (60.0 * ((g - b) / df) + 360.0).toInt() % 360
                }
                g -> {
                    (60.0 * ((b - r) / df) + 120.0).toInt() % 360
                }
                else -> {
                    (60.0 * ((r - g) / df) + 240.0).toInt() % 360
                }
            }
        }
        if (mx != 0.0) {
            s = df/ mx
        }

        var shift = 0
        if ((h/5)%2 == 0) {
            shift = 4
        }
        return adhocColorClassifierColormap!![(mx * 100).toInt()][(s*100).toInt()][h/10].shl(shift) and 0x0f
    }

    private fun adhocCardColorGuess(buffer: Bitmap, pixels: IntArray): LinkedList<Category> {
        var rc = 0
        var rg = 0
        var rp = 0
        var tc = 0

        if (buffer.height > buffer.width) {
            // go vertically
            assert(buffer.width>=7)
            for (x in buffer.width/2 -3 until buffer.width/2 + 3)
                for (y in 1*buffer.height/4 until 3*buffer.height/4) {
                    tc +=1
                    val px = pixels[y*buffer.width + x]
                    val flags = getColorFlagsByPixel(px)
                    if (flags != 1 && flags != 2 && flags != 4)
                        continue
                    if (flags and 0x4 != 0)
                        rc += 1
                    if (flags and 0x2 != 0)
                        rg += 1
                    if (flags and 0x1 != 0)
                        rp += 1
//                    //dbg
//                    pixels[y*buffer.width.toInt() + x] = Color.WHITE
                }
        }else {
            // go horizontally
            assert(buffer.height>=7)
            for (y in buffer.height/2 -3 until buffer.height/2 + 3)
                for (x in 1*buffer.width/4 until 3*buffer.width/4) {
                    tc +=1
                    val px = pixels[y*buffer.width + x]
                    val flags = getColorFlagsByPixel(px)
                    if (flags != 1 && flags != 2 && flags != 4)
                        continue
                    if (flags and 0x4 != 0)
                        rc += 1
                    if (flags and 0x2 != 0)
                        rg += 1
                    if (flags and 0x1 != 0)
                        rp += 1
//                    //dbg
//                    pixels[y*buffer.width.toInt() + x] = Color.WHITE
                }
        }
        if (tc == 0)
            return  LinkedList<Category>()

        val r = LinkedList<Category>()
        r.add(Category("red", rc.toFloat()/tc.toFloat()))
        r.add(Category("green", rg.toFloat()/tc.toFloat()))
        r.add(Category("purple", rp.toFloat()/tc.toFloat()))

        r.sortByDescending { it.score }

        return r
    }

    // Mutable buffers for fun classify
    private var buffer: Bitmap = Bitmap.createBitmap(1000, 1000, Bitmap.Config.ARGB_8888)
    private var pixels = IntArray(1000 * 1000)

    fun extractToBitmap(image: Bitmap, imageRotation: Int, border: RectF, outputBuffer: Bitmap): Boolean {
        var top = border.top.toInt()
        var bottom = border.bottom.toInt()
        var left = border.left.toInt()
        var right = border.right.toInt()

        // filter by picture size
        if (right - left>= outputBuffer.width || bottom - top >= outputBuffer.height)
            return false

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

        // filter by bitmap limitation (we could adjust them though)
        // the initial rect sometimes may have negative left/top or
        // too big right/bottom
        if (left < 0) left = 0
        if (right > image.width) right = image.width
        if (top < 0 ) top = 0
        if (bottom > image.height) bottom = image.height

        // those are not changeable
        val width = right - left
        val height = bottom - top

        if (width <= 0 || height <=0 )
            return false

        assert(left + width <= image.width &&
                top + height <= image.height) {
            "left+width:" + (left + width).toString() +
                    ", top+height" + (top + height).toString() +
                    ", image: " + image.width.toString() + "x" + image.height.toString() +
                    ", rot: " + imageRotation.toString() +
                    ", initial rect(LxTxRxBx): " + border.left.toInt().toString() + "x" + border.top.toInt().toString() + "x" + border.right.toInt().toString() + "x" + border.bottom.toInt().toString() +
                    ", left: " + left.toString() +
                    ", top: " + top.toString() +
                    ", width: " + width.toString() +
                    ", height: " + height.toString()
        }

        outputBuffer.width = width
        outputBuffer.height = height
        image.getPixels(pixels, 0,width,
                left,
                top,
                width,
                height)

        outputBuffer.setPixels(pixels, 0,width,0,0,
                width,
                height)

        return true
    }

    fun classify(image: Bitmap, imageRotation: Int, border: RectF): Array<MutableList<Category>>? {
        // reset to max
        buffer.width = 1000
        buffer.height = 1000

        if (!extractToBitmap(image, imageRotation, border, buffer)) {
            return null
        }
        // we want them to be vertical (this is weird - I think to trained horizontal)
        var classificationRotation = Surface.ROTATION_90
        if (buffer.width < buffer.height)
            classificationRotation = 0

        val r = Array<MutableList<Category>>(imageClassifiers.size) { LinkedList<Category>() }
        for (i in imageClassifiers.indices) {
            if (i == COLOR_CLASSIFIER)
                continue /*skip for now*/
            val res = classifyImage(i, buffer, classificationRotation)

            res?.let {
                if (it.isNotEmpty()) {
                    r[i] = it[0].categories
                }
            }
        }

        // if shape and fill are classified with good probability - good time to do adhoc
        if (    r[SHAPE_CLASSIFIER].size > 0 &&
                r[SHADING_CLASSIFIER].size > 0) {
            r[COLOR_CLASSIFIER] = adhocCardColorGuess(buffer, pixels)
//            //dbg
//            buffer.setPixels(pixels, 0,width,0,0,
//                    width,
//                    height)
        }

        return r
    }

    interface ClassifierErrorListener {
        fun onClassifierError(error: String)
    }

    companion object {
        const val NUMBER_CLASSIFIER = 0
        const val COLOR_CLASSIFIER = 1
        const val SHADING_CLASSIFIER = 2
        const val SHAPE_CLASSIFIER = 3
    }
}


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
package com.github.aodinokov.setgamesolver

// see
// https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/classifier/ImageClassifier.java
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.RectF
import android.util.Log
import android.view.Surface
import com.github.aodinokov.setgamesolver.fragments.DelegationMode
import com.google.gson.Gson
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.label.Category
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.lang.Double.max
import java.lang.Double.min
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel.MapMode
import java.util.*
import java.util.concurrent.locks.ReentrantLock

class ClassifierHelper(
        val context: Context,
        var threshold: Float = 0.1f,
        var numThreads: Int = 2,
        var currentDelegate: DelegationMode = DelegationMode.Cpu,
        var classifierErrorListener: ClassifierErrorListener? = null
) {
    private val OUTPUT_CLASSES: Int = 3

    private val classifierLock = ReentrantLock()
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: NnApiDelegate? = null

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.  */
    protected var imgData: ByteBuffer? = null
    protected fun getImageSizeX(): Int {
        return 224
    }
    protected fun getImageSizeY(): Int {
        return 224
    }
    protected fun getNumBytesPerChannel(): Int {
        return 4
    }
    // Pre-allocate your destination bitmap and a Canvas to write to it
    private val targetBitmap = Bitmap.createBitmap(getImageSizeX(), getImageSizeY(), Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(targetBitmap)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG) // Enables bilinear filtering for quality

    /** Dimensions of inputs.  */
    private val DIM_BATCH_SIZE: Int = 1
    private val DIM_PIXEL_SIZE: Int = 3
    /** Preallocated buffers for storing image data in.  */
    private val intValues = IntArray(getImageSizeX() * getImageSizeY())

    /** these values must be aligned with model config */
    private val IMAGE_MEAN: Float = 0f//127.5f
    private val IMAGE_STD: Float = 1f//127.5f
    fun addPixelValue(pixelValue: Int) {
        imgData!!.putFloat((((pixelValue shr 16) and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
        imgData!!.putFloat((((pixelValue shr 8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
        imgData!!.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
    }

    fun clearClassifier() {
        classifierLock.lock()

        val localInterpreter = interpreter
        interpreter = null
        val localGpuDelegate = gpuDelegate
        gpuDelegate = null
        val localNnapiDelegate = nnapiDelegate
        nnapiDelegate = null

        try {
            // 1. Close the Interpreter first (if you have one)
            localInterpreter?.close()
            // 2. Release Hardware Delegates
            localGpuDelegate?.close()
            localNnapiDelegate?.close()
            Thread.sleep(100)   // give some time to camera to stop
        } finally {
            classifierLock.unlock()
        }
    }

    // TODO: to check this - it must be quicker
//        // Create preprocessor for the image.
//        // See https://www.tensorflow.org/lite/inference_with_metadata/
//        //            lite_support#imageprocessor_architecture
//        val imageProcessor =
//                ImageProcessor.Builder()
//                        .build()
//
//        // Preprocess the image and convert it into a TensorImage for classification.
//        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
//
//        val imageProcessingOptions = ImageProcessingOptions.builder()
//                .setOrientation(getOrientationFromRotation(rotation))
//                .build()
//
//        return imageClassifier?.classify(tensorImage, imageProcessingOptions)

    @Throws(IOException::class)
    private fun loadModelFile(modelPath: String): ByteBuffer {
    return context.assets.openFd(modelPath).use { fileDescriptor ->
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            fileChannel.map(MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }

    private fun setupClassifier() {
        // create input buffer
        imgData = ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE
                    * getImageSizeX()
                    * getImageSizeY()
                    * DIM_PIXEL_SIZE
                    * getNumBytesPerChannel());
        imgData!!.order(ByteOrder.nativeOrder());

        // load and config model
        val modelBuffer: ByteBuffer = loadModelFile("setgame-classify.tflite")

        val options: Interpreter.Options = Interpreter.Options()
        options.setNumThreads(numThreads)

        when (currentDelegate) {
            DelegationMode.Cpu -> {
                // Default
            }
            DelegationMode.Gpu -> {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            }
            DelegationMode.Nnapi -> {
                nnapiDelegate = NnApiDelegate()
                options.addDelegate(nnapiDelegate)
            }
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    private fun getDegreeFromRotation(rotation: Int) : Float {
        return when (rotation) {
            Surface.ROTATION_270 ->
                0f//270f
            Surface.ROTATION_180 ->
                90f//180f
            Surface.ROTATION_90 ->
                0f//90f
            else ->
                90f//0f
        }
    }

    private fun scaleRotateAndConvert(sourceImage: Bitmap, rotation: Int) {
        matrix.reset()

        // 1. Calculate Scaling factors
        val scaleX = getImageSizeX().toFloat() / sourceImage.width
        val scaleY = getImageSizeY().toFloat() / sourceImage.height
        matrix.postScale(scaleX, scaleY)

        // 2. Rotate 90 degrees around the center
        // Note: rotating 90 deg changes the aspect ratio context,
        // so ensure your targetBitmap dimensions match your model's expected input!
        matrix.postRotate(getDegreeFromRotation(rotation), getImageSizeX() / 2f, getImageSizeY() / 2f)

        // 3. Draw onto the reused targetBitmap
        // This clears the previous frame and draws the new transformed one
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        canvas.drawBitmap(sourceImage, matrix, paint)

        // 4. Proceed to your ByteBuffer conversion
        convertBitmapToByteBuffer(targetBitmap)
    }

    /** Writes Image data into a `ByteBuffer`.  */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        if (imgData == null) {
            return
        }
        imgData!!.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        // Convert the image to floating point.
        var pixel = 0

        for (i in 0 until getImageSizeX()) {
            for (j in 0 until getImageSizeY()) {
                val `val`: Int = intValues.get(pixel++)
                addPixelValue(`val`)
            }
        }
    }

    private fun getArgmax(probabilities: FloatArray): Int {
        return probabilities.indices.maxByOrNull { probabilities[it] } ?: -1
    }

    private val labelMap = mapOf<String, Array<String>>(
        Pair("count", arrayOf("1", "2", "3")),
        Pair("color", arrayOf("green", "purple", "red")),
        Pair("fill", arrayOf("empty", "striped", "solid")),
        Pair("shape", arrayOf("diamond", "oval", "squiggle")))
    private fun getCategoryFromByteBuffer(labelId: String, buffer: ByteBuffer): Category? {
        buffer.rewind()
        val floats = FloatArray(buffer.capacity()/Float.SIZE_BYTES)
        buffer.asFloatBuffer().get(floats)
        val maxEl =  getArgmax(floats)
        if (maxEl < 0) return null
        if (floats[maxEl] < threshold) return null;
        return Category(labelMap[labelId]!![maxEl], floats[maxEl])
    }

    private fun classifyImage(image: Bitmap, rotation: Int): Array<List<Category?>> {
        if (!classifierLock.tryLock()) {
            return arrayOf(mutableListOf(), mutableListOf(), mutableListOf(), mutableListOf())
        }
        try {
            if (interpreter == null) {
                setupClassifier()   // TODO: hmm???? not sure. what if we competing with clearClassifier
            }

            // scale bitmap to the model size
            scaleRotateAndConvert(image, rotation)
            // store scaled image to the buffer
            convertBitmapToByteBuffer(targetBitmap)

            // we have 4 heads
            val countOut = ByteBuffer.allocateDirect(OUTPUT_CLASSES * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            val colorOut = ByteBuffer.allocateDirect(OUTPUT_CLASSES * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            val fillOut = ByteBuffer.allocateDirect(OUTPUT_CLASSES * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            val shapeOut = ByteBuffer.allocateDirect(OUTPUT_CLASSES * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            val outputs = mutableMapOf<Int, Any>(
                // the same sequence that in model (I've made a mistake - swapped count and color)
                0 to colorOut,
                1 to countOut,
                3 to fillOut,
                2 to shapeOut
            )
            try {
                interpreter?.runForMultipleInputsOutputs(arrayOf(imgData), outputs)
                return arrayOf(
                    listOfNotNull(getCategoryFromByteBuffer("count", countOut)),
                    listOfNotNull(getCategoryFromByteBuffer("color", colorOut)),
                    listOfNotNull(getCategoryFromByteBuffer("fill", fillOut)),
                    listOfNotNull(getCategoryFromByteBuffer("shape", shapeOut))
                )
            } catch (e: Exception) {
                Log.e("TFLite", "Inference failed", e)
                return arrayOf(mutableListOf(), mutableListOf(), mutableListOf(), mutableListOf())
            }
        } finally {
            classifierLock.unlock()
        }
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

        val r = Array<MutableList<Category>>(4) { LinkedList<Category>() }
        val res = classifyImage(buffer, classificationRotation)
        for (i in NUMBER_CLASSIFIER .. SHAPE_CLASSIFIER) {
//            if (i == COLOR_CLASSIFIER)
//                continue /*skip for now*/

            if (res[i] != null && res[i].isNotEmpty()) {
                val category = res[i].first()
                if (category != null) {
                    r[i] = mutableListOf(category)
                }
            }
        }
        return r


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


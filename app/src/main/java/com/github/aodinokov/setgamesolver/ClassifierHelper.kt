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
import android.graphics.RectF
import android.util.Log
import android.view.Surface
import com.github.aodinokov.setgamesolver.fragments.DelegationMode
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.FileInputStream
import java.io.IOException
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

    protected fun getImageSizeX(): Int {
        return 224
    }
    protected fun getImageSizeY(): Int {
        return 224
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

    private var modelOutputIndexesMap = mutableMapOf<String, Int>()
    private fun updataModelOutputIndexesMap(modelBuffer: ByteBuffer) {
        val metadataExtractor = MetadataExtractor(modelBuffer)

        val outputCount = metadataExtractor.outputTensorCount
        for (i in 0 until outputCount) {
            val tensorMetadata = metadataExtractor.getOutputTensorMetadata(i)
            val humanName = tensorMetadata?.name()
            if (humanName != null) {
                modelOutputIndexesMap[humanName] = i
            }
        }
    }

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
        // load and config model
        val modelBuffer: ByteBuffer = loadModelFile("setgame-classify.tflite")
        // we don't want to guess idexes - lets read them
        updataModelOutputIndexesMap(modelBuffer)

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

    private fun getRotArgFromRotation(rotation: Int) : Int {
        return when (rotation) {
            Surface.ROTATION_270 ->
                0
            Surface.ROTATION_180 ->
                1
            Surface.ROTATION_90 ->
                0
            else ->
                1
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
                setupClassifier()   // hmm???? not sure. what if we competing with clearClassifier
            }

            val imageProcessor =
                ImageProcessor.Builder()
                    .add(Rot90Op(getRotArgFromRotation(rotation)))
                    .add(ResizeOp(getImageSizeX(), getImageSizeY(), ResizeOp.ResizeMethod.BILINEAR))
                    .build()

            // Preprocess the image and convert it into a TensorImage for classification.
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(image)
            val tensorImage = imageProcessor.process(ti)

            // we have 4 heads
            val countOut = ByteBuffer.allocateDirect(OUTPUT_CLASSES * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            val colorOut = ByteBuffer.allocateDirect(OUTPUT_CLASSES * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            val fillOut = ByteBuffer.allocateDirect(OUTPUT_CLASSES * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            val shapeOut = ByteBuffer.allocateDirect(OUTPUT_CLASSES * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            try {
                val outputs = mutableMapOf<Int, Any>(
                    // the same sequence that in model (I've made a mistake - swapped count and color)
                    0 to colorOut,
                    1 to countOut,
                    2 to shapeOut,
                    3 to fillOut
//                    modelOutputIndexesMap["count_output"]!! to countOut,
//                    modelOutputIndexesMap["color_output"]!! to colorOut,
//                    modelOutputIndexesMap["fill_output"]!! to fillOut,
//                    modelOutputIndexesMap["shape_output"]!! to shapeOut
                )

                interpreter?.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputs)
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
            if (res[i] != null && res[i].isNotEmpty()) {
                val category = res[i].first()
                if (category != null) {
                    r[i] = mutableListOf(category)
                }
            }
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


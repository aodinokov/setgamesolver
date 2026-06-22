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

import android.graphics.Bitmap
import android.content.Context
import android.graphics.RectF
import android.util.Log
import android.view.Surface
import com.github.aodinokov.setgamesolver.fragments.DelegationMode
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.vision.detector.Detection
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel.MapMode
import java.util.concurrent.locks.ReentrantLock

data class DetectionResult(
    val boundingBox: RectF,
    val confidence: Float,
    val classId: Int
)

class DetectorHelper(
        val context: Context,
        var threshold: Float = 0.5f,
        var maxResults: Int = 30,
        var numThreads: Int = 2,
        var currentDelegate: DelegationMode = DelegationMode.Cpu,
        var currentModel: Int = 0,
        var detectorErrorListener: DetectorErrorListener? = null,
        var detectorResultsListener: DetectorResultsListener? = null
) {
    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private val classifierLock = ReentrantLock()
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: NnApiDelegate? = null

    protected fun getImageSizeX(): Int {
        return 320
    }
    protected fun getImageSizeY(): Int {
        return 320
    }


    fun clearDetector() {
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

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    private fun setupDetector() {
        // load and config model
        val modelBuffer: ByteBuffer = loadModelFile("setgame-detect.tflite")

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
        return when (rotation/90) {
            Surface.ROTATION_270 ->
                3
            Surface.ROTATION_180 ->
                2
            Surface.ROTATION_90 ->
                1
            else ->
                0
        }
    }

    private val outputElementCount = 1 * 300 * 6
    private val outputBuffer: ByteBuffer = ByteBuffer.allocateDirect(outputElementCount * 4).apply {
        // CRITICAL: Native byte order (usually Little Endian on ARM/Android)
        order(ByteOrder.nativeOrder())
    }

    fun detect(image: Bitmap, imageRotation: Int): Triple<List<DetectionResult>?, Int, Int> {
        val rotation = getRotArgFromRotation(imageRotation)
        // Determine the target dimensions based on whether the image was flipped 90/270 degrees
        val isFlipped = rotation == 1 || rotation == 3
        val targetW = if (isFlipped) image.height else image.width
        val targetH = if (isFlipped) image.width else image.height

        if (!classifierLock.tryLock()) {
            return Triple(null, targetH, targetW)
        }
        try {
            if (interpreter == null) {
                setupDetector()   // hmm???? not sure. what if we're competing with clearClassifier
            }

            val imageProcessor =
                ImageProcessor.Builder()
                    .add(Rot90Op(-getRotArgFromRotation(imageRotation)))
                    .add(ResizeOp(getImageSizeX(), getImageSizeY(), ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0.0f, 255.0f))
                    .build()

            // Preprocess the image and convert it into a TensorImage for classification.
            val ti = TensorImage(DataType.FLOAT32)
            ti.load(image)
            val tensorImage = imageProcessor.process(ti)

            try {
                // Rewind buffers before use
                outputBuffer.rewind()
                // run
                interpreter?.run(tensorImage.buffer, outputBuffer)

                val results = parseOutputBuffer(threshold,
                    targetH, targetW, rotation)
                detectorResultsListener?.onDetectorResults(results,
                    targetH, targetW)
                return Triple(results, targetH, targetW)
            } catch (e: Exception) {
                Log.e("TFLite", "Inference failed", e)
                return Triple(null, targetH, targetW)
            }
        } finally {
            classifierLock.unlock()
        }
    }

    private fun parseOutputBuffer(confidenceThreshold: Float, targetH: Int, targetW: Int, rotation: Int): List<DetectionResult> {
        val detections = mutableListOf<DetectionResult>()

        // Rewind to read from the beginning
        outputBuffer.rewind()

        // View the ByteBuffer as a FloatBuffer for easy float reads
        val floatBuffer = outputBuffer.asFloatBuffer()

        // Loop through all 300 bounding box proposals
        for (i in 0 until 300) {
            val baseIndex = i * 6

            // Layout: [x1, y1, x2, y2, confidence, class_id]
            val confidence = floatBuffer.get(baseIndex + 4)

            if (confidence >= confidenceThreshold) {

                detections.add(
                    DetectionResult(
                        boundingBox = RectF(
                            floatBuffer.get(baseIndex + 0) * targetW,
                            floatBuffer.get(baseIndex + 1) * targetH,
                            floatBuffer.get(baseIndex + 2) * targetW,
                            floatBuffer.get(baseIndex + 3) * targetH
                        ),
                        confidence = confidence,
                        classId = floatBuffer.get(baseIndex + 5).toInt()
                    )
                )
            }
        }
        return detections
    }

    interface DetectorErrorListener {
        fun onDetectorError(error: String)
    }

    interface DetectorResultsListener {
        fun onDetectorResults(
                results: List<DetectionResult>?,
                imageHeight: Int,
                imageWidth: Int
        )
    }

    companion object {
        const val MODEL_SETGAME = 0
    }
}


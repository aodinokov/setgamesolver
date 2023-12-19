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
package com.github.aodinokov.setgameresolver

import android.graphics.Bitmap
import android.content.Context
import android.util.Log
import com.github.aodinokov.setgameresolver.fragments.DelegationMode
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector

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
    private var detector: ObjectDetector? = null

    fun clearDetector() {
        detector = null
    }

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    private fun setupDetector() {
        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DelegationMode.Cpu -> {
                // Default
            }
            DelegationMode.Gpu -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    detectorErrorListener?.onDetectorError("Detector creations: GPU is not supported on this device")
                }
            }
            DelegationMode.Nnapi -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName =
            when (currentModel) {
                MODEL_SETGAME -> "setgame-detect.tflite"
                MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
                else -> "mobilenetv1.tflite"
            }

        try {
            detector =
                ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            detectorErrorListener?.onDetectorError(
                "Object detector failed to initialize. See error logs for details"
            )
            Log.e("Test", "TFLite failed to load model with error: " + e.message)
        }
    }

    fun detect(image: Bitmap, imageRotation: Int): Triple<List<Detection>?, Int, Int> {
        // ensure all tools are ready
        if (detector == null) {
            setupDetector()
        }
        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        val imageProcessor =
            ImageProcessor.Builder()
                .add(Rot90Op(-imageRotation / 90))
                .build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val results = detector?.detect(tensorImage)
        detectorResultsListener?.onDetectorResults(
            results,
            tensorImage.height,
            tensorImage.width)
        return Triple(results, tensorImage.height, tensorImage.width)
    }

    interface DetectorErrorListener {
        fun onDetectorError(error: String)
    }

    interface DetectorResultsListener {
        fun onDetectorResults(
                results: List<Detection>?,
                imageHeight: Int,
                imageWidth: Int
        )
    }

    companion object {
        const val MODEL_SETGAME = 0
        const val MODEL_MOBILENETV1 = 1
    }
}


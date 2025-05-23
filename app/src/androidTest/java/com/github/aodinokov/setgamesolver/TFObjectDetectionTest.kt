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

package com.github.aodinokov.setgamesolver

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.InputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.vision.detector.Detection
/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class TFObjectDetectionTest {

    val controlResults = listOf<Detection>(
        Detection.create(RectF(69.0f, 58.0f, 227.0f, 171.0f),
            listOf<Category>(Category.create("cat", "cat", 0.77734375f))),
        Detection.create(RectF(15.0f, 5.0f, 285.0f, 214.0f),
            listOf<Category>(Category.create("couch", "couch", 0.5859375f))),
            Detection.create(RectF(45.0f, 27.0f, 257.0f, 184.0f),
            listOf<Category>(Category.create("chair", "chair", 0.55078125f)))
    )

    @Test
    @Throws(Exception::class)
    fun detectionResultsShouldNotChange() {
        val setgameDetectorHelper =
                DetectorHelper(
                context = InstrumentationRegistry.getInstrumentation().context,
                currentModel = DetectorHelper.MODEL_MOBILENETV1,
                detectorErrorListener =
                    object : DetectorHelper.DetectorErrorListener {
                        override fun onDetectorError(error: String) {
                            // no op
                        }
                    },
                detectorResultsListener =
                object : DetectorHelper.DetectorResultsListener {
                        override fun onDetectorResults(
                            results: List<Detection>?,
                            imageHeight: Int,
                            imageWidth: Int
                        ) {

                            assertEquals(controlResults.size, results!!.size)

                            // Loop through the detected and control data
                            for (i in controlResults.indices) {
                                // Verify that the bounding boxes are the same
                                assertEquals(controlResults[i].getBoundingBox(), results[i].getBoundingBox())

                                // Verify that the detected data and control
                                // data have the same number of categories
                                assertEquals(
                                        controlResults[i].getCategories().size,
                                        results[i].getCategories().size
                                )

                                // Loop through the categories
                                for (j in 0 until controlResults[i].categories.size - 1) {
                                    // Verify that the labels are consistent
                                    assertEquals(
                                            controlResults[i].getCategories()[j].label,
                                            results[i].getCategories()[j].label
                                    )
                                }
                            }
                        }
                    }
            )
        // Create Bitmap and convert to TensorImage
        val bitmap = loadImage("cat1.png")
        // Run the object detector on the sample image
        setgameDetectorHelper.detect(bitmap!!, 0)
    }

    @Test
    @Throws(Exception::class)
    fun detectedImageIsScaledWithinModelDimens() {
        val setgameDetectorHelper =
                DetectorHelper(
                context = InstrumentationRegistry.getInstrumentation().context,
                currentModel = DetectorHelper.MODEL_MOBILENETV1,
                    detectorErrorListener =
                    object : DetectorHelper.DetectorErrorListener {
                        override fun onDetectorError(error: String) {
                            // no op
                        }
                    },
                    detectorResultsListener =
                    object : DetectorHelper.DetectorResultsListener {
                        override fun onDetectorResults(
                                results: List<Detection>?,
                                imageHeight: Int,
                                imageWidth: Int
                        ) {
                            assertNotNull(results)
                            for (result in results!!) {
                                assertTrue(result.getBoundingBox().top <= imageHeight)
                                assertTrue(result.getBoundingBox().bottom <= imageHeight)
                                assertTrue(result.getBoundingBox().left <= imageWidth)
                                assertTrue(result.getBoundingBox().right <= imageWidth)
                            }
                        }
                    }
            )

            // Create Bitmap and convert to TensorImage
            val bitmap = loadImage("cat1.png")
            // Run the object detector on the sample image
            setgameDetectorHelper.detect(bitmap!!, 0)
    }

    @Test
    @Throws(Exception::class)
    fun someResultsForSetgame_test1() {
        val setgameDetectorHelper =
                DetectorHelper(
                        context = InstrumentationRegistry.getInstrumentation().context,
                        currentModel = DetectorHelper.MODEL_SETGAME,
                        detectorErrorListener =
                        object : DetectorHelper.DetectorErrorListener {
                            override fun onDetectorError(error: String) {
                                // no op
                            }
                        },
                        detectorResultsListener =
                        object : DetectorHelper.DetectorResultsListener {
                            override fun onDetectorResults(
                                    results: List<Detection>?,
                                    imageHeight: Int,
                                    imageWidth: Int
                            ) {
                                assertNotNull(results)
                                assert(results!!.isNotEmpty())
                            }
                        }
                )

        // Create Bitmap and convert to TensorImage
        val bitmap = loadImage("setgame_set1.jpg")
        // Run the object detector on the sample image
        setgameDetectorHelper.detect(bitmap!!, 0)
    }

    @Throws(Exception::class)
    private fun loadImage(fileName: String): Bitmap? {
        val assetManager: AssetManager =
            InstrumentationRegistry.getInstrumentation().context.assets
        val inputStream: InputStream = assetManager.open(fileName)
        return BitmapFactory.decodeStream(inputStream)
    }
}

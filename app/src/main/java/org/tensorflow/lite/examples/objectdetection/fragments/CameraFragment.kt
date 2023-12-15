/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
package org.tensorflow.lite.examples.objectdetection.fragments

import android.annotation.SuppressLint
import android.app.AlarmManager
import android.app.Dialog
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.preference.PreferenceManager
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import org.tensorflow.lite.examples.objectdetection.CardColor
import org.tensorflow.lite.examples.objectdetection.CardNumber
import org.tensorflow.lite.examples.objectdetection.CardShading
import org.tensorflow.lite.examples.objectdetection.CardShape
import org.tensorflow.lite.examples.objectdetection.CardValue
import org.tensorflow.lite.examples.objectdetection.OverlayView
import org.tensorflow.lite.examples.objectdetection.R
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding
import java.util.LinkedList
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.pow
import android.content.res.TypedArray
import android.graphics.RectF
import android.os.SystemClock
import com.google.gson.Gson
import org.tensorflow.lite.examples.objectdetection.AbstractCard
import org.tensorflow.lite.examples.objectdetection.ClassifierHelper
import org.tensorflow.lite.examples.objectdetection.DetectorHelper
import org.tensorflow.lite.examples.objectdetection.SimpleCard
import org.tensorflow.lite.examples.objectdetection.CardSet
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.vision.detector.Detection
import java.lang.Float.max
import kotlin.math.absoluteValue

enum class SetsFinderMode(val mode: Int) {
    AllSets(0),
    NonOverlappingSets(1),
}

/**
 * 9x9 Bitmap with adjustable order
 */
class ThumbnailsBitmapHelper(
        val thumbnailsBitmap: Bitmap,
        // maps & weights (higher - bigger)
        val numberWeight: Int = 3,
        val numberMap: Map<Int,Int>? = null,
        val colorWeight: Int = 2,
        val colorMap: Map<Int,Int>?= null,
        val shadingWeight: Int = 1,
        val shadingMap: Map<Int,Int>?= null,
        val shapeWeight: Int = 0,
        val shapeMap: Map<Int,Int>?= null) {

    fun getThumbIndex(cardValue: CardValue):Int {
        var number = cardValue.number.code - 1
        var color = cardValue.color.code - 1
        var shading = cardValue.shading.code - 1
        var shape = cardValue.shape.code - 1

        if (numberMap != null) number = numberMap[number]!!
        if (colorMap != null) color = colorMap[color]!!
        if (shadingMap != null) shading = shadingMap[shading]!!
        if (shapeMap != null) shape = shapeMap[shape]!!

        val idx = number * 3.0.pow(numberWeight).toInt() +
                color * 3.0.pow(colorWeight).toInt() +
                shading * 3.0.pow(shadingWeight).toInt() +
                shape * 3.0.pow(shapeWeight).toInt()
        assert(idx in 0..80)

        return idx
    }
    fun getThumbColumn(idx: Int): Int {
        return idx % 9
    }
    fun getThumbRow(idx: Int): Int {
        return idx / 9
    }
    fun getSingleThumbBitmap(idx: Int):Bitmap {
        val column = getThumbColumn(idx)
        val row = getThumbRow(idx)
        val src = Rect(
                thumbnailsBitmap.width / 9 * column,
                thumbnailsBitmap.height / 9 * row,
                thumbnailsBitmap.width / 9 * (column + 1),
                thumbnailsBitmap.height / 9 * (row + 1))
        return Bitmap.createBitmap(thumbnailsBitmap,
                thumbnailsBitmap.width / 9 * column,
                thumbnailsBitmap.height / 9 * row,
                thumbnailsBitmap.width / 9,
                thumbnailsBitmap.height / 9)
    }
}

class SelectedCamera(    val auto: Boolean = true,
                         val facing: Int = 0,
                         val cameraId: String = "",
                         val stringRepresentation: String = "") {
    override fun toString(): String {
        return stringRepresentation
    }
}

class SelectedCameraResolution(val parent: SelectedCamera,
                               val size: Size,
                               val stringRepresentation: String = ""
){
    override fun toString(): String {
        if (stringRepresentation == "") {
            return size.toString()
        }
        return stringRepresentation
    }
}

data class BoundingBoxTransformation(
        var deltaX: Float,
        var deltaY: Float,
        var scaleX:Float,
        var scaleY: Float)

class CardClassifierZone(var boundingBox: RectF) {
    private var prevBoundingBox: RectF? = null
    fun isWithinBoundingBox(boundingBox: RectF): Boolean {
        if ((this.boundingBox.centerX() - boundingBox.centerX()).absoluteValue < this.boundingBox.width()/2 + boundingBox.width()/2 &&
                (this.boundingBox.centerY() - boundingBox.centerY()).absoluteValue < this.boundingBox.height()/2 + boundingBox.height()/2) {
            return true
        }
        return false
    }
    fun updateBoundingBox(boundingBox: RectF) {
        prevBoundingBox = this.boundingBox
        this.boundingBox = boundingBox
    }
    fun getLastBoundingBoxTransformation(): BoundingBoxTransformation? {
        if (
                prevBoundingBox == null ||
                prevBoundingBox!!.width() == 0.0f ||
                prevBoundingBox!!.height() == 0.0f)
            return null
        return BoundingBoxTransformation(
                this.boundingBox.centerX() - prevBoundingBox!!.centerX(),
                this.boundingBox.centerY() - prevBoundingBox!!.centerY(),
                this.boundingBox.width()/prevBoundingBox!!.width(),
                this.boundingBox.height()/prevBoundingBox!!.height()
        )
    }
    fun applyBoundingBoxTransformation(t: BoundingBoxTransformation) {
        prevBoundingBox = RectF(this.boundingBox)
        val boundsTmp = RectF(
                prevBoundingBox!!.left + t.deltaX,
                prevBoundingBox!!.top + t.deltaY,
                prevBoundingBox!!.right + t.deltaX,
                prevBoundingBox!!.bottom + t.deltaY,
        )
        //now scale
        this.boundingBox.left = boundsTmp.centerX()-boundsTmp.width()*t.scaleX/2
        this.boundingBox.top = boundsTmp.centerY()-boundsTmp.height()*t.scaleX/2
        this.boundingBox.right = boundsTmp.centerX()+boundsTmp.width()*t.scaleX/2
        this.boundingBox.bottom = boundsTmp.centerY()+boundsTmp.height()*t.scaleX/2
    }

    // lock if edit is in progress
    var editIsProgress = false
    var deleteNextCycle = false

    var overriddenValue: CardValue? = null
    private var categoriesMax = Array<Category?>(4, {null})
    fun updateCategories(newCategories: Array<MutableList<Category>>?) {
        if (newCategories == null)
            return
        assert(newCategories.size == categoriesMax.size)
        for (i in 0 until categoriesMax.size) {
            if (newCategories[i].size == 0)
                continue
            val newCat = newCategories[i][0]
            if (categoriesMax[i] == null || categoriesMax[i]!!.score <= newCat.score)
                categoriesMax[i] = newCat
        }
    }
    fun getCategories(): MutableList<Category> {
        var res = LinkedList<Category>()
        if (categoriesMax[0]!= null &&
                categoriesMax[1]!= null &&
                categoriesMax[2]!= null &&
                categoriesMax[3]!= null) {
            res.add(Category(
                    categoriesMax[0]!!.label + "-" +
                            categoriesMax[1]!!.label + "-" +
                            categoriesMax[2]!!.label + "-" +
                            categoriesMax[3]!!.label,
                    categoriesMax[0]!!.score *
                            categoriesMax[1]!!.score *
                            categoriesMax[2]!!.score *
                            categoriesMax[3]!!.score))
        }
        return res
    }
    fun getValue(): CardValue? {
        if (overriddenValue != null)
            return overriddenValue!!
        val cats = getCategories()
        if (cats.size > 0) {
            return CardValue.fromString(cats[0].label)
        }
        return null
    }

    private var detectedTime: Long = SystemClock.uptimeMillis()
    fun updateDetectedTime(detectedTime: Long = SystemClock.uptimeMillis()) {
        this.detectedTime = detectedTime
    }
    fun isDetectionOutdated():Boolean {
        if (editIsProgress || overriddenValue  != null)
            return false
        // if time when it was last time re-detected or re-classified is more then const (e.g. 2sec)
        return  (SystemClock.uptimeMillis() - detectedTime) < 30000 // can be in cycles, not in real time. what is better?
    }
    fun isReClassifyCandidate(): Boolean {
        if (editIsProgress || overriddenValue  != null)
            return false
        return SystemClock.uptimeMillis() - detectedTime < 1500 // can be in cycles, not in real time. what is better?
    }

    // TODO: to remove
    public var groups = HashSet<Int>()
    // Group-able interface impl
    fun getGroupIds(): Set<Int> {
        return groups
    }
}

/**
 * Inherit SimpleCard, but also store list of Classifier Zones that produced that.
 * There can be several - in that case there are duplicates and we need
 * to highlight them
 */
class ClassifiedCard(v: CardValue): SimpleCard(v) {
    val zones = LinkedList<CardClassifierZone>()
}

enum class DelegationMode(val mode: Int) {
    Cpu(0),
    Gpu(1),
    Nnapi(2),
    ;
    companion object {
        fun fromInt(value: Int): DelegationMode? {
            return try {
                DelegationMode.values().first { it.mode == value }
            } catch (ex: NoSuchElementException) {
                null
            }
        }
    }
}

class CameraFragment : Fragment(),
        DetectorHelper.DetectorErrorListener,
        ClassifierHelper.ClassifierErrorListener {

    private val TAG = "ObjectDetection"

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    /** overlay painting helper objects*/
    private var thumbnailsBitmapHelper: ThumbnailsBitmapHelper? = null

    /** image work helpers */
    private lateinit var detectorHelper: DetectorHelper
    private lateinit var classifierHelper: ClassifierHelper

    private lateinit var bitmapBuffer: Bitmap
    /** Blocking camera operations are performed using this executor */
    private lateinit var cameraExecutor: ExecutorService

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    /** modes */
    private var setsFinderMode = SetsFinderMode.AllSets
    private var scanIsInProgress: Boolean = false

    /** overlay data to show*/
    private var scaleFactor: Float = 1f
    private var inferenceTime: Long = 0

    /* raw data after detection */
    private var rawDetectionResults: List<Detection> = LinkedList<Detection>()
    /* we're keeping special objects that classify cards. override is a part of functionality */
    private var cardClassifierZones = LinkedList<CardClassifierZone>()
    /* resulting cards we were able to collect from classifier zones */
    private var cards = HashMap<CardValue, ClassifiedCard>()

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(CameraFragmentDirections.actionCameraToPermissions())
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        cameraExecutor.shutdown()
    }

    override fun onCreateView(
      inflater: LayoutInflater,
      container: ViewGroup?,
      savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        detectorHelper = DetectorHelper(context = requireContext(), detectorErrorListener = this)
        classifierHelper = ClassifierHelper(context = requireContext(), classifierErrorListener = this)

        readPreferences()

        if (thumbnailsBitmapHelper == null) {
            val inputStream = resources.assets.open("setgame-cards.png")
            val thumbnailsBitmap = BitmapFactory.decodeStream(inputStream)
            thumbnailsBitmapHelper = ThumbnailsBitmapHelper(thumbnailsBitmap,
                    numberWeight = 0,
                    colorWeight = 1,
                    shapeWeight = 2,
                    shadingWeight = 3,
                    colorMap = mapOf<Int,Int>(Pair(0,2),Pair(1,1),Pair(2,0)),
                    shadingMap = mapOf<Int,Int>(Pair(0,2),Pair(1,0),Pair(2,1)),
                    shapeMap = mapOf<Int,Int>(Pair(0,1),Pair(1,2),Pair(2,0)))
        }

        // Initialize our background executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        // Attach listeners to UI control widgets
        initControlsUi()

        // Reuse the function to set text fields based on the current data
        updateTextControlsUi()
    }

    // settings read/ write support
    // add fun UnmarshalV<N>toState(yaml: String)  when needed
    private fun unmarshalV1toState(jsonString: String) {
        val gson = Gson()
        val state = gson.fromJson(jsonString, Map::class.java)
        if (state != null) {
            // read to the settings with conversion
            if (state.containsKey("setsFinderMode")) {
                try {
                    setsFinderMode = SetsFinderMode.valueOf(state["setsFinderMode"].toString())
                }catch (_: IllegalArgumentException) { }
            }
            if (state.containsKey("detectorMode")){
                detectorHelper.threshold = state["detThreshold"].toString().toFloat()
            }
            if (state.containsKey("detMaxResults")){
                detectorHelper.maxResults = state["detMaxResults"].toString().toInt()
            }
            if (state.containsKey("classThreshold")){
                classifierHelper.threshold = state["classThreshold"].toString().toFloat()
            }
            if (state.containsKey("numThreads")){
                detectorHelper.numThreads = state["numThreads"].toString().toInt()
                classifierHelper.numThreads = detectorHelper.numThreads
            }
            if (state.containsKey("currentDelegate")){
                val currentDelegate = DelegationMode.fromInt(state["currentDelegate"].toString().toInt())
                if (currentDelegate != null) {
                    detectorHelper.currentDelegate = currentDelegate
                    classifierHelper.currentDelegate = currentDelegate
                }
            }
            if (state.containsKey("currentModel")){
                detectorHelper.currentModel = state["currentModel"].toString().toInt()
            }
        }
    }

    private fun marshalV1toState(): String {
        val gson = Gson()
        val state = mutableMapOf<String, String>()
        state["setsFinderMode"] = setsFinderMode.toString()
        state["detThreshold"] = detectorHelper.threshold.toString()
        state["detMaxResults"] = detectorHelper.maxResults.toString()
        state["classThreshold"] = classifierHelper.threshold.toString()
        state["numThreads"] = detectorHelper.numThreads.toString()
        state["currentDelegate"] = detectorHelper.currentDelegate.mode.toString()
        state["currentModel"] = detectorHelper.currentModel.toString()
        return gson.toJson(state)
    }

    private fun readPreferences() {
        val preferences = PreferenceManager.getDefaultSharedPreferences(context)
        // Retrieve a string value
        val configVersion = preferences.getString("config_version", "")
        val config = preferences.getString("config_json", "")
        if (config != null) {
            if (configVersion == "1") {
                unmarshalV1toState(config)
            }
        }
    }

    private fun writePreferences() {
        val preferences = PreferenceManager.getDefaultSharedPreferences(context)
        val editor = preferences.edit()
        editor.putString("config_version", "1")
        editor.putString("config_json", marshalV1toState())
        editor.apply()
    }


    @SuppressLint("ClickableViewAccessibility")
    private fun initControlsUi() {
        // When clicked, change the underlying model used for object detection
        fragmentCameraBinding.bottomSheetLayout.spinnerMode.setSelection(setsFinderMode.mode, false)
        fragmentCameraBinding.bottomSheetLayout.spinnerMode.onItemSelectedListener =
                object : AdapterView.OnItemSelectedListener {
                    override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                        setsFinderMode =
                                when (p2) {
                                    0 -> SetsFinderMode.AllSets
                                    1 -> SetsFinderMode.NonOverlappingSets
                                    else -> SetsFinderMode.AllSets
                                }
                        updateTextControlsUi()
                        writePreferences()
                    }

                    override fun onNothingSelected(p0: AdapterView<*>?) {
                        /* no op */
                    }
                }

        // When clicked, lower detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detThresholdMinus.setOnClickListener {
            if (detectorHelper.threshold >= 0.1) {
                detectorHelper.threshold -= 0.1f
                detectorHelper.clearDetector()
                updateTextControlsUi()
                writePreferences()
            }
        }

        // When clicked, raise detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detThresholdPlus.setOnClickListener {
            if (detectorHelper.threshold <= 0.8) {
                detectorHelper.threshold += 0.1f
                detectorHelper.clearDetector()
                updateTextControlsUi()
                writePreferences()
            }
        }

        // When clicked, reduce the number of objects that can be detected at a time
        fragmentCameraBinding.bottomSheetLayout.detMaxResultsMinus.setOnClickListener {
            if (detectorHelper.maxResults > 1) {
                detectorHelper.maxResults--
                detectorHelper.clearDetector()
                updateTextControlsUi()
                writePreferences()
            }
        }

        // When clicked, increase the number of objects that can be detected at a time
        fragmentCameraBinding.bottomSheetLayout.detMaxResultsPlus.setOnClickListener {
            if (detectorHelper.maxResults < 32) {
                detectorHelper.maxResults++
                detectorHelper.clearDetector()
                updateTextControlsUi()
                writePreferences()
            }
        }

        // When clicked, lower classifier score threshold floor
        fragmentCameraBinding.bottomSheetLayout.classThresholdMinus.setOnClickListener {
            if (classifierHelper.threshold >= 0.1) {
                classifierHelper.threshold -= 0.1f
                classifierHelper.clearClassifier()
                updateTextControlsUi()
                writePreferences()
            }
        }

        // When clicked, raise classifier score threshold floor
        fragmentCameraBinding.bottomSheetLayout.classThresholdPlus.setOnClickListener {
            if (classifierHelper.threshold <= 0.8) {
                classifierHelper.threshold += 0.1f
                classifierHelper.clearClassifier()
                updateTextControlsUi()
                writePreferences()
            }
        }

        // When clicked, decrease the number of threads used for detection
        fragmentCameraBinding.bottomSheetLayout.threadsMinus.setOnClickListener {
            if (detectorHelper.numThreads > 1) {
                detectorHelper.numThreads--
                // sync classifier setting with detector
                classifierHelper.numThreads = detectorHelper.numThreads
                detectorHelper.clearDetector()
                classifierHelper.clearClassifier()
                updateTextControlsUi()
                writePreferences()
            }
        }

        // When clicked, increase the number of threads used for detection
        fragmentCameraBinding.bottomSheetLayout.threadsPlus.setOnClickListener {
            if (detectorHelper.numThreads < 4) {
                detectorHelper.numThreads++
                // sync classifier setting with detector
                classifierHelper.numThreads = detectorHelper.numThreads
                detectorHelper.clearDetector()
                classifierHelper.clearClassifier()
                updateTextControlsUi()
                writePreferences()
            }
        }

        // When clicked, change the underlying hardware used for inference. Current options are CPU
        // GPU, and NNAPI
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(detectorHelper.currentDelegate.mode, false)
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                    detectorHelper.currentDelegate = DelegationMode.fromInt(p2)!!
                    // sync classifier setting with detector
                    classifierHelper.currentDelegate = detectorHelper.currentDelegate
                    detectorHelper.clearDetector()
                    classifierHelper.clearClassifier()
                    updateTextControlsUi()
                    writePreferences()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        // When clicked, change the underlying model used for object detection
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.setSelection(detectorHelper.currentModel, false)
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                    detectorHelper.currentModel = p2
                    detectorHelper.clearDetector()

                    // it's better to clear it, because it's only relevant to setCards
                    cardClassifierZones.clear()

                    updateTextControlsUi()
                    writePreferences()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        fragmentCameraBinding.bottomSheetLayout.startstopButton.setOnClickListener {
            /* update button */
            scanIsInProgress = !scanIsInProgress

            /* reset all data */
            rawDetectionResults = LinkedList<Detection>()
            cardClassifierZones.clear()

            updateTextControlsUi()
        }

        val resolutionDialog = Dialog(requireContext())
        fragmentCameraBinding.bottomSheetLayout.resolutionButton.setOnClickListener {
            resolutionDialog.setContentView(R.layout.resolution_dialog)
            resolutionDialog.window!!.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            resolutionDialog.setCancelable(false)

            val spinnerCamera = resolutionDialog.findViewById<Spinner>(R.id.spinner_camera)
            val spinnerResolution = resolutionDialog.findViewById<Spinner>(R.id.spinner_resolution)
            val okayText = resolutionDialog.findViewById<TextView>(R.id.okay_text)
            val cancelText = resolutionDialog.findViewById<TextView>(R.id.cancel_text)

            okayText.setOnClickListener(View.OnClickListener {
                //store preferences
                val preferences = PreferenceManager.getDefaultSharedPreferences(context)
                val c = preferences.getString("camera", "")
                val r = preferences.getString("resolution", "")

                var configChanged = false
                if (c != spinnerCamera.adapter.getItem(spinnerCamera.selectedItemId.toInt()).toString() ||
                        r != spinnerResolution.adapter.getItem(spinnerResolution.selectedItemId.toInt()).toString()) {
                    configChanged = true
                    val editor = preferences.edit()
                    editor.putString("camera", spinnerCamera.adapter.getItem(spinnerCamera.selectedItemId.toInt()).toString())
                    editor.putString("resolution", spinnerResolution.adapter.getItem(spinnerResolution.selectedItemId.toInt()).toString())
                    editor.apply()
                }

                resolutionDialog.dismiss()

                if (configChanged) {
                    Toast.makeText(requireContext(), "restarting to apply changes", Toast.LENGTH_SHORT).show()
                    // restart app to apply the new settings
                    doRestart(requireContext())
                }
            })

            cancelText.setOnClickListener(View.OnClickListener {
                resolutionDialog.dismiss()
            })

            val cm = context?.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val camerasList = buildCamerasList(cm)
            val camerasAdapter = ArrayAdapter<SelectedCamera>(requireContext(), R.layout.spinner_item,  camerasList)
            spinnerCamera.adapter = camerasAdapter
            spinnerCamera.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                    val upperAdapter = p0!!.adapter
                    val camera = upperAdapter.getItem(p2)
                    // update resolution list if resolution has a different selected camera?
                    if (!spinnerResolution.adapter.isEmpty && (spinnerResolution.adapter.getItem(0) as SelectedCameraResolution).parent != camera) {
                        val adapter = ArrayAdapter<SelectedCameraResolution>(requireContext(), R.layout.spinner_item, buildCameraResolutionList(cm, camera as SelectedCamera))
                        spinnerResolution.adapter = adapter
                    }
                }
                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }
            if (camerasList.isNotEmpty()) {
                // read and set dialog
                val preferences = PreferenceManager.getDefaultSharedPreferences(context)

                val c = preferences.getString("camera", camerasList[0].toString())
                // find the current id
                var ci = camerasList.indexOfFirst { it.toString() == c }
                if (ci < 0){ ci = 0}
                spinnerCamera.setSelection(ci, false)

                val resolutionList = buildCameraResolutionList(cm, camerasList[ci])
                val resolutionAdapter = ArrayAdapter<SelectedCameraResolution>(requireContext(), R.layout.spinner_item, resolutionList)
                spinnerResolution.adapter = resolutionAdapter

                val r = preferences.getString("resolution", resolutionList[0].toString())
                // find the resolution id
                var ri = resolutionList.indexOfFirst { it.toString() == r }
                if (ri < 0){ ri = 0}
                spinnerResolution.setSelection(ri, false)
            }
            resolutionDialog.show()
        }

        // overlay draw procedure
        fragmentCameraBinding.overlay.setOnDrawListener(object : OverlayView.DrawListener {
            // colors
            private var boxPaint = Paint()
            private var textBackgroundPaint = Paint()
            private var textPaint = Paint()
            private var groupBoxPaintMap = HashMap<Int, Paint>()
            // tmp
            private var bounds = Rect()
            init {
                textBackgroundPaint.color = Color.BLACK
                textBackgroundPaint.style = Paint.Style.FILL
                textBackgroundPaint.textSize = 50f

                textPaint.color = Color.WHITE
                textPaint.style = Paint.Style.FILL
                textPaint.textSize = 50f

                boxPaint.color = ContextCompat.getColor(requireContext(), R.color.bounding_box_color)
                boxPaint.strokeWidth = 8F
                boxPaint.style = Paint.Style.STROKE

                // init colors for groups
                groupBoxPaintMap.clear()
                val colors: TypedArray = resources.obtainTypedArray(R.array.groupColors)
                for (i in 0 until colors.length()) {
                    val p = Paint()
                    p.color = colors.getColor(i, 0)
                    p.strokeWidth = 8F
                    p.style = Paint.Style.STROKE
                    groupBoxPaintMap[i] = p
                }
                colors.recycle()
            }
            override fun onDraw(canvas: Canvas){
                if (!scanIsInProgress)
                    return

                // show fps
                val fpsTop = 0f
                var fpsText = "fps:inf"
                if (inferenceTime > 0) {
                    // inferenceTime is in ms
                    fpsText = String.format("fps:%2.0f", 1000.0/inferenceTime.toFloat())
                }
                // Draw rect behind display text
                textBackgroundPaint.getTextBounds(fpsText, 0, fpsText.length, bounds)
                val fpsTextWidth = bounds.width()
                val fpsTextHeight = bounds.height()
                canvas.drawRect(
                        0f,
                        fpsTop,
                        0f + fpsTextWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                        fpsTop + fpsTextHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                        textBackgroundPaint
                )
                // Draw text for detected object
                canvas.drawText(fpsText, 0f, fpsTop+bounds.height(), textPaint)

                //show raw results
                for (result in rawDetectionResults) {
                    val boundingBox = result.getBoundingBox()

                    val top = boundingBox.top * scaleFactor
                    val bottom = boundingBox.bottom * scaleFactor
                    val left = boundingBox.left * scaleFactor
                    val right = boundingBox.right * scaleFactor

                    // Draw bounding box around detected objects
                    canvas.drawRect(RectF(left, top, right, bottom), boxPaint)

                    if (detectorHelper.currentModel != DetectorHelper.MODEL_SETGAME && result.categories.size > 0) {
                        // Create text to display alongside detected objects
                        val shiftX = 0
                        val label = result.categories[0].label + " "
                        val drawableText = label + String.format("%.2f", result.categories[0].score)

                        // Draw rect behind display text
                        textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
                        val textWidth = bounds.width()
                        val textHeight = bounds.height()
                        canvas.drawRect(
                                left + shiftX,
                                top,
                                left + shiftX + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                                top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                                textBackgroundPaint
                        )

                        // Draw text for detected object
                        canvas.drawText(drawableText, left + shiftX, top + bounds.height(), textPaint)
                    }
                }
                // TODO: to rework
                // show cards
                assert(thumbnailsBitmapHelper != null)
                for (cardClassifierZone in cardClassifierZones) {
                    val boundingBox = cardClassifierZone.boundingBox

                    val top = boundingBox.top * scaleFactor
                    val bottom = boundingBox.bottom * scaleFactor
                    val left = boundingBox.left * scaleFactor
                    val right = boundingBox.right * scaleFactor

                    // Draw bounding box around detected objects
                    val drawableRect = RectF(left, top, right, bottom)
                        for(groupId in cardClassifierZone.getGroupIds()) {
                            assert(groupBoxPaintMap.size > 0)
                            val pb = groupBoxPaintMap.get(groupId%groupBoxPaintMap.size)!!
                            canvas.drawRect(drawableRect, pb)
                            // make all groups visible
                            drawableRect.left -= pb.strokeWidth
                            drawableRect.top -= pb.strokeWidth
                            drawableRect.bottom += pb.strokeWidth
                            drawableRect.right += pb.strokeWidth
                        }
                    val cardValue = cardClassifierZone.getValue()
                    if (cardValue != null) {
                        // Create text to display alongside detected objects
                        val shiftX = thumbnailsBitmapHelper!!.thumbnailsBitmap.width / 9 // we have put 9 cards in the row

                        val idx = thumbnailsBitmapHelper!!.getThumbIndex(cardValue)
                        val column = thumbnailsBitmapHelper!!.getThumbColumn(idx)
                        val row = thumbnailsBitmapHelper!!.getThumbRow(idx)

                        val src = Rect(
                                thumbnailsBitmapHelper!!.thumbnailsBitmap.width / 9 * column,
                                thumbnailsBitmapHelper!!.thumbnailsBitmap.height / 9 * row,
                                thumbnailsBitmapHelper!!.thumbnailsBitmap.width / 9 * (column + 1),
                                thumbnailsBitmapHelper!!.thumbnailsBitmap.height / 9 * (row + 1))
                        val dst = RectF(
                                left,
                                top,
                                left + thumbnailsBitmapHelper!!.thumbnailsBitmap.width / 9,
                                top + thumbnailsBitmapHelper!!.thumbnailsBitmap.height / 9)

                        canvas.drawBitmap(thumbnailsBitmapHelper!!.thumbnailsBitmap,
                                src,
                                dst,
                                textBackgroundPaint
                        )

                        // show only if it's not override
                        if (cardClassifierZone.overriddenValue == null) {
                            val drawableText = String.format("%.2f", cardClassifierZone.getCategories()[0].score)

                            // Draw rect behind display text
                            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
                            val textWidth = bounds.width()
                            val textHeight = bounds.height()
                            canvas.drawRect(
                                    left + shiftX,
                                    top,
                                    left + shiftX + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                                    top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                                    textBackgroundPaint
                            )

                            // Draw text for detected object
                            canvas.drawText(drawableText, left + shiftX, top + bounds.height(), textPaint)
                        }
                    }
                }
            }
        })

        // override dialog
        val overrideDialog = Dialog(requireContext())
        fragmentCameraBinding.overlay.setOnTouchListener(View.OnTouchListener { _, event ->
            assert(thumbnailsBitmapHelper != null)
            if (event != null &&
                    event.action == MotionEvent.ACTION_DOWN &&
                    this.scanIsInProgress) {
                overrideDialog.setContentView(R.layout.card_override_dialog)
                overrideDialog.window!!.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                overrideDialog.setCancelable(false)

                // controls
                val updateText = overrideDialog.findViewById<TextView>(R.id.update_text)
                val deleteText = overrideDialog.findViewById<TextView>(R.id.delete_text)
                val cancelText = overrideDialog.findViewById<TextView>(R.id.cancel_text)
                // variations per axis
                val minusCount = overrideDialog.findViewById<ImageButton>(R.id.minus_count)
                val plusCount = overrideDialog.findViewById<ImageButton>(R.id.plus_count)
                val minusColor = overrideDialog.findViewById<ImageButton>(R.id.minus_color)
                val plusColor = overrideDialog.findViewById<ImageButton>(R.id.plus_color)
                val minusFill = overrideDialog.findViewById<ImageButton>(R.id.minus_fill)
                val plusFill = overrideDialog.findViewById<ImageButton>(R.id.plus_fill)
                val minusShape = overrideDialog.findViewById<ImageButton>(R.id.minus_shape)
                val plusShape = overrideDialog.findViewById<ImageButton>(R.id.plus_shape)

                var currentDlgCardValue: CardValue? = null
                fun setView(currentCardValue: CardValue) {
                    currentDlgCardValue = currentCardValue

                    val current = overrideDialog.findViewById<ImageView>(R.id.current)
                    current.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(currentCardValue)))
                    // their values
                    val minusCountCard = CardValue(CardNumber.previous(currentCardValue.number), currentCardValue.color, currentCardValue.shading, currentCardValue.shape)
                    val plusCountCard = CardValue(CardNumber.next(currentCardValue.number), currentCardValue.color, currentCardValue.shading, currentCardValue.shape)
                    val minusColorCard = CardValue(currentCardValue.number, CardColor.previous(currentCardValue.color), currentCardValue.shading, currentCardValue.shape)
                    val plusColorCard = CardValue(currentCardValue.number, CardColor.next(currentCardValue.color), currentCardValue.shading, currentCardValue.shape)
                    val minusFillCard = CardValue(currentCardValue.number, currentCardValue.color, CardShading.previous(currentCardValue.shading), currentCardValue.shape)
                    val plusFillCard = CardValue(currentCardValue.number, currentCardValue.color, CardShading.next(currentCardValue.shading), currentCardValue.shape)
                    val minusShapeCard = CardValue(currentCardValue.number, currentCardValue.color, currentCardValue.shading, CardShape.previous(currentCardValue.shape))
                    val plusShapeCard = CardValue(currentCardValue.number, currentCardValue.color, currentCardValue.shading, CardShape.next(currentCardValue.shape))
                    // set the picture
                    minusCount.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(minusCountCard)))
                    minusCount.setOnClickListener(View.OnClickListener {
                        setView(minusCountCard)
                    })
                    plusCount.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(plusCountCard)))
                    plusCount.setOnClickListener(View.OnClickListener {
                        setView(plusCountCard)
                    })
                    minusColor.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(minusColorCard)))
                    minusColor.setOnClickListener(View.OnClickListener {
                        setView(minusColorCard)
                    })
                    plusColor.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(plusColorCard)))
                    plusColor.setOnClickListener(View.OnClickListener {
                        setView(plusColorCard)
                    })
                    minusFill.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(minusFillCard)))
                    minusFill.setOnClickListener(View.OnClickListener {
                        setView(minusFillCard)
                    })
                    plusFill.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(plusFillCard)))
                    plusFill.setOnClickListener(View.OnClickListener {
                        setView(plusFillCard)
                    })
                    minusShape.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(minusShapeCard)))
                    minusShape.setOnClickListener(View.OnClickListener {
                        setView(minusShapeCard)
                    })
                    plusShape.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(plusShapeCard)))
                    plusShape.setOnClickListener(View.OnClickListener {
                        setView(plusShapeCard)
                    })
                }

                // find if we pressed within any detected card? if so - propose to override
                for (cardClassifierZone in this.cardClassifierZones) {

                    val top = cardClassifierZone.boundingBox.top * scaleFactor
                    val bottom = cardClassifierZone.boundingBox.bottom * scaleFactor
                    val left = cardClassifierZone.boundingBox.left * scaleFactor
                    val right = cardClassifierZone.boundingBox.right * scaleFactor

                    if (event.rawX > left && event.rawX < right &&
                            event.rawY > top && event.rawY < bottom) {

                        // check that it's a card at all
                        val cardValue = cardClassifierZone.getValue() ?: continue
                        cardClassifierZone.editIsProgress = true
                        setView(cardValue)

                        cancelText.setOnClickListener(View.OnClickListener {
                            cardClassifierZone.editIsProgress = false
                            overrideDialog.dismiss()
                        })
                        updateText.setOnClickListener(View.OnClickListener {
                            cardClassifierZone.overriddenValue = currentDlgCardValue
                            cardClassifierZone.editIsProgress = false
                            overrideDialog.dismiss()
                        })
                        deleteText.setOnClickListener(View.OnClickListener {
                            //this.cardClassifierZones.remove(cardClassifierZone)
                            // doesn't work because we're working in the copy of the list
                            // instead we need to mark this object as forDeletion
                            cardClassifierZone.deleteNextCycle = true
                            cardClassifierZone.editIsProgress = false
                            overrideDialog.dismiss()
                        })

                        overrideDialog.show()
                        break
                    }
                }
            }
            return@OnTouchListener true
        })
    }

    // Update the values displayed in the bottom sheet. Reset detector.
    private fun updateTextControlsUi() {
        fragmentCameraBinding.bottomSheetLayout.detMaxResultsValue.text =
            detectorHelper.maxResults.toString()
        fragmentCameraBinding.bottomSheetLayout.detThresholdValue.text =
            String.format("%.2f", detectorHelper.threshold)
        fragmentCameraBinding.bottomSheetLayout.classThresholdValue.text =
                String.format("%.2f", classifierHelper.threshold)
        fragmentCameraBinding.bottomSheetLayout.threadsValue.text =
            detectorHelper.numThreads.toString()

        if (scanIsInProgress) {
            fragmentCameraBinding.bottomSheetLayout.startstopButton.text = getString(R.string.label_startstop_btn_stop)
        } else {
            fragmentCameraBinding.bottomSheetLayout.startstopButton.text = getString(R.string.label_startstop_btn_start)
        }

        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        detectorHelper.clearDetector()
        classifierHelper.clearClassifier()

        fragmentCameraBinding.overlay.invalidate()
    }

    // restart app to apply new camera settings
    private fun doRestart(c: Context?) {
        try {
            //check if the context is given
            if (c != null) {
                //fetch the packagemanager so we can get the default launch activity
                // (you can replace this intent with any other activity if you want
                val pm = c.packageManager
                //check if we got the PackageManager
                if (pm != null) {
                    //create the intent with the default start activity for your application
                    val mStartActivity = pm.getLaunchIntentForPackage(
                            c.packageName
                    )
                    if (mStartActivity != null) {
                        mStartActivity.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
                        //create a pending intent so the application is restarted after System.exit(0) was called.
                        // We use an AlarmManager to call this intent in 100ms
                        val mPendingIntentId = 223344
                        val mPendingIntent = PendingIntent
                                .getActivity(c, mPendingIntentId, mStartActivity,
                                        PendingIntent.FLAG_CANCEL_CURRENT or PendingIntent.FLAG_IMMUTABLE)
                        val mgr = c.getSystemService(Context.ALARM_SERVICE) as AlarmManager
                        mgr[AlarmManager.RTC, System.currentTimeMillis() + 100] = mPendingIntent
                        //kill the application - actually doesn't work if next line is uncommented
                        //System.exit(0)
                    } else {
                        Log.e(TAG, "Was not able to restart application, mStartActivity null")
                    }
                } else {
                    Log.e(TAG, "Was not able to restart application, PM null")
                }
            } else {
                Log.e(TAG, "Was not able to restart application, Context null")
            }
        } catch (ex: java.lang.Exception) {
            Log.e(TAG, "Was not able to restart application")
        }
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    private fun convertFacing(apiFacing: Int): Int {
        if (apiFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            return 1
        } else if (apiFacing == CameraCharacteristics.LENS_FACING_BACK) {
            return 0
        } else if (apiFacing == CameraCharacteristics.LENS_FACING_EXTERNAL) {
            return 2
        }
        return -1
    }

    private fun buildCamerasList(cameraManager: CameraManager): Array<SelectedCamera> {
        val cameraIdPerFacing = mutableMapOf<Int, MutableList<String>>()
        val cameraSelectFacingConst = resources.getStringArray(R.array.camera_select_facing)

        for (facingId in 0 until cameraSelectFacingConst.size) {
            cameraIdPerFacing[facingId] = mutableListOf<String>()
        }

        // Get the list of camera IDs.
        val cameraIds = cameraManager.cameraIdList

        for (cameraId in cameraIds) {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
            cameraIdPerFacing[convertFacing(facing!!)]!!.add(cameraId)
        }

        val autoConst = resources.getString(R.string.label_camera_auto)
        val cameras = mutableListOf<SelectedCamera>()
        for (facingId in 0 until cameraSelectFacingConst.size) {
            if (cameraIdPerFacing[facingId]!!.size > 0){
                // add auto
                cameras.add(SelectedCamera(
                        auto = true,
                        facing = facingId,
                        cameraId = "n/a",
                        stringRepresentation = autoConst + " " + cameraSelectFacingConst[facingId]))
                for (id in 0 until cameraIdPerFacing[facingId]!!.size) {
                    cameras.add(
                            SelectedCamera(
                                    auto = false,
                                    facing = facingId,
                                    cameraId = cameraIdPerFacing[facingId]!!.get(id),
                                    stringRepresentation = cameraIdPerFacing[facingId]!!.get(id)+ "-"+cameraSelectFacingConst[facingId]))
                }
            }
        }

        return cameras.toTypedArray()
    }

    private fun buildCameraResolutionList(cameraManager: CameraManager, camera: SelectedCamera): Array<SelectedCameraResolution> {

        val cameraResolutions = mutableMapOf<String, SelectedCameraResolution>()

        // Get the list of camera IDs.
        val cameraIds = cameraManager.cameraIdList

        for (cameraId in cameraIds) {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)

            if (camera.auto && convertFacing(facing!!) != camera.facing)
                continue
            if (!camera.auto && camera.cameraId != cameraId)
                continue

            // Get the SCALER_STREAM_CONFIGURATION_MAP from the CameraCharacteristics object.
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            // Get the list of output sizes from the SCALER_STREAM_CONFIGURATION_MAP object.
            val outputSizes = map?.getOutputSizes(SurfaceTexture::class.java)

            // merge sizes into cameraResolutions
            outputSizes?.forEach {
                cameraResolutions.put(it.toString(), SelectedCameraResolution(
                    parent = camera,
                    size = it)) }

        }

        if (cameraResolutions.size == 0) {
            return emptyArray()
        }

        val res = mutableListOf <SelectedCameraResolution>()
        // Recommended is using the 4:3 ratio because this is the closest to our models
        // the classifier has a resolution 224x224 and typically we have 3 x 4 or 4x4 cards with gaps
        // that gives us something around 2000-1000x 2000-1000 - no need to have higher resolutions
        // e.g. 1920x1440 works ok (it's 4:3 - 480 is base)
        var recommended: SelectedCameraResolution? = null
        var max: SelectedCameraResolution? = null

        var curMaxS = 0
        var curRecS = 0
        for (x in cameraResolutions.values) {
            //add
            res.add(x)
            // find max
            if ( curMaxS < x.size.height * x.size.width) {
                curMaxS = x.size.height * x.size.width
                max = x
            }
            //find recommended
            if (x.size.height < 2000 && x.size.width < 2000 &&
                    x.size.width *3 == x.size.height * 4 &&
                    curRecS < x.size.height * x.size.width) {
                curRecS = x.size.height * x.size.width
                recommended = x
            }
        }

        //sort res by S
        res.sortBy { it.size.width*it.size.height }

        if (max != null) {
            res.add(SelectedCameraResolution(max.parent, max.size, "Max"))
        }
        if (recommended != null) {
            res.add(SelectedCameraResolution(recommended.parent, recommended.size, "Auto"))
        }
        res.reverse()
        return res.toTypedArray()
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        // read settings from preferences
        val cm = context?.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val preferences = PreferenceManager.getDefaultSharedPreferences(context)

        val cameras_list = buildCamerasList(cm)
        val c = preferences.getString("camera", cameras_list[0].toString())
        // find the current id
        var ci = cameras_list.indexOfFirst { it.toString() == c }
        if (ci < 0){ ci = 0}

        val resolution_list = buildCameraResolutionList(cm, cameras_list[ci])
        val r = preferences.getString("resolution", resolution_list[0].toString())
        // find the resolution id
        var ri = resolution_list.indexOfFirst { it.toString() == r }
        if (ri < 0){ ri = 0}

        val selectedCamera = cameras_list[ci]
        val selectedCameraResolution = resolution_list[ri]
        val failsafeStartMode = preferences.getBoolean("failsafe_start", false)

        // CameraProvider
        val cameraProvider =
            cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val availableCameraInfos = cameraProvider.getAvailableCameraInfos();

        val editor = preferences.edit()
        // starting the critical initialization part
        editor.putBoolean("failsafe_start", true)
        editor.apply()

        // CameraSelector - makes assumption that we're only using the back camera
        var cameraSelector: CameraSelector? = null
        if (selectedCamera.auto) {
            var facing = CameraSelector.LENS_FACING_BACK
            if (selectedCamera.facing == 1) {
                facing = CameraSelector.LENS_FACING_FRONT
            }
            cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(facing)
                    .build()
        } else {
            cameraSelector = availableCameraInfos.get(selectedCamera.cameraId.toInt()).cameraSelector
        }

        var previewBuilder = Preview.Builder()
        var imageAnalyzerBuilder = ImageAnalysis.Builder()
        if (failsafeStartMode) {
            previewBuilder =  previewBuilder.setTargetAspectRatio(AspectRatio.RATIO_4_3)
            imageAnalyzerBuilder = imageAnalyzerBuilder.setTargetAspectRatio(AspectRatio.RATIO_4_3)
        } else {
            // switching h and w
            previewBuilder =  previewBuilder.setTargetResolution(Size(selectedCameraResolution.size.height, selectedCameraResolution.size.width ))
            imageAnalyzerBuilder = imageAnalyzerBuilder.setTargetResolution(Size(selectedCameraResolution.size.height, selectedCameraResolution.size.width))
        }
        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview = previewBuilder
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer = imageAnalyzerBuilder
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        if (!::bitmapBuffer.isInitialized) {
                            // The image rotation and RGB image buffer are initialized only once
                            // the analyzer has started running
                            bitmapBuffer = Bitmap.createBitmap(
                                    image.planes[0].rowStride/image.planes[0].pixelStride,
                                    image.height,
                              Bitmap.Config.ARGB_8888
                            )
                        }
                        scanObjects(image)
                    }
                }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }

        // passed the critical initialization part
        editor.putBoolean("failsafe_start", false)
        editor.apply()
    }

    private fun scanObjects(image: ImageProxy) {
        // Copy out RGB bits to the shared bitmap buffer
        // NOTE: we must do it even if we don't detect anything
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

        if(!this.scanIsInProgress)
            return

        val imageRotation = image.imageInfo.rotationDegrees

        // count time
        val startTime = SystemClock.uptimeMillis()

        // Pass Bitmap and rotation to the object detector helper for processing and detection
        val detectedTriple = detectorHelper.detect(bitmapBuffer, imageRotation)

        // Update UI after objects have been detected. Extracts original image height/width
        // to scale and place bounding boxes properly through OverlayView
        this.rawDetectionResults = detectedTriple.first as List<Detection>? ?: LinkedList<Detection>()

        val imageHeight: Int = detectedTriple.second
        val imageWidth: Int = detectedTriple.third
        this.scaleFactor = max(
                fragmentCameraBinding.overlay.width * 1f / imageWidth,
                fragmentCameraBinding.overlay.height * 1f / imageHeight)

        if (detectorHelper.currentModel == DetectorHelper.MODEL_SETGAME) {
            scanForSets(bitmapBuffer, imageRotation)
        }

        this.inferenceTime = SystemClock.uptimeMillis() - startTime
        activity?.runOnUiThread {
            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }

    private fun scanForSets(
            image: Bitmap,
            imageRotation: Int) {
        // update classification Zones list with new detection info
        updateWithNewDetections(image, imageRotation)
        // recreate list of cards (and find duplicates if exist)
        recreateCardsMap()
        // updateSets()
        // TODO: to remove: find sets and mark them as groups
        findSetsOldImplementation()
    }

    private fun updateWithNewDetections(
            image: Bitmap,
            imageRotation: Int) {

        val newDet = LinkedList<Detection>()

        val previousCardZones = LinkedList<CardClassifierZone>(this.cardClassifierZones)
        val newCardZones = LinkedList<CardClassifierZone>()
        val reDetectedCardZones = LinkedList<CardClassifierZone>()

        // clean up
        outer@while (true){
            for (card in previousCardZones) {
                if (card.deleteNextCycle) {
                    previousCardZones.remove(card)
                    continue@outer
                }
            }
            break
        }

        outer@for (det in this.rawDetectionResults) {
            for (card in previousCardZones) {
                if (card.isWithinBoundingBox(det.boundingBox)) {
                    // move to the re-detected list
                    previousCardZones.remove(card)
                    reDetectedCardZones.add(card)

                    // mark as re-detected & update all info
                    card.updateBoundingBox(det.boundingBox)
                    card.updateDetectedTime()

                    continue@outer
                }
            }
            // new card didn't find any matching card - add a new one
            newDet.add(det)
        }

        // try to reClassify redetectedCardZones if they're timed out
        // limit this to 5 cardZones at a time - we'll update them next detection period
        var reclassifiedCounter = 0
        for (cardZone in reDetectedCardZones) {
            if (!cardZone.isReClassifyCandidate())
                continue
            val res = classifierHelper.classify(image, imageRotation, cardZone.boundingBox)
            if (res != null) {
                cardZone.updateCategories(res)
            }
            reclassifiedCounter++
            if (reclassifiedCounter > 5)
                break
        }

        // classify the newly appeared cardZones in newDet and add the to newCards
        outer@for (det in newDet) {
            // filter overriding detections
            for (addedCardZone in newCardZones) {
                if (addedCardZone.isWithinBoundingBox(det.boundingBox))
                    continue@outer
            }
            val res = classifierHelper.classify(image, imageRotation, det.boundingBox)
            if (res != null/*&& res.classifications.size > 0 */) {
                val cardZone = CardClassifierZone(det.boundingBox)
                cardZone.updateCategories(res)
                newCardZones.add(cardZone)
            }
        }
        // try to identify their new position based on the trajectory of redetected cards
        // and redetect them and add to reDetectedCards
        var t: BoundingBoxTransformation? = null
        for (cardZone in reDetectedCardZones) {
            // TBD: we're handling only move without zooming, rotating and etc. even though it's possible to try those as well later
            val xt = cardZone.getLastBoundingBoxTransformation()
            if (xt != null) {
                t = xt
                // reset scaling for now to make a more stable result
                t.scaleX = 1.0f
                t.scaleY = 1.0f
            }
        }

        // we can try to transform and classify
        for (cardZone in previousCardZones) {
            if (t != null)
                cardZone.applyBoundingBoxTransformation(t)
            val res = classifierHelper.classify(image, imageRotation, cardZone.boundingBox)
            if (res != null &&
                    res[ClassifierHelper.SHAPE_CLASSIFIER].size > 0 &&
                    res[ClassifierHelper.SHAPE_CLASSIFIER][0].score > 0.8){
                cardZone.updateCategories(res)
                reDetectedCardZones.add(cardZone)
                cardZone.updateDetectedTime()
            }else {
                if (cardZone.overriddenValue != null || !cardZone.isDetectionOutdated()){
                    reDetectedCardZones.add(cardZone)
                }
            }
        }

        // update the internal list with all we found
        this.cardClassifierZones = reDetectedCardZones
        this.cardClassifierZones.addAll(newCardZones)
    }

    private fun recreateCardsMap(){
        val cards = HashMap<CardValue, ClassifiedCard>()
        for (zone in this.cardClassifierZones){
            val cardValue = zone.getValue()
            if (cardValue != null) {
                if (!cards.containsKey(cardValue)) {
                    cards[cardValue] = ClassifiedCard(cardValue)
                }
                cards[cardValue]?.zones?.add(zone)
            }
        }
        this.cards = cards
    }

    private fun updateSets(){
        val sets = CardSet.findAllSets(this.cards.values.toSet())
        if (setsFinderMode == SetsFinderMode.AllSets) {
            // TODO: update existing groups with all sets (keep groupId - color)
        }else {
            // TODO: update existing with nonoveralpping sets (keep groupId - color)
            val nonOverlappingSets = CardSet.findAllNonOverlappingSets(sets)
        }
    }


    private fun findSetsOldImplementation(): Boolean {
        var vCardsByName = HashMap<AbstractCard,CardClassifierZone>()
        var inSet = HashSet<AbstractCard>()
        // store all cards to set
        for (vCard in cardClassifierZones) {
            var cardVal = vCard.getValue()
            // add only classified cards
            if (cardVal != null) {
                val card = SimpleCard(cardVal)
                inSet.add(card)
                vCardsByName.put(card, vCard)
            }
        }

        val solutions = CardSet.findAllSets(inSet)
        val solutionsMode1 = solutions.sortedBy {
            it.toString()
        }

        // clean groups
        for (vCard in cardClassifierZones) {
            vCard.groups.clear()
        }

        // mode 1
        if (setsFinderMode == SetsFinderMode.AllSets) {
            // TODO: how to keep the same group from scan to scan?
            var groupId = 0
            for (g in solutionsMode1) {
                for (c in g) {
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
        val solutionsMode2 = CardSet.findAllNonOverlappingSets(solutions).sortedBy {
            it.toString()
        }
        for (ss in solutionsMode2) {
            // for each solutionset
            for (s in ss) {
                for (c in s) {
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

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = fragmentCameraBinding.viewFinder.display.rotation
    }

    override fun onDetectorError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }
    override fun onClassifierError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }
}

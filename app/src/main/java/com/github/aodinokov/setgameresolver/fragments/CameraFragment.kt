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
package com.github.aodinokov.setgameresolver.fragments

import android.annotation.SuppressLint
import android.app.AlarmManager
import android.app.Dialog
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.res.Configuration
import android.content.res.TypedArray
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.SystemClock
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
import com.google.gson.Gson
import com.github.aodinokov.setgameresolver.CardColor
import com.github.aodinokov.setgameresolver.CardNumber
import com.github.aodinokov.setgameresolver.CardSet
import com.github.aodinokov.setgameresolver.CardShading
import com.github.aodinokov.setgameresolver.CardShape
import com.github.aodinokov.setgameresolver.CardValue
import com.github.aodinokov.setgameresolver.ClassifierHelper
import com.github.aodinokov.setgameresolver.DetectorHelper
import com.github.aodinokov.setgameresolver.OverlayView
import com.github.aodinokov.setgameresolver.R
import com.github.aodinokov.setgameresolver.SimpleCard
import com.github.aodinokov.setgameresolver.databinding.FragmentCameraBinding
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.vision.detector.Detection
import java.lang.Float.max
import java.util.LinkedList
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.absoluteValue
import kotlin.math.pow

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
/**
 * 9x9 Bitmap with adjustable order
 */
class ThumbnailsBitmapHelper(
        val thumbnailsBitmap: Bitmap,
        // maps & weights (higher - bigger)
        val numberWeight: Int = 3,
        private val numberMap: Map<Int,Int>? = null,
        val colorWeight: Int = 2,
        private val colorMap: Map<Int,Int>?= null,
        val shadingWeight: Int = 1,
        private val shadingMap: Map<Int,Int>?= null,
        val shapeWeight: Int = 0,
        private val shapeMap: Map<Int,Int>?= null) {

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
        return Bitmap.createBitmap(thumbnailsBitmap,
                thumbnailsBitmap.width / 9 * column,
                thumbnailsBitmap.height / 9 * row,
                thumbnailsBitmap.width / 9,
                thumbnailsBitmap.height / 9)
    }
}

class CardClassifierZone(var boundingBox: RectF) {
    private var prevBoundingBox: RectF? = null

    companion object {
        data class BoundingBoxTransformation(
                var deltaX: Float,
                var deltaY: Float,
                var scaleX:Float,
                var scaleY: Float)
    }

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
    private var categoriesMax = Array<Category?>(4) { null }
    fun updateCategories(newCategories: Array<MutableList<Category>>?) {
        if (newCategories == null)
            return
        assert(newCategories.size == categoriesMax.size)
        for (i in categoriesMax.indices) {
            if (newCategories[i].size == 0)
                continue
            val newCat = newCategories[i][0]
            if (categoriesMax[i] == null || categoriesMax[i]!!.score <= newCat.score)
                categoriesMax[i] = newCat
        }
    }
    fun getCategories(): MutableList<Category> {
        val res = LinkedList<Category>()
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
}
class CameraFragment : Fragment(),
        DetectorHelper.DetectorErrorListener,
        ClassifierHelper.ClassifierErrorListener {
    private val tag = "Scanner"
    companion object {
        enum class SetsFinderMode(val mode: Int) {
            AllSets(0),
            NonOverlappingSets(1),
        }
        enum class ScanMode {
            Idle,
            Camera,
            StaticPicture;
        }
        /**
         * Inherit SimpleCard, but also store list of Classifier Zones that produced that.
         * There can be several - in that case there are duplicates and we need
         * to highlight them
         */
        class ClassifiedCard(v: CardValue): SimpleCard(v) {
            val zones = LinkedList<CardClassifierZone>()
            val groupIds = HashSet<Int>()
        }

        private const val BOUNDING_RECT_TEXT_PADDING = 8

        class SelectedCamera(    val auto: Boolean = true,
                                 val facing: Int = 0,
                                 val cameraId: String = "",
                                 private val stringRepresentation: String = "") {
            override fun toString(): String {
                return stringRepresentation
            }
        }
        class SelectedCameraResolution(val parent: SelectedCamera,
                                       val size: Size,
                                       private val stringRepresentation: String = ""
        ){
            override fun toString(): String {
                if (stringRepresentation == "") {
                    return size.toString()
                }
                return stringRepresentation
            }
        }
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    /** overlay painting helper objects*/
    private var thumbnailsBitmapHelper: ThumbnailsBitmapHelper? = null

    /** image work helpers */
    private lateinit var detectorHelper: DetectorHelper
    private lateinit var classifierHelper: ClassifierHelper

    private lateinit var bitmapBuffer: Bitmap
    /** for static picture mode */
    private var bitmapStaticBufferOriginalImageRotation: Int = 0
    private var bitmapStaticBufferOriginal: Bitmap? = null
    private var bitmapStaticBufferRotated: Bitmap? = null
    /** Blocking camera operations are performed using this executor */
    private lateinit var cameraExecutor: ExecutorService

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    /** modes */
    private var setsFinderMode = SetsFinderMode.AllSets
    private var scanMode = ScanMode.Idle

    /** last detection results */
    private var imageHeight: Int = 0
    private var imageWidth: Int = 0
    private val scaleFactor
        get() = if (imageHeight == 0 || imageWidth == 0) 1f else  max(
                fragmentCameraBinding.overlay.width * 1f / imageWidth,
                fragmentCameraBinding.overlay.height * 1f / imageHeight)
    /** overlay data to show*/
    private var inferenceTime: Long = 0

    /* raw data after detection */
    private var rawDetectionResults: List<Detection> = LinkedList<Detection>()
    /* we're keeping special objects that classify cards. override is a part of functionality */
    private var cardClassifierZones = LinkedList<CardClassifierZone>()
    /* resulting cards we were able to collect from classifier zones */
    private var cards = HashMap<CardValue, ClassifiedCard>()
    /* map stores groupId (identifies the color) and set of CardSet
       in AllSets Mode the set will always contain 1 CardSet
       in NonOverlappingSets it contains the sorted result of findAllNonOverlappingSets
       groupId is unique */
    private var sets = HashMap<String, Pair<Int,Set<CardSet>>>()
    /* set of deallocated group Ids
       if group is empty - the next allocated groupID will be sets.size() (nextId)
       because it contains all unique allocated groups */
    private val freeGroupIds = HashSet<Int>()
    private fun allocateGroupId(nextId: Int): Int {
        if (freeGroupIds.isEmpty())
            return nextId
        val id = freeGroupIds.first()
        val rmResult = freeGroupIds.remove(id)
        assert(rmResult)
        return id
    }
    private fun freeGroupId(id: Int) {
        assert(!freeGroupIds.contains(id))
        freeGroupIds.add(id)
    }

    private fun forceRedrawIfNeeded() {
        // in case of camera we are permanently re-drawing
        if (scanMode == ScanMode.StaticPicture) {
            // clean up
            handleMarkedZones()
            // recalculate sets
            updateSets()
            activity?.runOnUiThread {  // Force a redraw
                fragmentCameraBinding.overlay.invalidate()
            }
        }
    }

    private fun minMaxDetectionRectF(): RectF {
        if (imageWidth == 0 || imageHeight == 0) {
            // some very arbitrary limits
            return RectF(20f, 20f, 100f, 100f)
        }
        return RectF(imageWidth/12f, imageHeight/12f, imageWidth/3f, imageHeight/3f)
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
                    colorMap = mapOf(Pair(0,2),Pair(1,1),Pair(2,0)),
                    shadingMap = mapOf(Pair(0,2),Pair(1,0),Pair(2,1)),
                    shapeMap = mapOf(Pair(0,1),Pair(1,2),Pair(2,0)))
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
            scanMode = if (scanMode == ScanMode.Idle) {
                ScanMode.Camera
            } else {
                ScanMode.Idle
            }

            /* reset all data */
            rawDetectionResults = LinkedList<Detection>()
            cardClassifierZones.clear()
            updateSets()

            updateTextControlsUi()
        }

        val resolutionDialog = Dialog(requireContext())
        fragmentCameraBinding.bottomSheetLayout.resolutionButton.setOnClickListener {
            when (scanMode) {
                ScanMode.Idle -> {
                    resolutionDialog.setContentView(R.layout.resolution_dialog)
                    resolutionDialog.window!!.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                    resolutionDialog.setCancelable(false)

                    val spinnerCamera = resolutionDialog.findViewById<Spinner>(R.id.spinner_camera)
                    val spinnerResolution = resolutionDialog.findViewById<Spinner>(R.id.spinner_resolution)
                    val okayText = resolutionDialog.findViewById<TextView>(R.id.okay_text)
                    val cancelText = resolutionDialog.findViewById<TextView>(R.id.cancel_text)

                    okayText.setOnClickListener {
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
                    }

                    cancelText.setOnClickListener {
                        resolutionDialog.dismiss()
                    }

                    val cm = context?.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                    val camerasList = buildCamerasList(cm)
                    val camerasAdapter = ArrayAdapter(requireContext(), R.layout.spinner_item, camerasList)
                    spinnerCamera.adapter = camerasAdapter
                    spinnerCamera.onItemSelectedListener =
                            object : AdapterView.OnItemSelectedListener {
                                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                                    val upperAdapter = p0!!.adapter
                                    val camera = upperAdapter.getItem(p2)
                                    // update resolution list if resolution has a different selected camera?
                                    if (!spinnerResolution.adapter.isEmpty && (spinnerResolution.adapter.getItem(0) as SelectedCameraResolution).parent != camera) {
                                        val adapter = ArrayAdapter(requireContext(), R.layout.spinner_item, buildCameraResolutionList(cm, camera as SelectedCamera))
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
                        if (ci < 0) {
                            ci = 0
                        }
                        spinnerCamera.setSelection(ci, false)

                        val resolutionList = buildCameraResolutionList(cm, camerasList[ci])
                        val resolutionAdapter = ArrayAdapter(requireContext(), R.layout.spinner_item, resolutionList)
                        spinnerResolution.adapter = resolutionAdapter

                        val r = preferences.getString("resolution", resolutionList[0].toString())
                        // find the resolution id
                        var ri = resolutionList.indexOfFirst { it.toString() == r }
                        if (ri < 0) {
                            ri = 0
                        }
                        spinnerResolution.setSelection(ri, false)
                    }
                    resolutionDialog.show()
                }
                ScanMode.Camera ->{
                    scanMode = ScanMode.StaticPicture
                    updateTextControlsUi()
                }
                ScanMode.StaticPicture ->{
                    scanMode = ScanMode.Camera
                    updateTextControlsUi()
                }
            }
        }

        // overlay draw procedure
        fragmentCameraBinding.overlay.setOnDrawListener(object : OverlayView.DrawListener {
            // colors
            private var boxPaint = Paint()
            private var duplicatePaint = Paint()
            private var textBackgroundPaint = Paint()
            private var textPaint = Paint()
            private var groupBoxPaintMap = HashMap<Int, Paint>()
            // make tmp vars here to save time in draw function
            private var bounds = Rect()
            private var boundsF = RectF()
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

                duplicatePaint.color = ContextCompat.getColor(requireContext(), R.color.duplicate_box_color)
                duplicatePaint.style = Paint.Style.FILL
                duplicatePaint.alpha = 63

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

            private fun drawCardValue(canvas: Canvas,
                                      boundingBox: RectF,
                                      cardValue: CardValue,
                                      duplicate: Boolean = false){
                val top = boundingBox.top * scaleFactor
                val left = boundingBox.left * scaleFactor

                assert(thumbnailsBitmapHelper != null)

                // Create text to display alongside detected objects
                val idx = thumbnailsBitmapHelper!!.getThumbIndex(cardValue)
                val column = thumbnailsBitmapHelper!!.getThumbColumn(idx)
                val row = thumbnailsBitmapHelper!!.getThumbRow(idx)

                bounds.set(
                        thumbnailsBitmapHelper!!.thumbnailsBitmap.width / 9 * column,
                        thumbnailsBitmapHelper!!.thumbnailsBitmap.height / 9 * row,
                        thumbnailsBitmapHelper!!.thumbnailsBitmap.width / 9 * (column + 1),
                        thumbnailsBitmapHelper!!.thumbnailsBitmap.height / 9 * (row + 1))
                boundsF.set(
                        left,
                        top,
                        left + thumbnailsBitmapHelper!!.thumbnailsBitmap.width / 9,
                        top + thumbnailsBitmapHelper!!.thumbnailsBitmap.height / 9)

                canvas.drawBitmap(thumbnailsBitmapHelper!!.thumbnailsBitmap,
                        bounds,
                        boundsF,
                        textBackgroundPaint
                )
                if (duplicate) {
                    canvas.drawRect(
                            left,
                            top,
                            left + thumbnailsBitmapHelper!!.thumbnailsBitmap.width / 9,
                            top + thumbnailsBitmapHelper!!.thumbnailsBitmap.height / 9,
                            duplicatePaint)
                }
            }

            private fun drawCardText(canvas: Canvas,
                                     boundingBox: RectF,
                                     text: String) {
                assert(thumbnailsBitmapHelper != null)
                val shiftX = thumbnailsBitmapHelper!!.thumbnailsBitmap.width / 9 // we have put 9 cards in the row

                val top = boundingBox.top * scaleFactor
                val left = boundingBox.left * scaleFactor

                // Draw rect behind display text
                textBackgroundPaint.getTextBounds(text, 0, text.length, bounds)
                val textWidth = bounds.width()
                val textHeight = bounds.height()
                canvas.drawRect(
                        left + shiftX,
                        top,
                        left + shiftX + textWidth + BOUNDING_RECT_TEXT_PADDING,
                        top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                        textBackgroundPaint
                )

                // Draw text for detected object
                canvas.drawText(text, left + shiftX, top + bounds.height(), textPaint)
            }

            private fun drawCardGroups(canvas: Canvas, boundingBox: RectF, groupIds: Set<Int>) {
                val top = boundingBox.top * scaleFactor
                val bottom = boundingBox.bottom * scaleFactor
                val left = boundingBox.left * scaleFactor
                val right = boundingBox.right * scaleFactor

                // Draw bounding box around detected objects
                boundsF.set(left, top, right, bottom)
                val drawableRect = boundsF
                for(groupId in groupIds.sortedBy { it }) {
                    assert(groupBoxPaintMap.size > 0)
                    val pb = groupBoxPaintMap[groupId%groupBoxPaintMap.size]!!
                    canvas.drawRect(drawableRect, pb)
                    // make all groups visible
                    drawableRect.left -= pb.strokeWidth
                    drawableRect.top -= pb.strokeWidth
                    drawableRect.bottom += pb.strokeWidth
                    drawableRect.right += pb.strokeWidth
                }
            }

            override fun onDraw(canvas: Canvas){
                if (scanMode == ScanMode.Idle)
                    return

                if (scanMode == ScanMode.Camera) {
                    // Generate fps text
                    var fpsText = "fps:inf"
                    if (inferenceTime > 0) {  // inferenceTime is in ms
                        fpsText = String.format("fps:%2.0f", 1000.0 / inferenceTime.toFloat())
                    }
                    textBackgroundPaint.getTextBounds(fpsText, 0, fpsText.length, bounds)
                    // Draw rect behind display text
                    val fpsTop = 0f
                    val fpsTextWidth = bounds.width()
                    val fpsTextHeight = bounds.height()
                    canvas.drawRect(
                            0f,
                            fpsTop,
                            0f + fpsTextWidth + BOUNDING_RECT_TEXT_PADDING,
                            fpsTop + fpsTextHeight + BOUNDING_RECT_TEXT_PADDING,
                            textBackgroundPaint
                    )
                    // Draw display text
                    canvas.drawText(fpsText, 0f, fpsTop + bounds.height(), textPaint)

                    //show raw results
                    for (result in rawDetectionResults) {
                        boundsF.top = result.boundingBox.top * scaleFactor
                        boundsF.bottom = result.boundingBox.bottom * scaleFactor
                        boundsF.left = result.boundingBox.left * scaleFactor
                        boundsF.right = result.boundingBox.right * scaleFactor

                        // Draw bounding box around detected objects
                        canvas.drawRect(boundsF, boxPaint)

                        // Create text to display alongside detected objects
                        if (detectorHelper.currentModel != DetectorHelper.MODEL_SETGAME && result.categories.size > 0) {
                            val shiftX = 0
                            val label = result.categories[0].label + " "
                            val drawableText = label + String.format("%.2f", result.categories[0].score)

                            // Draw rect behind display text
                            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
                            val textWidth = bounds.width()
                            val textHeight = bounds.height()
                            canvas.drawRect(
                                    boundsF.left + shiftX,
                                    boundsF.top,
                                    boundsF.left + shiftX + textWidth + BOUNDING_RECT_TEXT_PADDING,
                                    boundsF.top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                                    textBackgroundPaint
                            )

                            // Draw text for detected object
                            canvas.drawText(drawableText, boundsF.left + shiftX, boundsF.top + bounds.height(), textPaint)
                        }
                    }
                }

                if (scanMode == ScanMode.StaticPicture && bitmapStaticBufferRotated != null) {
                    // scale if needed to match sized of view_finder or overlay
                    // because without that it doesn't show the image properly in horizontal view
                    //canvas.drawBitmap(bitmapStaticBufferRotated!!, 0.0f, 0.0f, textBackgroundPaint)
                    val scale = max(fragmentCameraBinding.overlay.width.toFloat()/bitmapStaticBufferRotated!!.width.toFloat(),
                            fragmentCameraBinding.overlay.height.toFloat()/bitmapStaticBufferRotated!!.height.toFloat())
                    bounds.set(0,0, bitmapStaticBufferRotated!!.width, bitmapStaticBufferRotated!!.height)
                    boundsF.set(0f, 0f,
                            bitmapStaticBufferRotated!!.width.toFloat()*scale, bitmapStaticBufferRotated!!.height.toFloat()*scale)
                    canvas.drawBitmap(bitmapStaticBufferRotated!!, bounds, boundsF, textBackgroundPaint)
                }

                // show zones, cardValues, duplicates and sets
                if (detectorHelper.currentModel == DetectorHelper.MODEL_SETGAME) {
                   for (card in cards.values){
                       for (zone in card.zones){
                           drawCardValue(canvas,
                                   zone.boundingBox,
                                   card.getValue(),
                                   duplicate = card.zones.size > 1)
                           if (scanMode == ScanMode.Camera && zone.overriddenValue == null) {
                               val drawableText = String.format("%.2f", zone.getCategories()[0].score)
                               drawCardText(canvas, zone.boundingBox, drawableText)
                           }
                           drawCardGroups(canvas,
                                   zone.boundingBox,
                                   card.groupIds)
                       }
                   }
                }
            }
        })

        // create/update dialog
        val overrideDialog = Dialog(requireContext())
        fragmentCameraBinding.overlay.setOnTouchListener(View.OnTouchListener { _, event ->
            // ignore touches in Idle mode
            if (scanMode == ScanMode.Idle)
                return@OnTouchListener true

            assert(thumbnailsBitmapHelper != null)
            if (event != null &&
                    event.action == MotionEvent.ACTION_DOWN) {
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
                    minusCount.setOnClickListener {
                        setView(minusCountCard)
                    }
                    plusCount.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(plusCountCard)))
                    plusCount.setOnClickListener {
                        setView(plusCountCard)
                    }
                    minusColor.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(minusColorCard)))
                    minusColor.setOnClickListener {
                        setView(minusColorCard)
                    }
                    plusColor.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(plusColorCard)))
                    plusColor.setOnClickListener {
                        setView(plusColorCard)
                    }
                    minusFill.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(minusFillCard)))
                    minusFill.setOnClickListener {
                        setView(minusFillCard)
                    }
                    plusFill.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(plusFillCard)))
                    plusFill.setOnClickListener {
                        setView(plusFillCard)
                    }
                    minusShape.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(minusShapeCard)))
                    minusShape.setOnClickListener {
                        setView(minusShapeCard)
                    }
                    plusShape.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndex(plusShapeCard)))
                    plusShape.setOnClickListener {
                        setView(plusShapeCard)
                    }
                }

                // to calculate avg sizes
                var width = 0.0f
                var height = 0.0f

                // find if we pressed within any detected card? if so - propose to override
                for (cardClassifierZone in this.cardClassifierZones) {

                    val top = cardClassifierZone.boundingBox.top * scaleFactor
                    val bottom = cardClassifierZone.boundingBox.bottom * scaleFactor
                    val left = cardClassifierZone.boundingBox.left * scaleFactor
                    val right = cardClassifierZone.boundingBox.right * scaleFactor

                    width += right - left
                    height += bottom - top

                    if (event.x > left && event.x < right &&
                            event.y > top && event.y < bottom) {

                        // check that it's a card at all
                        val cardValue = cardClassifierZone.getValue() ?: continue
                        cardClassifierZone.editIsProgress = true
                        setView(cardValue)

                        cancelText.setOnClickListener {
                            cardClassifierZone.editIsProgress = false
                            overrideDialog.dismiss()
                        }
                        updateText.setOnClickListener {
                            cardClassifierZone.overriddenValue = currentDlgCardValue
                            cardClassifierZone.editIsProgress = false
                            overrideDialog.dismiss()
                            forceRedrawIfNeeded()
                        }
                        deleteText.setOnClickListener {
                            //this.cardClassifierZones.remove(cardClassifierZone)
                            // doesn't work because we're working in the copy of the list
                            // instead we need to mark this object as forDeletion
                            cardClassifierZone.deleteNextCycle = true
                            cardClassifierZone.editIsProgress = false
                            overrideDialog.dismiss()
                            forceRedrawIfNeeded()
                        }

                        updateText.text = getString(R.string.label_update_btn)
                        deleteText.visibility = View.VISIBLE
                        overrideDialog.show()
                        return@OnTouchListener true
                    }
                }
                // wasn't able to find anything - but in Freeze mode we can create
                if (scanMode == ScanMode.StaticPicture &&
                        scaleFactor != 0.0f &&
                        bitmapStaticBufferOriginal != null) {
                    if (this.cardClassifierZones.size > 0) {
                        // get avg
                        width /= cardClassifierZones.size.toFloat()
                        height /= cardClassifierZones.size.toFloat()
                    }

                    // filter and use default if it's not ok
                    val minMax = minMaxDetectionRectF()
                    if (width < minMax.left || width > minMax.right ||
                            height < minMax.top || height > minMax.bottom) {
                        width = minMax.left * 2
                        height = minMax.top * 2
                    }

                    // try to create new Zone
                    val cardZone = CardClassifierZone(RectF(
                            event.x/scaleFactor - width/2, event.y/scaleFactor - height/2,
                            event.x/scaleFactor + width/2, event.y/scaleFactor + height/2))

                    val res = classifierHelper.classify(
                            bitmapStaticBufferOriginal!!, bitmapStaticBufferOriginalImageRotation,
                            cardZone.boundingBox)
                    if (res != null) {
                        cardZone.updateCategories(res)
                        val cardValue = cardZone.getValue() ?: CardValue(
                                CardNumber.ONE, CardColor.GREEN,
                                CardShading.EMPTY, CardShape.DIAMOND)
                        cardZone.editIsProgress = true
                        setView(cardValue)

                        cancelText.setOnClickListener {
                            cardZone.editIsProgress = false
                            overrideDialog.dismiss()
                        }
                        updateText.setOnClickListener {
                            cardZone.overriddenValue = currentDlgCardValue
                            cardZone.editIsProgress = false
                            // finally add it!!!
                            this.cardClassifierZones.add(cardZone)
                            overrideDialog.dismiss()
                            forceRedrawIfNeeded()
                        }

                        updateText.text = getString(R.string.label_create_btn)
                        deleteText.visibility = View.GONE
                        overrideDialog.show()
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

        if (scanMode == ScanMode.Idle) {
            fragmentCameraBinding.bottomSheetLayout.startstopButton.text = getString(R.string.label_startstop_btn_start)
            fragmentCameraBinding.bottomSheetLayout.resolutionButton.text = getString(R.string.label_resolution_btn)
        } else {
            fragmentCameraBinding.bottomSheetLayout.startstopButton.text = getString(R.string.label_startstop_btn_stop)
            if (scanMode == ScanMode.Camera) {
                fragmentCameraBinding.bottomSheetLayout.resolutionButton.text = getString(R.string.label_freeze_btn)
            }else {
                fragmentCameraBinding.bottomSheetLayout.resolutionButton.text = getString(R.string.label_unfreeze_btn)
            }
        }

        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        detectorHelper.clearDetector()
        classifierHelper.clearClassifier()

        if (scanMode == ScanMode.StaticPicture) {
            updateSets()
        }
        fragmentCameraBinding.overlay.invalidate()
    }

    // restart app to apply new camera settings
    private fun doRestart(c: Context?) {
        try {
            //check if the context is given
            if (c != null) {
                //fetch the package manager so we can get the default launch activity
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
                        Log.e(tag, "Was not able to restart application, mStartActivity null")
                    }
                } else {
                    Log.e(tag, "Was not able to restart application, PM null")
                }
            } else {
                Log.e(tag, "Was not able to restart application, Context null")
            }
        } catch (ex: java.lang.Exception) {
            Log.e(tag, "Was not able to restart application")
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
        return when (apiFacing) {
            CameraCharacteristics.LENS_FACING_FRONT -> {
                1
            }
            CameraCharacteristics.LENS_FACING_BACK -> {
                0
            }
            CameraCharacteristics.LENS_FACING_EXTERNAL -> {
                2
            }
            else -> -1
        }
    }

    private fun buildCamerasList(cameraManager: CameraManager): Array<SelectedCamera> {
        val cameraIdPerFacing = mutableMapOf<Int, MutableList<String>>()
        val cameraSelectFacingConst = resources.getStringArray(R.array.camera_select_facing)

        for (facingId in cameraSelectFacingConst.indices) {
            cameraIdPerFacing[facingId] = mutableListOf()
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
        for (facingId in cameraSelectFacingConst.indices) {
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
                                    cameraId = cameraIdPerFacing[facingId]!![id],
                                    stringRepresentation = cameraIdPerFacing[facingId]!![id] + "-"+cameraSelectFacingConst[facingId]))
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
                cameraResolutions[it.toString()] = SelectedCameraResolution(
                    parent = camera,
                    size = it)
            }

        }

        if (cameraResolutions.isEmpty()) {
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

        val camerasList = buildCamerasList(cm)
        val c = preferences.getString("camera", camerasList[0].toString())
        // find the current id
        var ci = camerasList.indexOfFirst { it.toString() == c }
        if (ci < 0){ ci = 0}

        val resolutionList = buildCameraResolutionList(cm, camerasList[ci])
        val r = preferences.getString("resolution", resolutionList[0].toString())
        // find the resolution id
        var ri = resolutionList.indexOfFirst { it.toString() == r }
        if (ri < 0){ ri = 0}

        val selectedCamera = camerasList[ci]
        val selectedCameraResolution = resolutionList[ri]
        val failsafeStartMode = preferences.getBoolean("failsafe_start", false)

        // CameraProvider
        val cameraProvider =
            cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val availableCameraInfo = cameraProvider.availableCameraInfos

        val editor = preferences.edit()
        // starting the critical initialization part
        editor.putBoolean("failsafe_start", true)
        editor.apply()

        // CameraSelector - makes assumption that we're only using the back camera
        val cameraSelector: CameraSelector?
        if (selectedCamera.auto) {
            var facing = CameraSelector.LENS_FACING_BACK
            if (selectedCamera.facing == 1) {
                facing = CameraSelector.LENS_FACING_FRONT
            }
            cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(facing)
                    .build()
        } else {
            cameraSelector = availableCameraInfo[selectedCamera.cameraId.toInt()].cameraSelector
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
            Log.e(tag, "Use case binding failed", exc)
        }

        // passed the critical initialization part
        editor.putBoolean("failsafe_start", false)
        editor.apply()
    }

    private fun scanObjects(image: ImageProxy) {
        val imageRotation = image.imageInfo.rotationDegrees
        // Copy out RGB bits to the shared bitmap buffer
        // NOTE: we must do it even if we don't detect anything
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

        if (scanMode == ScanMode.StaticPicture) {
            if (bitmapStaticBufferOriginal == null) {
                bitmapStaticBufferOriginal = Bitmap.createBitmap(bitmapBuffer)
                bitmapStaticBufferOriginalImageRotation = imageRotation

                if (bitmapStaticBufferRotated != null) {
                    bitmapStaticBufferRotated!!.recycle()
                    bitmapStaticBufferRotated = null
                }
                val matrix = Matrix()
                matrix.postRotate(imageRotation.toFloat())
                // copy bitmapBuffer and rotate
                bitmapStaticBufferRotated = Bitmap.createBitmap(bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true)
                activity?.runOnUiThread {
                    // Force a redraw
                    fragmentCameraBinding.overlay.invalidate()
                }
            }
        } else {
            if (bitmapStaticBufferRotated != null){
                bitmapStaticBufferRotated!!.recycle()
                bitmapStaticBufferRotated = null
            }
            if (bitmapStaticBufferOriginal != null){
                bitmapStaticBufferOriginal!!.recycle()
                bitmapStaticBufferOriginal = null
            }
        }

        if(scanMode != ScanMode.Camera)
            return

        // count time
        val startTime = SystemClock.uptimeMillis()

        // Pass Bitmap and rotation to the object detector helper for processing and detection
        val detectedTriple = detectorHelper.detect(bitmapBuffer, imageRotation)

        // Update UI after objects have been detected. Extracts original image height/width
        // to scale and place bounding boxes properly through OverlayView
        this.rawDetectionResults = detectedTriple.first ?: LinkedList<Detection>()

        this.imageHeight = detectedTriple.second
        this.imageWidth = detectedTriple.third

        if (detectorHelper.currentModel == DetectorHelper.MODEL_SETGAME) {
            // update classification Zones list with new detection info
            updateWithNewDetections(bitmapBuffer, imageRotation)
            // recreate list of cards (and find duplicates if exist)
            // update set map and update cards with their groupId
            updateSets()
        }

        this.inferenceTime = SystemClock.uptimeMillis() - startTime
        activity?.runOnUiThread {
            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }

    private fun handleMarkedZones() {
        // clean up
        outer@while (true){
            for (card in cardClassifierZones) {
                if (card.deleteNextCycle) {
                    cardClassifierZones.remove(card)
                    continue@outer
                }
            }
            break
        }
    }

    private fun updateWithNewDetections(
            image: Bitmap,
            imageRotation: Int) {
        val newDet = LinkedList<Detection>()

        // clean up
        handleMarkedZones()

        val previousCardZones = LinkedList(this.cardClassifierZones)
        val newCardZones = LinkedList<CardClassifierZone>()
        val reDetectedCardZones = LinkedList<CardClassifierZone>()

        val minMax = minMaxDetectionRectF()
        outer@for (det in this.rawDetectionResults) {
            // filter by size and skip if doesn't match
            if (det.boundingBox.width() < minMax.left ||
                    det.boundingBox.width() > minMax.right ||
                    det.boundingBox.height() < minMax.top ||
                    det.boundingBox.height() > minMax.bottom)
                continue

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

        // try to reClassify reDetectedCardZones if they're timed out
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
        // try to identify their new position based on the trajectory of reDetected cards
        // and re-detect them and add to reDetectedCards
        var t: CardClassifierZone.Companion.BoundingBoxTransformation? = null
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

    private fun updateSets() {
        val cards = HashMap<CardValue, ClassifiedCard>()
        for (zone in this.cardClassifierZones){
            val cardValue = zone.getValue()
            if (cardValue != null) {
                if (!cards.containsKey(cardValue)) {
                    // let's try to re-use the objects if they exist so we do less mem allocations
                    if (this.cards.containsKey(cardValue)) {
                        val oldCardObj = this.cards[cardValue]!!
                        oldCardObj.zones.clear()
                        oldCardObj.groupIds.clear()
                        cards[cardValue] = oldCardObj
                    }else{
                        cards[cardValue] = ClassifiedCard(cardValue)
                    }
                }
                cards[cardValue]?.zones?.add(zone)
            }
        }

        val newSets = HashMap<String, Pair<Int, Set<CardSet>>>()
        val prevSets = HashMap(this.sets)
        val allSets = CardSet.findAllSets(cards.values.toSet())

        val hashSets = if (setsFinderMode == SetsFinderMode.AllSets) {
            val hashSets = HashSet<Set<CardSet>>()
            for (set in allSets) {
                hashSets.add(setOf(set))
            }
            hashSets
        }else {
            val hashSets = HashSet<Set<CardSet>>()
            for (set in CardSet.findAllNonOverlappingSets(allSets)) {
                hashSets.add(set.sortedBy { it.toString() }.toSet())
            }
            hashSets
        }

        // move repeated sets from Prev to New
        val hashOfNewSets = HashSet<Set<CardSet>>()
        for (sets in hashSets) {
            val key = sets.toString()
            if (prevSets.containsKey(key)) {
                newSets[key] = prevSets[key]!!
                prevSets.remove(key)
            } else {
                hashOfNewSets.add(sets)
            }
        }

        // free Ids from non-repeated sets
        for (pair in prevSets.values) {
            freeGroupId(pair.first)
        }
        prevSets.clear()

        // create newly added
        for (sets in hashOfNewSets) {
            val key = sets.toString()
            newSets[key] = Pair(allocateGroupId(newSets.size), sets)
        }

        // update cards with groups
        for (pair in newSets.values){
            val groupId = pair.first
            for (set in pair.second) {
                for (card in set) {
                    if (cards.containsKey(card.getValue())){
                        cards[card.getValue()]!!.groupIds.add(groupId)
                    }
                }
            }
        }
        this.cards = cards
        this.sets = newSets
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

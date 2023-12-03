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
import org.tensorflow.lite.examples.objectdetection.Detected
import org.tensorflow.lite.examples.objectdetection.DetectorMode
import org.tensorflow.lite.examples.objectdetection.OverlayView
import org.tensorflow.lite.examples.objectdetection.R
import org.tensorflow.lite.examples.objectdetection.SetgameDetectorHelper
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding
import java.util.LinkedList
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.pow
import kotlin.math.max
import android.content.res.TypedArray
import android.graphics.RectF
import org.tensorflow.lite.examples.objectdetection.Grouppable

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

    fun getThumbIndx(cardValue: CardValue):Int {
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
                thumbnailsBitmap!!.width / 9 * column,
                thumbnailsBitmap!!.height / 9 * row,
                thumbnailsBitmap!!.width / 9 * (column + 1),
                thumbnailsBitmap!!.height / 9 * (row + 1))
        return Bitmap.createBitmap(thumbnailsBitmap!!,
                thumbnailsBitmap!!.width / 9 * column,
                thumbnailsBitmap!!.height / 9 * row,
                thumbnailsBitmap!!.width / 9,
                thumbnailsBitmap!!.height / 9)
    }
}

class SelectedCamera(    val auto: Boolean = true,
                         val facing: Int = 0,
                         val cameraId: String = "",
                         val stringRepr: String = "") {
    override fun toString(): String {
        return stringRepr
    }
}

class SelectedCameraResolution(val parent: SelectedCamera,
                               val size: Size,
                               val stringRepr: String = ""
){
    override fun toString(): String {
        if (stringRepr == "") {
            return size.toString()
        }
        return stringRepr
    }
}

class CameraFragment : Fragment(), SetgameDetectorHelper.DetectorListener {

    private val TAG = "ObjectDetection"

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var setgameDetectorHelper: SetgameDetectorHelper
    private lateinit var bitmapBuffer: Bitmap
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    /** Blocking camera operations are performed using this executor */
    private lateinit var cameraExecutor: ExecutorService

    /** overlay data */
    private var showDetections: Boolean = false
    private var scaleFactor: Float = 1f
    private var inferenceTime: Long = 0
    private var results: List<Detected> = LinkedList<Detected>()

    /** overlay painting helper objects*/
    private var thumbnailsBitmapHelper: ThumbnailsBitmapHelper? = null

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

        setgameDetectorHelper = SetgameDetectorHelper(
            context = requireContext(),
            objectDetectorListener = this)

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

    @SuppressLint("ClickableViewAccessibility")
    private fun initControlsUi() {
        // When clicked, change the underlying model used for object detection
        fragmentCameraBinding.bottomSheetLayout.spinnerMode.setSelection(setgameDetectorHelper.detectorMode.mode, false)
        fragmentCameraBinding.bottomSheetLayout.spinnerMode.onItemSelectedListener =
                object : AdapterView.OnItemSelectedListener {
                    override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                        setgameDetectorHelper.detectorMode =
                                when (p2) {
                                    0 -> DetectorMode.AllSets
                                    1 -> DetectorMode.NonOverlappingSets
                                    else -> DetectorMode.AllSets
                                }
                        setgameDetectorHelper.WritePreferences()
                        updateTextControlsUi()
                    }

                    override fun onNothingSelected(p0: AdapterView<*>?) {
                        /* no op */
                    }
                }


        // When clicked, lower detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detThresholdMinus.setOnClickListener {
            if (setgameDetectorHelper.detThreshold >= 0.1) {
                setgameDetectorHelper.detThreshold -= 0.1f
                setgameDetectorHelper.WritePreferences()
                updateTextControlsUi()
            }
        }

        // When clicked, raise detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detThresholdPlus.setOnClickListener {
            if (setgameDetectorHelper.detThreshold <= 0.8) {
                setgameDetectorHelper.detThreshold += 0.1f
                setgameDetectorHelper.WritePreferences()
                updateTextControlsUi()
            }
        }

        // When clicked, reduce the number of objects that can be detected at a time
        fragmentCameraBinding.bottomSheetLayout.detMaxResultsMinus.setOnClickListener {
            if (setgameDetectorHelper.detMaxResults > 1) {
                setgameDetectorHelper.detMaxResults--
                setgameDetectorHelper.WritePreferences()
                updateTextControlsUi()
            }
        }

        // When clicked, increase the number of objects that can be detected at a time
        fragmentCameraBinding.bottomSheetLayout.detMaxResultsPlus.setOnClickListener {
            if (setgameDetectorHelper.detMaxResults < 32) {
                setgameDetectorHelper.detMaxResults++
                setgameDetectorHelper.WritePreferences()
                updateTextControlsUi()
            }
        }

        fragmentCameraBinding.bottomSheetLayout.classThresholdMinus.setOnClickListener {
            if (setgameDetectorHelper.classThreshold >= 0.1) {
                setgameDetectorHelper.classThreshold -= 0.1f
                setgameDetectorHelper.WritePreferences()
                updateTextControlsUi()
            }
        }

        // When clicked, raise detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.classThresholdPlus.setOnClickListener {
            if (setgameDetectorHelper.classThreshold <= 0.8) {
                setgameDetectorHelper.classThreshold += 0.1f
                setgameDetectorHelper.WritePreferences()
                updateTextControlsUi()
            }
        }


        // When clicked, decrease the number of threads used for detection
        fragmentCameraBinding.bottomSheetLayout.threadsMinus.setOnClickListener {
            if (setgameDetectorHelper.numThreads > 1) {
                setgameDetectorHelper.numThreads--
                setgameDetectorHelper.WritePreferences()
                updateTextControlsUi()
            }
        }

        // When clicked, increase the number of threads used for detection
        fragmentCameraBinding.bottomSheetLayout.threadsPlus.setOnClickListener {
            if (setgameDetectorHelper.numThreads < 4) {
                setgameDetectorHelper.numThreads++
                setgameDetectorHelper.WritePreferences()
                updateTextControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference. Current options are CPU
        // GPU, and NNAPI
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(setgameDetectorHelper.currentDelegate, false)
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                    setgameDetectorHelper.currentDelegate = p2
                    setgameDetectorHelper.WritePreferences()
                    updateTextControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        // When clicked, change the underlying model used for object detection
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.setSelection(setgameDetectorHelper.currentModel, false)
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                    setgameDetectorHelper.currentModel = p2
                    setgameDetectorHelper.WritePreferences()
                    updateTextControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }


        fragmentCameraBinding.bottomSheetLayout.startstopButton.setOnClickListener {
            /* update button*/
            setgameDetectorHelper.scanEnabled = !setgameDetectorHelper.scanEnabled
            setgameDetectorHelper.clearCards()
            updateTextControlsUi()
        }

        val resolutionDialog = Dialog(requireContext())
        fragmentCameraBinding.bottomSheetLayout.resolutionButton.setOnClickListener {
            resolutionDialog.setContentView(R.layout.resolution_dialog)
            resolutionDialog.window!!.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            resolutionDialog.setCancelable(false)
            //dialog.window!!.attributes.windowAnimations = R.style.animation

            val spinner_camera = resolutionDialog.findViewById<Spinner>(R.id.spinner_camera)
            val spinner_resolution = resolutionDialog.findViewById<Spinner>(R.id.spinner_resolution)
            val okay_text = resolutionDialog.findViewById<TextView>(R.id.okay_text)
            val cancel_text = resolutionDialog.findViewById<TextView>(R.id.cancel_text)

            okay_text.setOnClickListener(View.OnClickListener {
                //store preferences
                val preferences = PreferenceManager.getDefaultSharedPreferences(context)
                val c = preferences.getString("camera", "")
                val r = preferences.getString("resolution", "")

                var configChanged = false
                if (c != spinner_camera.adapter.getItem(spinner_camera.selectedItemId.toInt()).toString() ||
                        r != spinner_resolution.adapter.getItem(spinner_resolution.selectedItemId.toInt()).toString()) {
                    configChanged = true
                    val editor = preferences.edit()
                    editor.putString("camera", spinner_camera.adapter.getItem(spinner_camera.selectedItemId.toInt()).toString())
                    editor.putString("resolution", spinner_resolution.adapter.getItem(spinner_resolution.selectedItemId.toInt()).toString())
                    editor.apply()
                }

                resolutionDialog.dismiss()

                if (configChanged) {
                    Toast.makeText(requireContext(), "restarting to apply changes", Toast.LENGTH_SHORT).show()
                    // restart app to apply the new settings
                    doRestart(requireContext())
                }
            })

            cancel_text.setOnClickListener(View.OnClickListener {
                resolutionDialog.dismiss()
            })

            val cm = context?.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameras_list = BuildCamerasList(cm)
            var cameras_adapter = ArrayAdapter<SelectedCamera>(requireContext(), R.layout.spinner_item,  cameras_list)
            spinner_camera.adapter = cameras_adapter
            spinner_camera.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long) {
                    val upperAdapter = p0!!.adapter
                    val camera = upperAdapter.getItem(p2)
                    // update resolution list if resolution has a different selected camera?
                    if (!spinner_resolution.adapter.isEmpty && (spinner_resolution.adapter.getItem(0) as SelectedCameraResolution).parent != camera) {
                        val cm = context?.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                        var adapter = ArrayAdapter<SelectedCameraResolution>(requireContext(), R.layout.spinner_item, BuildCameraResolutionList(cm, camera as SelectedCamera))
                        spinner_resolution.adapter = adapter
                    }
                }
                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }
            if (cameras_list.size > 0) {
                // read and set dialog
                val preferences = PreferenceManager.getDefaultSharedPreferences(context)

                val c = preferences.getString("camera", cameras_list[0].toString())
                // find the current id
                var ci = cameras_list.indexOfFirst { it.toString() == c }
                if (ci < 0){ ci = 0}
                spinner_camera.setSelection(ci, false)

                val resolution_list = BuildCameraResolutionList(cm, cameras_list[ci])
                var resolution_adapter = ArrayAdapter<SelectedCameraResolution>(requireContext(), R.layout.spinner_item, resolution_list)
                spinner_resolution.adapter = resolution_adapter

                val r = preferences.getString("resolution", resolution_list[0].toString())
                // find the resolution id
                var ri = resolution_list.indexOfFirst { it.toString() == r }
                if (ri < 0){ ri = 0}
                spinner_resolution.setSelection(ri, false)
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
                if (!showDetections)
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
                val textWidth = bounds.width()
                val textHeight = bounds.height()
                canvas.drawRect(
                        0f,
                        fpsTop,
                        0f + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                        fpsTop + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                        textBackgroundPaint
                )
                // Draw text for detected object
                canvas.drawText(fpsText, 0f, fpsTop+bounds.height(), textPaint)

                //show results
                for (result in results) {
                    val boundingBox = result.getBoundingBox()

                    val top = boundingBox.top * scaleFactor
                    val bottom = boundingBox.bottom * scaleFactor
                    val left = boundingBox.left * scaleFactor
                    val right = boundingBox.right * scaleFactor

                    // Draw bounding box around detected objects
                    val drawableRect = RectF(left, top, right, bottom)
                    if (result is Grouppable) {
                        for(groupId in result.getGroupIds()) {
                            assert(groupBoxPaintMap.size > 0)
                            val pb = groupBoxPaintMap.get(groupId%groupBoxPaintMap.size)!!
                            canvas.drawRect(drawableRect, pb)
                            // make all groups visible
                            drawableRect.left -= pb.strokeWidth
                            drawableRect.top -= pb.strokeWidth
                            drawableRect.bottom += pb.strokeWidth
                            drawableRect.right += pb.strokeWidth
                        }
                    }else{
                        canvas.drawRect(drawableRect, boxPaint)
                    }

                    if (result.getCategories().size > 0) {
                        // Create text to display alongside detected objects
                        var shiftX = 0
                        var label = result.getCategories()[0].label + " "
                        val crd = CardValue.fromString(result.getCategories()[0].label)
                        if (crd != null && thumbnailsBitmapHelper != null) {
                            // don't need label - we'll draw a picture instead
                            label = ""
                            shiftX = thumbnailsBitmapHelper!!.thumbnailsBitmap!!.width / 9 // we have put 9 cards in the row

                            val idx = thumbnailsBitmapHelper!!.getThumbIndx(crd)
                            val column = thumbnailsBitmapHelper!!.getThumbColumn(idx)
                            val row = thumbnailsBitmapHelper!!.getThumbRow(idx)

                            val src = Rect(
                                    thumbnailsBitmapHelper!!.thumbnailsBitmap!!.width / 9 * column,
                                    thumbnailsBitmapHelper!!.thumbnailsBitmap!!.height / 9 * row,
                                    thumbnailsBitmapHelper!!.thumbnailsBitmap!!.width / 9 * (column + 1),
                                    thumbnailsBitmapHelper!!.thumbnailsBitmap!!.height / 9 * (row + 1))
                            val dst = RectF(
                                    left,
                                    top,
                                    left + thumbnailsBitmapHelper!!.thumbnailsBitmap!!.width / 9,
                                    top + thumbnailsBitmapHelper!!.thumbnailsBitmap!!.height / 9)

                            canvas.drawBitmap(thumbnailsBitmapHelper!!.thumbnailsBitmap!!,
                                    src,
                                    dst,
                                    textBackgroundPaint
                            )
                        }

                        val drawableText = label + String.format("%.2f", result.getCategories()[0].score)

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
        })

        // overlay touch dialog
        val overrideDialog = Dialog(requireContext())
        fragmentCameraBinding.overlay.setOnTouchListener(View.OnTouchListener { view, event ->
            if (event != null && thumbnailsBitmapHelper != null) {
                val dialog = overrideDialog
                if (event.action == MotionEvent.ACTION_DOWN) {
                    //Toast.makeText(context, (event.rawX/scaleFactor).toString()+ ", "+ (event.rawY/scaleFactor).toString() +" touched", Toast.LENGTH_SHORT).show()
                    overrideDialog.setContentView(R.layout.card_override_dialog)
                    overrideDialog.window!!.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                    overrideDialog.setCancelable(false)

                    val okay_text = overrideDialog.findViewById<TextView>(R.id.okay_text)
                    val cancel_text = overrideDialog.findViewById<TextView>(R.id.cancel_text)

                    // variations per axis
                    val minusCount = overrideDialog.findViewById<ImageButton>(R.id.minus_count)
                    val plusCount = overrideDialog.findViewById<ImageButton>(R.id.plus_count)
                    val minusColor = overrideDialog.findViewById<ImageButton>(R.id.minus_color)
                    val plusColor = overrideDialog.findViewById<ImageButton>(R.id.plus_color)
                    val minusFill = overrideDialog.findViewById<ImageButton>(R.id.minus_fill)
                    val plusFill = overrideDialog.findViewById<ImageButton>(R.id.plus_fill)
                    val minusShape = overrideDialog.findViewById<ImageButton>(R.id.minus_shape)
                    val plusShape = overrideDialog.findViewById<ImageButton>(R.id.plus_shape)

                    fun setView(currentCardValue: CardValue) {
                        val current = overrideDialog.findViewById<ImageView>(R.id.current)
                        current.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndx(currentCardValue)))
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
                        minusCount.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndx(minusCountCard)))
                        minusCount.setOnClickListener(View.OnClickListener {
                            setView(minusCountCard)
                        })
                        plusCount.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndx(plusCountCard)))
                        plusCount.setOnClickListener(View.OnClickListener {
                            setView(plusCountCard)
                        })
                        minusColor.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndx(minusColorCard)))
                        minusColor.setOnClickListener(View.OnClickListener {
                            setView(minusColorCard)
                        })
                        plusColor.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndx(plusColorCard)))
                        plusColor.setOnClickListener(View.OnClickListener {
                            setView(plusColorCard)
                        })
                        minusFill.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndx(minusFillCard)))
                        minusFill.setOnClickListener(View.OnClickListener {
                            setView(minusFillCard)
                        })
                        plusFill.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndx(plusFillCard)))
                        plusFill.setOnClickListener(View.OnClickListener {
                            setView(plusFillCard)
                        })
                        minusShape.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndx(minusShapeCard)))
                        minusShape.setOnClickListener(View.OnClickListener {
                            setView(minusShapeCard)
                        })
                        plusShape.setImageBitmap(thumbnailsBitmapHelper!!.getSingleThumbBitmap(thumbnailsBitmapHelper!!.getThumbIndx(plusShapeCard)))
                        plusShape.setOnClickListener(View.OnClickListener {
                            setView(plusShapeCard)
                        })
                    }

                    cancel_text.setOnClickListener(View.OnClickListener {
                        overrideDialog.dismiss()
                    })

                    // TODO: find if we pressed within any detected card? if so - propose to override
                    setView(CardValue(
                            CardNumber.ONE,
                            CardColor.GREEN,
                            CardShading.SOLID,
                            CardShape.DIAMOND))

                    overrideDialog.show()
                }
            }
            return@OnTouchListener true
        })
    }

    // Update the values displayed in the bottom sheet. Reset detector.
    private fun updateTextControlsUi() {

        fragmentCameraBinding.bottomSheetLayout.detMaxResultsValue.text =
            setgameDetectorHelper.detMaxResults.toString()
        fragmentCameraBinding.bottomSheetLayout.detThresholdValue.text =
            String.format("%.2f", setgameDetectorHelper.detThreshold)
        fragmentCameraBinding.bottomSheetLayout.classThresholdValue.text =
                String.format("%.2f", setgameDetectorHelper.classThreshold)
        fragmentCameraBinding.bottomSheetLayout.threadsValue.text =
            setgameDetectorHelper.numThreads.toString()

        if (setgameDetectorHelper.scanEnabled) {
            fragmentCameraBinding.bottomSheetLayout.startstopButton.text =
                getString(R.string.label_startstop_btn_stop)
        } else {
            fragmentCameraBinding.bottomSheetLayout.startstopButton.text =
                getString(R.string.label_startstop_btn_start)
        }

        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        setgameDetectorHelper.clearObjectDetector()
        showDetections = setgameDetectorHelper.scanEnabled
        fragmentCameraBinding.overlay.invalidate()
        //TBD: fragmentCameraBinding.overlay.clear()
    }

    // restart app to apply new camera settings
    fun doRestart(c: Context?) {
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

    private fun BuildCamerasList(cameraManager: CameraManager): Array<SelectedCamera> {
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
                        stringRepr = autoConst + " " + cameraSelectFacingConst[facingId]))
                for (id in 0 until cameraIdPerFacing[facingId]!!.size) {
                    cameras.add(
                            SelectedCamera(
                                    auto = false,
                                    facing = facingId,
                                    cameraId = cameraIdPerFacing[facingId]!!.get(id),
                                    stringRepr = cameraIdPerFacing[facingId]!!.get(id)+ "-"+cameraSelectFacingConst[facingId]))
                }
            }
        }

        return cameras.toTypedArray()
    }

    private fun BuildCameraResolutionList(cameraManager: CameraManager, camera: SelectedCamera): Array<SelectedCameraResolution> {

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

        val cameras_list = BuildCamerasList(cm)
        val c = preferences.getString("camera", cameras_list[0].toString())
        // find the current id
        var ci = cameras_list.indexOfFirst { it.toString() == c }
        if (ci < 0){ ci = 0}

        val resolution_list = BuildCameraResolutionList(cm, cameras_list[ci])
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
                                    //image.width+16,
                                    image.planes[0].rowStride/image.planes[0].pixelStride,
                                    image.height,
                              Bitmap.Config.ARGB_8888
                            )
                        }
                        detectObjects(image)
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

    private fun detectObjects(image: ImageProxy) {
//        if(!setgameDetectorHelper.scanEnabled)
//            return

        // Copy out RGB bits to the shared bitmap buffer
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }


        val imageRotation = image.imageInfo.rotationDegrees
        // Pass Bitmap and rotation to the object detector helper for processing and detection
        setgameDetectorHelper.detect(bitmapBuffer, imageRotation)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = fragmentCameraBinding.viewFinder.display.rotation
    }

    // Update UI after objects have been detected. Extracts original image height/width
    // to scale and place bounding boxes properly through OverlayView
    override fun onResults(
            results: MutableList<Detected>?,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
    ) {
        // if button is enabled
        if(!setgameDetectorHelper.scanEnabled) {
            return
        }

        activity?.runOnUiThread {
            this.inferenceTime = inferenceTime
            this.results = results?: LinkedList<Detected>()

            // PreviewView is in FILL_START mode. So we need to scale up the bounding box to match with
            // the size that the captured images will be displayed.
            scaleFactor = max(fragmentCameraBinding.overlay.width * 1f / imageWidth,
                    fragmentCameraBinding.overlay.height * 1f / imageHeight)

            // Force a redraw
            fragmentCameraBinding.overlay.invalidate()
        }
    }

    override fun onError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }
}

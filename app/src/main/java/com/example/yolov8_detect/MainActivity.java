package com.example.yolov8_detect;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.media.Image;
import android.os.Bundle;
import android.view.WindowManager;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;

import com.google.common.util.concurrent.ListenableFuture;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {
    private ProcessCameraProvider processCameraProvider;
    private PreviewView previewView;
    private RectView rectView;
    private SupportOnnx supportOnnx;
    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.previewView);
        rectView = findViewById(R.id.rectView);

        //자동꺼짐 해제
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        //권한 확인
        permissionCheck();

        //Onnx 처리 지원 객체
        supportOnnx = new SupportOnnx(this);

        //모델 불러오기
        load();

        //카메라 빌드
        setCamera();

        //카메라 켜기
        startCamera();
    }

    public void permissionCheck() {
        PermissionSupport permissionSupport = new PermissionSupport(this, this);
        permissionSupport.checkPermissions();
    }

    public void load() {
        //model, label 불러오기
        supportOnnx.loadModel();
        supportOnnx.loadLabel();
        try {
            //onnxRuntime 활성화
            ortEnvironment = OrtEnvironment.getEnvironment();
            ortSession = ortEnvironment.createSession(this.getFilesDir().getAbsolutePath() + "/" + SupportOnnx.fileName,
                    new OrtSession.SessionOptions());
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    public void setCamera() {
        try {
            ListenableFuture<ProcessCameraProvider> cameraProviderListenableFuture = ProcessCameraProvider.getInstance(this);
            processCameraProvider = cameraProviderListenableFuture.get();
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public void startCamera() {
        //화면 중앙
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        //후면 카메라
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        Preview preview = new Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9).build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        //이미지 분석 빌드
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();

        //label 정보 전달
        rectView.setLabels(supportOnnx.getLabels());

        //분석
        imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), imageProxy -> {
            imageProcessing(imageProxy);
            imageProxy.close();
        });

        //생명 주기 설정
        processCameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    @SuppressLint("UnsafeOptInUsageError")
    public void imageProcessing(ImageProxy imageProxy) {
        Image image = imageProxy.getImage();
        if (image != null) {
            // image -> bitmap
            Bitmap bitmap = supportOnnx.imageToBitmap(image);
            Bitmap bitmap_640 = supportOnnx.rescaleBitmap(bitmap);
            // bitmap -> float buffer
            FloatBuffer imgDataFloat = supportOnnx.bitmapToFloatBuffer(bitmap_640);

            //모델명
            String inputName = ortSession.getInputNames().iterator().next();
            //모델의 요구 입력값
            long[] shape = {SupportOnnx.BATCH_SIZE, SupportOnnx.PIXEL_SIZE, SupportOnnx.INPUT_SIZE, SupportOnnx.INPUT_SIZE};

            try {
                // float buffer -> tensor
                OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnvironment, imgDataFloat, shape);
                // 추론
                OrtSession.Result result = ortSession.run(Collections.singletonMap(inputName, inputTensor));
                // 결과 (v8 의 출력은 [1][xywh + label 의 개수][8400] 입니다.
                float[][][] output = (float[][][]) result.get(0).getValue();

                int rows = output[0][0].length; //8400
                // tensor -> label, score, rectF
                ArrayList<Result> results = supportOnnx.outputsToNMSPredictions(output, rows);

                // rectF 를 보이는 화면의 비율에 맞게 수정
                results = rectView.transFormRect(results);

                // Result(label, score, rectF) -> 화면에 출력
                rectView.clear();
                rectView.resultToList(results);
                rectView.invalidate();

            } catch (OrtException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    protected void onStop() {
        try {
            ortSession.endProfiling();
        } catch (OrtException e) {
            e.printStackTrace();
        }
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        try {
            ortSession.close();
            ortEnvironment.close();
        } catch (OrtException e) {
            e.printStackTrace();
        }
        super.onDestroy();
    }
}
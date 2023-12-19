package com.example.yolov8_detect;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.Image;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

public class SupportOnnx {
    static final String fileName = "fire_640_v8.onnx";
    static final String labelName = "label_fire_v8.txt";
    static final int INPUT_SIZE = 640;
    static final int BATCH_SIZE = 1;
    static final int PIXEL_SIZE = 3;
    static final int FLOAT_SIZE = 4;

    public float iouThresh = 0.5f;
    public float objectThresh = 0.4f;
    private final Context context;
    private String[] labels;

    public SupportOnnx(Context context) {
        this.context = context;
    }

    //모델 불러오기
    public void loadModel() {
        //assets 안의 모델을 byte 형태로 읽어서 가져온다.
        AssetManager assetManager = context.getAssets();
        File outputFile = new File(context.getFilesDir() + "/" + fileName);

        try {
            InputStream inputStream = assetManager.open(fileName);
            OutputStream outputStream = new FileOutputStream(outputFile);
            byte[] buffer = new byte[1024 * 4];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }

            inputStream.close();
            outputStream.flush();
            outputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //라벨 불러오기
    public void loadLabel() {
        try {
            //assets 파일안에 있는 label (문자열) 을 한줄씩 읽어서 저장
            BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(labelName)));
            String line;
            List<String> labelList = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                labelList.add(line);
            }
            // arraylist -> array
            labels = new String[labelList.size()];
            labelList.toArray(labels);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //비트맵 변환
    public Bitmap imageToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();

        //YUV 평면 얻기
        // 1. Bytebuffer -> byte[]
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        byte[] yuvBytes = new byte[ySize + uSize + vSize];
        yBuffer.get(yuvBytes, 0, ySize);
        vBuffer.get(yuvBytes, ySize, vSize);
        uBuffer.get(yuvBytes, ySize + vSize, uSize);

        // 2. byte[] -> yuvImage -> byteArray
        YuvImage yuvImage = new YuvImage(yuvBytes, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 90, out);

        // 3. byteArray -> bitmap
        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    // [가로 * 세로] -> [640 * 640] 으로 변환
    public Bitmap rescaleBitmap(Bitmap bitmap) {
        return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
    }

    // bitmap -> float buffer
    public FloatBuffer bitmapToFloatBuffer(Bitmap bitmap) {
        int cap = BATCH_SIZE * PIXEL_SIZE * INPUT_SIZE * INPUT_SIZE;
        ByteOrder order = ByteOrder.nativeOrder();
        FloatBuffer buffer = ByteBuffer.allocate(cap * FLOAT_SIZE).order(order).asFloatBuffer();

        int area = INPUT_SIZE * INPUT_SIZE;
        int[] bitmapData = new int[area];

        bitmap.getPixels(bitmapData, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < INPUT_SIZE - 1; i++) {
            for (int j = 0; j < INPUT_SIZE - 1; j++) {
                //0~640*640 차례대로 하나씩 가져오기
                int idx = INPUT_SIZE * i + j;
                int pixelValue = bitmapData[idx];
                float imageMean = 0.0f;
                float imageSTD = 255.0f;
                buffer.put(idx, (((pixelValue >> 16 & 0xFF) - imageMean) / imageSTD)); //r값을 정수형 -> 실수형
                buffer.put(idx + area, (((pixelValue >> 8 & 0xFF) - imageMean) / imageSTD)); //g값을 정수형 -> 실수형
                buffer.put(idx + area * 2, (((pixelValue & 0xFF) - imageMean) / imageSTD)); //b값을 정수형 -> 실수형
            }
        }
        buffer.rewind();
        return buffer;
    }

    // 추론 결과 (3차원 텐서) -> 결과 추출
    public ArrayList<Result> outputsToNMSPredictions(float[][][] output, int rows) {
        ArrayList<Result> results = new ArrayList<>();

        float[][][] outputV8 = new float[1][rows][output[0].length];

        // 행렬 전치 (가로 세로 -> 세로 가로)
        for (int l = 0; l < output[0].length ; l++) {
            for (int m = 0; m < rows; m++) {
                outputV8[0][m][l] = output[0][l][m];
            }
        }

        //그대로 output 을 입력으로 하지 않고 전치를 시킨 이유는 yolo v8 이 되면서 출력 행렬이 [1][xywh + label 개수][8400] 이 되었기 때문입니다.
        // yolo v5 의 경우 [1][25200][xywh + conf + label 개수] 형태였습니다.
        // v5의 함수와 함께 사용하기 위해 위와 같이 전치를 시킵니다. 물론 해당 앱은 v5 는 실행 되지 않습니다.
        // conf 값에 대한 내용을 뺐기 때문입니다. v5 를 보고싶으시면 다른 깃허브 코드를 참고해주시면 감사하겠습니다.

        // 각 bounding box 에 대해 가장 확률이 높은 Class 예측
        for (int i = 0; i < rows; ++i) {
            int detectionClass = -1;
            float maxClass = 0;

            float[] _classes = new float[labels.length];
            //3차원 output 배열에서 4번부터있는 label 만 따로 빼서 새로운 1차원 클래스를 만든다.
            System.arraycopy(outputV8[0][i], 4, _classes, 0, labels.length);

            //그 label 중에서 가장 값이 큰 값을 선정한다.
            for (int c = 0; c < labels.length; ++c) {
                if (_classes[c] > maxClass) {
                    detectionClass = c;
                    maxClass = _classes[c];
                }
            }

            //실제 확률 값은 4번의 확률값과 해당 label 의 확률값의 곱이다.
            float confidenceInClass = maxClass;
            //만약 그 확률 값이 특정 확률을 넘어서면 List 형태로 저장한다.
            if (confidenceInClass > objectThresh) {
                float xPos = outputV8[0][i][0];
                float yPos = outputV8[0][i][1];
                float width = outputV8[0][i][2];
                float height = outputV8[0][i][3];

                //사각형은 화면 밖으로 나갈 수 없으니 화면을 넘기면 최대 화면 값을 가지게 한다.
                RectF rectF = new RectF(Math.max(0, xPos - width / 2), Math.max(0, yPos - height / 2),
                        Math.min(INPUT_SIZE - 1, xPos + width / 2), Math.min(INPUT_SIZE - 1, yPos + height / 2));
                Result recognition = new Result(detectionClass, confidenceInClass, rectF);
                results.add(recognition);
            }
        }

        return nms(results);
    }

    // NMS (비최대 억제)
    public ArrayList<Result> nms(ArrayList<Result> results) {
        ArrayList<Result> nmsList = new ArrayList<>();

        for (int k = 0; k < labels.length; k++) {
            //1.find max confidence per class
            PriorityQueue<Result> pq =
                    new PriorityQueue<>(50,
                            (o1, o2) -> Float.compare(o1.getScore(), o2.getScore()));

            for (int i = 0; i < results.size(); i++) {
                if (results.get(i).getLabel() == k) {
                    pq.add(results.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Result[] a = new Result[pq.size()];
                Result[] detections = pq.toArray(a);
                Result max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Result detection = detections[j];
                    RectF b = detection.getRectF();
                    if (box_iou(max.getRectF(), b) < iouThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    // IOU = 교집합/합집합
    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    // 결과 box의 교집합
    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        return w * h;
    }

    // 결과 box의 합집합
    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = Math.max(l1, l2);
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = Math.min(r1, r2);
        return right - left;
    }

    //라벨 전달
    public String[] getLabels() {
        return labels;
    }
}

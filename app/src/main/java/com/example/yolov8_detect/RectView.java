package com.example.yolov8_detect;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class RectView extends View {
    private final Map<RectF, String> fireMap = new HashMap<>();
    private final Map<RectF, String> smokeMap = new HashMap<>();
    private final Paint firePaint = new Paint();
    private final Paint smokePaint = new Paint();
    private final Paint textPaint = new Paint();

    private String[] labels;

    public RectView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        firePaint.setStyle(Paint.Style.STROKE);     //빈 사각형 그림
        firePaint.setStrokeWidth(10.0f);            //굵기 10
        firePaint.setColor(Color.RED);              //빨간색
        firePaint.setStrokeCap(Paint.Cap.ROUND);    //끝을 뭉특하게
        firePaint.setStrokeJoin(Paint.Join.ROUND);  //끝 주위도 뭉특하게
        firePaint.setStrokeMiter(100);              //뭉특한 정도 100도

        smokePaint.setStyle(Paint.Style.STROKE);
        smokePaint.setStrokeWidth(10.0f);
        smokePaint.setColor(Color.GRAY);
        smokePaint.setStrokeCap(Paint.Cap.ROUND);
        smokePaint.setStrokeJoin(Paint.Join.ROUND);
        smokePaint.setStrokeMiter(100);

        textPaint.setTextSize(60.0f);
        textPaint.setColor(Color.WHITE);
    }

    public void setLabels(String[] labels) {
        this.labels = labels;
    }

    // rectF 비율 수정
    public ArrayList<Result> transFormRect(ArrayList<Result> resultArrayList) {
        //핸드폰의 기종에 따라 PreviewView 의 크기는 변한다.
        float scaleX = getWidth() / (float) SupportOnnx.INPUT_SIZE;
        // float scaleY = getHeight() / (float) SupportOnnx.INPUT_SIZE;
        float scaleY = scaleX * 9f / 16f;
        float realY = getWidth() * 9f / 16f;
        float diffY = realY - getHeight();

        for (Result result : resultArrayList) {
            result.getRectF().left *= scaleX;
            result.getRectF().right *= scaleX;
            result.getRectF().top = result.getRectF().top * scaleY - (diffY / 2f);
            result.getRectF().bottom = result.getRectF().bottom * scaleY - (diffY / 2f);
        }
        return resultArrayList;
    }

    //초기화
    public void clear() {
        fireMap.clear();
        smokeMap.clear();
    }

    // Result -> 각각의 해시맵 (fireMap, smokeMap)
    public void resultToList(ArrayList<Result> results) {
        //rectF에는 상자의 좌표값 , String 에는 객체명(화재 or 연기) 과 확률을 적는다.
        for (Result result : results) {
            if (result.getLabel() == 0) { // fire
                fireMap.put(result.getRectF(), labels[0] + ", " + Math.round(result.getScore() * 100) + "%");
            } else {                      // smoke
                smokeMap.put(result.getRectF(), labels[1] + ", " + Math.round(result.getScore() * 100) + "%");
            }
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        // fire(HashMap) -> canvas
        for (Map.Entry<RectF, String> fire : fireMap.entrySet()) {
            canvas.drawRect(fire.getKey(), firePaint);
            canvas.drawText(fire.getValue(), fire.getKey().left + 10.0f, fire.getKey().top + 60.0f, textPaint);
        }
        // smoke(HashMap) -> canvas
        for (Map.Entry<RectF, String> smoke : smokeMap.entrySet()) {
            canvas.drawRect(smoke.getKey(), smokePaint);
            canvas.drawText(smoke.getValue(), smoke.getKey().left + 10.0f, smoke.getKey().top + 60.0f, textPaint);
        }
        super.onDraw(canvas);
    }
}

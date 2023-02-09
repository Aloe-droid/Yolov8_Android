package com.example.yolov8_detect;

import android.graphics.RectF;

public class Result {
    private final int label;
    private final float score;
    private final RectF rectF;

    public Result(int label, float score, RectF rectF) {
        this.label = label;
        this.score = score;
        this.rectF = rectF;
    }

    public int getLabel() {
        return label;
    }

    public float getScore() {
        return score;
    }

    public RectF getRectF() {
        return rectF;
    }
}

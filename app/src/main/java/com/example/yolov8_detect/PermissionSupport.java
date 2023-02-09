package com.example.yolov8_detect;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;

import androidx.core.app.ActivityCompat;

public class PermissionSupport {
    private final Context context;
    private final Activity activity;
    private final String[] permissions;

    public PermissionSupport(Context context, Activity activity){
        this.context = context;
        this.activity = activity;

        permissions = new String[1];
        permissions[0] = Manifest.permission.CAMERA;
    }

    public void checkPermissions(){
        for(String permission : permissions){
            if(ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED){
                if(!ActivityCompat.shouldShowRequestPermissionRationale(activity,permission)){
                    ActivityCompat.requestPermissions(activity,permissions,1);
                }
            }
        }
    }

}

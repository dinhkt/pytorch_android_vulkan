// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.vision.hpestimation;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;


public class JointsView extends View {
    private float[][] joints_loc;

    public JointsView(Context context) {
        super(context);
    }

    public JointsView(Context context, AttributeSet attrs){
        super(context, attrs);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        Paint paint_red = new Paint();
        paint_red.setColor((Color.RED));
        paint_red.setStrokeWidth(10);
        Paint paint_blue = new Paint();
        paint_blue.setColor(Color.BLUE);
        paint_blue.setStrokeWidth(10);
        if (joints_loc == null) return;
        int f[][] = {{0,1,2,3,4},{0,5,6,7,8},{0,9,10,11,12},{0,13,14,15,16},{0,17,18,19,20}};
        for (int i=0;i<5;i++) {
            for (int j = 0; j < 4; j++) {
                if (i%2==0) {
                    canvas.drawLine(joints_loc[f[i][j]][0], joints_loc[f[i][j]][1], joints_loc[f[i][j + 1]][0], joints_loc[f[i][j + 1]][1], paint_red);
                }
                else{
                    canvas.drawLine(joints_loc[f[i][j]][0], joints_loc[f[i][j]][1], joints_loc[f[i][j + 1]][0], joints_loc[f[i][j + 1]][1], paint_blue);
                }
            }
        }
    }

    public void setResults(float[][] results) {
        joints_loc = results.clone();
    }
}

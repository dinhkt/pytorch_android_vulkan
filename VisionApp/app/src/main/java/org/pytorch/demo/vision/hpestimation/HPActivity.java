package org.pytorch.demo.vision.hpestimation;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.TextureView;
import android.view.ViewStub;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.Device;
import org.pytorch.IValue;
//import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Map;

import org.pytorch.MemoryFormat;
import org.pytorch.demo.vision.*;


public class HPActivity extends AbstractCameraXActivity<HPActivity.JointsResult> {
    private Module mModule = null;
    private JointsView jointsView;
    private TextView textview1;
    private TextView textview2;
    private int avg_time=0;
    private int count_frame=0;
    private int total_time=0;
    private String mode="GPU Mode";

    static class JointsResult {
        private final float[][] jointResults;

        public JointsResult(float[][] joints) {
            jointResults=joints.clone();
        }
    }
    @Override
    protected void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        long startTime=System.nanoTime();
        try {
            if (mModule == null) {
                //mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "detnet_script_hmap.ptl"));
                mModule=Module.load(MainActivity.assetFilePath(getApplicationContext(), "detnet_vulkan.pt"),null, Device.VULKAN);
            }
        } catch (IOException e) {
            Log.e("HandPose Estimation", "Error reading assets", e);
            return;
        }
        long stopTime=System.nanoTime();
        Log.i("TIME","Handpose Model Loading time: "+(stopTime-startTime));
    }
    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_hp_estimation;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        jointsView=findViewById(R.id.jointsView);
        return ((ViewStub) findViewById(R.id.hp_estimation_texture_view_stub))
                .inflate()
                .findViewById(R.id.vision_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(JointsResult result) {
        jointsView.setResults(result.jointResults);
        jointsView.invalidate();

        textview1=findViewById(R.id.mode3);
        textview1.setText(mode);

        textview2=findViewById(R.id.time2);
        textview2.setText("Avg time: "+avg_time+" ms");
    }

    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    @Override
    @WorkerThread
    @Nullable
    protected JointsResult analyzeImage(ImageProxy image, int rotationDegrees) {
        Bitmap bitmap = imgToBitmap(image.getImage());
        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 128, 128, true);

        // preparing input tensor
        float[] norm_mean= {0.5F,0.5F,0.5F};
        float[] norm_std= {1.F,1.F,1.F};
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                norm_mean, norm_std, MemoryFormat.CHANNELS_LAST);
        // running the model
        long startTime=System.nanoTime();
        Map<String,IValue> output = mModule.forward(IValue.from(inputTensor)).toDictStringKey();
        long stopTime=System.nanoTime();

        count_frame+=1;
        total_time+=(stopTime-startTime)/1000000;
        if (count_frame==10){
            avg_time=total_time/10;
            Log.i("TIME","Avg Execution time:"+avg_time +"ms");
            count_frame=0;
            total_time=0;
        }

        //process hmap
        final Tensor output_hmap=output.get(new String("h_map")).toTensor();
        final float[] hmap= output_hmap.getDataAsFloatArray();

        final float[][] joints = new float[21][2];
        JointsView f=findViewById(R.id.jointsView);
        if (f!=null){
            int w = f.getWidth();
            int h = f.getHeight();
            for (int i = 0; i < 21; i++) {
                int max_index=0;
                float max_val=hmap[i*1024];
                for (int j=0;j<1024;j++){
                    if (hmap[i*1024+j]>max_val){
                        max_val=hmap[i*1024+j];
                        max_index=j;
                    }
                }
                joints[i][0] = (float) (max_index % 32)*w/ 32;
                joints[i][1] = (float) (max_index / 32)*h/ 32;
            }
        }

//        final Tensor output_hmap=output.get(new String("hmap_")).toTensor();
//        final long[] hmap= output_hmap.getDataAsLongArray();
//
//        final float[][] joints = new float[21][2];
//        JointsView j=findViewById(R.id.jointsView);
//        if (j!=null) {
//            int w = j.getWidth();
//            int h = j.getHeight();
//            for (int i = 0; i < 21; i++) {
//                joints[i][0] = (float) hmap[i * 2] * w / 32;
//                joints[i][1] = (float) hmap[i * 2 + 1] * h / 32;
//            }
//        }
        return new JointsResult(joints);
    }
}

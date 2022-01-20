package org.pytorch.demo.vision.imgclassification;
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
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import org.pytorch.MemoryFormat;
import org.pytorch.demo.vision.*;
public class ICActivity extends AbstractCameraXActivity<String> {
    private Module mModule = null;
    private TextView textview1;
    private TextView textview2;
    private TextView textview3;
    private int avg_time=0;
    private int count_frame=0;
    private int total_time=0;
    private String mode="GPU Mode";
    @Override
    protected void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        long startTime=System.nanoTime();
        try {
            if (mModule == null) {
                //mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "mobilenet_v2.ptl"));
                mModule=Module.load(MainActivity.assetFilePath(getApplicationContext(), "mobilenet_v2_vulkan.pt"),null, Device.VULKAN);
                //mModule=Module.load(MainActivity.assetFilePath(getApplicationContext(), "mobilenet2-cpu.pt"),null, Device.CPU);
            }
        } catch (Exception e) {
            Log.e("Image Classification", "Error reading assets", e);
            return;
        }
        long stopTime=System.nanoTime();
        Log.i("TIME"," Mobilenet Model Loading time: "+(stopTime-startTime)/1000000+"ms");
    }
    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_image_classification;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        return ((ViewStub) findViewById(R.id.image_classification_texture_view_stub))
                .inflate()
                .findViewById(R.id.vision_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(String result) {
        textview1=findViewById(R.id.textView);
        textview1.setText(result);

        textview2=findViewById(R.id.mode);
        textview2.setText(mode);

        textview3=findViewById(R.id.time);
        textview3.setText("Avg time: "+avg_time+" ms");
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
    protected String analyzeImage(ImageProxy image, int rotationDegrees) {
        Bitmap bitmap = imgToBitmap(image.getImage());
        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        // preparing input tensor
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

        // running the model
        long startTime=System.nanoTime();
        final Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor();
        long stopTime=System.nanoTime();
        //Log.i("TIME","Execution time:"+(stopTime-startTime)/1000000 +"ms");
        count_frame+=1;
        total_time+=(stopTime-startTime)/1000000;
        if (count_frame==10){
            avg_time=total_time/10;
            Log.i("TIME","Avg Execution time:"+avg_time +"ms");
            count_frame=0;
            total_time=0;
        }
        // getting tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray();

        // searching for the index with maximum score
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }

        String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];
        return className;
    }
}

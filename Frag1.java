package com.aaa.aaa;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;

import com.aaa.aaa.ml.ConvertedAppleModel;
import com.aaa.aaa.ml.ConvertedCabbageModel;
import com.aaa.aaa.ml.ConvertedChineseCabbageModel;
import com.aaa.aaa.ml.ConvertedFirstModel;
import com.aaa.aaa.ml.ConvertedGarlicModel;
import com.aaa.aaa.ml.ConvertedMandarineModel;
import com.aaa.aaa.ml.ConvertedOnionModel;
import com.aaa.aaa.ml.ConvertedPearModel;
import com.aaa.aaa.ml.ConvertedPersimmonModel;
import com.aaa.aaa.ml.ConvertedPotatoModel;
import com.aaa.aaa.ml.ConvertedRadishModel;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class Frag1 extends Fragment implements View.OnClickListener {

    TextView cate1;
    TextView cate2;
    TextView cate3;
    int imageSize = 224;

    private View view;
    private ImageView img;

    private boolean mIsOpenCVReady = false;

    static {
        System.loadLibrary("opencv_java4");
    }

    Mat image= new Mat();
    Mat result= new Mat();
    Mat bgModel= new Mat();
    Mat fgModel= new Mat();
    Rect rect;


    /*// OpcnCV 선언
   private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(getActivity()) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                    //imageMat=new Mat();
                    Mat image= new Mat();
                    Mat result= new Mat();
                    Mat bgModel= new Mat();
                    Mat fgModel= new Mat();
                    Rect rect;

                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };*/

    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //OpenCVLoader.initDebug();
    }


    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {

        view = inflater.inflate(R.layout.frag1,container,false);
        init();
        return view;
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.btn_camera:
            {
                //카메라 앱 실행
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                //requestCode 0을 전송해줌 화면 돌아왔을때 구분해야함
                startActivityForResult(intent, 0);
                break;
            }
            case R.id.btn_gallery:
            {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                //
                intent.setType("image/*");
                //requestCode 1을 전송해줌 화면 돌아왔을때 구분해야함
                startActivityForResult(intent, 1);
                break;
            }
        }
    }
    private void init(){

        //각 뷰들 초기화
        img = view.findViewById(R.id.img);
        cate1 = view.findViewById(R.id.editTextNumber);
        cate2 = view.findViewById(R.id.editTextNumber2);
        cate3 = view.findViewById(R.id.editTextNumber3);

        ((Button)view.findViewById(R.id.btn_camera)).setOnClickListener(this);
        ((Button)view.findViewById(R.id.btn_gallery)).setOnClickListener(this);

    }


    public void classifyImage(Bitmap image){
        try {
            ConvertedFirstModel model = ConvertedFirstModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedFirstModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"사과", "양배추", "배추", "마늘", "감귤",
                    "양파", "배", "감", "감자", "무"};
            cate1.setText(classes[maxPos]);

            if(maxPos == 0){
                classifyApple(image);
            } else if(maxPos == 1){
                classifyCabbage(image);
            } else if(maxPos == 2){
                classifyChineseCabbage(image);
            } else if(maxPos == 3){
                classifyGarlic(image);
            } else if(maxPos == 4){
                classifyMandarine(image);
            } else if(maxPos == 5){
                classifyOnion(image);
            } else if(maxPos == 6){
                classifyPear(image);
            } else if(maxPos == 7){
                classifyPersimmon(image);
            } else if(maxPos == 8){
                classifyPotato(image);
            } else{
                classifyRadish(image);
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }

    public void classifyApple(Bitmap image){
        try {
            ConvertedAppleModel model = ConvertedAppleModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedAppleModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"apple_fuji_L", "apple_fuji_M", "apple_fuji_S",
                    "apple_yanggwang_L", "apple_yanggwang_M", "apple_yanggwang_S"};

            if(maxPos == 0){
                cate2.setText("부사");
                cate3.setText("특");
            } else if(maxPos == 1){
                cate2.setText("부사");
                cate3.setText("상");
            } else if(maxPos == 2){
                cate2.setText("부사");
                cate3.setText("보통");
            } else if(maxPos == 3){
                cate2.setText("양광");
                cate3.setText("특");
            } else if(maxPos == 4){
                cate2.setText("양광");
                cate3.setText("상");
            } else{
                cate2.setText("양광");
                cate3.setText("보통");
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }

    public void classifyCabbage(Bitmap image){
        try {
            ConvertedCabbageModel model = ConvertedCabbageModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedCabbageModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"cabbage_green_L", "cabbage_green_M", "cabbage_green_S",
                    "cabbage_red_L", "cabbage_red_M", "cabbage_red_S"};

            if(maxPos == 0){
                cate2.setText("일반 양배추");
                cate3.setText("특");
            } else if(maxPos == 1){
                cate2.setText("일반 양배추");
                cate3.setText("상");
            } else if(maxPos == 2){
                cate2.setText("일반 양배추");
                cate3.setText("보통");
            } else if(maxPos == 3){
                cate2.setText("적색 양배추");
                cate3.setText("특");
            } else if(maxPos == 4){
                cate2.setText("적색 양배추");
                cate3.setText("상");
            } else{
                cate2.setText("적색 양배추");
                cate3.setText("보통");
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }

    public void classifyChineseCabbage(Bitmap image){
        try {
            ConvertedChineseCabbageModel model = ConvertedChineseCabbageModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedChineseCabbageModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"chinese-cabbage_L", "chinese-cabbage_M", "chinese-cabbage_S"};

            if(maxPos == 0){
                cate2.setText("일반 배추");
                cate3.setText("특");
            } else if(maxPos == 1){
                cate2.setText("일반 배추");
                cate3.setText("상");
            } else{
                cate2.setText("일반 배추");
                cate3.setText("보통");
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }

    public void classifyGarlic(Bitmap image){
        try {
            ConvertedGarlicModel model = ConvertedGarlicModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedGarlicModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"garlic_uiseong_L", "garlic_uiseong_M", "garlic_uiseong_S"};

            if(maxPos == 0){
                cate2.setText("의성");
                cate3.setText("특");
            } else if(maxPos == 1){
                cate2.setText("의성");
                cate3.setText("상");
            } else{
                cate2.setText("의성");
                cate3.setText("보통");
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }

    public void classifyMandarine(Bitmap image){
        try {
            ConvertedMandarineModel model = ConvertedMandarineModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedMandarineModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"mandarine_hallabong_L", "mandarine_hallabong_M", "mandarine_hallabong_S",
                    "mandarine_onjumilgam_L", "mandarine_onjumilgam_M", "mandarine_onjumilgam_S"};

            if(maxPos == 0){
                cate2.setText("한라봉");
                cate3.setText("특");
            } else if(maxPos == 1){
                cate2.setText("한라봉");
                cate3.setText("상");
            } else if(maxPos == 2){
                cate2.setText("한라봉");
                cate3.setText("보통");
            } else if(maxPos == 3){
                cate2.setText("온주 밀감");
                cate3.setText("특");
            } else if(maxPos == 4){
                cate2.setText("온주 밀감");
                cate3.setText("상");
            } else{
                cate2.setText("온주 밀감");
                cate3.setText("보통");
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }

    public void classifyOnion(Bitmap image){
        try {
            ConvertedOnionModel model = ConvertedOnionModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedOnionModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"onion_red_L", "onion_red_M", "onion_red_S",
                    "onion_white_L", "onion_white_M", "onion_white_S"};

            if(maxPos == 0){
                cate2.setText("적색 양파");
                cate3.setText("특");
            } else if(maxPos == 1){
                cate2.setText("적색 양파");
                cate3.setText("상");
            } else if(maxPos == 2){
                cate2.setText("적색 양파");
                cate3.setText("보통");
            } else if(maxPos == 3){
                cate2.setText("일반 양파");
                cate3.setText("특");
            } else if(maxPos == 4){
                cate2.setText("일반 양파");
                cate3.setText("상");
            } else{
                cate2.setText("일반 양파");
                cate3.setText("보통");
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }

    public void classifyPear(Bitmap image){
        try {
            ConvertedPearModel model = ConvertedPearModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedPearModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"pear_chuhwang_L", "pear_chuhwang_M", "pear_chuhwang_S",
                    "pear_singo_L", "pear_singo_M", "pear_singo_S"};

            if(maxPos == 0){
                cate2.setText("추황");
                cate3.setText("특");
            } else if(maxPos == 1){
                cate2.setText("추황");
                cate3.setText("상");
            } else if(maxPos == 2){
                cate2.setText("추황");
                cate3.setText("보통");
            } else if(maxPos == 3){
                cate2.setText("신고");
                cate3.setText("특");
            } else if(maxPos == 4){
                cate2.setText("신고");
                cate3.setText("상");
            } else{
                cate2.setText("신고");
                cate3.setText("보통");
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }

    public void classifyPersimmon(Bitmap image){
        try {
            ConvertedPersimmonModel model = ConvertedPersimmonModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedPersimmonModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"persimmon_bansi_L", "persimmon_bansi_M", "persimmon_bansi_S",
                    "persimmon_booyu_L", "persimmon_booyu_M", "persimmon_booyu_S",
                    "persimmon_daebong_L", "persimmon_daebong_M", "persimmon_daebong_S"};

            if(maxPos == 0){
                cate2.setText("반시");
                cate3.setText("특");
            } else if(maxPos == 1){
                cate2.setText("반시");
                cate3.setText("상");
            } else if(maxPos == 2){
                cate2.setText("반시");
                cate3.setText("보통");
            } else if(maxPos == 3){
                cate2.setText("부유");
                cate3.setText("특");
            } else if(maxPos == 4){
                cate2.setText("부유");
                cate3.setText("상");
            } else if(maxPos == 5){
                cate2.setText("부유");
                cate3.setText("보통");
            } else if(maxPos == 6){
                cate2.setText("대봉");
                cate3.setText("특");
            } else if(maxPos == 7){
                cate2.setText("대봉");
                cate3.setText("상");
            } else{
                cate2.setText("대봉");
                cate3.setText("보통");
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }

    public void classifyPotato(Bitmap image){
        try {
            ConvertedPotatoModel model = ConvertedPotatoModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedPotatoModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"potato_seolbong_L", "potato_seolbong_M", "potato_seolbong_S",
                    "potato_sumi_L", "potato_sumi_M", "potato_sumi_S"};

            if(maxPos == 0){
                cate2.setText("설봉");
                cate3.setText("특");
            } else if(maxPos == 1){
                cate2.setText("설봉");
                cate3.setText("상");
            } else if(maxPos == 2){
                cate2.setText("설봉");
                cate3.setText("보통");
            } else if(maxPos == 3){
                cate2.setText("수미");
                cate3.setText("특");
            } else if(maxPos == 4){
                cate2.setText("수미");
                cate3.setText("상");
            } else{
                cate2.setText("수미");
                cate3.setText("보통");
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }

    public void classifyRadish(Bitmap image){
        try {
            ConvertedRadishModel model = ConvertedRadishModel.newInstance(getActivity());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ConvertedRadishModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"radish_winter-radish_L", "radish_winter-radish_M", "radish_winter-radish_S"};

            if(maxPos == 0){
                cate2.setText("겨울 무");
                cate3.setText("특");
            } else if(maxPos == 1){
                cate2.setText("겨울 무");
                cate3.setText("상");
            } else{
                cate2.setText("겨울 무");
                cate3.setText("보통");
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(getActivity(),"classify 오류",Toast.LENGTH_SHORT).show();
        }
    }





    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (!mIsOpenCVReady) {
            return;
        }

        switch (requestCode) {
            // 사진업로드 이벤트
            case 0:
                // 사진 선택
                if (resultCode == Activity.RESULT_OK) {

                    Bitmap imageBitmap = null;
                    try {
                        // Image 상대경로를 가져온다
                        Bundle extras = data.getExtras();
                        imageBitmap = (Bitmap) extras.get("data");

                    } catch (Exception e) {
                        // 대기메시지 종료
                    }

                    int dimension = Math.min(imageBitmap.getWidth(), imageBitmap.getHeight());
                    imageBitmap = ThumbnailUtils.extractThumbnail(imageBitmap, dimension, dimension);
                    img.setImageBitmap(imageBitmap);

                    imageBitmap = Bitmap.createScaledBitmap(imageBitmap, imageSize, imageSize, false);

                    // OpenCV 배경제거 처리
                    rect = new Rect(0, 0, 223, 223);
                    Utils.bitmapToMat(imageBitmap, image);
                    Imgproc.cvtColor(image, image, Imgproc.COLOR_RGBA2RGB);
                    System.out.println(image.type());

                    Imgproc.grabCut(image, result, rect, bgModel, fgModel, 5, Imgproc.GC_INIT_WITH_RECT);
                    Core.compare(result, new Scalar(Imgproc.GC_PR_FGD), result, Core.CMP_EQ);
                    Mat foreground = new Mat(image.size(), CvType.CV_8UC3,
                            new Scalar(255, 255, 255));
                    image.copyTo(foreground, result);

                    Utils.matToBitmap(foreground, imageBitmap);
                    img.setImageBitmap(imageBitmap);

                    classifyImage(imageBitmap);

                } // 사진 선택 취소
                else if (resultCode == Activity.RESULT_CANCELED) {

                }
                break;
            case 1:
                if (resultCode == Activity.RESULT_OK) {
                    Uri dat = data.getData();
                    Bitmap bitmap = null;
                    try {
                        // Image 상대경로를 가져온다
                        //Uri uri = data.getData();
                        // img.setImageURI(uri);

                        bitmap = MediaStore.Images.Media.getBitmap(getActivity().getContentResolver(), dat);

                    } catch (Exception e) {
                        // 대기메시지 종료
                    }

                    //img.setImageBitmap(bitmap);

                    bitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, false);

                    // OpenCV 배경제거 처리
                    rect = new Rect(0, 0, 223, 223);
                    Utils.bitmapToMat(bitmap, image);
                    Imgproc.cvtColor(image, image, Imgproc.COLOR_RGBA2RGB);
                    System.out.println(image.type());

                    Imgproc.grabCut(image, result, rect, bgModel, fgModel, 5, Imgproc.GC_INIT_WITH_RECT);
                    Core.compare(result, new Scalar(Imgproc.GC_PR_FGD), result, Core.CMP_EQ);
                    Mat foreground = new Mat(image.size(), CvType.CV_8UC3,
                            new Scalar(255, 255, 255));
                    image.copyTo(foreground, result);

                    Utils.matToBitmap(foreground, bitmap);
                    img.setImageBitmap(bitmap);

                    // 품질 판별 분류기
                    classifyImage(bitmap);

                } // 사진 선택 취소
                else if (resultCode == Activity.RESULT_CANCELED) {

                }
                break;
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        FragmentActivity activity = getActivity();
        if (activity != null) {
            ((BottomActivity) activity).setActionBarTitle("품질 판별");
        }

        if (OpenCVLoader.initDebug()) {
            mIsOpenCVReady = true;
        }

        /*// OpenCV load check
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, getActivity(), mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }*/

    }

}

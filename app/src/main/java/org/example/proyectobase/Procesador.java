package org.example.proyectobase;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jamarfal on 19/4/17.
 */

public class Procesador {
    public Procesador() { //Constructor
    }

    public Mat procesa(Mat entrada) {

        // Mat original
        Mat Gx = new Mat();
        Mat Gy = new Mat();

        Imgproc.Sobel(entrada, Gx, CvType.CV_32FC1, 1, 0); //Derivada primera rto x
        Imgproc.Sobel(entrada, Gy, CvType.CV_32FC1, 0, 1); //Derivada primera rto y

        Mat ModGrad = new Mat();
        Mat AngGrad = new Mat();

        Core.cartToPolar(Gx, Gy, ModGrad, AngGrad);


        return ModGrad;
    }
}
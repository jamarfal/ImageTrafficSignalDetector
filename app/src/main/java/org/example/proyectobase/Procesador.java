package org.example.proyectobase;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
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
        double tam = 11;

        // Elemento estructurante
        Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(tam, tam));
        // Dilatación
        Mat gray_dilation = new Mat(); // Result
        Imgproc.dilate(entrada, gray_dilation, SE); // 3x3 dilation
        // Cálculo del residuo
        Mat dilation_residue = new Mat();
        Core.subtract(gray_dilation, entrada, dilation_residue);
        return dilation_residue;
    }
}
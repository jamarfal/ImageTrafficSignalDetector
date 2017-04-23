package org.example.proyectobase;

import org.opencv.core.Core;
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
 *
 * Aumento Lineal de Contraste
 */

public class Procesador {

    Procesador() {

    }

    public Mat procesa(Mat entrada) {
        Mat salida = new Mat();
        Imgproc.equalizeHist(entrada, salida);
        return salida;
    }
}
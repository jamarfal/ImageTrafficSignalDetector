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

    MatOfInt canales;
    MatOfInt numero_bins;
    MatOfFloat intervalo;
    Mat hist;
    List<Mat> imagenes;
    float[] histograma;
    Mat paso_bajo;

    Procesador() {
        canales = new MatOfInt(0);
        numero_bins = new MatOfInt(256);
        intervalo = new MatOfFloat(0, 256);
        hist = new Mat();
        imagenes = new ArrayList<Mat>();
        histograma = new float[256];
        paso_bajo= new Mat();
    }

    public Mat procesa(Mat entrada) {
        Mat salida = new Mat();
        int filter_size = 17;
        Size s=new Size(filter_size,filter_size);
        Imgproc.blur(entrada, paso_bajo, s);
// Hacer la resta. Los valores negativos saturan a cero
        Core.subtract(paso_bajo, entrada, salida);
//Aplicar Ganancia para ver mejor. La multiplicacion satura
        Scalar ganancia = new Scalar(2);
        Core.multiply(salida, ganancia, salida);
        return salida;
    }

}
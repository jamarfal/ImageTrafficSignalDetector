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

    void mitadMitad(Mat entrada, Mat salida) {
//Representar la entrada en la mitad izquierda
        Rect mitad_izquierda = new Rect();
        mitad_izquierda.x = 0; mitad_izquierda.y = 0;
        mitad_izquierda.height = entrada.height();
        mitad_izquierda.width = entrada.width()/2;
        Mat salida_mitad_izquierda = salida.submat( mitad_izquierda );
        Mat entrada_mitad_izquierda = entrada.submat( mitad_izquierda );
        entrada_mitad_izquierda.copyTo(salida_mitad_izquierda);
    }
}
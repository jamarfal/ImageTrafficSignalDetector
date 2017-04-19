package org.example.proyectobase;

import org.opencv.core.Mat;

/**
 * Created by jamarfal on 19/4/17.
 */

public class Procesador {

    public Procesador() { //Constructor
    }

    public Mat procesa(Mat entrada) {
        Mat salida = entrada.clone();
        return salida;
    }
}

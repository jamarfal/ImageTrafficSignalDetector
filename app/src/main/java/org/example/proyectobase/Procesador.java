package org.example.proyectobase;

import org.opencv.core.Core;
import org.opencv.core.Mat;

/**
 * Created by jamarfal on 19/4/17.
 */

public class Procesador {
    Mat red;
    Mat green;
    Mat blue;
    Mat maxGB;
    Mat maxRB;
    public Procesador() { //Constructor
        red = new Mat();
        green = new Mat();
        blue = new Mat();
        maxGB = new Mat();
        maxRB = new Mat();
    }
    public Mat procesa(Mat entrada) {
        Mat salida = new Mat();
        Core.extractChannel(entrada, red, 0);
        Core.extractChannel(entrada, green, 1);
        Core.extractChannel(entrada, blue, 2);
        Core.max(green, blue, maxGB);
        Core.subtract( red , maxGB , salida );
        return salida;
    }

    public Mat procesaGreen(Mat entrada) {
        Mat salida = new Mat();
        Core.extractChannel(entrada, red, 0);
        Core.extractChannel(entrada, green, 1);
        Core.extractChannel(entrada, blue, 2);
        Core.max(red, blue, maxRB);
        Core.subtract( green , maxRB , salida );
        return salida;
    }


}

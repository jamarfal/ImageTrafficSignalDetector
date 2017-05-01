package org.example.proyectobase;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

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
    public Mat procedaRed(Mat entrada) {
        Mat salida = new Mat();
        Core.extractChannel(entrada, red, 0);
        Core.extractChannel(entrada, green, 1);
        Core.extractChannel(entrada, blue, 2);
        Core.max(green, blue, maxGB);
        Core.subtract( red , maxGB , salida );
        Core.MinMaxLocResult minMax = Core.minMaxLoc(salida);
        int maximum = (int) minMax.maxVal;
        int thresh = maximum / 4;
        Imgproc.threshold(salida, salida, thresh, 255, Imgproc.THRESH_BINARY);
        return salida;
    }

//    public Mat procesaGreen(Mat entrada) {
//        Mat salida = new Mat();
//        Core.extractChannel(entrada, red, 0);
//        Core.extractChannel(entrada, green, 1);
//        Core.extractChannel(entrada, blue, 2);
//        Core.max(red, blue, maxRB);
//        Core.subtract( green , maxRB , salida );
//        return salida;
//    }


}

package org.example.proyectobase;

import android.support.annotation.NonNull;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
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
    Mat red;
    Mat green;
    Mat blue;
    Mat maxGB;
    Rect rectCirculo;
    Mat localizada;

    public Procesador() { //Constructor
        red = new Mat();
        green = new Mat();
        blue = new Mat();
        maxGB = new Mat();
        rectCirculo = new Rect();
        localizada = new Mat();
    }

    public Mat procesa(Mat entrada) { //entrada: imagen color
        if (!localizarCirculoRojo(entrada))
            return entrada.clone();
        Mat salida = segmentarInteriorDisco(rectCirculo);
        return salida;
    }

    public boolean localizarCirculoRojo(Mat entrada) {
        Mat binaria = binarizacionZonasRojas(entrada);

        //Seleccion a candidatos a circulos
        List<MatOfPoint> blobs = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        localizada = entrada.clone();//Copia porque finContours modifica entrada
        //Imgproc.cvtColor(salida, salida, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.findContours(binaria, blobs, hierarchy, Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_NONE);
        int minimumHeight = 30;
        float maxratio = (float) 0.75;
        boolean hayCirculo = false;
        // Seleccionar candidatos a circulos
        for (int c = 0; c < blobs.size(); c++) {
            double[] data = hierarchy.get(0, c);
            int parent = (int) data[3];
            if (parent < 0) //Contorno exterior: rechazar
                continue;
            Rect BB = Imgproc.boundingRect(blobs.get(c));
// Comprobar tamaÃ±o
            if (BB.width < minimumHeight || BB.height < minimumHeight)
                continue;
// Comprobar anchura similar a altura
            float wf = BB.width;
            float hf = BB.height;
            float ratio = wf / hf;
            if (ratio < maxratio || ratio > 1.0 / maxratio)
                continue;
// Comprobar no estÃ¡ cerca del borde
            if (BB.x < 2 || BB.y < 2)
                continue;
            if (entrada.width() - (BB.x + BB.width) < 3 || entrada.height() - (BB.y + BB.height) < 3)
                continue;
// Aqui cumple todos los criterios. Dibujamos

            rectCirculo = BB.clone();
            hayCirculo = true;
            final Point P1 = new Point(BB.x, BB.y);
            final Point P2 = new Point(BB.x + BB.width - 1, BB.y + BB.height - 1);
            Imgproc.rectangle(localizada, P1, P2, new Scalar(0, 0, 255));


        } // for
        hierarchy.release();
        return hayCirculo;
    }

    @NonNull
    private Mat binarizacionZonasRojas(Mat entrada) {
        Mat binaria = new Mat();
        Core.extractChannel(entrada, red, 0);
        Core.extractChannel(entrada, green, 1);
        Core.extractChannel(entrada, blue, 2);
        Core.max(green, blue, maxGB);
        Core.subtract(red, maxGB, binaria);
        green.release();
        blue.release();
        red.release();
        maxGB.release();
        //BinarizaciÃ³n de zonas rojas
        Core.MinMaxLocResult minMax = Core.minMaxLoc(binaria);
        int maximum = (int) minMax.maxVal;
        int thresh = maximum / 2;
        Imgproc.threshold(binaria, binaria, thresh, 255, Imgproc.THRESH_BINARY);
        return binaria;
    }

    public Mat segmentarInteriorDisco(Rect rectEntrada) {
        Mat miSenal = localizada.submat(rectEntrada);
        Mat aux = new Mat();
        Core.extractChannel(miSenal, red, 0);
        Core.extractChannel(miSenal, green, 1);
        Core.extractChannel(miSenal, blue, 2);
        Core.max(green, blue, maxGB);
        Core.subtract(red, maxGB, aux);
        Imgproc.threshold(aux, aux, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        green.release();
        blue.release();
        red.release();
        maxGB.release();
        //Seleccion a candidatos a circulos
        List<MatOfPoint> blobsd = new ArrayList<MatOfPoint>();
        Mat hierarchyd = new Mat();
        //Imgproc.cvtColor(salida, salida, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.findContours(aux, blobsd, hierarchyd, Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_NONE);
        aux.release();
        int maxHeight = miSenal.height();
// Seleccionar candidatos a circulos
        for (int c = 0; c < blobsd.size(); c++) {
            Rect BBd = Imgproc.boundingRect(blobsd.get(c));
// Comprobar tamaÃ±o
           if (BBd.height <= (maxHeight / 3)) {
                Log.i("Cuadro verde 1: ", "altura menor o igual que tercio");
                continue;
            }
            if (BBd.height <= 12) {
                Log.i("Cuadro verde 2: ", "altura menor o igual 12 pixels");
                continue;
            }
// Comprobar tamaÃ±o
           if (BBd.height < BBd.width){
                Log.i("Cuadro verde 3: ", "altura menor que anchura");
                continue;
            }
// Comprobar no estÃ¡ cerca del borde
            if (miSenal.width() - (BBd.x + BBd.width) < 3 || miSenal.height() - (BBd.y + BBd.height) < 3) {
                Log.i("Cuadro verde 4: ", "borde");
                continue;
            }
// Aqui cumple todos los criterios. Dibujamos
            final Point P1 = new Point(BBd.x + rectEntrada.x, BBd.y + rectEntrada.y);
            final Point P2 = new Point(BBd.x + rectEntrada.x + BBd.width - 1,
                    BBd.y + rectEntrada.y + BBd.height - 1);
            Imgproc.rectangle(localizada, P1, P2, new Scalar(255, 0, 0));
        }
        return localizada;
    }
}
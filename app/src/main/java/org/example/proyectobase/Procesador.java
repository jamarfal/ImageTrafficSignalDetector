package org.example.proyectobase;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
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
    Mat signal;

    public Procesador() { //Constructor
        red = new Mat();
        green = new Mat();
        blue = new Mat();
        maxGB = new Mat();
        signal = new Mat();
    }

    public Mat procesa(Mat entrada) {
        CheckCircle checkCircle = localizarCirculoRojo(entrada);
        if (!checkCircle.existsCircle()) {
            return entrada.clone();
        }

        Mat salida = segmentarInteriorDisco(checkCircle.getRectCircle());
        return salida;
    }

    public CheckCircle localizarCirculoRojo(Mat entrada) {
        Rect rectCircle = null;
        Mat binaria = binarizacionZonasRojas(entrada);

        //Seleccion a candidatos a circulos
        List<MatOfPoint> blobs = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        signal = entrada.clone();//Copia porque finContours modifica entrada

        Imgproc.findContours(binaria, blobs, hierarchy, Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_NONE);
        int minimumHeight = 30;
        float maxratio = (float) 0.75;
        boolean existsCircle = false;
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
            rectCircle = BB.clone();
            existsCircle = true;
            final Point P1 = new Point(BB.x, BB.y);
            final Point P2 = new Point(BB.x + BB.width - 1, BB.y + BB.height - 1);
            Imgproc.rectangle(signal, P1, P2, new Scalar(125, 125, 255));


        } // for
        hierarchy.release();
        return new CheckCircle(existsCircle, rectCircle);
    }

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
        //Binarización de zonas rojas
        Core.MinMaxLocResult minMax = Core.minMaxLoc(binaria);
        int maximum = (int) minMax.maxVal;
        int thresh = maximum / 2;
        Imgproc.threshold(binaria, binaria, thresh, 255, Imgproc.THRESH_BINARY);
        return binaria;
    }

    private Mat otsuTransformation(Mat detectedSignal) {
        Mat otsu = new Mat();
        Core.extractChannel(detectedSignal, red, 0);
        Core.extractChannel(detectedSignal, green, 1);
        Core.extractChannel(detectedSignal, blue, 2);
        Core.max(green, blue, maxGB);
        Core.subtract(red, maxGB, otsu);
        Imgproc.threshold(otsu, otsu, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        green.release();
        blue.release();
        red.release();
        maxGB.release();
        return otsu;
    }

    public Mat segmentarInteriorDisco(Rect rectEntrada) {
        // Recorta la imagen
        Mat detectedSignal = signal.submat(rectEntrada);

        // Aplica Otsu
        Mat aux = otsuTransformation(detectedSignal);

        //Seleccion a de objetos
        List<MatOfPoint> blobsd = new ArrayList<MatOfPoint>();
        Mat hierarchyd = new Mat();
        Imgproc.findContours(aux, blobsd, hierarchyd, Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_NONE);
        aux.release();
        int maxHeight = detectedSignal.height();

        for (int c = 0; c < blobsd.size(); c++) {
            Rect BBd = Imgproc.boundingRect(blobsd.get(c));

            // Tener una altura mayor que la tercera parte del círculo.
            if (BBd.height <= (maxHeight / 3)) {
                continue;
            }

            // Tener una altura mayor de 12 píxeles.
            if (BBd.height <= 12) {
                continue;
            }

            // Tener una altura mayor que su anchura.
            if (BBd.height < BBd.width) {
                continue;
            }

            // No tocar el borde del rectángulo del círculo.
            if (detectedSignal.width() - (BBd.x + BBd.width) < 3 || detectedSignal.height() - (BBd.y + BBd.height) < 3) {
                continue;
            }

            // Aqui cumple todos los criterios. Dibujamos
            final Point P1 = new Point(BBd.x + rectEntrada.x, BBd.y + rectEntrada.y);
            final Point P2 = new Point(BBd.x + rectEntrada.x + BBd.width - 1,
                    BBd.y + rectEntrada.y + BBd.height - 1);
            Imgproc.rectangle(signal, P1, P2, new Scalar(0, 255, 0));
        }
        return signal;
    }
}
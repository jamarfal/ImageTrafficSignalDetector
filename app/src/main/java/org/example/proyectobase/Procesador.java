package org.example.proyectobase;

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

    double tam =11;
    Mat gray_dilation;
    Mat dilation_residue;
    int contraste = 11;
    int tamano = 7;

    public Procesador() {
        gray_dilation = new Mat();
        dilation_residue = new Mat();
    }
    public Mat procesa(Mat entrada) {
        Mat binaria = new Mat();
        Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(tam,tam));
        Imgproc.dilate(entrada, gray_dilation, SE );

        Core.subtract(gray_dilation, entrada, dilation_residue);

        Imgproc.adaptiveThreshold(dilation_residue, binaria, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY, tamano, -contraste );

        List<MatOfPoint> blobs = new ArrayList< MatOfPoint > () ;
        Mat hierarchy = new Mat();
        Mat salida = binaria.clone();//Copia porque finContours modifica entrada
        Imgproc.cvtColor(salida, salida, Imgproc.COLOR_GRAY2RGBA);
        Imgproc.findContours(binaria, blobs, hierarchy, Imgproc.RETR_CCOMP,
                Imgproc.CHAIN_APPROX_NONE );
        int minimumHeight = 30;
        float maxratio = (float) 0.75;
// Seleccionar candidatos a circulos
        for (int c= 0; c< blobs.size(); c++ ) {
            double[] data = hierarchy.get(0,c);
            int parent = (int) data[3];
            if(parent < 0) //Contorno exterior: rechazar
                continue;
            Rect BB = Imgproc.boundingRect(blobs.get(c) );
// Comprobar tamaño
            if ( BB.width < minimumHeight || BB.height < minimumHeight)
                continue;
// Comprobar anchura similar a altura
            float wf = BB.width;
            float hf = BB.height;
            float ratio = wf / hf;
            if(ratio < maxratio || ratio > 1.0/maxratio)
                continue;
// Comprobar no está cerca del borde
            if(BB.x < 2 || BB.y < 2)
                continue;
            if(entrada.width() - (BB.x + BB.width) < 3 || entrada.height() - (BB.y + BB.height) < 3)
                continue;
// Aqui cumple todos los criterios. Dibujamos
            final Point P1 = new Point(BB.x, BB.y);
            final Point P2 = new Point(BB.x+BB.width-1, BB.y+BB.height-1);
            Imgproc.rectangle(salida, P1, P2, new Scalar(255,0,0) );
        } // for
        return salida;
    }
}
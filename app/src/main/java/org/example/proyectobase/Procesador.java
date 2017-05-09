package org.example.proyectobase;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jamarfal on 19/4/17.
 */

public class Procesador {

    private Mat gris;
    private Mat salidaintensidad;
    private Mat salidatrlocal;
    private Mat salidabinarizacion;
    private Mat salidasegmentacion;
    private Mat salidaocr;

    public enum Salida {ENTRADA, INTENSIDAD, OPERADOR_LOCAL, BINARIZACION, SEGMENTACION, RECONOCIMIENTO}

    public enum TipoIntensidad {SIN_PROCESO, LUMINANCIA, AUMENTO_LINEAL_CONSTRASTE, EQUALIZ_HISTOGRAMA, ZONAS_ROJAS}

    public enum TipoOperadorLocal {SIN_PROCESO, PASO_BAJO, PASO_ALTO, GRADIENTES, MORFOLOGICO_3, MORFOLOGICO_11}

    public enum TipoBinarizacion {SIN_PROCESO, ADAPTATIVA, MAXIMO}

    public enum TipoSegmentacion {SIN_PROCESO}

    public enum TipoReconocimiento {SIN_PROCESO}

    private Salida mostrarSalida;
    private TipoIntensidad tipoIntensidad;
    private TipoOperadorLocal tipoOperadorLocal;
    private TipoBinarizacion tipoBinarizacion;
    private TipoSegmentacion tipoSegmentacion;
    private TipoReconocimiento tipoReconocimiento;


    // Aumento Lineal Contraste
    MatOfInt canales;
    MatOfInt numero_bins;
    MatOfFloat intervalo;
    Mat hist;
    List<Mat> imagenes;
    float[] histograma;

    // Detección Zonas rojas
    Mat red;
    Mat green;
    Mat blue;
    Mat maxGB;
    Mat maxRB;

    // Paso Alto
    Mat paso_bajo;

    // Modulo Gradiente Sobel
    private final Mat Gx;
    private final Mat Gy;
    private final Mat AngGrad;

    public Procesador() {
        mostrarSalida = Salida.INTENSIDAD;
        tipoIntensidad = TipoIntensidad.LUMINANCIA;
        tipoOperadorLocal = TipoOperadorLocal.SIN_PROCESO;
        tipoBinarizacion = TipoBinarizacion.SIN_PROCESO;
        tipoSegmentacion = TipoSegmentacion.SIN_PROCESO;
        tipoReconocimiento = TipoReconocimiento.SIN_PROCESO;
        salidaintensidad = new Mat();
        salidatrlocal = new Mat();
        salidabinarizacion = new Mat();
        salidasegmentacion = new Mat();
        salidaocr = new Mat();
        gris = new Mat();

        canales = new MatOfInt(0);
        numero_bins = new MatOfInt(256);
        intervalo = new MatOfFloat(0, 256);
        hist = new Mat();
        imagenes = new ArrayList<Mat>();
        histograma = new float[256];


        red = new Mat();
        green = new Mat();
        blue = new Mat();
        maxGB = new Mat();
        maxRB = new Mat();

        paso_bajo = new Mat();

        Gx = new Mat();
        Gy = new Mat();
        AngGrad = new Mat();
    }

    public Mat procesa(Mat entrada) {

        if (mostrarSalida == Salida.ENTRADA) {
            return entrada;
        }

        // Transformación intensidad
        switch (tipoIntensidad) {
            case SIN_PROCESO:
                salidaintensidad = entrada;
                break;
            case LUMINANCIA:
                Imgproc.cvtColor(entrada, salidaintensidad, Imgproc.COLOR_RGBA2GRAY);
                break;

            case AUMENTO_LINEAL_CONSTRASTE:
                Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
                aumentoLinealContraste(gris); //resultado en salidaintensidad
                break;
            case EQUALIZ_HISTOGRAMA:
                //Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY); //Eq. Hist necesita gris
                Imgproc.equalizeHist(gris, salidaintensidad);
                break;
            case ZONAS_ROJAS:
                zonaRoja(entrada); //resultado en salidaintensidad
                break;
            default:
                salidaintensidad = entrada;
        }

        if (mostrarSalida == Salida.INTENSIDAD) {
            return salidaintensidad;
        }

        // Operador local
        switch (tipoOperadorLocal) {
            case SIN_PROCESO:
                salidatrlocal = salidaintensidad;
                break;
            case PASO_BAJO:
                pasoBajo(salidaintensidad); //resultado en salidatrlocal
                break;
            case PASO_ALTO:
                Imgproc.cvtColor(entrada, gris, Imgproc.COLOR_RGBA2GRAY);
                pasoAlto(salidaintensidad);
            case GRADIENTES:
                sobel(salidaintensidad);
                break;
            case MORFOLOGICO_3:

                gradienteMorfolico(3, salidaintensidad);
                break;

            case MORFOLOGICO_11:

                gradienteMorfolico(11, salidaintensidad);
                break;
        }

        if (mostrarSalida == Salida.OPERADOR_LOCAL) {
            return salidatrlocal;
        }

        // Binarización
        switch (tipoBinarizacion) {
            case SIN_PROCESO:
                salidabinarizacion = salidatrlocal;
                break;
            default:
                salidabinarizacion = salidatrlocal;
                break;
        }

        if (mostrarSalida == Salida.BINARIZACION) {
            return salidabinarizacion;
        }

        // Segmentación
        switch (tipoSegmentacion) {
            case SIN_PROCESO:
                salidasegmentacion = salidabinarizacion;
                break;
        }

        if (mostrarSalida == Salida.SEGMENTACION) {
            return salidasegmentacion;
        }

        // Reconocimiento OCR
        switch (tipoReconocimiento) {
            case SIN_PROCESO:
                salidaocr = salidabinarizacion;
                break;
        }
        return salidaocr;
    }


    void zonaRoja(Mat entrada) { //Ejemplo para ser rellenada en curso
        Core.extractChannel(entrada, red, 0);
        Core.extractChannel(entrada, green, 1);
        Core.extractChannel(entrada, blue, 2);
        Core.max(green, blue, maxGB);
        Core.subtract(red, maxGB, salidaintensidad);
    }

    void aumentoLinealContraste(Mat entrada) {//Ejemplo para ser rellenada en curso
        imagenes.clear(); //Eliminar imagen anterior si la hay
        imagenes.add(entrada); //Añadir imagen actual
        Imgproc.calcHist(imagenes, canales, new Mat(), hist,
                numero_bins, intervalo);
//Lectura del histograma a un array de float
        hist.get(0, 0, histograma);
//Calcular xmin y xmax
        int total_pixeles = entrada.cols() * entrada.rows();
        float porcentaje_saturacion = (float) 0.05;
        int pixeles_saturados = (int) (porcentaje_saturacion * total_pixeles);
        int xmin = 0;
        int xmax = 255;
        float acumulado = 0f;
        for (int n = 0; n < 256; n++) { //xmin
            acumulado = acumulado + histograma[n];
            if (acumulado > pixeles_saturados) {
                xmin = n;
                break;
            }
        }
        acumulado = 0;
        for (int n = 255; n >= 0; n--) { //xmax
            acumulado = acumulado + histograma[n];
            if (acumulado > pixeles_saturados) {
                xmax = n;
                break;
            }
        }
//Calculo de la salida
        Core.subtract(entrada, new Scalar(xmin), salidaintensidad);
        float pendiente = ((float) 255.0) / ((float) (xmax - xmin));
        Core.multiply(salidaintensidad, new Scalar(pendiente), salidaintensidad);
    }

    void pasoBajo(Mat entrada) {//Ejemplo para ser rellenada en curso
        salidatrlocal = entrada;
    }

    private void pasoAlto(Mat entrada) {
        int filter_size = 17;
        Size s = new Size(filter_size, filter_size);
        Imgproc.blur(entrada, paso_bajo, s);
// Hacer la resta. Los valores negativos saturan a cero
        Core.subtract(paso_bajo, entrada, salidatrlocal);
//Aplicar Ganancia para ver mejor. La multiplicacion satura
        Scalar ganancia = new Scalar(2);
        Core.multiply(salidatrlocal, ganancia, salidatrlocal);
    }

    private void sobel(Mat entrada) {
        // Derivadas
        Imgproc.Sobel(entrada, Gx, CvType.CV_32FC1, 1, 0); //Derivada primera rto x
        Imgproc.Sobel(entrada, Gy, CvType.CV_32FC1, 0, 1); //Derivada primera rto y

        // Módulo de Gradiente
        Core.cartToPolar(Gx, Gy, salidatrlocal, AngGrad);

        salidatrlocal.convertTo(salidatrlocal, CvType.CV_8UC1);

    }

    private void gradienteMorfolico(int tam, Mat entrada) {

        // Elemento estructurante
        Mat SE = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(tam, tam));
        // Dilatación
        Mat gray_dilation = new Mat(); // Result
        Imgproc.dilate(entrada, gray_dilation, SE); // 3x3 dilation
        // Cálculo del residuo
        Mat dilation_residue = new Mat();
        Core.subtract(gray_dilation, entrada, dilation_residue);

        //Calculo del gradiente morfológico.
        int contraste = 2;
        int tamano = 7;
        Imgproc.adaptiveThreshold(gray_dilation, dilation_residue,255, Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY, tamano, -contraste );
        salidatrlocal = dilation_residue.clone();

    }

    public Mat getGris() {
        return gris;
    }

    public void setGris(Mat gris) {
        this.gris = gris;
    }

    public Mat getSalidaintensidad() {
        return salidaintensidad;
    }

    public void setSalidaintensidad(Mat salidaintensidad) {
        this.salidaintensidad = salidaintensidad;
    }

    public Mat getSalidatrlocal() {
        return salidatrlocal;
    }

    public void setSalidatrlocal(Mat salidatrlocal) {
        this.salidatrlocal = salidatrlocal;
    }

    public Mat getSalidabinarizacion() {
        return salidabinarizacion;
    }

    public void setSalidabinarizacion(Mat salidabinarizacion) {
        this.salidabinarizacion = salidabinarizacion;
    }

    public Mat getSalidasegmentacion() {
        return salidasegmentacion;
    }

    public void setSalidasegmentacion(Mat salidasegmentacion) {
        this.salidasegmentacion = salidasegmentacion;
    }

    public Mat getSalidaocr() {
        return salidaocr;
    }

    public void setSalidaocr(Mat salidaocr) {
        this.salidaocr = salidaocr;
    }

    public Salida getMostrarSalida() {
        return mostrarSalida;
    }

    public void setMostrarSalida(Salida mostrarSalida) {
        this.mostrarSalida = mostrarSalida;
    }

    public TipoIntensidad getTipoIntensidad() {
        return tipoIntensidad;
    }

    public void setTipoIntensidad(TipoIntensidad tipoIntensidad) {
        this.tipoIntensidad = tipoIntensidad;
    }

    public TipoOperadorLocal getTipoOperadorLocal() {
        return tipoOperadorLocal;
    }

    public void setTipoOperadorLocal(TipoOperadorLocal tipoOperadorLocal) {
        this.tipoOperadorLocal = tipoOperadorLocal;
    }

    public TipoBinarizacion getTipoBinarizacion() {
        return tipoBinarizacion;
    }

    public void setTipoBinarizacion(TipoBinarizacion tipoBinarizacion) {
        this.tipoBinarizacion = tipoBinarizacion;
    }

    public TipoSegmentacion getTipoSegmentacion() {
        return tipoSegmentacion;
    }

    public void setTipoSegmentacion(TipoSegmentacion tipoSegmentacion) {
        this.tipoSegmentacion = tipoSegmentacion;
    }

    public TipoReconocimiento getTipoReconocimiento() {
        return tipoReconocimiento;
    }

    public void setTipoReconocimiento(TipoReconocimiento tipoReconocimiento) {
        this.tipoReconocimiento = tipoReconocimiento;
    }
}

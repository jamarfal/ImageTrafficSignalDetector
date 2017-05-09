package org.example.proyectobase;

import org.opencv.core.Rect;

/**
 * Created by jamarfal on 9/5/17.
 */

public class CheckCircle {

    private boolean existsCircle;
    private Rect rectCircle;

    public CheckCircle(boolean existsCircle, Rect rectCircle) {
        this.existsCircle = existsCircle;
        this.rectCircle = rectCircle;
    }

    public boolean existsCircle() {
        return existsCircle;
    }

    public void setExistsCircle(boolean existsCircle) {
        this.existsCircle = existsCircle;
    }

    public Rect getRectCircle() {
        return rectCircle;
    }

    public void setRectCircle(Rect rectCircle) {
        this.rectCircle = rectCircle;
    }
}

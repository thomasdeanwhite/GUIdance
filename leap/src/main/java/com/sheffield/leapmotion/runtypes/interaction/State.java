package com.sheffield.leapmotion.runtypes.interaction;

import java.io.Serializable;

/**
 * Created by thoma on 21/06/2017.
 */
public class State implements Serializable {

    public static State ORIGIN = new State(-1, null, -1);

    private int lastState;

    private int stateNumber;
    private double[] image;

    public State(int stateNumber, double[] image, int lastState){
        this.stateNumber = stateNumber;
        this.image = image;
        this.lastState = lastState;
    }

    public int getLastState() {
        return lastState;
    }

    public int getStateNumber() {
        return stateNumber;
    }

    public double[] getImage() {
        return image;
    }

    public boolean screenshotIdentical (double[] screenshot){

        if (screenshot.length != image.length){
            return false;
        }

        for (int i = 0; i < screenshot.length; i++){
            double x = screenshot[i];
            double y = image[i];

            if (x != y){
                return false;
            }
        }
        return true;
    }
}

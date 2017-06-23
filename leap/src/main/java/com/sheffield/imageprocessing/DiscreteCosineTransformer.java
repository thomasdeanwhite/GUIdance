package com.sheffield.imageprocessing;

import java.awt.*;
import java.util.ArrayList;

/**
 * Created by thomas on 02/03/2016.
 */
public class   DiscreteCosineTransformer {

    public static final int BLOCKS = 8;

    private double[] originalImage;
    private int imageWidth = 0;
    private int imageHeight = 0;

    private int newWidth = 0;
    private double[] dctImage;
    private double[] coefficients;

    public static final int THRESHOLD = 100;

    private static final double[][] COSINES = new double[BLOCKS][BLOCKS];
    private static final double[][] COEFFICIENTS = new double[BLOCKS][BLOCKS];

    static {
        for (int i = 0; i < BLOCKS; i++){
            for (int j = 0; j < BLOCKS; j++){
                COSINES[i][j] = Math.cos((i*Math.PI*(2*j+1))/(double)(2*BLOCKS));
                COEFFICIENTS[i][j] = c(i) * c(j);
            }
        }
    }

    public DiscreteCosineTransformer(double[] image, int width, int height){
        originalImage = image;

        if (originalImage.length != width * height){
            throw new IllegalArgumentException("Size of image array must be equal to width * height");
        }

        imageWidth = width;
        imageHeight = height;
        newWidth = imageWidth;// - BLOCKS;
    }

    public void updateImage(double[] image){
        originalImage = image;
    }

    public void calculateDct(){

        coefficients = new double[(imageWidth+BLOCKS) * (imageHeight + BLOCKS)];

        for (int i = 0; i < 1 + imageWidth/BLOCKS; i++) {
            for (int j = 0; j < 1 + imageHeight/BLOCKS; j++) {
                dct(i * BLOCKS, j * BLOCKS);
            }
        }


    }

    public void calculateDctFromChanges(ArrayList<Point> changes){

        for (Point p : changes){
            int x = (int)p.getX();
            int y = (int)p.getY();

            dct(x * BLOCKS, y * BLOCKS);
        }


    }

    public double[] getCoefficients (){
        return coefficients;
    }

    private int i(int x, int y){
        return (y * imageWidth) + x;
    }

    private int i2(int x, int y) { return (y * (imageWidth+BLOCKS)) + x;}

    private static double c(int u){
        return u == 0 ? 1/Math.sqrt(BLOCKS) : Math.sqrt(2 / (double) BLOCKS);
    }

    public double[] inverse(int coeffQuantity){
        if (coefficients == null){
            calculateDct();
        }

        double[] newImage = new double[coefficients.length];

        for (int i = 0; i < imageWidth/BLOCKS; i++){
            for (int j = 0; j < imageHeight/BLOCKS; j++){
                    idct(i * BLOCKS, j * BLOCKS, newImage, coeffQuantity);
            }
        }



        return newImage;

    }

    private void idct(int xpos, int ypos, double[] newImage, int coeffQuantity){
        //double divider = 1 / Math.sqrt(2 * BLOCKS);
        for (int x = 0; x < BLOCKS; x++){
            if (x + xpos >= imageWidth){
                continue;
            }
            for (int y = 0; y < BLOCKS; y++){
                if (y + ypos >= imageHeight)
                    continue;
                double value = 0;
                for (int i = 0; i < coeffQuantity; i++){
                    for (int j = 0; j < coeffQuantity; j++){
                        value += COEFFICIENTS[i][j] * coefficients[i2(i+xpos, j+ypos)] *
                                COSINES[i][x] * COSINES[j][y];
                    }
                }
                //value = (value*divider);
                    newImage[i(xpos + x, ypos + y)] = Math.abs(value);
            }
        }
    }

    private void dct(int xpos, int ypos){
        //double divider = 1 / Math.sqrt(2 * BLOCKS);

        for (int i = 0; i < BLOCKS; i++){
            for (int j = 0; j < BLOCKS; j++){
                double value = 0;
                for (int x = 0; x < BLOCKS; x++){
                    for (int y = 0; y < BLOCKS; y++){
                        int xv = x + xpos;
                        if (xv >= imageWidth){
                            xv = imageWidth - (xv - imageWidth) - 1;
                        }

                        int yv = y + ypos;
                        if (yv >= imageHeight){
                            yv = imageHeight - (yv - imageHeight) - 1;
                        }
                        double multi =  COSINES[i][x] * COSINES[j][y];
                        value += originalImage[i(xv, yv)] * multi;
                    }
                }
                value *= COEFFICIENTS[i][j];

                if (Math.abs(value) > THRESHOLD) {
                    coefficients[i2(xpos + i, ypos + j)] = value;
                }
            }
        }

    }

    public double[] getInterleavedData(){
        double[] highFrequencies = new double[imageWidth/BLOCKS * imageHeight/BLOCKS];

        int width = imageWidth/BLOCKS;
        int height = imageHeight/BLOCKS;

        for (int i = 0; i < width; i++){
            for (int j = 0; j < height; j++){
                highFrequencies[(j * width) + i] = coefficients[i2(i*BLOCKS, j*BLOCKS)];
            }
        }

        return highFrequencies;
    }
}

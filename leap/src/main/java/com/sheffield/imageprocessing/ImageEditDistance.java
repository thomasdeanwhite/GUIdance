package com.sheffield.imageprocessing;

/**
 * Created by thoma on 27/04/2016.
 */
public class ImageEditDistance {


    private static int currentWidth = 0;
    public static long calculateEditDistance(int[] i1, int[] i2, int width, int blockSize){
        long distance = 0;
        int height = i1.length / width;
        currentWidth = width;

        for (int i = 0; i < width; i++){
            for (int j = 0; j < height; j++){

            }
        }

        return distance;
    }

    public static int i(int x, int y){
        return (y * currentWidth) + x;
    }

}

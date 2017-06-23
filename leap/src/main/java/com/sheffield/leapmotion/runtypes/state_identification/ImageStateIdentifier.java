package com.sheffield.leapmotion.runtypes.state_identification;

import java.awt.image.BufferedImage;
import java.util.HashMap;

/**
 * Created by thomas on 9/26/2016.
 */
public interface ImageStateIdentifier {

    int identifyImage(BufferedImage bi, HashMap<Integer, BufferedImage> seenStates);

    String getOutputFilename();
}

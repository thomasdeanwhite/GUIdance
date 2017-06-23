package com.sheffield.leapmotion.output;

import com.sheffield.leapmotion.Properties;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import static com.sheffield.leapmotion.Properties.*;

/**
 * Created by thomas on 15/03/2016.
 */
public class StateComparator {

    public static int statesFound = 0;

    private static int SCREENSHOT_PADDING = 10;

    private static ArrayList<Integer[]> states;

    private static final boolean WRITE_SCREENSHOTS_TO_FILE = false;

    private static int currentState;

    public static String SCREENSHOT_DIRECTORY;

    public static HashMap<Integer, Integer> statesVisits;

    public static final int X_SIZE = 32;

    public static final int Y_SIZE = 32;


    private static ArrayList<Integer> statesVisitedThisRun;

    static {
        cleanUp();
    }

    public static int getCurrentState() {
        return currentState;
    }

    public static Integer[] getState(int state) {
        return states.get(state);
    }

    public static ArrayList<Integer[]> getStates() {
        return states;
    }

    public static void cleanUp() {
        statesFound = 0;
        states = new ArrayList<Integer[]>();
        currentState = -1;
        SCREENSHOT_DIRECTORY = Properties.TESTING_OUTPUT + "/screenshots";
        statesVisits =
                new HashMap<Integer, Integer>();
        statesVisitedThisRun =
                new ArrayList<Integer>();
    }

    /**
     * Returns if s1 == s2. Note for efficienc, percentage difference is
     * calculated off
     * s1 only, so sum(s1) != sum(s2), isSameState(s1, s2) may not be equals to
     * isSameState(s2, s1).
     *
     * @param s1
     * @param s2
     * @return
     */
    public boolean isSameState(Integer[] s1, Integer[] s2) {
        int difference = calculateStateDifference(s1, s2);

        return isSameState(difference, sum(s1));
    }

    /**
     * returns candidates in s
     *
     * @param s
     * @return
     */
    public static int sum(Integer[] s) {
        int total = 0;
        for (int i : s) {
            total += i;
        }
        return total;
    }


    private static boolean isSameState(int difference, int size) {
        double diffPercentage = difference / (double) size;

        //App.out.println(difference);

        return diffPercentage < HISTOGRAM_THRESHOLD;
    }

    public static int calculateStateDifference(Integer[] s1, Integer[] s2) {
        int differences = 0;
        int limit = Math.min(s1.length, s2.length);
        for (int j = 0; j < limit; j++) {
            int result = s2[j];
            int s = s1[j];
            differences += Math.abs(result - s);
        }
        return differences;
    }

    private static Integer[] shrink(Integer[] state) {
        Integer[] newState = new Integer[HISTOGRAM_BINS];

        for (int i = 0; i < newState.length; i++) {
            newState[i] = 0;
        }

        float mod = (float) (HISTOGRAM_BINS) / (float) state.length;

        for (int i = 0; i < state.length; i++) {
            int index = (int) (i * mod);
            newState[index] += state[i];
        }

        return newState;
    }

    public static int addState(Integer[] state) {
        if (HISTOGRAM_BINS < state.length) {
            state = shrink(state);
        }
        int closestState = -1;

        int maxDifference = Integer.MAX_VALUE;

        int totalValues = sum(state);

        for (int i = 0; i < states.size(); i++) {
            Integer[] ss = states.get(i);
            int differences = 0;
            int limit = Math.min(state.length, ss.length);
            for (int j = 0; j < limit; j++) {
                int result = ss[j];
                int s = state[j];
                differences += Math.abs(result - s);
                //totalValues += ss[j];
            }

            ///App.out.println(i + ":" + ((float)differences/(float)
            // resultData.length));

            if (differences < maxDifference) {
                maxDifference = differences;
                closestState = i;
            }

        }

//

        int stateNumber = states.size();

        if (!isSameState(maxDifference, totalValues) || states.size() == 0) {
            statesVisits.put(stateNumber, 0);
            states.add(state);
        } else {
            stateNumber = closestState;
        }
        return stateNumber;
    }

    public static int changeContrast(int blackAndWhite, int iterations) {
        for (int s = 0; s < iterations; s++) {
            blackAndWhite = (int) (255 * (1 + Math.sin(
                    (((blackAndWhite) * Math.PI) / 255d) - Math.PI / 2d)));
        }
        return blackAndWhite;
    }


    private static int SCREENSHOTS_WROTE = 0;


    public static BufferedImage screenshot() {
        Window activeWindow = javax.swing.FocusManager.getCurrentManager().getActiveWindow();


        Robot robot = null;
        try {
            robot = new Robot();
        } catch (AWTException e) {
            e.printStackTrace();
        }

        Rectangle bounds = new Rectangle(Toolkit.getDefaultToolkit()
                .getScreenSize());

        if (activeWindow != null) {
            bounds = new Rectangle(
                    (int) activeWindow.getBounds().getX(),
                    (int) activeWindow.getBounds().getY(),
                    (int) activeWindow.getBounds().getWidth(),
                    (int) activeWindow.getBounds().getHeight());
        }

        return robot.createScreenCapture(bounds);
    }

    /**
     * Captures the current screen and returns it as a JSON Integer Array
     *
     * @return
     */
    public static String peekState() {
        BufferedImage screenShot = screenshot();
        String state = peekState(screenShot);


        return state;
    }

    public static String peekState(BufferedImage newState) {

        int[] data = ((DataBufferInt) newState.getRaster().getDataBuffer()).getData();

        int width = newState.getWidth();
        int height = newState.getHeight();

        newState.flush();

        newState = null;

        final int X_LIM = X_SIZE;
        final int Y_LIM = Y_SIZE;

        double[] dImage = new double[X_LIM * Y_LIM];

        int xCompress = width / X_LIM;
        int yCompress = height / Y_LIM;

        for (int i = 0; i < X_LIM; i++) {
            for (int j = 0; j < Y_LIM; j++) {
                int blackAndWhite = data[((j * xCompress) * width) + (i * yCompress)];
                blackAndWhite = (int) ((0.333 * ((blackAndWhite >> 16) &
                        0x0FF) +
                        0.333 * ((blackAndWhite >> 8) & 0x0FF) +
                        0.333 * (blackAndWhite & 0x0FF)));

                dImage[(j * X_LIM) + i] = blackAndWhite;
            }

        }


        Integer[] bins = new Integer[HISTOGRAM_BINS];
        for (int i = 0; i < bins.length; i++) {
            bins[i] = 0;
        }

        float mod = ((float) (HISTOGRAM_BINS - 1) / 255f);
        for (int i = 0; i < dImage.length; i++) {
            bins[(int) (dImage[i] * mod)]++;
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bins.length; i++) {
            sb.append(bins[i] + ",");
        }
        String output = sb.toString();
        return output.substring(0, output.length() - 1);
    }

    /**
     * Captures the current screen and returns it as a JSON Integer Array
     *
     * @return
     */
    public static String captureState() {
        BufferedImage screenShot = screenshot();
        String state = captureState(screenShot);

        if (WRITE_SCREENSHOTS_TO_FILE) {
            try {
                File f = new File(
                        SCREENSHOT_DIRECTORY + "/" + Properties.FRAME_SELECTION_STRATEGY + "/" + CURRENT_RUN + "/" +
                                "raw/SCREEN" + (SCREENSHOTS_WROTE++) + ".png");
                if (f.getParentFile() != null)
                    f.getParentFile().mkdirs();
                ImageIO.write(screenShot, "png", f);

                screenShot.flush();

                screenShot = null;
            } catch (IOException e) {
                e.printStackTrace();
            }
        }


        return state;
    }


    public static void captureState(double[] screen, int state) {

        BufferedImage compressed = new BufferedImage(X_SIZE, Y_SIZE,
                BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < X_SIZE; i++) {
            for (int j = 0; j < Y_SIZE; j++) {
                int value = (int) (screen[(j * X_SIZE) + i]*255f);
                int newValue = 255;
                newValue = newValue << 8;
                newValue += value;
                newValue = newValue << 8;
                newValue += value;
                newValue = newValue << 8;
                newValue += value;
                compressed.setRGB(i, j, newValue);
            }
        }

        try {
            File f = new File(
                    SCREENSHOT_DIRECTORY + "/" + state + ".png");
            if (f.getParentFile() != null)
                f.getParentFile().mkdirs();
            ImageIO.write(compressed, "png", f);

            compressed.flush();

            compressed = null;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static double[] screenshotState() {

        BufferedImage newState = screenshot();

        int[] data = ((DataBufferInt) newState.getRaster().getDataBuffer()).getData();

        int width = newState.getWidth();
        int height = newState.getHeight();

        newState.flush();

        newState = null;

        //System.gc();

        final int X_LIM = X_SIZE;
        final int Y_LIM = Y_SIZE;

        double[] dImage = new double[X_LIM * Y_LIM];

        int xCompress = (width / X_LIM);
        int yCompress = (height / Y_LIM);


        for (int i = 0; i < X_LIM; i++) {
            for (int j = 0; j < Y_LIM; j++) {
                int index = ((j * yCompress) * width) + (i * xCompress);

                assert index < width * height;

                int blackAndWhite = data[index];
                blackAndWhite = (int) ((0.333 * ((blackAndWhite >> 16) &
                        0x0FF) +
                        0.333 * ((blackAndWhite >> 8) & 0x0FF) +
                        0.333 * (blackAndWhite & 0x0FF)));

                dImage[(j * X_LIM) + i] = blackAndWhite / (float) 255;
            }

        }

        return dImage;
    }

    public static String captureState(BufferedImage newState) {

        int[] data = ((DataBufferInt) newState.getRaster().getDataBuffer()).getData();

        int width = newState.getWidth();
        int height = newState.getHeight();

        newState.flush();

        newState = null;

        //System.gc();

        final int X_LIM = width / SCREENSHOT_COMPRESSION;
        final int Y_LIM = height / SCREENSHOT_COMPRESSION;

        double[] dImage = new double[X_LIM * Y_LIM];


        for (int i = 0; i < X_LIM; i++) {
            for (int j = 0; j < Y_LIM; j++) {
                int blackAndWhite = data[((j * SCREENSHOT_COMPRESSION) * width) + (i * SCREENSHOT_COMPRESSION)];
                blackAndWhite = (int) ((0.333 * ((blackAndWhite >> 16) &
                        0x0FF) +
                        0.333 * ((blackAndWhite >> 8) & 0x0FF) +
                        0.333 * (blackAndWhite & 0x0FF)));

                dImage[(j * X_LIM) + i] = blackAndWhite;
            }

        }

        Integer[] bins = new Integer[255];
        for (int i = 0; i < bins.length; i++) {
            bins[i] = 0;
        }

        for (int i = 0; i < dImage.length; i++) {
            bins[(int) (dImage[i])]++;
        }

        TestingStateComparator.captureState(bins);

        bins = shrink(bins);

        int closestState = -1;

        int maxDifference = Integer.MAX_VALUE;

        int totalValues = 0;


        for (int i = 0; i < bins.length; i++) {
            totalValues += bins[i];
        }

        for (int i = 0; i < states.size(); i++) {
            Integer[] ss = states.get(i);
            int differences = calculateStateDifference(bins, ss);

            ///App.out.println(i + ":" + ((float)differences/(float)
            // resultData.length));

            if (differences < maxDifference) {
                maxDifference = differences;
                closestState = i;
            }

        }

        int stateNumber = states.size();

        int totalStates = states.size();

        currentState = addState(bins);

        if (WRITE_SCREENSHOTS_TO_FILE) {
            BufferedImage compressed = new BufferedImage(X_LIM, Y_LIM,
                    BufferedImage.TYPE_INT_RGB);
            for (int i = 0; i < X_LIM; i++) {
                for (int j = 0; j < Y_LIM; j++) {
                    int value = (int) (dImage[(j * X_LIM) + i]);
                    int newValue = 255;
                    newValue = newValue << 8;
                    newValue += value;
                    newValue = newValue << 8;
                    newValue += value;
                    newValue = newValue << 8;
                    newValue += value;
                    compressed.setRGB(i, j, newValue);
                }
            }
            try {
                int currentTestingState = TestingStateComparator
                        .getCurrentState();
                File f = new File(
                        SCREENSHOT_DIRECTORY + "/" + Properties.FRAME_SELECTION_STRATEGY + "/" + CURRENT_RUN + "/" +
                                "STATE" + stateNumber + "-" + statesVisits
                                .get(currentState) + "-" +
                                +currentTestingState + "-" +
                                TestingStateComparator.getStatesVisited().get
                                        (currentTestingState) + ".png");
                if (f.getParentFile() != null)
                    f.getParentFile().mkdirs();
                ImageIO.write(compressed, "png", f);
            } catch (IOException e) {
                e.printStackTrace();
            }

            compressed.flush();

            compressed = null;
        }

        //new state found?
        if (currentState == totalStates) {

            statesVisits.put(currentState, 1);
            statesVisitedThisRun.add(currentState);
            statesFound++;

        } else {
            currentState = closestState;

            if (!statesVisits.containsKey(currentState)) {
                statesVisits.put(currentState, 0);
            }

            statesVisits
                    .put(currentState, statesVisits.get(currentState) + 1);
            if (!statesVisitedThisRun.contains(currentState)) {
                statesVisitedThisRun.add(currentState);
            }
        }

        bins = states.get(currentState);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bins.length; i++) {
            sb.append(bins[i] + ",");
        }
        String output = sb.toString();
        return output.substring(0, output.length() - 1);
    }

    public static ArrayList<Integer> getStatesVisits() {
        return statesVisitedThisRun;
    }

}

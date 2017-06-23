package com.sheffield.leapmotion.output;

import com.sheffield.leapmotion.Properties;

import java.util.ArrayList;
import java.util.HashMap;

import static com.sheffield.leapmotion.Properties.TESTING_HISTOGRAM_BINS;
import static com.sheffield.leapmotion.Properties.TESTING_HISTOGRAM_THRESHOLD;

/**
 * Created by thomas on 15/03/2016.
 */
public class TestingStateComparator {

    public static int statesFound = 0;

    private static ArrayList<Integer[]> states;

    private static int currentState;

    public static String SCREENSHOT_DIRECTORY;

    public static HashMap<Integer, Integer> statesVisited;


    private static ArrayList<Integer> statesActuallyVisited;

    static {
        cleanUp();
    }

    public static int getCurrentState() {
        return currentState;
    }

    public static Integer[] getState(int state) {
        return states.get(state);
    }

    public static ArrayList<Integer[]> getStates(){
        return states;
    }

    public static void cleanUp(){
        statesFound = 0;
        states = new ArrayList<Integer[]>();
        currentState = -1;
        SCREENSHOT_DIRECTORY = Properties.TESTING_OUTPUT + "/screenshots";
        statesVisited =
                new HashMap<Integer, Integer>();
        statesActuallyVisited =
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


    public static boolean isSameState(int difference, int size) {
        double diffPercentage = difference / (double) size;

        //App.out.println(difference);

        return diffPercentage < TESTING_HISTOGRAM_THRESHOLD;
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

    private static Integer[] shrink(Integer[] state){
        Integer[] newState = new Integer[TESTING_HISTOGRAM_BINS];

        for (int i = 0; i < newState.length; i++) {
            newState[i] = 0;
        }

        float mod = (float) (TESTING_HISTOGRAM_BINS) / (float) state.length;

        for (int i = 0; i < state.length; i++) {
            int index = (int) (i * mod);
            newState[index] += state[i];
        }

        return newState;
    }

    public static int addState(Integer[] state) {
        if (TESTING_HISTOGRAM_BINS < state.length) {
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
            statesVisited.put(stateNumber, 0);
            states.add(state);
        } else {
            stateNumber = closestState;
        }
        return stateNumber;
    }

    public static void captureState(Integer[] bins) {

        int totalStates = states.size();

        currentState = addState(bins);

        //new state found?
        if (currentState == totalStates) {

            statesVisited.put(currentState, 1);
            statesActuallyVisited.add(currentState);
            statesFound++;

        } else {
            statesVisited
                    .put(currentState, statesVisited.get(currentState) + 1);
            if (!statesActuallyVisited.contains(currentState)) {
                statesActuallyVisited.add(currentState);
            }
        }
    }

    public static HashMap<Integer, Integer> getStatesVisited() {
        return statesVisited;
    }

}

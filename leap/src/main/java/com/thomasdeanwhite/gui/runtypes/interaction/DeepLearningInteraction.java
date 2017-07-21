package com.thomasdeanwhite.gui.runtypes.interaction;

import com.google.gson.Gson;
import com.thomasdeanwhite.gui.App;
import com.thomasdeanwhite.gui.Properties;
import com.thomasdeanwhite.gui.output.StateComparator;
import com.thomasdeanwhite.gui.sampler.MouseEvent;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Created by thoma on 21/06/2017.
 */
public class DeepLearningInteraction extends UserInteraction {
    private long minTime = Long.MAX_VALUE;

    private float mouseSpeed = 5f;

    private static final float CLICK_THRESHOLD = 0.25f;

    private static final float RANDOM_PROBABILITY = 0.005f;

    private static final float JITTER = 0.0000001f;

    private Event lastEvent = Event.NONE;
    private Event secondLastEvent = lastEvent;

    private int iteration = 0;

    private Event nextEvent;

    private Process pythonProcess;

    Gson gson;

    @Override
    public void load() throws IOException {
        super.load();

        lastState = State.ORIGIN;
        states = new HashMap<>();

        gson = new Gson();

        lastState = captureState(Event.NONE);

        //TODO: Load any database info from previous runs
    }

    private boolean debugHeader = true;

    @Override
    public Event interact(long timePassed) {

        Event e;

        if (rawEvents.size() == 0) {
            return Event.NONE;
        }

        if (Math.random() <= RANDOM_PROBABILITY || lastEvent.equals(Event.NONE)) {
            int eventIndex = 1 + (int) Math.round(Math.random() * (rawEvents.size()-2));
            e = rawEvents.get(eventIndex);
            nextEvent = e;
            lastEvent = rawEvents.get(eventIndex-1);
        } else {
            double[] img = lastState.getImage();

            String input = "";

            for (double d : img)
                input += d + " ";


            String mouseInfo = lastEvent.toCsv(secondLastEvent.getMouseX() + (float)(JITTER - Math.random() * JITTER * 2f),
                    secondLastEvent.getMouseY() + (float)(JITTER - Math.random() * JITTER * 2f))
                    .replace(",", " ");

            input += mouseInfo;

            try {
                if (pythonProcess == null) {
                    String pythonCommand = "%s";// + input;
                    Process process;

                    String pathToPythonScript = DeepLearningInteraction.class.getResource("tensor_play.py").getPath();

                    try {


                        ProcessBuilder builder = new ProcessBuilder(String.format(pythonCommand, "python3"), pathToPythonScript);
                        builder.redirectErrorStream(true);
                        builder.directory(new File(System.getProperty("user.dir")));
                        App.out.println(System.getProperty("user.dir"));
                        process = builder.start();
                    } catch (IOException e2){
                        ProcessBuilder builder = new ProcessBuilder(String.format(pythonCommand, "python"), pathToPythonScript);
                        builder.redirectErrorStream(true);
                        builder.directory(new File(System.getProperty("user.dir")));
                        process = builder.start();
                    }
                    pythonProcess = process;



                    new Thread(() -> {
                        BufferedReader br = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));
                        BufferedReader be = new BufferedReader(new InputStreamReader(pythonProcess.getErrorStream()));

                        String line;

                        try {
                            while ((line = br.readLine()) != null && line.trim().length() > 0) {
                                if (!line.toLowerCase().contains("dl-result")){
                                    continue;
                                }
                                DeepLearningInteraction.this.processLine(line.substring(line.indexOf(" ")+1));
                            }

                            while ((line = be.readLine()) != null) {
                                App.out.println(line);
                            }
                        } catch (IOException e1) {
                            e1.printStackTrace();
                        }

                        try {
                            pythonProcess.waitFor();
                        } catch (InterruptedException e1) {
                            e1.printStackTrace();
                        }
                        pythonProcess.destroy();
                        pythonProcess = null;
                    }).start();
                }

                pythonProcess.getOutputStream().write((input + "\n").getBytes());
                pythonProcess.getOutputStream().flush();

            } catch (IOException e1) {
                e1.printStackTrace();
                nextEvent = Event.NONE;
            }

        }

        return nextEvent == null ? Event.NONE : nextEvent;
    }

    public void processLine (String line){
        if (Properties.SHOW_OUTPUT) {
            App.out.println("\rlem: " + lastEvent.toString() + " disp: " + line);
        }
        //output: x y lmm rmm
        //x y lmm rmm
        String[] eles = line.split(" ");
        float lmm = Float.parseFloat(eles[2]);
        float rmm = Float.parseFloat(eles[3]);

        MouseEvent me = MouseEvent.MOVE;

        if (lastEvent.getEvent().equals(MouseEvent.LEFT_DOWN) || lastEvent.getEvent().equals(MouseEvent.DRAGGED)){
            me = MouseEvent.DRAGGED;
        }


        if (lmm > rmm && lmm > CLICK_THRESHOLD) {
            me = MouseEvent.LEFT_DOWN;
        } else if (rmm > lmm && rmm > CLICK_THRESHOLD) {
            me = MouseEvent.RIGHT_CLICK;
        } else if (lmm < rmm && lmm < CLICK_THRESHOLD) {
            me = MouseEvent.LEFT_UP;
        } else if (rmm < lmm && rmm < CLICK_THRESHOLD) {
            me = MouseEvent.RIGHT_UP;
        }

        int diffx = (int) (Float.parseFloat(eles[0]) * Event.bounds.getWidth());
        int diffy = (int) (Float.parseFloat(eles[1]) * Event.bounds.getHeight());

        float mag = (float) Math.sqrt(diffx * diffx + diffy * diffy);

        diffx = (int)(mouseSpeed * diffx / mag);
        diffy = (int)(mouseSpeed * diffy / mag);

        int mx = (int)Math.max(Math.min(Event.bounds.getWidth(), lastEvent.getMouseX() + diffx), 0);
        int my = (int)Math.max(Math.min(Event.bounds.getHeight(), lastEvent.getMouseY() + diffy), 0);

        nextEvent = new Event(me,
                mx,
                my,
                System.currentTimeMillis(),
                iteration);
    }

    @Override
    public void postInteraction(Event e) {
        //super.postInteraction(e);

        State state = captureState(e);

        lastState = state;

        secondLastEvent = lastEvent;

        lastEvent = e;

        iteration++;

    }

    public State captureState(Event e) {
        double[] newImage = StateComparator.screenshotState(e.getMouseX(), e.getMouseY());

        int stateNumber = states.size();

        State state = new State(stateNumber, newImage, lastState.getStateNumber());

        states.put(stateNumber, state);
        StateComparator.captureState(state.getImage(), state.getStateNumber());

        return state;
    }
}

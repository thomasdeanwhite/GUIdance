package com.sheffield.leapmotion.runtypes.interaction;

import com.google.gson.Gson;
import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.output.StateComparator;
import com.sheffield.leapmotion.sampler.MouseEvent;
import com.sheffield.leapmotion.util.FileHandler;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;

/**
 * Created by thoma on 21/06/2017.
 */
public class UserInteraction implements Interaction {

    protected ArrayList<Event> rawEvents;
    protected Event lastEvent = Event.NONE;
    protected long minTime = Long.MAX_VALUE;
    protected State lastState = State.ORIGIN;
    protected HashMap<Integer, State> states;
    protected File trainingDataInputFile;
    protected File trainingDataOutputFile;

    @Override
    public void load() throws IOException {
        File userData = new File(Properties.TESTING_OUTPUT + "/" + Properties.INPUT[0] + "/user_interactions.csv");

        trainingDataInputFile = new File(Properties.TESTING_OUTPUT + "/" + Properties.INPUT[0] + "/training_inputs.csv");

        trainingDataOutputFile = new File(Properties.TESTING_OUTPUT + "/" + Properties.INPUT[0] + "/training_outputs.csv");

        FileHandler.createFile(trainingDataOutputFile);
        FileHandler.createFile(trainingDataInputFile);

        states = new HashMap<>();

        String[] contents = FileHandler.readFile(userData).split("\n");

        Gson gson = new Gson();

        rawEvents = new ArrayList<>();

        for (String line : contents){
            Event e = gson.fromJson(line, Event.class);

            e.moveMouse((int)e.bounds.getX() + 1280, (int)e.bounds.getY());

            if (e.getTimestamp() < minTime){
                minTime = e.getTimestamp();
            }

            rawEvents.add(e);
    }

        rawEvents.parallelStream().forEach(e ->
            e.reduceTimestamp(UserInteraction.this.minTime)
        );


        rawEvents.sort(new Comparator<Event>() {
            @Override
            public int compare(Event o1, Event o2) {
                return (int)(o1.getTimestamp() - o2.getTimestamp());
            }
        });

        for (int i = 0; i < rawEvents.size(); i++){
            rawEvents.get(i).setEventIndex(i);
        }

    }

    @Override
    public Event interact(long timePassed) {

        Event e = null;

        if (rawEvents.size() == 0){
            return Event.NONE;
        }

        if (rawEvents.size() > 0 && rawEvents.get(0).getTimestamp() <
                timePassed){
            e = rawEvents.remove(0);
        }

        if (e == null){
            try {
                Thread.sleep(rawEvents.get(0).getTimestamp() - timePassed);
            } catch (InterruptedException e1) {
                e1.printStackTrace();
            }
            return Event.NONE;
        }

        lastState = captureState(e);

        return e;
    }

    @Override
    public void postInteraction(Event e) {

        if (e.equals(Event.NONE)){
            return;
        }

        String trainingInputRow = "";
        //String

        for (double d : lastState.getImage()){
            trainingInputRow += d + ",";
        }

        trainingInputRow = trainingInputRow + e.toCsv(lastEvent.getMouseX(), lastEvent.getMouseY());

        try {
            FileHandler.appendToFile(trainingDataInputFile, trainingInputRow + "\n");
        } catch (IOException e1) {
            e1.printStackTrace();
        }



        try {
            FileHandler.appendToFile(trainingDataOutputFile, rawEvents.get
                    (0).toCsv(e.getMouseX(), e.getMouseY()) + "\n");
        } catch (IOException e1) {
            e1.printStackTrace();
        }

        lastEvent = e;

    }


    public State captureState(Event e) {
        double[] newImage = StateComparator.screenshotState(e.getMouseX(), e.getMouseY());

        int stateNumber = states.size();

        boolean found = false;

        State state = State.ORIGIN;

//        for (State s : states.values()) {
//            if (s.screenshotIdentical(newImage)) {
//                state = s;
//                found = true;
//                break;
//            }
//        }
//
//        if (!found) {
            state = new State(stateNumber, newImage, lastState.getStateNumber());
            states.put(states.size(), state);
            //StateComparator.captureState(state.getImage(), state.getStateNumber());
//        }

        return state;
    }


}

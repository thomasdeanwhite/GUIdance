package com.thomasdeanwhite.gui.runtypes.interaction;

import com.google.gson.Gson;
import com.thomasdeanwhite.gui.App;
import com.thomasdeanwhite.gui.Properties;
import com.thomasdeanwhite.gui.output.StateComparator;
import com.thomasdeanwhite.gui.sampler.MouseEvent;
import com.thomasdeanwhite.gui.util.FileHandler;

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

    public Event getLastEvent() {
        return lastEvent;
    }

    protected File trainingDataInputFile;
    protected File trainingDataOutputFile;

    protected int lastMouseX = 0;
    protected int lastMouseY = 0;

    @Override
    public void load() throws IOException {

        App.out.print("Loaded data for user: " + Properties.INPUT[0]);

        File userData = new File(Properties.TESTING_OUTPUT + "/data/" + Properties.INPUT[0] + "/user_interactions.csv");

        trainingDataInputFile = new File(Properties.TESTING_OUTPUT + "/data/" + Properties.INPUT[0] + "/training_inputs.csv");

        trainingDataOutputFile = new File(Properties.TESTING_OUTPUT + "/data/" + Properties.INPUT[0] + "/training_outputs.csv");

        FileHandler.createFile(trainingDataOutputFile);
        FileHandler.createFile(trainingDataInputFile);

        states = new HashMap<>();

        String[] contents = FileHandler.readFile(userData).split("\n");

        Gson gson = new Gson();

        rawEvents = new ArrayList<>();

        for (String line : contents){
            Event e = gson.fromJson(line, Event.class);

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

        if (rawEvents.size() > 0 && rawEvents.get(0).getTimestamp() <=
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

        if (!e.getEvent().equals(MouseEvent.KEYBOARD_INPUT) && !e.getEvent().equals(MouseEvent.SHORTCUT_INPUT)){
            lastMouseX = e.getMouseX();
            lastMouseY = e.getMouseY();
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

        trainingInputRow = trainingInputRow.substring(0, trainingInputRow.length()-1);

        try {
            FileHandler.appendToFile(trainingDataInputFile, trainingInputRow + "\n");
        } catch (IOException e1) {
            e1.printStackTrace();
        }



        try {
            FileHandler.appendToFile(trainingDataOutputFile, e.toCsv() + "\n");
        } catch (IOException e1) {
            e1.printStackTrace();
        }

        lastEvent = e;

    }


    public State captureState(Event e) {
        double[] newImage = StateComparator.screenshotState(lastMouseX, lastMouseY);

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

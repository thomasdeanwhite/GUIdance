package com.sheffield.leapmotion.runtypes.interaction;

import com.google.gson.Gson;
import com.sheffield.leapmotion.Properties;
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
    protected long minTime = Long.MAX_VALUE;

    @Override
    public void load() throws IOException {
        File userData = new File(Properties.TESTING_OUTPUT + "/" + Properties.INPUT[0] + "/user_interactions.csv");

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

        while(rawEvents.size() > 0 && rawEvents.get(0).getTimestamp() < timePassed){
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

        return e;
    }

    @Override
    public void postInteraction(Event e) {

    }
}

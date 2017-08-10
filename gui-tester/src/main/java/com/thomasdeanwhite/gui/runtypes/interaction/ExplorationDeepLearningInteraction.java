package com.thomasdeanwhite.gui.runtypes.interaction;

import com.google.gson.Gson;
import com.thomasdeanwhite.gui.App;
import com.thomasdeanwhite.gui.Properties;
import com.thomasdeanwhite.gui.output.StateComparator;
import com.thomasdeanwhite.gui.sampler.MouseEvent;
import com.thomasdeanwhite.gui.util.FileHandler;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.*;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by thoma on 21/06/2017.
 */
public class ExplorationDeepLearningInteraction extends DeepLearningInteraction {

    private File heatmapFile = new File(Properties.TESTING_OUTPUT + "/heatmap.csv");

    private Random random = new Random();

    @Override
    public void load() throws IOException {

        //eliminate randomness from parent class
        RANDOM_PROBABILITY = 0f;

        //setup as normal
        super.load();

        //seed random first event
        nextEvent = new Event(MouseEvent.MOVE,
                Event.bounds.x + random.nextInt((int) Event.bounds.getWidth()),
                Event.bounds.y + random.nextInt((int) Event.bounds.getHeight()),
                0,
                0);
        lastEvent = nextEvent;
        secondLastEvent = lastEvent;
    }

    @Override
    public Event interact(long timePassed) {

        if (!heatmapFile.exists()) {
            if (heatmapFile.getParentFile().exists()) {
                heatmapFile.getParentFile().mkdirs();
            }
            try {



                heatmapFile.createNewFile();
                FileHandler.writeToFile(heatmapFile, "x,y,leftClick,rightClick\n");

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        Event newEvent = super.interact(timePassed);

        try {
            FileHandler.appendToFile(heatmapFile, newEvent.getMouseX() + ","
                    + newEvent.getMouseY() + ","
                    + newEvent.leftClickToFloat() + ","
                    + newEvent.rightClickToFloat() + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }

        return new Event(MouseEvent.MOVE,
                newEvent.getMouseX(),
                newEvent.getMouseY(),
                System.currentTimeMillis(),
                newEvent.getEventIndex());


    }
}

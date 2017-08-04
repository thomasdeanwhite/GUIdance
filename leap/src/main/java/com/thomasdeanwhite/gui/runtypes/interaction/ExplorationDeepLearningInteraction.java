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

/**
 * Created by thoma on 21/06/2017.
 */
public class ExplorationDeepLearningInteraction extends DeepLearningInteraction {

    private File heatmapFile = new File(Properties.TESTING_OUTPUT + "/heatmap.csv");

    @Override
    public Event interact(long timePassed) {

        if (!heatmapFile.exists()){
            if (heatmapFile.getParentFile().exists()){
                heatmapFile.getParentFile().mkdirs();
            }
            try {
                File screenshot = new File(Properties.TESTING_OUTPUT + "/screenshot.csv");

                if (!screenshot.exists()){
                    screenshot.createNewFile();
                }

                BufferedImage screen = StateComparator.screenshot();
                int[] data = ((DataBufferInt) screen.getRaster().getDataBuffer()).getData();

                StringBuilder sb = new StringBuilder();
                int width = screen.getWidth();

                for (int i = 0; i < screen.getWidth(); i++){
                    for(int j = 0; j < screen.getHeight(); j++){
                        sb.append(data[(j * width) + i] + ",");
                    }
                    sb.append("\n");
                }

                FileHandler.writeToFile(screenshot, sb.toString());


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

package com.sheffield.leapmotion.runtypes.interaction;

import com.sheffield.leapmotion.App;
import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.sampler.MouseEvent;
import com.sheffield.leapmotion.util.FileHandler;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Created by thoma on 17/07/2017.
 */
public class TestUserInteraction {

    private static File file;
    private static File tmpFile;

    @BeforeAll
    public static void setup(){
        try {

            String tmpDir = System.getProperty("java.io.tmpdir") +  Long.toString(System.nanoTime());

            Properties.INPUT = new String[]{"test"};

            String fullFileName = tmpDir + "/data/" + Properties.INPUT[0] + "/";

            file = new File(fullFileName, "user_interactions.csv");

            file.getParentFile().mkdirs();

            file.createNewFile();

            Properties.TESTING_OUTPUT = tmpDir;

            tmpFile = new File(Properties.TESTING_OUTPUT);
            tmpFile.deleteOnExit();


            String data = "";

            for (int i = 0; i < 10; i++){
                data += "{\"timestamp\":" + i + ",\"event\":\"MOVE\",\"mouseX\":" + (10+i) + ",\"mouseY\":" + (20 - i) + ",\"eventIndex\":" + (20 + i) + "}\n";
            }

            FileHandler.writeToFile(file, data);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testLoad(){
        UserInteraction ui = new UserInteraction();

        try {
            ui.load();
        } catch (IOException e) {
            e.printStackTrace();
            fail("File should exist");
        }

        Event e = ui.interact(0);

        assertEquals(10, e.getMouseX());
        assertEquals(20, e.getMouseY());
        assertEquals(MouseEvent.MOVE, e.getEvent());
    }


    @AfterAll
    public static void tearDown(){
        tmpFile.delete();
    }

}

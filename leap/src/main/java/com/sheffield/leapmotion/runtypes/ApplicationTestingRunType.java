package com.sheffield.leapmotion.runtypes;

import com.sheffield.leapmotion.App;
import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.output.StateComparator;
import com.sheffield.leapmotion.runtypes.interaction.DeepQNetworkInteraction;
import com.sheffield.leapmotion.runtypes.interaction.Event;
import com.sheffield.leapmotion.runtypes.interaction.Interaction;
import com.sheffield.leapmotion.runtypes.interaction.UserInteraction;
import com.sheffield.leapmotion.sampler.MouseEvent;

import java.awt.*;

import java.awt.event.InputEvent;
import java.io.IOException;


/**
 * Created by thoma on 20/06/2017.
 */
public class ApplicationTestingRunType implements RunType {
    @Override
    public int run() {

        Interaction interaction = new DeepQNetworkInteraction();

        ApplicationThread appThread = new ApplicationThread();

        App.out.println("- Using exec: " + Properties.EXEC);

        if (!appThread.isAppRunning()) {
            appThread.run();
        }

        try {
            interaction.load();
        } catch (IOException e) {
            e.printStackTrace(App.out);
            return -1;
        }

        long startTime = System.currentTimeMillis();

        long finishTime = startTime + Properties.RUNTIME;

        long currentTime;


        Robot r;
        try {
            r = new Robot();
        } catch (AWTException e) {
            e.printStackTrace(App.out);
            return -2;
        }

        do {

            if (!appThread.isAppRunning()) {
                appThread.run();
            }
            currentTime = System.currentTimeMillis();
            long timePassed = currentTime - startTime;

            Event e = interaction.interact(timePassed);


            r.mouseMove(e.getMouseX(), e.getMouseY());

            if (e.getEvent().equals(MouseEvent.LEFT_CLICK)) {
                click(r, InputEvent.BUTTON1_MASK);
            } else if (e.getEvent().equals(MouseEvent.RIGHT_CLICK)) {
                click(r, InputEvent.BUTTON2_MASK);
            } else if (e.getEvent().equals(MouseEvent.LEFT_DOWN)) {
                //mouseDown(r, InputEvent.BUTTON1_MASK);
                click(r, InputEvent.BUTTON1_MASK);
            } else if (e.getEvent().equals(MouseEvent.RIGHT_DOWN)) {
                mouseDown(r, InputEvent.BUTTON2_MASK);
                click(r, InputEvent.BUTTON2_MASK);
            } else if (e.getEvent().equals(MouseEvent.LEFT_UP)) {
                mouseUp(r, InputEvent.BUTTON1_MASK);
            } else if (e.getEvent().equals(MouseEvent.RIGHT_UP)) {
                mouseUp(r, InputEvent.BUTTON2_MASK);
            }

            interaction.postInteraction(e);

        } while (currentTime < finishTime);

        appThread.kill();

        return 0;
    }


    public void click(Robot r, int button) {
        r.mousePress(button);

        try {
            Thread.sleep(10);
        } catch (InterruptedException exc) {
        }
        ;

        r.mouseRelease(button);
    }

    public void mouseDown(Robot r, int button) {
        r.mousePress(button);
    }

    public void mouseUp(Robot r, int button) {
        r.mouseRelease(button);
    }

}

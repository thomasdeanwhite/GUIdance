package com.sheffield.leapmotion.runtypes;

import com.sheffield.leapmotion.App;
import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.runtypes.interaction.*;
import com.sheffield.leapmotion.runtypes.interaction.Event;
import com.sheffield.leapmotion.sampler.MouseEvent;
import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;
import org.jnativehook.keyboard.NativeKeyEvent;
import org.jnativehook.keyboard.NativeKeyListener;

import java.awt.*;

import java.awt.event.InputEvent;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;


/**
 * Created by thoma on 20/06/2017.
 */
public class ApplicationTestingRunType implements RunType, NativeKeyListener {

    private Rectangle bounds;
    private boolean running = true;
    private ApplicationThread appThread;

    public boolean isRunning() {
        return running;
    }

    @Override
    public int run() {

        try {
            GlobalScreen.registerNativeHook();
        } catch (NativeHookException e) {
            e.printStackTrace();
        }

        // Get the logger for "org.jnativehook" and set the level to warning.
        Logger logger = Logger.getLogger(GlobalScreen.class.getPackage().getName());
        logger.setLevel(Level.WARNING);
        logger.setUseParentHandlers(false);

        GlobalScreen.addNativeKeyListener(this);



        Interaction interaction;

        switch (Properties.INTERACTION){
            case DEEP_LEARNING:
                interaction = new DeepLearningInteraction();
                break;
            case MONKEY:
                interaction = new MonkeyInteraction();
                break;
            case USER:
            default:
                interaction = new UserInteraction();
        }



        appThread = new ApplicationThread();

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

            if (bounds == null){
                Window activeWindow = javax.swing.FocusManager.getCurrentManager().getActiveWindow();

                bounds = new Rectangle(Toolkit.getDefaultToolkit()
                        .getScreenSize());

                if (activeWindow != null) {
                    bounds = new Rectangle(
                            (int) activeWindow.getBounds().getX(),
                            (int) activeWindow.getBounds().getY(),
                            (int) activeWindow.getBounds().getWidth(),
                            (int) activeWindow.getBounds().getHeight());
                }
            }

            r.mouseMove((int)(bounds.getX() + e.getMouseX()), (int)(bounds.getY() + e.getMouseY()));

            if (e.getEvent().equals(MouseEvent.LEFT_CLICK)) {
                click(r, InputEvent.BUTTON1_MASK);
            } else if (e.getEvent().equals(MouseEvent.RIGHT_CLICK)) {
                click(r, InputEvent.BUTTON2_MASK);
            } else if (e.getEvent().equals(MouseEvent.LEFT_DOWN)) {
                mouseDown(r, InputEvent.BUTTON1_MASK);
                //click(r, InputEvent.BUTTON1_MASK);
            } else if (e.getEvent().equals(MouseEvent.RIGHT_DOWN)) {
                mouseDown(r, InputEvent.BUTTON2_MASK);
                //click(r, InputEvent.BUTTON2_MASK);
            } else if (e.getEvent().equals(MouseEvent.LEFT_UP)) {
                mouseUp(r, InputEvent.BUTTON1_MASK);
            } else if (e.getEvent().equals(MouseEvent.RIGHT_UP)) {
                mouseUp(r, InputEvent.BUTTON2_MASK);
            }

            interaction.postInteraction(e);

        } while (currentTime < finishTime && running);

        appThread.kill();

        return 0;
    }


    public void click(Robot r, int button) {
        r.mousePress(button);

        try {
            Thread.sleep(10);
        } catch (InterruptedException exc) {
        }

        r.mouseRelease(button);
    }

    public void mouseDown(Robot r, int button) {
        r.mousePress(button);
    }

    public void mouseUp(Robot r, int button) {
        r.mouseRelease(button);
    }


    private int keyPressedToExit = 0;


    @Override
    public void nativeKeyPressed(NativeKeyEvent nativeKeyEvent) {
        if (nativeKeyEvent.getKeyCode() == NativeKeyEvent.VC_ESCAPE) {
            keyPressedToExit++;
        } else {
            keyPressedToExit = 0;
        }

        if (keyPressedToExit >= 3){
            running = false;
            App.out.println("!!! User Initiated Exit !!!");
            appThread.kill();
            App.getApp().end();
            System.exit(0);
        }
    }

    @Override
    public void nativeKeyReleased(NativeKeyEvent nativeKeyEvent) {

    }

    @Override
    public void nativeKeyTyped(NativeKeyEvent nativeKeyEvent) {

    }
}

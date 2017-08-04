package com.thomasdeanwhite.gui.sampler;


import com.google.gson.Gson;
import com.thomasdeanwhite.gui.App;
import com.thomasdeanwhite.gui.Properties;
import com.thomasdeanwhite.gui.runtypes.interaction.Event;
import com.thomasdeanwhite.gui.util.FileHandler;
import org.jnativehook.GlobalScreen;
import org.jnativehook.NativeHookException;
import org.jnativehook.keyboard.NativeKeyEvent;
import org.jnativehook.keyboard.NativeKeyListener;
import org.jnativehook.mouse.NativeMouseEvent;
import org.jnativehook.mouse.NativeMouseInputListener;
import org.jnativehook.mouse.NativeMouseWheelEvent;
import org.jnativehook.mouse.NativeMouseWheelListener;

import java.awt.*;
import java.io.File;
import java.io.IOException;

public class SamplerApp implements NativeKeyListener, NativeMouseInputListener, NativeMouseWheelListener {


    private File output = null;
    private MouseEvent mouseEvent = MouseEvent.NONE;
    private Gson gson = new Gson();

    public static final int MOVEMENT_DELAY = 50;
    private int delay = 0;
    private long startMove = 0;

    private MouseEvent lastEvent = mouseEvent;
    private int lastX = 0;
    private int lastY = 0;

    private Rectangle bounds;

    public SamplerApp () {
        try {
            GlobalScreen.registerNativeHook();
        } catch (NativeHookException e) {
            e.printStackTrace();
        }

        GlobalScreen.addNativeMouseListener(this);

        GlobalScreen.addNativeMouseWheelListener(this);

        GlobalScreen.addNativeMouseMotionListener(this);
    }

    @Override
    public void nativeKeyPressed(NativeKeyEvent nativeKeyEvent) {

    }

    @Override
    public void nativeKeyReleased(NativeKeyEvent nativeKeyEvent) {

    }

    @Override
    public void nativeKeyTyped(NativeKeyEvent nativeKeyEvent) {

    }

    @Override
    public void nativeMouseClicked(NativeMouseEvent nativeMouseEvent) {
//        if (nativeMouseEvent.getButton() == 1) {//left click
//            mouseEvent = MouseEvent.LEFT_CLICK;
//        } else if (nativeMouseEvent.getButton() == 2){
//            mouseEvent = MouseEvent.RIGHT_CLICK;
//        } else {
//            mouseEvent = MouseEvent.OTHER_CLICK;
//        }
//        outputXYLocation(nativeMouseEvent);
    }

    @Override
    public void nativeMousePressed(NativeMouseEvent nativeMouseEvent) {
        if (nativeMouseEvent.getButton() == 1) {//left click
            mouseEvent = MouseEvent.LEFT_DOWN;
        } else if (nativeMouseEvent.getButton() == 2){
            mouseEvent = MouseEvent.RIGHT_DOWN;
        } else {
            mouseEvent = MouseEvent.OTHER_DOWN;
        }
        outputXYLocation(nativeMouseEvent);
    }

    @Override
    public void nativeMouseReleased(NativeMouseEvent nativeMouseEvent) {

        App.out.println(nativeMouseEvent.getButton());

        if (nativeMouseEvent.getButton() == 1) {//left click
            mouseEvent = MouseEvent.LEFT_UP;
        } else if (nativeMouseEvent.getButton() == 2){
            mouseEvent = MouseEvent.RIGHT_UP;
        } else {
            mouseEvent = MouseEvent.OTHER_UP;
        }
        outputXYLocation(nativeMouseEvent);
    }

    @Override
    public void nativeMouseMoved(NativeMouseEvent nativeMouseEvent) {
        mouseEvent = MouseEvent.MOVE;

        if (!lastEvent.equals(MouseEvent.MOVE)){
            startMove = System.currentTimeMillis() - MOVEMENT_DELAY;
        }
        delay = (int)(System.currentTimeMillis() - startMove);

        if (MOVEMENT_DELAY <= delay){
            delay = 0;
            outputXYLocation(nativeMouseEvent);
        }
    }

    @Override
    public void nativeMouseDragged(NativeMouseEvent nativeMouseEvent) {
        mouseEvent = MouseEvent.DRAGGED;
        outputXYLocation(nativeMouseEvent);
    }

    @Override
    public void nativeMouseWheelMoved(NativeMouseWheelEvent nativeMouseWheelEvent) {
        mouseEvent = MouseEvent.MOUSE_WHEEL;

        writeToOutput(nativeMouseWheelEvent.getWheelRotation(), 0);
    }

    public void outputXYLocation(NativeMouseEvent nme){
        writeToOutput(nme.getX(), nme.getY());
    }

    public synchronized void writeToOutput(int x, int y){

        if (lastEvent.equals(mouseEvent) && ((lastX == x && lastY == y))){
            return;
        }

        if (bounds == null){
            Window activeWindow = javax.swing.FocusManager.getCurrentManager().getActiveWindow();


            Robot robot = null;
            try {
                robot = new Robot();
            } catch (AWTException e) {
                e.printStackTrace();
            }

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

        com.thomasdeanwhite.gui.runtypes.interaction.Event me = new Event(mouseEvent, (int)(x - bounds.getX()), (int)(y - bounds.getY()), System.currentTimeMillis(), 0);
        //Event me = new Event(mouseEvent, x , y, System.currentTimeMillis(), 0);

        if (output == null) {
            output = new File(Properties.TESTING_OUTPUT + "/data/" + Properties.INPUT[0] + "/user_interactions.csv");

            if (!output.exists()) {
                if (!output.getParentFile().exists()) {
                    output.getParentFile().mkdirs();
                }
                try {
                    output.createNewFile();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        try {
            FileHandler.appendToFile(output, gson.toJson(me) + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }

        lastEvent = mouseEvent;
        lastX = x;
        lastY = y;
        mouseEvent = MouseEvent.NONE;
    }

    public void cleanup(){
        GlobalScreen.removeNativeMouseListener(this);
        GlobalScreen.removeNativeMouseMotionListener(this);
        GlobalScreen.removeNativeMouseWheelListener(this);
    }
}

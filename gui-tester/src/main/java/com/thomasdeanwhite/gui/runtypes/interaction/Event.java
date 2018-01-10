package com.thomasdeanwhite.gui.runtypes.interaction;

import com.thomasdeanwhite.gui.sampler.MouseEvent;

import java.awt.*;
import java.io.Serializable;

/**
 * Created by thoma on 21/06/2017.
 */
public class Event implements Serializable {

    public static Event NONE = new Event(MouseEvent.NONE, 0, 0, 0, -1);

    private long timestamp;
    private MouseEvent event = MouseEvent.NONE;
    private int mouseX;
    private int mouseY;
    private int eventIndex = 0;

    public void moveMouse(int x, int y){
        mouseX += x;
        mouseY += y;
    }

    public Event(MouseEvent me, int x, int y, long time, int index) {
        event = me;
        mouseX = x;
        mouseY = y;
        timestamp = time;
        eventIndex = index;
    }

    public MouseEvent getEvent() {
        return event;
    }

    public int getMouseX() {
        return mouseX;
    }

    public int getMouseY() {
        return mouseY;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void reduceTimestamp(long x) {
        timestamp = timestamp - x;
    }

    public int getEventIndex() {
        return eventIndex;
    }

    public void setEventIndex(int eventIndex) {
        this.eventIndex = eventIndex;
    }

    public String csvHeader(){
        return "leftDown,leftUp,rightDown,rightUp,leftClick,rightClick,keyboardInput,shortcutInput";
    }

    public String toCsv() {

        return eventToFloat(MouseEvent.LEFT_DOWN) + "," +
                eventToFloat(MouseEvent.LEFT_UP) + "," +
                eventToFloat(MouseEvent.RIGHT_DOWN) + "," +
                eventToFloat(MouseEvent.RIGHT_UP) + "," +
                eventToFloat(MouseEvent.LEFT_CLICK) + "," +
                eventToFloat(MouseEvent.RIGHT_CLICK) + "," +
                eventToFloat(MouseEvent.KEYBOARD_INPUT) + "," +
                eventToFloat(MouseEvent.SHORTCUT_INPUT);
    }

    public String toString() {
        return "(" + eventToFloat(MouseEvent.LEFT_DOWN) + "," +
                eventToFloat(MouseEvent.LEFT_UP) + "," +
                eventToFloat(MouseEvent.RIGHT_DOWN) + "," +
                eventToFloat(MouseEvent.RIGHT_UP) + "," +
                eventToFloat(MouseEvent.LEFT_CLICK) + "," +
                eventToFloat(MouseEvent.RIGHT_CLICK) + "," +
                eventToFloat(MouseEvent.KEYBOARD_INPUT) + "," +
                eventToFloat(MouseEvent.SHORTCUT_INPUT) + ")";
    }

    private float eventToFloat(MouseEvent event) {
        if (event.equals(this.event)) {
            return 1;
        }
        return 0;
    }
}

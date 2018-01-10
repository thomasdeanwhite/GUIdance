package com.sheffield.leapmotion.runtypes.interaction;

import com.thomasdeanwhite.gui.runtypes.interaction.Event;
import com.thomasdeanwhite.gui.sampler.MouseEvent;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestEvent {

    @Test
    public void testEventZerosToCsv(){
        Event e = new Event(MouseEvent.NONE, 0, 0, 0, 0);

        assertEquals("0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0", e.toCsv());
    }

    @Test
    public void testEventLeftClickToCsv(){
        Event e = new Event(MouseEvent.LEFT_DOWN, 0, 0, 0, 0);

        assertEquals("1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0", e.toCsv());
    }

    @Test
    public void testEventRightClickToCsv(){
        Event e = new Event(MouseEvent.RIGHT_DOWN, 0, 0, 0, 0);

        assertEquals("0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0", e.toCsv());
    }


    @Test
    public void testEventLeftUpToCsv(){
        Event e = new Event(MouseEvent.LEFT_UP, 0, 0, 0, 0);

        assertEquals("0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0", e.toCsv());
    }

    @Test
    public void testEventRightUpToCsv(){
        Event e = new Event(MouseEvent.RIGHT_UP, 0, 0, 0, 0);

        assertEquals("0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0", e.toCsv());
    }

}

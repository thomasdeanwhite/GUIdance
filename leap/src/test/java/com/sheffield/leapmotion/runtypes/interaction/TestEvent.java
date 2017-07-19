package com.sheffield.leapmotion.runtypes.interaction;

import com.sheffield.leapmotion.sampler.MouseEvent;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestEvent {

    @Test
    public void testEventZerosToCsv(){
        Event e = new Event(MouseEvent.MOVE, 0, 0, 0, 0);

        assertEquals("0.0,0.0,0.0,0.0", e.toCsv(0, 0));
    }

    @Test
    public void testEventLeftClickToCsv(){
        Event e = new Event(MouseEvent.LEFT_DOWN, 0, 0, 0, 0);

        assertEquals("0.0,0.0,1.0,0.0", e.toCsv(0, 0));
    }

    @Test
    public void testEventRightClickToCsv(){
        Event e = new Event(MouseEvent.RIGHT_DOWN, 0, 0, 0, 0);

        assertEquals("0.0,0.0,0.0,1.0", e.toCsv(0, 0));
    }


    @Test
    public void testEventLeftUpToCsv(){
        Event e = new Event(MouseEvent.LEFT_UP, 0, 0, 0, 0);

        assertEquals("0.0,0.0,-1.0,0.0", e.toCsv(0, 0));
    }

    @Test
    public void testEventRightUpToCsv(){
        Event e = new Event(MouseEvent.RIGHT_UP, 0, 0, 0, 0);

        assertEquals("0.0,0.0,0.0,-1.0", e.toCsv(0, 0));
    }

}

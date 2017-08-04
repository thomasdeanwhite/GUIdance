package com.thomasdeanwhite.gui.runtypes.interaction;

import com.thomasdeanwhite.gui.sampler.MouseEvent;

import java.io.IOException;
import java.util.Random;

public class MonkeyInteraction implements Interaction {

    private static final MouseEvent[] MOUSE_EVENTS = MouseEvent.values();
    private Random random;
    private int events = 0;

    @Override
    public void load() throws IOException {
        random = new Random();
    }

    @Override
    public Event interact(long timePassed) {
        return new Event(MOUSE_EVENTS[random.nextInt(MOUSE_EVENTS.length)],
                Event.bounds.x + random.nextInt(Event.bounds.width),
                Event.bounds.y + random.nextInt(Event.bounds.height),
                timePassed,
                events++);
    }

    @Override
    public void postInteraction(Event e) {

    }
}

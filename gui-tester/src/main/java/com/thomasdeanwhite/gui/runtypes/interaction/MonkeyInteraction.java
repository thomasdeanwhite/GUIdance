package com.thomasdeanwhite.gui.runtypes.interaction;

import com.thomasdeanwhite.gui.sampler.MouseEvent;

import java.awt.*;
import java.io.IOException;
import java.util.Random;

public class MonkeyInteraction implements Interaction {

    private Rectangle bounds;

    private static final MouseEvent[] MOUSE_EVENTS = MouseEvent.values();
    private Random random;
    private int events = 0;

    @Override
    public void load() throws IOException {
        random = new Random();
    }

    @Override
    public Event interact(long timePassed) {
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
        return new Event(MOUSE_EVENTS[random.nextInt(MOUSE_EVENTS.length)],
                bounds.x + random.nextInt(bounds.width),
                bounds.y + random.nextInt(bounds.height),
                timePassed,
                events++);
    }

    @Override
    public void postInteraction(Event e) {

    }
}

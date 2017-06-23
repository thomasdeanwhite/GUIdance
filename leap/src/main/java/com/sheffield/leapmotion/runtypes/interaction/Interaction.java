package com.sheffield.leapmotion.runtypes.interaction;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Created by thoma on 21/06/2017.
 */
public interface Interaction {

    void load() throws IOException;

    Event interact(long timePassed);

    void postInteraction(Event e);
}

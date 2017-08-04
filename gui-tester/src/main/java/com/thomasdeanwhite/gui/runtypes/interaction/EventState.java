package com.thomasdeanwhite.gui.runtypes.interaction;

import java.io.Serializable;

/**
 * Created by thoma on 22/06/2017.
 */
public class EventState extends State implements Serializable {

    private double[] newState;
    private float reward;
    private Event lastEvent;

    public EventState(int stateNumber, double[] image, int lastState, Event lastEvent, double[] newState, float reward) {
        super(stateNumber, image, lastState);
        this.newState = newState;
        this.reward = reward;
        this.lastEvent = lastEvent;
    }

    public EventState(State s, Event lastEvent, double[] newState, float reward) {
        this(s.getStateNumber(), s.getImage(), s.getStateNumber(), lastEvent, newState, reward);
    }

    public double[] getNewState() {
        return newState;
    }

    public float getReward() {
        return reward;
    }

    public Event getLastEvent() {
        return lastEvent;
    }
}

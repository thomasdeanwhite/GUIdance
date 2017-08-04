package com.thomasdeanwhite.gui.runtypes.interaction;

/**
 * Created by thoma on 22/06/2017.
 */
public class EventReward {

    private Event event;
    private float reward;

    public EventReward (Event event, float reward){
        this.event = event;
        this.reward = reward;
    }

    public Event getEvent() {
        return event;
    }

    public float getReward() {
        return reward;
    }

    public void setReward(float reward) {
        this.reward = reward;
    }

    public EventReward copy(){
        EventReward er = new EventReward(event, reward);

        return er;
    }
}

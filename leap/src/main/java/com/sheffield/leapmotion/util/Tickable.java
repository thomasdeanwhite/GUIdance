package com.sheffield.leapmotion.util;

/**
 * Created by thomas on 9/1/2016.
 */
public interface Tickable {
    void tick(long lastTime);

    long lastTick();
}

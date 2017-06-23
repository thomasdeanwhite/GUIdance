package com.sheffield.leapmotion.runtypes;

import com.sheffield.leapmotion.App;
import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.sampler.SamplerApp;

/**
 * Created by thomas on 18/11/2016.
 */
public class SamplingRunType implements RunType {
    @Override
    public int run() {
        App.out.println("- Sampling data");
        SamplerApp sa = new SamplerApp();

        ApplicationThread appThread = new ApplicationThread();

        App.out.println("- Using exec: " + Properties.EXEC);

        long startTime = System.currentTimeMillis();

        long finishTime = startTime + Properties.RUNTIME;

        long currentTime;


        if (!appThread.isAppRunning()) {
            appThread.run();
        }

        do {
            currentTime = System.currentTimeMillis();

            if (!appThread.isAppRunning()) {
                appThread.run();
            }

            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        } while (currentTime < finishTime);

        sa.cleanup();

        appThread.kill();

        return 0;
    }
}

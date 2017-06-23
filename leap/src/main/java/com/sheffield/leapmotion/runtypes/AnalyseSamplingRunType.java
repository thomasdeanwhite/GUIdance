package com.sheffield.leapmotion.runtypes;

import com.sheffield.leapmotion.App;
import com.sheffield.leapmotion.sampler.SamplerApp;

/**
 * Created by thomas on 18/11/2016.
 */
public class AnalyseSamplingRunType implements RunType {
    @Override
    public int run() {
        App.out.println("- Sampling data");
        //SamplerApp.main(new String[]{});
        return 0;
    }
}

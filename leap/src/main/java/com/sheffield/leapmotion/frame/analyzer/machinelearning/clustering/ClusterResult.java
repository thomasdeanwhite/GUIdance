package com.sheffield.leapmotion.frame.analyzer.machinelearning.clustering;

import java.util.HashMap;
import weka.core.Instance;

/**
 * Created by thomas on 08/02/17.
 */
public class ClusterResult {

    private HashMap<String, String> assignments;
    private HashMap<String, Instance> centroids;


    public ClusterResult(HashMap<String, String> assignments, HashMap<String, Instance> centroids){
        this.assignments = assignments;
        this.centroids = centroids;
    }

    public HashMap<String, String> getAssignments() {
        return assignments;
    }

    public HashMap<String, Instance> getCentroids() {
        return centroids;
    }
}

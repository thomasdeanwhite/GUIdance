package com.sheffield.leapmotion.frame.analyzer.machinelearning.clustering;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import com.sheffield.leapmotion.App;
import com.sheffield.leapmotion.Properties;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Cobweb;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils;

/**
 * Created by thomas on 08/02/17.
 */
public class WekaClusterer {

    private String filename;
    private List<String> files;
    private int clusters = 0;

    public WekaClusterer(String filename){
        this.filename = filename;
        setClusters(Properties.CLUSTERS);
        files = new ArrayList<>();
    }

    public WekaClusterer(String filename, List<String> files){
        this(filename);
        this.files.addAll(files);
    }


    public void setClusters(int clusters){
        this.clusters = clusters;
    }

    public ClusterResult cluster() throws Exception {

        int iterations = 50;

        File f = new File(filename);

        if (!f.exists()){
            throw new IllegalArgumentException("File " + f.getAbsolutePath() + " does not exist!");
        }

        ArffLoader loader = new ArffLoader();

        loader.setFile(f);

        Instances data = loader.getDataSet();

        ArrayList<String> ids = new ArrayList<String>(data.size());


            for (Instance s : data) {
                String id = s.stringValue(0);
                if (files.size() > 0) {
                    id += filename;
                }
                ids.add(id);
            }

//        data.deleteStringAttributes();

        for (String s : files) {
            File fi = new File(s);

            if (!fi.exists()) {
                throw new IllegalArgumentException("File " + fi.getAbsolutePath() + " does not exist!");
            }

            ArffLoader fiLoader = new ArffLoader();

            loader.setFile(fi);

            Instances fiData = loader.getDataSet();

            ConverterUtils.DataSource fiS = new ConverterUtils.DataSource(fiData);
            Instances struct = fiS.getStructure();
//            struct.deleteStringAttributes();

            ArrayList<String> fiIds = new ArrayList<String>(data.size());

            for (Instance sfi : fiData){
                ids.add(sfi.stringValue(0) + s);
            }

            while(fiS.hasMoreElements(struct)){
                Instance next = fiS.nextElement(struct);

                data.add(next);
            }
        }

        data.deleteStringAttributes();


        //data.setClassIndex(0);

        SimpleKMeans clusterer = new SimpleKMeans();

        clusterer.setSeed(0);


        clusterer.setNumClusters(clusters);
        clusterer.setPreserveInstancesOrder(true);
        clusterer.setMaxIterations(iterations);

        clusterer.buildClusterer(data);

        HashMap<String, String> assignments = new HashMap<String, String>(data.size());

        int[] assigns = clusterer.getAssignments();

        for (int i = 0; i < assigns.length; i++){
            assignments.put(ids.get(i), assigns[i] + "");
        }

        HashMap<String, Instance> centroids = new HashMap<>();

        Instances centrs = clusterer.getClusterCentroids();
        for (int i = 0; i < centrs.size(); i++){
            centroids.put("" + i, centrs.get(i));
        }

        return new ClusterResult(assignments, centroids);
    }


}

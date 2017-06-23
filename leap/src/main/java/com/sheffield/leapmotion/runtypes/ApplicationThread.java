package com.sheffield.leapmotion.runtypes;

import com.sheffield.leapmotion.App;
import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.util.FileHandler;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.File;

/**
 * Created by thoma on 20/06/2017.
 */
public class ApplicationThread {

    private File output = new File(Properties.TESTING_OUTPUT + "/app.out");
    private File error = new File(Properties.TESTING_OUTPUT + "/app.err");
    private Process process;
    private boolean appRunning = false;

    public boolean isAppRunning() {
        return appRunning;
    }

    public ApplicationThread() {

    }

    public void run() {
        if (process != null || appRunning){
            return;
        }

        //assert process == null;
        appRunning = true;
        Thread t = new Thread(new Runnable() {
            @Override
            public void run() {
                String exec = Properties.EXEC;

                try {
                    process = Runtime.getRuntime().exec(exec);

                    output();
                } catch (IOException e) {
                    e.printStackTrace(App.out);
                }
                //process = null;
                //ApplicationThread.this.appRunning = false;
            }
        });

        t.start();
    }

    public void output() {
        BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream()));
        BufferedReader be = new BufferedReader(new InputStreamReader(process.getInputStream()));

        if (!output.exists()){
            if (!output.getParentFile().exists()){
                output.getParentFile().mkdirs();
            }
            try {
                output.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        if (!error.exists()){
            if (!error.getParentFile().exists()){
                error.getParentFile().mkdirs();
            }

            try {
                error.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        String line;
        try {
            while ((line = br.readLine()) != null) {
                FileHandler.appendToFile(output, line);
            }

            while ((line = be.readLine()) != null) {
                FileHandler.appendToFile(error, line);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void kill() {
        if (process != null) {
            output();
            process.destroyForcibly();
        }
    }

}
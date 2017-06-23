package com.sheffield.leapmotion.frame.analyzer.machinelearning.ngram;

import com.sheffield.leapmotion.App;

import java.io.PrintStream;
import java.io.PrintWriter;

/**
 * Created by thoma on 20/06/2017.
 */
public class DataSparsityException extends RuntimeException {
    String error;
    public DataSparsityException(String s) {
        error = s;
    }

    @Override
    public void printStackTrace() {
        printStackTrace(System.out);
    }

    @Override
    public void printStackTrace(PrintStream s) {
        s.println(error);
        super.printStackTrace(s);
    }

    @Override
    public void printStackTrace(PrintWriter s) {
        s.println(error);
        super.printStackTrace(s);
    }
}

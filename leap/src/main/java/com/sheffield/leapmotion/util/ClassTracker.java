package com.sheffield.leapmotion.util;

/**
 * Created by thomas on 18/04/2016.
 */
public class ClassTracker {
    private String className = "";
    private int lines = 0;
    private int branches = 0;

    public String getClassName() {
        return className;
    }

    public void setClassName(String className) {
        this.className = className;
    }

    public int getLines() {
        return lines;
    }

    public void setLines(int lines) {
        this.lines = lines;
    }

    public int getBranches() {
        return branches;
    }

    public void setBranches(int branches) {
        this.branches = branches;
    }

    public ClassTracker (String cl, int lines, int branches){
        className = cl;
        this.lines = lines;
        this.branches = branches;
    }
}
package com.sheffield.leapmotion.runtypes.state_identification;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Created by thomas on 9/26/2016.
 */
public class StateShowingFrame extends JFrame {

    private BufferedImage bi;

    public StateShowingFrame (){
        super("New Screenshot States");
    }

    @Override
    public void paint(Graphics g) {
        //super.paint(g);

        Graphics2D g2d = (Graphics2D) g;

        g2d.clearRect(0, 0, getWidth(), getHeight());

        if (bi != null){
            g2d.drawImage(bi, 0, 0, getWidth(), getHeight(), null);
        }

    }

    public void setImage(BufferedImage i){
        bi = i;
    }
}

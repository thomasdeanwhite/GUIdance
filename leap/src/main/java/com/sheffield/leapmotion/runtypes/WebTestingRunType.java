package com.sheffield.leapmotion.runtypes;

import com.sheffield.leapmotion.App;
import com.sheffield.leapmotion.Properties;
import com.sheffield.leapmotion.runtypes.web.WebServer;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

/**
 * Created by thomas on 24/04/17.
 */
public class WebTestingRunType implements RunType {

    WebDriver wd;

    @Override
    public int run() {

        if (Properties.WEBSITE == null){
            App.out.println("Please include\n\t-webpage [arg]\noption when testing a website");
            return -1;
        }

        WebServer ws = new WebServer();

        Thread thread = new Thread(ws);

        System.setProperty("webdriver.chrome.driver", "chrome/chromedriver");

        //setup Selenium web browser
        wd = new ChromeDriver();

        wd.get(Properties.WEBSITE);

        App.getApp().setup(true);

        //start data seeding
        ws.run();

        return 0;
    }
}

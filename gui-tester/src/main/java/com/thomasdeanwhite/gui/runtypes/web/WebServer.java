package com.thomasdeanwhite.gui.runtypes.web;

import com.thomasdeanwhite.gui.App;
import com.thomasdeanwhite.gui.util.AppStatus;
import org.apache.commons.codec.binary.Base64;
import org.apache.commons.codec.digest.DigestUtils;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by thomas on 24/04/17.
 */
public class WebServer implements Runnable {

    private ServerSocket server;

    private HashMap<Socket, PrintWriter> clients = new HashMap<Socket, PrintWriter>();
    private HashMap<Socket, BufferedReader> clientReader = new HashMap<Socket, BufferedReader>();

    private ArrayList<String> connected = new ArrayList<String>();


    public static String deriveHttpKey(String key){
        String websocketKey = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

        byte[] hashedKey = DigestUtils.sha1(websocketKey);

        String encodedKey = Base64.encodeBase64String(hashedKey);

        return encodedKey;
    }

    public static final String getHttpHeaders(String key){



        String header = "HTTP/1.1 101 Switching Protocols\r\n" +
                "Connection: Upgrade\r\n" +
                "Sec-WebSocket-Accept: " + deriveHttpKey(key) + "\r\n" +
                "Sec-WebSocket-Key: " + key + "\r\n" +
                "Server: WebSocket++/0.2.2dev\r\n" +
                "Upgrade: websocket\r\n\r\n";

        return header;
}

    public WebServer() {
    }

    @Override
    public void run() {

        App.setOutput();

        //6347 is the gui server port
        try {
            server = new ServerSocket();

            InetSocketAddress ins = new InetSocketAddress("127.0.0.1", 6437);

            server.bind(ins);

            App.out.println("[] Web Server has started: " + server.getLocalSocketAddress().toString() + ":" + server.getLocalPort());


            while (App.getApp().status() != AppStatus.FINISHED) {
                final Socket client = server.accept();

                if (!connected.contains(client.getInetAddress().toString())) {
                    //handshake

                    App.out.println("[] Client " + client.getInetAddress().toString() + " connected.");

                    clients.put(client, new PrintWriter(client.getOutputStream(), true));
                    clientReader.put(client, new BufferedReader(new InputStreamReader(client.getInputStream())));


                    String line;

                    String websocketKey = "";

                    while ((line = clientReader.get(client).readLine()).length() > 0) {
                        App.out.println(line);
                        if (line.contains("Sec-WebSocket-Key")) {
                            websocketKey = line.split(":")[1].trim();
                        }
                    }

                    clients.get(client).write(getHttpHeaders(websocketKey));

                    clients.get(client).flush();

                    connected.add(client.getInetAddress().toString());
                }
            }

        } catch (IOException e) {
            e.printStackTrace(App.out);
        }
    }
}

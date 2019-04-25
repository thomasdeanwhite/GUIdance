#!/usr/bin/sudo python3
import os
import sys
import random
from fpdf import FPDF


apps = [("00", "bellmanzadeh"),
        ("02", "JabRef"),
        ("03", "Java Fabled Lands"),
        ("05", "ordrumbox"),
        ("11", "Dietetics"),
        ("12", "Minesweeper"),
        ("13", "SQuiz"),
        ("14", "blackjack"),
        ("15", "UPM"),
        ("16", "Simple Calculator")]

line_height = 7

if __name__ == '__main__':

    wd = os.getcwd()

    if len(sys.argv) < 4:
        print("Must include three parameters: input task location, output task location, number of users.", file=sys.stderr)
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_users = int(sys.argv[3])

    for u in range(num_users):
        applications = apps[:]
        random.shuffle(applications)

        user_out = output_dir + "/" + str(u+1)

        if not os.path.isdir(user_out):
            os.makedirs(user_out, exist_ok=True)
            try:
                os.mkdir(user_out)
            except OSError as e:
                #folder exists
                pass

        pdf = FPDF(orientation="P", unit="mm", format="A4")

        pdf.add_page()

        pdf.set_font("Courier", style="u", size=32)

        pdf.cell(200, 32, txt="GUI Interaction Observations", ln=1, align="C")
        pdf.set_font("Courier", size=32)
        pdf.cell(200, 32, txt="Participant " + str(u+1), ln=1, align="C")

        pdf.add_page()
        pdf.cell(200, 32, txt="Information", ln=1, align="C")

        pdf.set_font("Courier", size=14)
        pdf.write(line_height, "You will be interacting with each application twice, once to familiarise " +
                               "yourself with the application and again to perform a set of tasks.\n\nWhen a set of " +
                               "tasks is complete, hit ESCAPE three (3) times to finish interaction " +
                               "with that application. You may take a break between each application. DO NOT press " +
                               "ESCAPE during the application warm up phase, the application will automatically close " +
                               "when this phase is finished.\n\nPlease give the application 15 seconds to load before " +
                               "you start interacting with it.\n\nIf file input/output is required, please use " +
                               "/home/thomas/tmp to save and load files.\n\nYour home directory is located at: " +
                                user_out + "/")

        counter = 0

        for app in applications:

            a = app[0]
            name = app[1]
            script_name = name.replace(" ", "_")

            counter += 1

            pre = []
            tasks = []
            post = []

            with open(input_dir + "/" + a + ".txt") as f:
                for line in f:
                    c = line.strip()

                    if len(c) == 0:
                        continue

                    if c.startswith("##"):
                        post.append(c[2:])
                    elif c.startswith("#"):
                        pre.append(c[1:])
                    else:
                        tasks.append(c)

            random.shuffle(tasks)

            pdf.add_page()

            pdf.set_font("Courier", style="u", size=32)

            pdf.cell(200, 32, txt="Application {}".format(counter), ln=1, align="c")

            pdf.set_font("Courier", size=32)

            pdf.cell(200, 32, txt=name, ln=1, align="c")

            pdf.set_font("Courier", size=24)

            if len(pre) > 0:
                contents = "Information"
                pdf.cell(200, 18, txt=contents, ln=1, align="c")
                pdf.set_font("Courier", size=14)
                contents = ""
                for c in pre:
                    contents += "- " + c + "\n"

                pdf.write(line_height, contents)
            else :
                pdf.set_font("Courier", size=14)
                pdf.write(line_height, "No information is required for this application.\n")

            pdf.set_font("Courier", size=14)
            pdf.cell(200, 10, txt="Please run {}.sh from the user directory.".format(script_name), ln=1, align="c")
            pdf.cell(200, 10, txt="Click \"Run\".", ln=1, align="c")



            pdf.set_font("Courier", size=24)

            pdf.cell(200, 18, txt="3 Minute Warm Up", ln=1, align="c")

            pdf.add_page()

            contents = "Application {}: {}".format(counter, name)

            pdf.set_font("Courier", size=24)
            pdf.cell(200, 18, txt=contents, ln=1, align="c")



            contents = "Tasks:"
            pdf.set_font("Courier", size=24)
            pdf.cell(200, 18, txt=contents, ln=1, align="c")
            contents = ""
            for c in tasks:
                contents += "[ ] - " + c + "\n"
            pdf.set_font("Courier", size=14)
            pdf.write(line_height, contents)

            if len(post) > 0:
                pdf.set_font("Courier", size=24)
                pdf.cell(200, 18, txt="Finally/Optional:", ln=1, align="c")
                contents = ""
                for c in post:
                    contents += "[ ] - " + c + "\n"
                pdf.set_font("Courier", size=14)
                pdf.write(line_height, contents)

            with open(user_out + "/" + script_name + ".sh", "w+") as f:
                f.write("#!/bin/bash\nbash ~/t2s.sh \"Starting warmup for " + name + "\"\n")
                f.write("cd ../..\n./run_manual.sh {} user-models/participant-{}/warmup 00 180\n".format(a, u+1))
                f.write("bash ~/t2s.sh \"Warm up complete. Starting tasks for " + name + "\"\n")
                f.write("./run_manual.sh {} user-models/participant-{}/tasks 00\n".format(a, u+1))
                f.write("bash ~/t2s.sh \"Data gathering for {}: complete!".format(name) + "\"\n")


        pdf.output(user_out + "/participant-" + str(u+1) + "-tasks.pdf")








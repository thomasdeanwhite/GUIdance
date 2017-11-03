package com.thomasdeanwhite.gui.runtypes;

import com.scythe.instrumenter.analysis.ClassAnalyzer;
import com.scythe.instrumenter.analysis.ClassNode;
import com.scythe.instrumenter.analysis.DependencyTree;
import com.scythe.instrumenter.instrumentation.ClassReplacementTransformer;
import com.thomasdeanwhite.gui.App;
import com.thomasdeanwhite.gui.util.FileHandler;
import com.thomasdeanwhite.gui.Properties;
import com.scythe.util.ClassNameUtils;

import java.io.File;
import java.util.*;

/**
 * Created by thomas on 18/11/2016.
 */
public class InstrumentingRunType implements RunType {
    @Override
    public int run() {
        App.out.println("- Instrumenting JAR");
        for (String s : Properties.FORBIDDEN_PACKAGES) {
            ClassReplacementTransformer.addForbiddenPackage(s);
        }
        App.ENABLE_APPLICATION_OUTPUT = true;
        App.IS_INSTRUMENTING = true;
        Properties.LOG = false;
        ClassAnalyzer.setOut(App.out);


        Properties.INSTRUMENTATION_APPROACH = Properties.InstrumentationApproach.ARRAY;
        try {
            String dir = Properties.JAR_UNDER_TEST.substring(0, Properties.JAR_UNDER_TEST.lastIndexOf("/") + 1);

            Properties.WRITE_CLASS = true;
            Properties.BYTECODE_DIR = dir + "classes";

            String related = dir + "related_classes.csv";

            File relatedFile = new File(related);

            if (relatedFile.exists() && !Properties.DEPENDENCY_TREE_OVERRIDE) {
                Properties.SKIP_DEPENDENCY_TREE = true;
            }


            String output = Properties.TESTING_OUTPUT + "/NuiMimic/branches.csv";
            String output2 = Properties.TESTING_OUTPUT + "/NuiMimic/lines.csv";
            App.out.print("+ Writing output to: " + dir + " {branches.csv, lines.csv}");
            ClassAnalyzer.output(output, output2, Properties.UNTRACKED_PACKAGES);
            App.out.println("\r+ Written output to: " + dir + " {branches.csv, lines.csv}");

            HashSet<ClassNode> nodes;
            if (Properties.SLICE_ROOT == null) {
                ArrayList<ClassNode> superNodes = DependencyTree.getDependencyTree().getPackageNodes(DependencyTree.getClassMethodId("com.gui.leap.Controller", "<init>"));

                nodes = new HashSet<ClassNode>();

                nodes.addAll(superNodes);
//
//                for (ClassNode cn : superNodes){
//                    nodes.addAll(cn.getDependencies());
//                }
            } else {
                App.out.println();
                nodes = new HashSet<ClassNode>();
                for (String s : Properties.SLICE_ROOT.split(";")) {
                    s = ClassNameUtils.standardise(s);
                    App.out.print("\r[Getting dependencies for: " + s + "]");
                    nodes.addAll(DependencyTree.getDependencyTree().getDependencies(s));
                }
            }

            Set<String> lines = new HashSet<String>();

//            HashSet<String> lines = new HashSet<String>();
            ArrayList<String> relatedClasses = new ArrayList<String>();
            for (ClassNode cn : nodes) {

                HashSet<String> classes = new HashSet<String>();

                String[] clazzes = cn.toString(relatedClasses).split("\n");

                classes.addAll(Arrays.asList(clazzes));

//                String[] link = cn.toNomnoml().split("\N");
//                for (String N : link) {
//                    if (!lines.contains(N)) {
//                        lines.add(N);
//                    }
//                }

                for (String s : classes) {


                    if (s.length() > 0) {
                        if (!ClassReplacementTransformer.isForbiddenPackage(s) && !relatedClasses.contains(s)) {
                            relatedClasses.add(s);
                            //App.out.println(s);
                        }
                    }
                }

                //App.out.println(cn.toNomnoml());
            }

//            ArrayList<ClassNode> options = DependencyTree.getDependencyTree().getPackageNodes("JOptionsPane");
//
//            for (ClassNode cn : options) {
//                App.out.println("OPTIONS: " + cn.toNomnoml());
//            }


//            for (String s : lines) {
//                App.out.println(s);
//            }

            if (Properties.SKIP_DEPENDENCY_TREE && !Properties.DEPENDENCY_TREE_OVERRIDE){
                return 0;
            }

            relatedFile.createNewFile();

            String classes = "";

            String[] forbid = new String[]{};

            if (Properties.UNTRACKED_PACKAGES != null){
                forbid = Properties.UNTRACKED_PACKAGES.split(",");
            }

            for (int i = 0; i < forbid.length; i++){
                forbid[i] = ClassNameUtils.standardise(forbid[i]);
            }

            relatedClasses.sort(new Comparator<String>() {
                @Override
                public int compare(String o1, String o2) {
                    for (int i = 0; i < Math.min(o1.length(), o2.length()); i++){
                        if (o1.charAt(i) == o2.charAt(i)){
                            continue;
                        }
                        return o1.charAt(i) - o2.charAt(i);
                    }
                    return 0;
                }
            });

            for (String c : relatedClasses) {

                boolean skip = false;
                for (String f : forbid){
                    if (c.startsWith(f)){
                        skip = true;
                        break;
                    }
                }

                if (c == null || c.length() == 0 || skip || c.startsWith("com/gui/leap")) {
                    continue;
                }

                String cName = DependencyTree.getClassName(c);
                String mName = DependencyTree.getMethodName(c);

                classes += c + "," + ClassAnalyzer.getCoverableLines(cName, mName).size() + "," + ClassAnalyzer.getCoverableBranches(cName, mName).size() + "\n";
            }
            FileHandler.writeToFile(relatedFile, "class,lines,branches\n");
            FileHandler.appendToFile(relatedFile, classes);


        } catch (Exception e) {
            e.printStackTrace(App.out);
        }
        return 0;
    }
}

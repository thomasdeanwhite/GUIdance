cd D:\work\instrumentation
call mvn clean install
cd D:\work\NuiMimic
call mvn clean package -DskipTests=true
cd gui-tester\target
del nuimimic.jar
rename leap-0.0.1-SNAPSHOT.jar nuimimic.jar
cd D:\work\NuiMimic
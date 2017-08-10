for i in `seq 1 10000`
do
    java -Xmx8G -jar nuimimic.jar -input tom-tst -exec 'bash tst.sh' -runtime 60000 -runtype TESTING -interaction EXPLORATION_DEEP_LEARNING
done

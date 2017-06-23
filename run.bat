FOR /L %%x IN (1,1,5) DO (
  call java -Xmx8G -jar nuimimic.jar -exec "D:\\calculator.bat" -runtime 60000 -runtype TESTING
  call taskkill /IM Calculator.exe /F
)
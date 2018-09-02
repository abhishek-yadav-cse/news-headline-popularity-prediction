******GENERAL INSTRUCTIONS AND INFORMATIONS FOR TERMINAL/JUPYTER NOTEBOOK BASED CODE******

The source code includes 'code.py'(can be executed from terminal) file and 'code.ipynb'(Jupyter Notebook file). The required data are kept inside 'dataset' folder which includes following files - 
1. final_news.csv
2. Facebook_Economy.csv
3. Facebook_Microsoft.csv
4. Facebook_Obama.csv
5. Facebook_Palestine.csv
6. GooglePlus_Economy.csv
7. GooglePlus_Microsoft.csv
8. GooglePlus_Obama.csv
9. GooglePlus_Palestine.csv
10. LinkedIn_Economy.csv
11. LinkedIn_Microsoft.csv
12. LinkedIn_Obama.csv
13. LinkedIn_Palestine.csv
14. test_sentences.csv

All other files are intermediate files generated while executing the code.

****These files are manually pre-processed in following ways - ****

1. Removed unnecessary columns
2. Removed NaN values
___________________________________________________________________________

Check all dependencies and install requirements-

Install all python important library - Numpy, Scipy, Pandas, Matplotlib, sklearn, Tensorflow, Keras, nltk

___________________________________________________________________________

****To run 'code.py' file.****

$ python3 code.py

Type the category from list of social media categories.

$ Facebook

Type a sample sentence from internet among four categories - Obama, Economy, Palestine or Microsoft. You can also select a test sentence from list of test sentences (test_sentence.csv).

$ U.S. President Barack Obama and German Chancellor Angela Merkel take their position during an arrival ceremony at Schloss Herrenhausen in Hannover, Germany, Sunday, April 24, 2016. Obama has arrived in Germany to mount a two-day push to sell his trans-Atlantic trade pact.

Several output are generated.

___________________________________________________________________________
****Ouput****

1. Category predicted
2. Sentiment Predicted
3. Time series analysis predictions graphs
 The source code only contain LSTM output. However, we have compared three time series analysis models.
4. Future time prediction graph

___________________________________________________________________________

****To run 'code.ipynb' file.****

Simply open the file in Jupyter Notebook and run each cell one by one. The respective output will be generated.


___________________________________________________________________________


The directory also contain a folder named - "WebApplication". This folder has a web application implementation of above python code using Flask framework. Instructions to run that will be mentioned in a separate readme file kept inside "WebApplication" folder.





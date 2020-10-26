# aspect-based-sentiment-analysis
This repository contains code for SemEval2016 Task 5: Aspect-Based Sentiment Analysis  <br />
This work was not an official submission but a project work for my own senior project. <br />
SKlearn libray was used for SVM, TF-IDF features and Grid Cross-Validation for hyperparamter tuning (Slots 1 and 3). <br />
Pure tensorflow was used to implement CNN and LSTM for Slots 1 and 3. <br />
Stanford Core-NLP was used to obtain POS, NER tags and Lemmas. We also used other semantical features (isLower, isTitle, isDigit). CRF (pycrfsuite) was trained for Slot 2 as a token classification task using IOB-labels format

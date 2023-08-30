# TWEET CLASSIFIERS

---

This project is divided into three parts:
1. **Data Study**
2. **Classification**
3. **Others**

First of all, you need to install the required packages. You can do it by running the following command:

```bash
pip install -r requirements.txt
```

Optionally, you can create a virtual environment to install the packages. To do it, you need to run the following command:

```bash
python3 -m venv venv # in Linux/Mac
py -m venv venv # in Windows
```

Then you must activate it and run the previous command to install the packages.

For the correct execution of the scripts, you must edit the file **config.py** and set the correct values. The attributes are the following:
    - **uri**: uri of the database where the data is stored. Example: "mysql+pymysql://\<user>:\<password><@\<host>:\<port>/<db_name>"
    - **protest_types**: object with the different types of protests. All types have a list with the corresponding subtypes.
    - **tree_dirs**: list with the possible directories/paths where the conversation tree data are stored.

## Data Study

In the module **descriptive analsys** there are many scripts to get plots, graphs and tables to understand the data. The scripts have different parameters, you can see them by running the following command:

```bash
python3 descriptive_analysis/<script_name>.py -h
```

All of them that produce a figure have the option of show, save it or both. In case of saving it, the figure will be saved in the directory `images`.

## Classification

### Tone (Sentiment + Emotional) Classifier

In this part we have two different approaches to classify the tweets. The first one is an approach to sentiment and emotional analysis. For this, we have used the library [Transformes](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment), exactly an NLP for sentiments anlysis based in the BERT model. Additionaly, we have included the [NRC Lex](https://pypi.org/project/NRCLex/) library which is based in the [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) to classify the tweets in the following emotions:
- :angry: anger
- :fearful: fear
- anticipation
- trust
- :open_mouth: surprise
- :sweat: sadness
- :joy: joy
- disgust
    
For classify tweets individually, you can use the script **sentiment_classifier.py**, where you must introduce the conversation id and you can choose the classifier to use. You can run it with -h to see the parameters.

### Protest Type Classifier

The second one is an approach to classify the tweets in the different types of protests. For this, we have used the paper ['World Protest. A Study of Key Protest Issues in the 21st Century'](https://library.fes.de/pdf-files/bueros/usa/19020.pdf) where describes the different types and subtypes of protest. The values indicated in the `config.json` file correspond to the results of this paper.

The script **tagging.py** allows to classify the tweets in the different subtypes of protests manually and save it in a json file. There is a file in the `data` directory called `labeled_tweets.json` with two objects. The first one represent an unlabeled tweet (with the necessary format to be able to classify it) and the second one is an example of a labeled tweet. The script **tagging.py** will show the first tweet (without a label) and you must to select the type and then the subtype. Later, the script will show the next tweet and so on. If you want to stop the script, you must to select the `exit` option. The script will save the labeled tweets in the file `labeled_tweets.json`.

The classifier for this approach is still under development, so it is not yet available.

## Others

The module `tree` contains all tools to create and show the conversation graph that genarate a independet root tweet. The obtained tree is saved in the directory `data/tree`. In this directory there is a file called `conversation_id.json` with an example of a conversation tree. The name of each file is the id of the conversation that must be introduced in the scripts, for example:

```bash
python3 tree/graph_tree.py -c <conversation_id>
```

The module `utils` contains all tools that are used in the other modules.

# SubjectLineGenerator
by Kathy Liu and Jonah Winchell  
Fall 2021 NLP final project

### Project Structure
*generate.py*: 
- input: .mbox file
- output: .json file of tokenized sentences of subject lines and bodies of emails from .edu addresses  
  
*extractkeys.py*:
- finds representative sentence(s) of email bodies using embedding, clustering, silhouette coefficient
- finds keyword of representative sent/subject line using rake algorithm and dependencies  
  
*languagemodel.py*:
- labels subject lines with keywords using extractkeys.py
- trains keytotext LM with labeled data
- uses LM to generate subject line given keywords

### How to Run   
- To create the model, you must have Jupyter Notebooks installed either locally, or have access to Jupyter Notebooks in a cloud-based environment.
- Run the `SubjectLineModel.ipynb` Notebook, making sure that a suitable `data.json` is in the same folder.
- This will save a model locally, which you can then access via the `languagemodel.py` script.
- NOTE: The model itself is far too large to include in this git repo, or to be able to upload/transport at all. Using the Jupyter Notebook to train a new 
version is the only viable way to recreate it, sadly.
- When running `languagemodel.py`, point the script towards a text file containing the email body you wish to process.
- The script will then output a recommended subject line to the command line.
- As long as there is a `data.json` file already included, you do not need to run either `generate.py` or `extractkeys.py` for any reason.

For questions or requests, please email jwinchell@oxy.edu or kliu4@oxy.edu

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

### Limitations

milestone 2 demo:
- run `extractkeys.py`
- if modulenotfounderror, run `python3.x -m spacy download en_core_web_sm` in shell
- expected to take an email body, embed, cluster, find "best" cluster(s), find representative sentence of those cluster(s), print head word of each of those sentences

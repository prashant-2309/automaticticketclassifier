---
license: mit
language:
- en
pipeline_tag: text-classification
widget:
- text: "Title: File server down Description: The internet is back up after we rebooted everything but nobody can access the file server now, please help."
- text: "Title: New mouse please Description: My mouse is acting up a little bit could I get a new one please?"
- text: "Title: Internet outage Description: The whole internet is down for everyone in the office"
---
# Model Name: DavinciTech/BERT_Categorizer

## Model Description

This model is designed to categorize IT support tickets into ITIL-standard categories to streamline ticket routing and resolution in the IT support process. It's based on BERT, a transformer-based model that has been widely successful for a variety of NLP tasks. This model takes the text of an it support ticket as input in the format `"Title: <title text> Description: <body text>"`, and returns values for ITIL Impact, Urgency, and Type.

### How to Use

The DavinciTech/BERT_Categorizer can be used directly after installation via the Hugging Face `transformers` library.  The model returns a packed array of predictions for all the categories which need to be decoded to find the most likely candidate for each class.  Please see the included 'inference.py' script for an example of how to use the model and decode the output.

### Intended Uses and Limitations

This model is intended for use in IT support ticket classification systems that align with the ITIL framework. While it has been fine-tuned to categorize various standard IT support requests, users should be aware of the potential limitations in context understanding and the need for additional fine-tuning when adapting to specific IT environments.

We trained this model at Davinci Technology Solutions to automate the first triage steps in our IT helpdesk. Davinci Technology Solutions is a Managed Service Provider supporting small businesses in Canada with their technology needs. The training data used to train this model reflects this context.

### Training Data

The model was trained using approximately 10k IT support tickets in our support ticketing system, unfortunately this data set cannot be released due to the privacy concerns of our customers. However examples of the dataset are provided here if you would like to continue the training with your own dataset.

The dataset was prepared using a hybrid data synthesis method involving GPT-4, to ensure the accuracy of the training data we used the OpenAI API to verify and update the categories of all 10k tickets in the training dataset by asking GPT-4 to categorize the data and then spot checked it manually.

The example below shows how the training dataset was provided to the model during training. The text was preprocessed to remove all formatting, html css etc. and provide a clean plaintext version of each ticket. The majority of the requests were received via email and our ticketing system naturally mapped the email subject to ticket title and email body to ticket description.


```csv
ID,text,impact,urgency,type
895,"Title: RE Internet office phone line Description: Hi team We did a restart on the computer and we are connected to the internet now Just need to know about the phoneline and see if there are spare phone consoles that are kicking around for that office so the accountant can use it when they come in ",I2,U2,T2
1710,"Title: System so Slow Description: Morning we are experiencing a lot of slow response time on our end on the system yesterday and today is there a reason ",I3,U3,T3
```

In the above example the keys for the ticket impact, urgency and type correspond to the LABEL_DICTIONARY provided in the model config file.

```python
LABEL_DICTIONARY = {
"I1": "Low Impact",
"I2": "Medium Impact",
"I3": "High Impact",
"I4": "Critical Impact",
"U1": "Low Urgency",
"U2": "Medium Urgency",
"U3": "High Urgency",
"U4": "Critical Urgency",
"T1": "Information",
"T2": "Incident",
"T3": "Problem",
"T4": "Request",
"T5": "Question",
}
```
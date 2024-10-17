

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

# Ask Your Humans: Using Human Instructions to Improve Generalization in Reinforcement Learning

This section of the repository contains code for the collection interface.

## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Starting the interface

To load the collection interfact described in the paper, run this command:

```train
python3 app_nodb.py
```

You should have the entrance codes generated beforehand and uploaded to AMT for HITs.

We hosted the app on an EC2 instance so that the server would be continuously running while annotations were collected.

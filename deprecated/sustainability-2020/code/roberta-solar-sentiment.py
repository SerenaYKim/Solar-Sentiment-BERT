# -*- coding: utf-8 -*-
# import

import tensorflow as tf
import pandas as pd
import numpy as np
import re


# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')

import torch

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

!pip install transformers
# !unzip training.1600000.processed.noemoticon.csv.zip

# Load the dataset into a pandas dataframe.

df = pd.read_csv('...', sep=',', encoding='latin-1', names=['phrase', 'sentid', 'sentiment', 'phraseid'], index_col=None)

#----------------------------------------------------------------------------------------
df = df.iloc[1:]
df1 = df[df['sentiment'] == '2']
df = df[df['sentiment'] != '2']
df = df.drop(['phraseid', 'sentid'], axis=1)
# df = pd.concat([df, df3], ignore_index = True)
train = df.sample(frac = 0.9, random_state=195)
test = df.drop(train.index)
print(len(train),len(test))
print(train['sentiment'].value_counts(), '\n', test['sentiment'].value_counts())
print(df.head())

from google.colab import drive
drive.mount('/content/drive')


contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "don't": "do not", "din't": "did not",
                       "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "i'd": "i would", "i'll": "i will",
                       "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "might've": "might have",
                       "needn't": "need not", "she'd": "she would", "she'll": "she will", "so've": "so have", "this's": "this is",
                       "that'd": "that would", "there's": "there  is", "here's": "here is", "they'd": "they would", "wasn't": "was not",
                       "we'd": "we would", "what're": "what are", "where've": "where have", "who'll": "who will", "would've": "would have",
                       "wouldn't": "would not", "y'all": "you all", "you'd": "you would", "he'd": "he would", "he'll": "he will", "it's": "it is",
                       "doesn't": "does not", "dont": "do not", "that's": "that is", "you're": "you are", "could've": "could have", "she's": "she is",
                       "he's": "he is", "who's": "who is", "won't": "would not", "didn't": "did not", "we've": "we have", "that'll": "that  will",
                       "there's": "there is", "they're": "they are", "its": "it is", "wont": "would not", "isnt": "is not", "we'll": "we will", "they'll": "they will",
                       "gov't": "government", "shouldn't": "should not", "we're": "we are", "couldn't": "could not", "you'll": "you will", "they've": "they have",

                       "Ain't": "Is not", "Aren't": "Are not", "Can't": "Cannot", "Don't": "Do not", "Din't": "Did not",
                       "Hadn't": "Had not", "Hasn't": "Has not", "Haven't": "Have not", "I'd": "I would", "I'll": "I will",
                       "I'm": "I am", "I've": "I have", "Isn't": "Is not", "It'd": "It would", "Might've": "Might have",
                       "Needn't": "Need not", "She'd": "She would", "She'll": "She will", "So've": "So have", "This's": "This is",
                       "That'd": "That would", "There's": "There  is", "Here's": "Here is", "They'd": "They would", "Wasn't": "Was not",
                       "We'd": "We would", "What're": "What are", "Where've": "Where have", "Who'll": "Who will", "Would've": "Would have",
                       "Wouldn't": "Would not", "Y'all": "You all", "You'd": "You would", "He'd": "He would", "He'll": "He will", "It's": "It is",
                       "Doesn't": "Does not", "Dont": "Do not", "That's": "That is", "You're": "You are", "Could've": "Could have", "She's": "She is",
                       "He's": "He is", "Who's": "Who is", "Won't": "Would not", "Didn't": "Did not", "We've": "We have", "That'll": "That  will",
                       "There's": "There is", "They're": "They are", "Its": "It is", "Wont": "Would not", "Isnt": "Is not", "We'll": "We will", "Gov't": "Government"
                       , "Shouldn't": "Should not", "We're": "We are", "Couldn't": "Could not", "You'll": "You will", "They've": "They have"}

def stemmingWords(sentence,dictionary):
    return " ".join([dictionary.get(w,w) for w in sentence.split()])

# Get the lists of sentences and their labels.

def _removeNonAscii(s): return "".join(i for i in s if ord(i)<128)
# rm_list = ',:.";|()$1234567890-@^#!?$=%~&+*/\[]{}'
rm_list = ''
sentences = train.phrase.values
labels = train.sentiment.values
for i in range(0,len(sentences)):
  sentences[i] = re.sub('@[^\s]+',' ',sentences[i])
  sentences[i] = re.sub('&[^\s]+',' ',sentences[i])
  sentences[i] = re.sub('https?://\S+',' ',sentences[i])
  sentences[i] = _removeNonAscii(sentences[i])
  for j in rm_list:
    sentences[i] = sentences[i].replace(j,' ')
  sentences[i] = ' '.join(sentences[i].split())
  if sentences[i][0] == ':':
    if sentences[i][1] == ' ':
      sentences[i] = sentences[i][2:]
    else:
      sentences[i] = sentences[i][1:]
remove_list = ['healing', 'jewelry','bracelet', 'worship', 'cosmic', 'chakra',
               'charizard', 'discos', 'sex', 'plexus', 'claw', 'dragon', 'eclipse', 'radiation', 'beam', 'superman', 'gravitational', 'system', 'orbit']
flag = []
print(len(sentences))

for sent in sentences:
  print(sent)

from transformers import RobertaTokenizer

# Load the BERT tokenizer.
print('Loading RoBERTa tokenizer...')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

max_len = 0
temp = []

# For every sentence...
for i in range(0,len(sentences)):

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sentences[i], add_special_tokens=True)

    # Update the maximum sentence length.
    if (len(input_ids)>128):
      temp.append(i)
      print(sentences[i])
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)

# Tokenize all of the sentences and map the tokens to thier word IDs.
labels = labels.astype(int)
labels = labels/4
labels = labels.astype(int)
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,          # Pad & truncate all sentences.
                        truncation = True,
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])

from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it
# here. For fine-tuning BERT on a specific task, the authors recommend a batch
# size of 16 or 32.
batch_size = 16

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

from transformers import RobertaConfig, RobertaForSequenceClassification, AdamW

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 6e-6, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

from transformers import get_cosine_schedule_with_warmup
# from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4.
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
# scheduler = get_constant_schedule_with_warmup(optimizer,
#                                             num_warmup_steps = 0)

import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random
import numpy as np

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 9887976

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        if (step + 1) % 2 == 0:
          optimizer.step()
        # Update the learning rate.
          scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)


    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

import pandas as pd

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap.
#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# Display the table.
df_stats

import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# Load the dataset into a pandas dataframe.
# df = pd.read_csv('1.csv', sep='\t', encoding='utf-8', names=['sentence', 'phraseid', 'sentid', 'sentiment'])
# df = df.iloc[1:]
# df = df[df['sentiment'] != '2']
# print(df[2:3]['sentence'])
# Report the number of sentences.
df = test
print('Number of test sentences: {:,}\n'.format(df.shape[0]))
# print(df.head())
# Create sentence and label lists
sentences = df.phrase.values
labels = df.sentiment.values
for i in range(0,len(sentences)):
  sentences[i] = re.sub('@[^\s]+',' ',sentences[i])
  sentences[i] = re.sub('&[^\s]+',' ',sentences[i])
  sentences[i] = re.sub('https?://\S+',' ',sentences[i])
  sentences[i] = _removeNonAscii(sentences[i])
  for j in rm_list:
    sentences[i] = sentences[i].replace(j,' ')
  sentences[i] = ' '.join(sentences[i].split())
  if sentences[i][0] == ':':
    if sentences[i][1] == ' ':
      sentences[i] = sentences[i][2:]
    else:
      sentences[i] = sentences[i][1:]

# flag = []
# for i in range(0,len(sentences)):
#   word_split = sentences[i].split()
#   for word in word_split:
#     if word.lower() in remove_list:
#       flag.append(i)
# sentences = np.delete(sentences, flag)
# labels = np.delete(labels, flag)
print(len(sentences))
# for  i in range(len(sentences)):
#   word_split = sentences[i].split()
#   for word in word_split:
#     if "'s" in word or "'S" in word:
#       sentences[i] = sentences[i].replace(word, word[:-2]+' '+'is')
#   sentences[i] = stemmingWords(sentences[i], contraction_mapping)
#   sentences[i] = sentences[i].replace("'",' ')
#   sentences[i] = ' '.join(sentences[i].split())

labels = labels.astype(int)
labels = labels/4
labels = labels.astype(int)

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        truncation = True,
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Set the batch size.
batch_size = 16

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

pd.options.display.max_colwidth = 400
for i in range(1, len(sentences)):
  print(sentences[i])

# Prediction on test set
from sklearn.metrics import f1_score
print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables
predictions , true_labels = [], []
total_eval_accuracy=0
# Predict
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch

  # Telling the model not to compute or store gradients, saving memory and
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  total_eval_accuracy += flat_accuracy(logits, label_ids)
  # Store predictions and true labels
  predictions.append(np.argmax(logits, axis=1).flatten())
  true_labels.append(label_ids)

predictions = [item for sublist in predictions for item in sublist]
true_labels= [item for sublist in true_labels for item in sublist]
f1 = f1_score(y_true=true_labels, y_pred=predictions)
avg_val_accuracy = total_eval_accuracy / len(prediction_dataloader)
print("  Accuracy: {0:.3f}".format(avg_val_accuracy))
print("  F1: {0:.3f}".format(f1))
print('    DONE.')


    # ========================================
    #               Prediction
    # ========================================


from itertools import chain
import os

for filename in os.listdir('...'):
  print("predicting file :", filename, '\n')
  with open('/.../'+filename) as f:
    df = pd.read_csv('/.../'+filename, sep=',', encoding='latin-1', names=['phrase', 'phraseid', 'state'])
    df = df[1:]
    df.insert(0, "label", [k for k in range(0,len(df))], True)
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))
    print(df.head())
    # Create sentence and label lists
    df['phraseid'] = df['phraseid'].astype(int)
    sentences = df.phrase.values
    phraseids = df.phraseid.values
    labels = df.label.values
    for i in range(0,len(sentences)):
      sentences[i] = re.sub('@[^\s]+',' ',sentences[i])
      sentences[i] = re.sub('&[^\s]+',' ',sentences[i])
      sentences[i] = re.sub('https?://\S+',' ',sentences[i])
      sentences[i] = _removeNonAscii(sentences[i])
      for j in rm_list:
        sentences[i] = sentences[i].replace(j,' ')
      sentences[i] = ' '.join(sentences[i].split())
      if sentences[i][0] == ':':
        if sentences[i][1] == ' ':
          sentences[i] = sentences[i][2:]
        else:
          sentences[i] = sentences[i][1:]

      # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

# For every sentence...
    for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        truncation = True,
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

    # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # print(labels, phraseids)
    labels = torch.tensor(labels)
    phraseids = torch.tensor(phraseids)
# Set the batch size.
    batch_size = 16

# Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels, phraseids)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
    model.eval()

# Tracking variables
    predictions , true_labels, phrase_id1 = [], [], []
# Predict
    for batch in prediction_dataloader:
  # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels, b_phraseids = batch

  # Telling the model not to compute or store gradients, saving memory and
  # speeding up prediction
      with torch.no_grad():
      # Forward pass, calculate logit predictions
          outputs = model(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_mask)

      logits = outputs[0]

  # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      phrase_ids = b_phraseids.to('cpu').numpy()

      pred_flat = np.argmax(logits, axis=1).flatten()
      labels_flat = label_ids.flatten()
      phrase_ids_flat = phrase_ids.flatten()
  # Store predictions and true labels
      predictions.append(pred_flat)
      true_labels.append(labels_flat)
      phrase_id1.append(phrase_ids_flat)

    predictions = list(chain.from_iterable(predictions))
    phrase_id1 = list(chain.from_iterable(phrase_id1))
    true_labels = list(chain.from_iterable(true_labels))
    df1 = pd.DataFrame(list(zip(sentences, predictions, phrase_id1)), columns=['tweet', 'sentiment', 'phraseid'])
    df1.to_csv('/.../'+filename)
print('    DONE.')

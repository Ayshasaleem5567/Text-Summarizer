# AI
## Test Summarization
Text summarization is an NLP application that generates a concise and insightful summary of a lengthy paragraph, allowing us to quickly grasp the essence of the subject. Automatic text summarising is essential to better support the discovery of relevant information and access meaningful data.

## Types of Text Summarization
1. *Abstrative Based*: In Abstractive based, we produce new phrases from the original text . It's possible that the sentences produced by abstractive summarization are missing from the source material.

2. *Extractive Based*: In extractive based, we only extract the text that contains the significant lines or phrases from the original text. That compiled text would serve as our summary.

## Problem Statement
Long and detailed reviews from customers are common. Manually assessing these evaluations takes a lot of work. Natural language processing can be used to create a succinct summary for in-depth reviews.
Our goal is to provide a summary for the "Amazon Fine Food Reviews" utilizing text summarization techniques that are abstraction-based.

Data Scource: [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews?select=Reviews.csv)
***
## Information about dataset
+ Some columns are important such as the text and summary one whereas others are relatively less important.
+ Total rows: 564827 rows
+ Toatal columns: 24 but 2 important
***
## Problems in dataset
+ Null values within reviews indicate missing information, which can mislead readers or create an incomplete picture of a product or service. This can lead to inaccurate expectations or biased opinions based on the limited available information.
+ Reading long, repetitive reviews with null values can result in a poor user experience. Users may feel overwhelmed, frustrated, or confused when trying to extract meaningful insights. This can discourage participation.
+ Repetitive reviews tend to provide the same information repeatedly, which adds little value to readers. This redundancy can be tiresome and time-consuming to navigate, diminishing the usefulness of the reviews.
***
## Benefits
Using a text summarizer based on LSTM (Long Short-Term Memory) models offers several advantages:
+ LSTM models give increased accuracy and efficiency in generating summaries.
+ LSTM models preserve context while summarizing text, creating more coherent and informative summaries.
+ LSTM models can summarize texts of different lengths.
+ LSTM models benefit from large-scale training data to learn diverse patterns and extract important information.
+ LSTM models produce high-quality summaries by considering context and grammar rules.
+ LSTM-based summarizers provide consistency and standardization, reducing bias and subjective interpretations.
+ Text summarizers automate the summarization process, saving time and effort for tasks such as news aggregation.
***
## Limitations
+ LSTM models lack domain-specific knowledge, leading to less accurate summaries.
+ LSTM-based summarizers limit their ability to generate abstractive summaries.
+ LSTM models may not fully understand the broader context of a document.
+ LSTM models may struggle with complex sentence structures, leading to oversimplified summaries.
+ LSTM-based summarizers can produce generic summaries that lack specificity or convey unique aspects.
+ LSTM models may not accurately summarize unfamiliar terms, leading to incomplete or misleading summaries.
+ LSTM models must be exposed to diverse training data to produce accurate summaries.
+ LSTM models may have issues with overfitting or difficulty in effectively training the model
***
## Limitations of encoder decoder architecture:
+ The encoder converts the entire input sequence into a fixed length vector and then the decoder predicts the output sequence. This works only for short sequences since the decoder is looking at the entire input sequence for the prediction.
+ It is difficult for the encoder to memorize long sequences into a fixed length vector.
***
## Project pipeline
1. Understanding Text Summarization
2. Text pre-processing
3. Abstractive Text Summarization using LSTM, ENCODER-DECODER architecture
4. Web scrape an article using BS4
5. Extractive Text Summarization using Transformer
***
## Text cleaner
It preprocesses text to remove HTML tags, converts alphabetic characters, tokenizes words, and joins them back into a string.
 ***
## Summary cleaner
Tokenizes text, removes short words, joins cleaned words back into string.
***
## Tokenizer and dictionary creation
Create a dictionary to map each unique word in the training data to an integer, iterate through each review text, pad integer sequences with zeros, and add 1 to the word_dict size to calculate vocabulary size.
***
## Encoder Decoder
Three LSTM layers encode the input sequence, with the output of the last layer used for decoding. The attention layer helps the decoder focus on relevant parts of the input sequence, producing output vectors the same length as the encoder input sequence. The RMSprop optimizer and sparse categorical cross-entropy loss are used to train a neural machine translation model for 50 epochs with a batch size of 512. Two dictionaries are created to map words to integer indices and vice versa, converting model indices into target language words.

![encoder decoder](https://github.com/aunali1932/project-text-summary/blob/main/Screenshot%202023-05-09%20024835.png)

![encoder decoder](https://github.com/aunali1932/project-text-summary/blob/main/Screenshot%202023-05-09%20025751.png)
***
## Encoder Decoder Inference using Softmax
Inference models for encoder and decoder are used to predict translations, taking encoder_inputs, hidden state, and cell state. Attention is used to pass decoder_hidden_state_input and outputs2 through the attention layer and concatenated tensor through the decoder_dense layer. Moreover, The function decodes a single input sequence using a trained encoder-decoder model, generates an empty target sequence, predicts the next token, and returns the decoded sentence.
***
## Seq2summary
Seq2summary and seq2text are functions used to convert sequences of tokens into their original text form. Seq2summary iterates over a sequence of tokens to check if it is present in the reverse_target_word_index dictionary and joins the list of words to form a string. Lastly, we use a trained model to generate summaries for validation examples, printing the review text, original summary, and predicted summary
***
## Conclusion
The text summarizer using an encoder-decoder LSTM architecture is a powerful tool for generating concise summaries from longer pieces of text, leveraging the strengths of LSTM networks to capture long-term dependencies.The encoder-decoder LSTM text summarizer is effective due to its ability to handle variable-length inputs and outputs, the LSTM architecture and attention mechanism, and the quality and diversity of the training data.LSTM-based text summarizer leverages LSTM networks and attention mechanisms to generate concise summaries.

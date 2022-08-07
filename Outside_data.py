import tensorflow as tf
from tensorflow import keras
import  numpy as np

imdb = keras.datasets.imdb

word_index = imdb.get_word_index()

work_index = {k:v+3 for k , v in word_index.items()}
#Assign own values
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
work_index["<UNUSED>"] = 3

model = keras.models.load_model("model.h5")

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded


with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])
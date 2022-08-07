import tensorflow as tf
from tensorflow import keras


imdb = keras.datasets.imdb


(train_data , train_labels) , (test_data , test_labels) = imdb.load_data(num_words=10000)



#find a way to reverse_words
word_index = imdb.get_word_index()

work_index = {k:v+3 for k , v in word_index.items()}
#Assign own values
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
work_index["<UNUSED>"] = 3


#swap all the values and the keys
#have the integer poitning to the word
reverse_word_index = dict([(value , key) for key , value in word_index.items() ])

def decode_review(text):

    readable_text = ''
    for i in text:

        word = reverse_word_index.get(i, "?")

        readable_text = readable_text + " " + word

    return readable_text
#
# def decode_review(text):
# 	return " ".join([reverse_word_index.get(i, "?") for i in text])

print(decode_review( test_data[0]))



# write a padding function
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# print(len(test_data[0]) )

#write a padding function
#get only the first 50 words

# if (len(test_data[0])>250):
#     copy = []
#     for value in range(250):
#         copy = test_data[0][value]
#     print(decode_review(copy))
#     #get only the first 250 data.
#
# if len(test_data[0])<250:
#     print("I am here")
#
#     for i in range(len(test_data) , 250):
#         test_data[0].append(1)
#
#     print(decode_review(test_data[0]))
#
#


print(decode_review(test_data[0]))

#create model
model = keras.Sequential()

import numpy as np



def translate(sentence, eng_tokenizer, hin_tokenizer, model):

    seq     = eng_tokenizer.texts_to_sequences([sentence])
    seq     = tf.keras.preprocessing.sequence.pad_sequences(
                  seq, maxlen=MAX_LEN, padding='post')        # (1, MAX_LEN)

    hin_idx  = hin_tokenizer.word_index
    idx_hin  = {v: k for k, v in hin_idx.items()}            # reverse lookup

    dec_input = np.zeros((1, MAX_LEN))
    dec_input[0, 0] = hin_idx['start']                       # first token is START

    result = []

    for i in range(1, MAX_LEN):
        predictions = model.predict([seq, dec_input], verbose=0)  # (1, MAX_LEN, VOCAB_SIZE)

        word_idx = np.argmax(predictions[0, i-1, :])
        word     = idx_hin.get(word_idx, '')

        if word == 'end' or word == '':
            break

        result.append(word)
        dec_input[0, i] = word_idx                           # feed predicted word back in

    print("Input  :", sentence)
    print("Output :", ' '.join(result))


translate("i love my country", eng_tokenizer, hin_tokenizer, model)

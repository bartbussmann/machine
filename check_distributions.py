file_24 = open('data/CLEANED-SCAN/length_split/increasing_lengths/24/tasks_train.txt', 'r')
file_25 = open('data/CLEANED-SCAN/length_split/increasing_lengths/25/tasks_train.txt', 'r')

all_words = []
other_words = []

for line in file_24:
    line.strip()
    words = line.split()
    for i in range(len(words) -4):
        word = words[i]
        next_word = words[i+1]
        next_word2 = words[i+2]
        next_word3 = words[i+3]
        next_word4 = words[i+4]
        tuple = (word, next_word, next_word2, next_word3, next_word4)
        if tuple not in all_words:
            all_words.append(tuple)

for line in file_25:
    line.strip()
    words = line.split()
    for i in range(len(words) -4):
        word = words[i]
        next_word = words[i+1]
        next_word2 = words[i+2]
        next_word3 = words[i+3]
        next_word4 = words[i+4]
        tuple = (word, next_word, next_word2, next_word3, next_word4)
        if tuple not in all_words:
            all_words.append(tuple)
            other_words.append(tuple)

print(other_words)
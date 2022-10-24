#!/usr/bin/env python3
# TM1: Martin Studna 55d956fd-25b4-11ec-986f-f39926f24a9c
# TM2: Roman Ruzica  2f67b427-a885-11e7-a937-00505601122b
import numpy as np


class TransformedDataset:
    @property
    def alphabet(self):
        return self._alphabet

    @property
    def text(self):
        return self._text

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._size

    def __init__(self, dataset, window, alphabet):
        self._window = window
        self._text = dataset.data
        self._size = len(dataset.data)

        # Create alphabet_map
        alphabet_map = {"<pad>": 0, "<unk>": 1}
        if not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index
        else:
            # Find most frequent characters
            freqs = {}
            for char in self._text.lower():
                freqs[char] = freqs.get(char, 0) + 1

            most_frequent = sorted(
                freqs.items(), key=lambda item: item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                alphabet_map[char] = i
                if alphabet and len(alphabet_map) >= alphabet:
                    break

        # Remap lowercased input characters using the alphabet_map
        lcletters = np.zeros(self._size + 2 * window, np.int16)
        diacritics_map = {}
        for i in range(self._size):

            if dataset.target[i] in "áéýíóúÁÉÝÍÓÚ":
                diacritics_map[dataset.target[i]] = 1
            elif dataset.target[i] in "ěščřžťňďžĚŠČŘŽŤŇĎ":
                diacritics_map[dataset.target[i]] = 2
            elif dataset.target[i] in "ůŮ":
                diacritics_map[dataset.target[i]] = 3
            else:
                diacritics_map[dataset.target[i]] = 0

            char = self._text[i].lower()
            if char not in alphabet_map:
                char = "<unk>"
            lcletters[i + window] = alphabet_map[char]

        # Generate batches data
        windows = np.zeros([self._size, 2 * window + 1], np.int16)
        labels = np.zeros((self._size, 4), np.uint16)
        for i in range(self._size):
            windows[i] = lcletters[i:i + 2 * window + 1]
            labels[i][diacritics_map[dataset.target[i]]] = 1

        self._data = {"windows": windows, "labels": labels}

        # Compute alphabet
        self._alphabet = [None] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self._alphabet[value] = key

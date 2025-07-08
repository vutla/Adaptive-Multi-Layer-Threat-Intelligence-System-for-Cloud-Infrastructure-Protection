import pickle


def save(name, value):
    with open('./Saved data/' + name + '.pkl', 'wb') as file:
        pickle.dump(value, file)


def load(name):
    with open('./Saved data/' + name + '.pkl', 'rb') as file:
        return pickle.load(file)

only_label = []
def set_only_label(label):
    global only_label 
    only_label.append(label)
def reset():
    global only_label
    only_label = []

def get_only_label(index):
    global only_label
    return only_label[index]


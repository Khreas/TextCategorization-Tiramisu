import cPickle
import os.path

def savingWeights(classifier):
    """ Saving weights function

    This function save the weights of the training (or not) ANN in a txt file
    """

    file = open('weightsValues.txt', 'wb')
    cPickle.dump(classifier.hiddenLayer.W.get_value(borrow=True),file , -1)   # the -1 is for HIGHEST_PROTOCOL and it triggers much more efficient storage than numpy's default
    cPickle.dump(classifier.hiddenLayer.b.get_value(borrow=True),file , -1)
    file.close()
    print("\n\n[Weigths saved successfully]")

def loadingWeights(self):
    """ Loading weights function

    This function load the weights of the previous saved training (or not) ANN in a txt file
    """

    if(os.path.isfile('weightsValues.txt')):
        file = open('weightsValues.txt', 'rb')
        self.hiddenLayer.W.set_value(cPickle.load(file), borrow=True)
        self.hiddenLayer.b.set_value(cPickle.load(file), borrow=True)
        file.close()
        print("\n\n[Weigths loaded succesfully]\n\n")
    else:
        print("\n\tSorry ''!\nAn error occured: the saved training ANN file has not been found.\n")
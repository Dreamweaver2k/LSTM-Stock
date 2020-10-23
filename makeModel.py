import sys
import Learning.Machine_Learning.stock_pred as get_model


def main():
    dataPath = sys.argv[1]
    savePath = sys.argv[2]
    epochs = sys.argv[3]
    batchsize = sys.argv[4]
    get_model.createModel(dataPath, savePath, epochs, batchsize)


if __name__ == '__main__':
    main()

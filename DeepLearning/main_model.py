import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
from AutoEncoder import AutoEncoder
from Cluster import ClusteringLayer
from utils.DataLoader import load_gaps
from new_ae import gen_model

import numpy as np
import metrics
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from sklearn.cluster import KMeans

import time

class URD:
    def __init__(self, ae_weights, URD_weights=None):
        '''
        URD Model 
        param ae_weights: the path to auto_encoder's weight file 
        '''
        self.ae = gen_model()
        self.ae.load_weights(ae_weights)
        self.encoder = Model(self.ae.inputs, self.ae.get_layer('encoder_output').output)
        model_input = Input(shape=(16,16,1))
        encoder_output = self.encoder(model_input)
        clustering = ClusteringLayer(2, name='clustering')(encoder_output)
        self.model = Model(model_input, clustering)
        self.model.compile(optimizer='sgd', loss="kld")
        self.model.summary()
        if URD_weights is not None:
            self.model.load_weights(URD_weights)

    def get_model(self):
        return self.model, self.ae, self.encoder

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def test_model(self):
        x, y = np.load('images/16px_image_x.npy'), np.load('images/16px_image_y.npy')
        x = np.reshape(x, (40000, 16, 16, 1))
        q = self.model.predict(x, verbose=0)
        p = self.target_distribution(q)  # update the auxiliary target distribution p
        # evaluate the clustering performance
        y_pred = q.argmax(1)

        sns.set(font_scale=3)
        confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

        plt.figure(figsize=(16, 14))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
        plt.title("Confusion matrix", fontsize=30)
        plt.ylabel('True label', fontsize=25)
        plt.xlabel('Clustering label', fontsize=25)
        plt.show()


    def train(self):
        x, y = np.load('images/x.npy'), np.load('images/y.npy')
        x = np.reshape(x, (x.shape[0], 16, 16, 1))
        kmeans = KMeans(n_clusters=2, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        loss = 0
        ae_loss = 0
        index = 0
        maxiter = 80000
        update_interval = 100
        index_array = np.arange(x.shape[0])
        batch_size = 16
        tol = 0.001

        # model.load_weights('DEC_model_final.h5')

        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q =self.model.predict(x, verbose=0)
                # update the auxiliary target distribution p
                p = self.target_distribution(q)

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f, loss=%.5f' %
                        (ite, acc, nmi, ari, loss))

                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(
                    np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            idx = np.random.randint(low=0, high=x.shape[0], size=batch_size)
            # ae_loss = ae.train_on_batch(x=x[idx], y=x[idx])
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        self.model.save_weights('DEC_model_final_x.h5')
        self.test_model()

    def speed_test(self):
        x, y = np.load('images/x.npy'), np.load('images/y.npy')
        x = np.reshape(x, (x.shape[0], 16, 16, 1))
        start = time.time()
        q = self.model.predict(x, verbose=0)
        end = time.time()
        print("%.3f" % (end - start))
        p = self.target_distribution(q)  # update the auxiliary target distribution p
        # evaluate the clustering performance
        y_pred = q.argmax(1)

        sns.set(font_scale=3)
        confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

        plt.figure(figsize=(16, 14))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
        plt.title("Confusion matrix", fontsize=30)
        plt.ylabel('True label', fontsize=25)
        plt.xlabel('Clustering label', fontsize=25)
        plt.show()


if __name__ == "__main__":
    urd = URD('models/ae_0112_new.h5')
    urd.train()
    urd.speed_test()
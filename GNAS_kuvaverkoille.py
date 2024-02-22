import random
import numpy as np
import time
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import Model
import tensorflow as tf
import gc
import sys
import logging


from tensorflow.keras.layers import Dense, Input, Conv2D, Concatenate, \
BatchNormalization, add, Multiply, Dropout, MaxPooling2D, Flatten, GlobalAveragePooling2D,\
AveragePooling2D, GlobalMaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

# Ulina pois päältä!
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Maksimimäärä ihmisiä koska dataa liian vähän vielä
MAARA = 4

"""
Nämä ovat globaaleja geneettisiä vakioita joita luokat käyttävät. Muuta tarpeen mukaan
"""
CROSSN = 3
GEENIALUSTUS = 0.5
YHDENMUTRATE = 0.04
INVAASIORAJA = 0.09
WHEEL = 42

# Kerrosten koko
KERROKSET = [4,4,4,4,4]

NNKERROKSET = [4,4]

# Optimoijan parametrit

OPTIMOIJA = 3
OPTIMOIJA_DICT = { "000" : "SGD",
                  "100" : "RMSprop",
                  "010" : "Adagrad",
                  "001" : "Adadelta",
                  "011" : "Adam",
                  "111" : "Adamax",
                  "110" : "Nadam"}

# Alustuksen parametrit
ALUSTUS = 3
ALUSTUS_DICT = {"000" : "glorot_normal",
                "100" : "glorot_uniform",
                "010" : "he_uniform",
                "001" : "lecun_uniform",
                "111" : "VarianceScaling",
                "110" : "he_normal",
                "101" : "TruncatedNormal"}

DROPPI = 4

PAALLA = 1
PAALLA_DICT = {"0" : False,
               "1" : True}

AKTIVOINTI = 3
AKTIVOINTI_DICT ={ "000" : "relu",
                    "100" : "sigmoid",
                    # "010" : "softplus",
                    # "001" : "softsign",
                    "111" : "tanh",
                    # "110" : "selu",
                   "101" : "elu"}
                    # "011" : "linear"}


KERNELI = 2
KERNELI_DICT = { "00" : 1,
                 "01" : 3,
                 "10" : 5,
                 "11" : 7}

NORMALISOINTI = 1
NORMALISOINTI_DICT = {"0": "On",
                    "1": "Off"}

POOLING = 1
POOLING_DICT = {"0": "Max",
                "1": "Avg"}

GLOBAL_POOLING = 1
GLOBAL_POOLING_DICT = {"0": "Max",
                       "1": "Avg"}


def class_acc(true_y, pred_y):

    erotus = true_y - pred_y
    eri, maara = np.unique(true_y, return_counts = True)
    eri = list(eri)

    tulokset = []


    for idx, y in enumerate(eri):

        paikat = np.where(true_y == y)
        arvaukset = erotus[paikat]
        oikein = np.where(arvaukset == 0)[0].shape

        tulokset.append(oikein / maara[idx])

    tulos = sum(tulokset) / len(tulokset)
    tulos = float(tulos)

    return tulos


def bin2logfloatnonzero(kromosomi, alkuIso, loppuIso, loppuPieni):

    ulos = 0.0

    for i in range(alkuIso, loppuIso):
        ulos += kromosomi[i] * 2**(i - alkuIso)

    for j in range(loppuIso, loppuPieni):
        ulos += kromosomi[j] * 2**(loppuIso - j - 1)

    if ulos == 0.0:
        return 1.0
    else:
        return ulos


"""
Muutta binit alle 1 floatiksi katkaisukohtien mukaan alkaen indeksistä
"""
def binPieneksi(kromosomi, alku, loppu, indeksi):

    ulos = 0.0
    for i in range(alku,loppu):
        ulos = ulos + kromosomi[i]*2.0**(alku-indeksi-i)

    return ulos

"""
Muutta binit floateiksi katkaisukohtien mukaan.
Poistaa nollatapaukset korvaamalla ne ykkösellä.
"""
def binFloatiksi(kromosomi, alkuIso, loppuIso, loppuPieni):

    isot = 0.0

    isotKerroin = binIntiksi(kromosomi, alkuIso, loppuIso)

    isot = 2**isotKerroin

    pienet = 0.0
    for i in range(loppuIso,loppuPieni):
        pienet = pienet + kromosomi[i]*2.0**(loppuIso-1-i)


    if pienet*isot == 0.0:

        if isot == 0 and pienet > 0:
            return pienet

        else:
            return 1
    if (isot*pienet < 0):
        print("vika")
        print(isot)
        print(pienet)
        print(isot*pienet)

    return isot*pienet


"""
Muuttaa binit intiksi katkaisukohtien nolla mukana
"""
def binIntiksi(kromosomi, alkuKohta, loppuKohta):

    ulos = 0
    for i in range(alkuKohta, loppuKohta):
        ulos = ulos + kromosomi[i]*2**(i-alkuKohta)

    return ulos

"""
Muuttaa binit intiksi katkaisukohtien mukaan positiivisina, ilman nollaa
"""
def binIntiksiPos(kromosomi, alkuKohta, loppuKohta):

    ulos = 1
    for i in range(alkuKohta, loppuKohta):
        ulos = ulos + kromosomi[i]*2**(i-alkuKohta)

    return int(ulos)

"""
Muutta binit 10 kantaiseksi pieneneväksi logaritmiseksi skaalaksi
"""

def binKymmenys(kromosomi, alkuKohta, loppuKohta):

    ulos = 0.0
    for i in range(alkuKohta, loppuKohta):

        ulos = ulos + kromosomi[i]*10**(alkuKohta-i)


    return ulos


def valinta(dikki, geeni):

    geeni = geeni.astype(int)
    geenistr = list(geeni.astype(str))
    geenistr = "".join(geenistr)

    if geenistr in dikki:
        return dikki[geenistr], None

    else:

        valinta = random.choice(list(dikki.keys()))
        geenilist = list(valinta)
        geenilist = [int(i) for i in geenilist]
        return dikki[valinta], geenilist


class GNAS():

    def __init__(self, n_gen,
                 size,
                 n_best,
                 n_rand,
                 input_dim,
                 n_children,
                 mutation_rate,
                 vuodet,
                 verbose = 3,
                 eliittikerroin = [],
                 verkonnimi = "perus",
                 loppusovitukset = 0,
                 loppuvuodet = 100,
                 opetusosuus = 1,
                 luokkapainot = None,
                 geenipohja = None):

        if geenipohja is not None:
            self.kaikkikromosomit = np.load(geenipohja)
        else:
            self.kaikkikromosomit = None

        self.luokkapainot = luokkapainot

        self.opetusosuus = opetusosuus

        # Dimensio inputille
        self.input_dim = input_dim

        # Kerroin eliitin lisääntymiselle
        self.eliittikerroin = eliittikerroin

        # Tulostuksen määrä
        self.verbose = verbose

        # Parhaat kromosomit per maailma
        self.kromosomis_best = []

        # Aikakauden paras
        self.parhaatverkot = []

        # Aikakausien määrä
        self.n_gen = n_gen

        # Yksilöiden määrä
        self.size = size

        # Montako parasta uuteen sukupolveen
        self.n_best = n_best

        # Montako satunnaista uuteen sukupolveen
        self.n_rand = n_rand

        self.vuodet = vuodet

        # Montako lasta per uuden sukupolven yksilö
        self.n_children = n_children

        # Mutaation todennäköisyys per yksilö
        self.mutation_rate = mutation_rate

        # Monesko sukupolvi menossa
        self.generation = 0

        # Paras löydetty kromosomi
        self.parasKromosomi = 0

        # Paras tulos
        self.paras = 0

        # Paras verkko joka fitattu
        self.parasVerkko = 0

        self.parasmaara = self.n_best[0]

        self.verkonnimi = verkonnimi

        self.yhteensa_yksiloita = 0

        self.loppusovitukset = loppusovitukset

        self.loppuvuodet = loppuvuodet

        self.n_genes = OPTIMOIJA + ALUSTUS + DROPPI + sum(KERROKSET) \
            + len(KERROKSET)*(AKTIVOINTI + KERNELI + POOLING + PAALLA + NORMALISOINTI) + GLOBAL_POOLING - 1\
                + sum(NNKERROKSET) + len(NNKERROKSET) * (PAALLA + NORMALISOINTI + AKTIVOINTI) 
        
        
        loggers = logging.StreamHandler(sys.stdout), logging.FileHandler(verkonnimi + ".log")
        logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=loggers)


    def initilize(self):

        first_generation = np.zeros((self.size, self.n_genes + 1))

        for idx in range(first_generation.shape[0]):
            first_generation[idx, 1:] = np.random.randint(2, size = self.n_genes).astype(dtype = float)

        return first_generation



    def tulostaKromosomi(self, kromosomi, score = 0):

        paikka = 0

        optimoija, geeni = valinta(OPTIMOIJA_DICT, kromosomi[paikka:paikka+OPTIMOIJA])
        if geeni != None:
            kromosomi[paikka:paikka+OPTIMOIJA] = geeni
        paikka += OPTIMOIJA

        alustus, geeni = valinta(ALUSTUS_DICT, kromosomi[paikka:paikka+ALUSTUS])
        if geeni != None:
            kromosomi[paikka:paikka+ALUSTUS] = geeni
        paikka += ALUSTUS

        droppi = binPieneksi(kromosomi, paikka, paikka+DROPPI, 1)
        paikka += DROPPI

        tulostus = f"{100*score:2.2f}% {optimoija:8} {alustus:15} D:{droppi:.2f}|"

        for idx, N in enumerate(KERROKSET):

            if idx == 0:

                neuroneita = binIntiksiPos(kromosomi, paikka, paikka+N) * 8 # Pienennys
                paikka += N

                aktivointi, geeni = valinta(AKTIVOINTI_DICT, kromosomi[paikka:paikka+AKTIVOINTI])
                if geeni != None:
                    kromosomi[paikka:paikka+AKTIVOINTI] = geeni
                paikka += AKTIVOINTI

                kerneli, geeni = valinta(KERNELI_DICT, kromosomi[paikka:paikka+KERNELI])
                if geeni != None:
                    kromosomi[paikka:paikka+KERNELI] = geeni
                paikka += KERNELI

                normalisointi, geeni = valinta(NORMALISOINTI_DICT, kromosomi[paikka:paikka+NORMALISOINTI])
                if geeni != None:
                    kromosomi[paikka:paikka+NORMALISOINTI] = geeni
                paikka += NORMALISOINTI

                pooling, geeni = valinta(POOLING_DICT, kromosomi[paikka:paikka+POOLING])
                if geeni != None:
                    kromosomi[paikka:paikka+POOLING] = geeni
                paikka += POOLING

                tulostus += f" {aktivointi:8}:{neuroneita:3}({kerneli:1})N:{normalisointi:3}:{pooling:3}|"

            else:

                paalla, geeni = valinta(PAALLA_DICT, kromosomi[paikka:paikka+PAALLA])
                if geeni != None:
                    kromosomi[paikka:paikka+PAALLA] = geeni
                paikka += PAALLA

                if not paalla:
                    paikka += N + AKTIVOINTI + KERNELI + NORMALISOINTI + POOLING
                    continue

                neuroneita = binIntiksiPos(kromosomi, paikka, paikka+N) * 4 # Pienennys
                paikka += N

                aktivointi, geeni = valinta(AKTIVOINTI_DICT, kromosomi[paikka:paikka+AKTIVOINTI])
                if geeni != None:
                    kromosomi[paikka:paikka+AKTIVOINTI] = geeni
                paikka += AKTIVOINTI

                kerneli, geeni = valinta(KERNELI_DICT, kromosomi[paikka:paikka+KERNELI])
                if geeni != None:
                    kromosomi[paikka:paikka+KERNELI] = geeni
                paikka += KERNELI

                normalisointi, geeni = valinta(NORMALISOINTI_DICT, kromosomi[paikka:paikka+NORMALISOINTI])
                if geeni != None:
                    kromosomi[paikka:paikka+NORMALISOINTI] = geeni
                paikka += NORMALISOINTI

                pooling, geeni = valinta(POOLING_DICT, kromosomi[paikka:paikka+POOLING])
                if geeni != None:
                    kromosomi[paikka:paikka+POOLING] = geeni
                paikka += POOLING

                tulostus += f"{aktivointi:8}:{neuroneita:3}({kerneli:1})N:{normalisointi:3}:{pooling:3}|"

        globalpool, geeni = valinta(GLOBAL_POOLING_DICT, kromosomi[paikka:paikka+GLOBAL_POOLING])
        if geeni != None:
            kromosomi[paikka:paikka+GLOBAL_POOLING] = geeni
        paikka += GLOBAL_POOLING

        tulostus +=  f"GP:{globalpool:3}"

        for idx, NN in enumerate(NNKERROKSET):

            if not paalla:
                paikka += NN + AKTIVOINTI + NORMALISOINTI
                continue

            paikka += PAALLA

            neuroneita = binIntiksiPos(kromosomi, paikka, paikka+NN) * 8 # Pienennys
            paikka += NN

            aktivointi, geeni = valinta(AKTIVOINTI_DICT, kromosomi[paikka:paikka+AKTIVOINTI])
            if geeni != None:
                kromosomi[paikka:paikka+AKTIVOINTI] = geeni
            paikka += AKTIVOINTI

            normalisointi, geeni = valinta(NORMALISOINTI_DICT, kromosomi[paikka:paikka+NORMALISOINTI])
            if geeni != None:
                kromosomi[paikka:paikka+NORMALISOINTI] = geeni
            paikka += NORMALISOINTI

            tulostus += f"|NN:{aktivointi:8}:{neuroneita:3}N:{normalisointi:3}"

        # print(tulostus)
        logging.info(tulostus)


    def luoUusi(self):

        kromosomi = np.zeros(self.n_genes)
        mask = np.random.rand(len(kromosomi)) < GEENIALUSTUS
        kromosomi[mask] = 1
        return kromosomi


    def luoVerkko(self, kromosomi):

        paikka = 0

        optimoija, geeni = valinta(OPTIMOIJA_DICT, kromosomi[paikka:paikka+OPTIMOIJA])
        if geeni != None:
            kromosomi[paikka:paikka+OPTIMOIJA] = geeni
        paikka += OPTIMOIJA

        alustus, geeni = valinta(ALUSTUS_DICT, kromosomi[paikka:paikka+ALUSTUS])
        if geeni != None:
            kromosomi[paikka:paikka+ALUSTUS] = geeni
        paikka += ALUSTUS

        droppi = binPieneksi(kromosomi, paikka, paikka+DROPPI, 1)
        paikka += DROPPI

        for idx, N in enumerate(KERROKSET):

            if idx == 0:

                neuroneita = binIntiksiPos(kromosomi, paikka, paikka+N) * 8 #Piennenys
                paikka += N

                aktivointi, geeni = valinta(AKTIVOINTI_DICT, kromosomi[paikka:paikka+AKTIVOINTI])
                if geeni != None:
                    kromosomi[paikka:paikka+AKTIVOINTI] = geeni
                paikka += AKTIVOINTI

                kerneli, geeni = valinta(KERNELI_DICT, kromosomi[paikka:paikka+KERNELI])
                if geeni != None:
                    kromosomi[paikka:paikka+KERNELI] = geeni
                paikka += KERNELI


                sisaan = Input(shape =  (self.input_dim[0], self.input_dim[1], self.input_dim[2]), name = 'sisaan')

                x = Conv2D(
                            filters = neuroneita,
                            kernel_size = (kerneli,kerneli),
                            padding = 'same',
                            activation = aktivointi,
                            kernel_initializer = alustus
                            )(sisaan)

                normalisointi, geeni = valinta(NORMALISOINTI_DICT, kromosomi[paikka:paikka+NORMALISOINTI])
                if geeni != None:
                    kromosomi[paikka:paikka+NORMALISOINTI] = geeni
                paikka += NORMALISOINTI

                if normalisointi == "On":
                    x = BatchNormalization()(x)

                pooling, geeni = valinta(POOLING_DICT, kromosomi[paikka:paikka+POOLING])
                if geeni != None:
                    kromosomi[paikka:paikka+POOLING] = geeni
                paikka += POOLING

                if pooling == "Max":
                    x = MaxPooling2D(pool_size = (2,2))(x)
                else:
                    x = AveragePooling2D(pool_size = (2,2))(x)


            else:

                paalla, geeni = valinta(PAALLA_DICT, kromosomi[paikka:paikka+PAALLA])
                if geeni != None:
                    kromosomi[paikka:paikka+PAALLA] = geeni
                paikka += PAALLA

                if not paalla:
                    paikka += N + AKTIVOINTI + KERNELI + NORMALISOINTI + POOLING
                    continue

                neuroneita = binIntiksiPos(kromosomi, paikka, paikka+N) * 4 # Piennennys
                paikka += N

                aktivointi, geeni = valinta(AKTIVOINTI_DICT, kromosomi[paikka:paikka+AKTIVOINTI])
                if geeni != None:
                    kromosomi[paikka:paikka+AKTIVOINTI] = geeni
                paikka += AKTIVOINTI

                kerneli, geeni = valinta(KERNELI_DICT, kromosomi[paikka:paikka+KERNELI])
                if geeni != None:
                    kromosomi[paikka:paikka+KERNELI] = geeni
                paikka += KERNELI

                x = Conv2D(
                            filters = neuroneita,
                            kernel_size = (kerneli,kerneli),
                            padding = 'same',
                            activation = aktivointi,
                            kernel_initializer = alustus
                            )(x)

                normalisointi, geeni = valinta(NORMALISOINTI_DICT, kromosomi[paikka:paikka+NORMALISOINTI])
                if geeni != None:
                    kromosomi[paikka:paikka+NORMALISOINTI] = geeni
                paikka += NORMALISOINTI

                if normalisointi == "On":
                    x = BatchNormalization()(x)

                pooling, geeni = valinta(POOLING_DICT, kromosomi[paikka:paikka+POOLING])
                if geeni != None:
                    kromosomi[paikka:paikka+POOLING] = geeni
                paikka += POOLING

                if pooling == "Max":
                    x = MaxPooling2D(pool_size = (2,2))(x)
                else:
                    x = AveragePooling2D(pool_size = (2,2))(x)


        globalpool, geeni = valinta(GLOBAL_POOLING_DICT, kromosomi[paikka:paikka+GLOBAL_POOLING])
        if geeni != None:
            kromosomi[paikka:paikka+GLOBAL_POOLING] = geeni
        paikka += GLOBAL_POOLING

        if globalpool == "Max":
            x = GlobalMaxPooling2D()(x)
        else:
            x = GlobalAveragePooling2D()(x)


        for idx, NN in enumerate(NNKERROKSET):

            if not paalla:
                paikka += NN + AKTIVOINTI + NORMALISOINTI
                continue

            paikka += PAALLA

            neuroneita = binIntiksiPos(kromosomi, paikka, paikka+NN) * 8 # Piennennys
            paikka += NN

            aktivointi, geeni = valinta(AKTIVOINTI_DICT, kromosomi[paikka:paikka+AKTIVOINTI])
            if geeni != None:
                kromosomi[paikka:paikka+AKTIVOINTI] = geeni
            paikka += AKTIVOINTI

            normalisointi, geeni = valinta(NORMALISOINTI_DICT, kromosomi[paikka:paikka+NORMALISOINTI])
            if geeni != None:
                kromosomi[paikka:paikka+NORMALISOINTI] = geeni
            paikka += NORMALISOINTI

            x = Dense(units=neuroneita,
                      activation = aktivointi)(x)

            x = Dropout(rate = droppi)(x)

            if normalisointi == "On":
                x = BatchNormalization()(x)

        ulos = Dense(
                    units = MAARA + 1,
                    activation = "softmax",
                    kernel_initializer = alustus)(x)

        verkko = Model([sisaan], [ulos])

        if optimoija == "SGD":
            optimizer = tf.keras.optimizers.SGD()
        elif optimoija == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop()
        elif optimoija == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad()
        elif optimoija == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta()
        elif optimoija == "Adam":
            optimizer = tf.keras.optimizers.Adam()
        elif optimoija == "Adamax":
            optimizer = tf.keras.optimizers.Adamax()
        else:
            optimizer = tf.keras.optimizers.Nadam()


        verkko.compile(loss = "categorical_crossentropy",
                       optimizer = optimizer,
                       metrics = ["accuracy"])


        return verkko


    def fitness(self, population):
        X_osa, y_osa = self.osadata
        X_val, y_val = self.validation
        scores = []


        datagen = ImageDataGenerator(
            horizontal_flip = True,
            vertical_flip = True,
            rotation_range = 30
            )

        for idx in range(population.shape[0]):

            self.yhteensa_yksiloita +=1

            verkko = self.luoVerkko(population[idx, 1:])

            tallennusPaikka = "parasverkko"
            tallennus = ModelCheckpoint( monitor = 'val_accuracy',
                save_best_only = True, mode = 'max', filepath = tallennusPaikka,
                verbose = 0)

            if self.generation < len(self.vuodet):
                vuosia = self.vuodet[self.generation]
            else:
                vuosia = self.vuodet[-1]

            elama = verkko.fit(datagen.flow(X_osa, y_osa, batch_size = 64),
                               validation_data = (X_val, y_val),
                               epochs = vuosia,
                               verbose = 0,
                               callbacks = [tallennus],
                               steps_per_epoch = int(np.ceil(len(X_osa) / 64)),
                               class_weight = self.luokkapainot)


            verkko = tf.keras.models.load_model("parasverkko")

            tulos = max(elama.history["val_accuracy"])
            # pred_tulos = verkko.predict(X_val)
            # pred_tulos = np.argmax(pred_tulos, axis = -1)
            # pred_oikein = np.argmax(y_val, axis = -1)
            # tulos = class_acc(pred_oikein, pred_tulos)
            
            population[idx, 0] = tulos

            if self.kaikkikromosomit is not None:
                self.kaikkikromosomit = np.vstack((self.kaikkikromosomit, population[idx]))
            else:
                self.kaikkikromosomit = population[idx]

            if self.verbose > 2:
                self.tulostaKromosomi(population[idx, 1:], tulos)

            if tulos > self.paras:

                if self.verbose > 0:
                    # print("")                    
                    # print("*** UUSI PARAS ***")
                    logging.info("\n*** NEW BEST FOUND!!! ***")
                    self.tulostaKromosomi(population[idx, 1:], tulos)
                    verkko.summary()
                    # print("")
                    

                    plt.figure(figsize = (12,8), dpi = 125)
                    plt.plot(elama.history["loss"])
                    plt.plot(elama.history["val_loss"])
                    plt.plot(elama.history["accuracy"])
                    plt.plot(elama.history["val_accuracy"])
                    plt.title(f"Parhaan verkon elämä jonka tarkkuus: {100*tulos:.2f}%")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.legend(["Opetus loss", "Validointi loss", "Opetus acc", "Validointi acc"], loc = "upper left")
                    plt.savefig(f"{self.verkonnimi}.png")

                self.parasVerkko = verkko
                tf.keras.models.save_model(verkko, self.verkonnimi)
                verkko.save(self.verkonnimi)
                self.paras = tulos
                # print(f"Uusi paras verkko tallennettu nimellä {self.verkonnimi}")
                logging.info(f"New best found neuralnetwork saved with name:  {self.verkonnimi}")
                self.parasKromosomi = population[idx]

            del verkko
            del elama
            tf.keras.backend.clear_session()
            gc.collect()

            np.save(self.verkonnimi, self.kaikkikromosomit)

        # print(type(population))
        scores = population[:,0]
        inds = np.argsort(scores, axis = 0)
        inds = np.flip(inds)
        return population[inds]


    def select(self, population_sorted):

        population_next = []

        if self.generation < len(self.n_best):
            self.parasmaara = self.n_best[self.generation]
        else:
            self.parasmaara = self.n_best[-1]

        for i in range(0, self.parasmaara):
            population_next.append(population_sorted[i])

        for i in range(0, self.n_rand):
            population_next.append(random.choice(population_sorted))

        population_next = np.array(population_next)

        return population_next


    def N_crossover(self, eka, toka, N):

        eka = np.copy(eka[1:])
        toka = np.copy(toka[1:])

        crosspoints = sorted(random.sample(range(1,eka.shape[0]-1), N))
        order = random.randint(0,1)
        result = np.ones(len(eka), dtype = int) * 42

        for idx in range(len(crosspoints)):

            if idx == 0:
                if order == 0:
                    result[0:crosspoints[idx]] = eka[0:crosspoints[idx]]
                else:
                    result[0:crosspoints[idx]] = toka[0:crosspoints[idx]]

            else:
                if order == 0:
                    result[crosspoints[idx-1]:crosspoints[idx]] = eka[crosspoints[idx-1]:crosspoints[idx]]
                else:
                    result[crosspoints[idx-1]:crosspoints[idx]] = toka[crosspoints[idx-1]:crosspoints[idx]]

            if order == 0:
                order = 1
            else:
                order = 0

        if order == 0:
           result[crosspoints[-1]:] = eka[crosspoints[-1]:]
        else:
           result[crosspoints[-1]:] = toka[crosspoints[-1]:]

        result = np.hstack((0.0, result))

        return result



    def crossover(self, population):

        population_next = []

        # Acc < 1 joten lisätään yksi
        pisteet = list(self.kaikkikromosomit[:,0])

        kaanne = [(ms+1)**WHEEL for ms in pisteet]
        maksimi = sum(kaanne)
        todnak = [ms/maksimi for ms in kaanne]

        for i in range(0, len(population)):


            if len(self.eliittikerroin) > 0 and i < len(self.eliittikerroin):

                for _ in range(self.eliittikerroin[i]):

                    isa = np.array(population[i])
                    aiti = self.kaikkikromosomit[np.random.choice(len(self.kaikkikromosomit), p = todnak)]
                    lapsi =  self.N_crossover(isa, aiti, CROSSN)
                    population_next.append(lapsi)


            else:

                for _ in range(self.n_children):

                    isa = np.array(population[i])
                    aiti = self.kaikkikromosomit[np.random.choice(len(self.kaikkikromosomit), p = todnak)]
                    lapsi = self.N_crossover(isa, aiti, CROSSN)
                    population_next.append(lapsi)

        population_next = np.array(population_next)
        return population_next

    def mutate_one(self, gene, mutprob):

        tulos = np.copy(gene[1:])
        antigene = 1 - tulos
        mask = np.random.rand(len(tulos)) < mutprob
        tulos[mask] = antigene[mask]
        tulos = np.hstack((0.0, tulos))

        return tulos


    def mutate(self, population, kaikki = False):

        population_next = []

        if kaikki == False:

            for i in range(population.shape[0]):

                kromosomi = population[i]

                if random.random() < self.mutation_rate:
                     kromosomi = self.mutate_one(kromosomi, YHDENMUTRATE)

                population_next.append(kromosomi)
        else:

            if self.verbose>1:
                # print("mutantti-invaasio!")
                logging.info("Mutant invasion!")

            for i in range(len(population)):

                kromosomi = np.array(population[i])
                kromosomi = self.mutate_one(kromosomi, YHDENMUTRATE)
                population_next.append(kromosomi)


        population_next = np.array(population_next)
        return population_next


    def generate(self, population):

        geneaika = time.time()

        # valinta, lisääntyminen, mutaatio
        population_sorted = self.fitness(population)
        # print(f"koko {population.shape} fitness")
        population = self.select(population_sorted)
        # print(f"koko {population.shape} select")
        population = self.crossover(population)
        # print(f"koko {population.shape} crossover")
        population = self.mutate(population)
        # print(f"koko {population.shape} mutation")

        # Historiaa
        self.generation += 1

        if self.verbose > 0:
            logging.info("")
            logging.info(f"Generation {self.generation:4d}")

        if self.verbose >= 2:
    
            logging.info("--- Best inviduals that continue ---")

            for i in range(self.parasmaara):
                self.tulostaKromosomi(population_sorted[i, 1:], population_sorted[i, 0])

            logging.info("... rest ...")

            for i in range(self.parasmaara, len(population_sorted)):
                self.tulostaKromosomi(population_sorted[i, 1:], population_sorted[i, 0])


        geneloppuaika = time.time()
        if self.verbose > 0:
            # print("")
            logging.info("\n")
            # print("Tulos: {:2.2f}%   Keskiarvo: {:2.2f}%   Keskihajonta: {:2.2f}%".format(100*population_sorted[0,0],100*np.mean(population_sorted[:,0]), 100*np.std(population_sorted[:,0])))
            logging.info(f"Result:{100*population_sorted[0,0]:2.2f}% Mean:{100*np.mean(population_sorted[:,0]):2.2f} SD:{100*np.std(population_sorted[:,0]):2.2f}")
            # print("Aikaa meni sukupolvelle: {:5.3f} min".format((geneloppuaika - geneaika)/60))
            logging.info(f"Time for this generation: {(geneloppuaika - geneaika)/60:5.3f} min")
            # print(f"Sukupolven koko: {len(population_sorted)}")
            logging.info(f"Generation size: {len(population_sorted)}")
            # print(f"Aikaa per yksilö {((geneloppuaika - geneaika)/60) / len(population_sorted):5.3f} min")
            logging.info(f"Time per invidual {((geneloppuaika - geneaika)/60) / len(population_sorted):5.3f} min")
            # print(f"Yksilöitä tähän asti: {self.yhteensa_yksiloita}")
            logging.info(f"Amount of inviduals {self.yhteensa_yksiloita}")            
            # print("--- --- ---")
            logging.info("--- --- ---")
            # print("")
            logging.info("\n")

        # Mutatatoidaan koko populaatio, jos alle std rajan heitto
        if np.std(population_sorted[:,0]) < INVAASIORAJA and self.generation > 3:
            population = self.mutate(population, True)

        return population


    def tuoEstimaattori(self):

        return self.parasVerkko

    def tuoParasKromosomi(self):

        return self.parasKromosomi


    def fit(self, X, y, X_val, y_val):

        self.dataset = X,y
        self.validation = X_val, y_val

        if self.opetusosuus != 1:
            pituus = int(len(X) * self.opetusosuus)
            X_osa = X[:pituus]
            y_osa = y[:pituus]
            self.osadata = X_osa, y_osa
            self.kokodata = X, y
        else:
            self.osadata = X, y
            self.kokodata = X, y

        # print(f"Geenin pituus {self.n_genes} #{2**self.n_genes:.5E}")
        logging.info(f"Gene length {self.n_genes} #{2**self.n_genes:.5E}")

        start = time.time()

        self.scores_best, self.scores_avg  = [], []

        population = self.initilize()

        for i in range(self.n_gen):
            population = self.generate(population)

        # indeksit = np.argsort(np.array(self.kaikkipisteet), axis = 0)
        # parhaat = np.array(self.kaikkikromosomit)
        # parhaat = list(parhaat[indeksit])
        # kaikkitulokset = np.array(self.kaikkipisteet)
        # kaikkitulokset = list(kaikkitulokset[indeksit])

        end = time.time()
        if self.verbose > 0:
            # print()
            logging.info("\n")
            # print("Aikaa meni yhteensä: {:5.3f} min".format((end - start)/60))
            logging.info(f"Total time spend: {(end - start)/60:5.3f} min")
            # print(f"Yksilöitä: {self.yhteensa_yksiloita}")
            logging.info(f"Amount of inviduals: {self.yhteensa_yksiloita}")
            # print(f"Aikaa per yksilö: {((end - start)/60) / self.yhteensa_yksiloita:5.3f} min")
            logging.info(f"Time per invidual: {((end - start)/60) / self.yhteensa_yksiloita:5.3f} min")
            
            print("Paras löydetty kromosomi:")
            logging.info("Best found chromosome: \n")
            self.tulostaKromosomi(self.parasKromosomi[1:], self.paras)
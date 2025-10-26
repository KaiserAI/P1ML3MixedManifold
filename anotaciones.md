- [ ] Implementar el variational autoencoder por si lo pide en la modificación de la práctica y el kae.

- [ ] Falta la comparativa entre TSNE y LLE por si solos, no con un autoencoder delante. entonces el experimento C  sería para elegir hiperparámetros de los manifold para el algoritmo mixto y un nuevo experimento D que compare TSNE y LLE por si solos, que eso tardará una barbaridad.

Entonces el experimento A es para elegir que combinación de autoencoder y manifold es la mejor.
El B es para elegir que hiperparámetros de los autoencoders son los mejores,
El C es para elegir que hiperparámetros de los manifolds son los mejores.
El D es para comparar TSNE y LLE por si solos pq lo pide él y ya.

Entonces creo que tendría más sentido hacer el B y el C primero y con esos valores configurar el A y el D.

- [ ] Tengo que poner las representaciones 2D de los experimentos.

# Latex
- Introducción.
- Estructura del Proyecto.
- Estructura de los experimentos.
- Resultados.
- Conclusiones.

justificar las 50 épocas de entreno con estos valores:
/home/walter/anaconda3/envs/P1ML3MixedManifold/bin/python /home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/exp_C_hyperparams_manifold.py 

--- Ejecutando Experimento C: Hiperparámetros del Manifold ---

==================== DATASET: MNIST (Sampling: 50.0%) ====================
   -> has_header: False. Saltando 0 fila(s).
Cargando train data desde: /home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/../data/MNIST/mnist_train.csv...
/home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/experiment_utils.py:41: DtypeWarning: Columns (0) have mixed types. Specify dtype option on import or set low_memory=False.
  df = pd.read_csv(full_path, header=None, skiprows=skip_rows)
Muestreo aplicado: 50.0%. Usando 30000 muestras.
Datos cargados (Shape: (30000, 784))
   -> has_header: False. Saltando 0 fila(s).
Cargando test data desde: /home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/../data/MNIST/mnist_test.csv...
/home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/experiment_utils.py:41: DtypeWarning: Columns (0) have mixed types. Specify dtype option on import or set low_memory=False.
  df = pd.read_csv(full_path, header=None, skiprows=skip_rows)
Muestreo aplicado: 50.0%. Usando 5000 muestras.
Datos cargados (Shape: (5000, 784))
INPUT_DIM detectado: 784

-> C.1: Explorando TSNE (Perplexity)
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=5
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.047097
Epoch 2/50, Loss: 0.026634
Epoch 3/50, Loss: 0.022230
Epoch 4/50, Loss: 0.019504
Epoch 5/50, Loss: 0.017887
Epoch 6/50, Loss: 0.016811
Epoch 7/50, Loss: 0.016082
Epoch 8/50, Loss: 0.015491
Epoch 9/50, Loss: 0.014963
Epoch 10/50, Loss: 0.014591
Epoch 11/50, Loss: 0.014251
Epoch 12/50, Loss: 0.013968
Epoch 13/50, Loss: 0.013710
Epoch 14/50, Loss: 0.013475
Epoch 15/50, Loss: 0.013274
Epoch 16/50, Loss: 0.013091
Epoch 17/50, Loss: 0.012923
Epoch 18/50, Loss: 0.012801
Epoch 19/50, Loss: 0.012667
Epoch 20/50, Loss: 0.012563
Epoch 21/50, Loss: 0.012448
Epoch 22/50, Loss: 0.012354
Epoch 23/50, Loss: 0.012246
Epoch 24/50, Loss: 0.012175
Epoch 25/50, Loss: 0.012098
Epoch 26/50, Loss: 0.012028
Epoch 27/50, Loss: 0.011955
Epoch 28/50, Loss: 0.011874
Epoch 29/50, Loss: 0.011814
Epoch 30/50, Loss: 0.011759
Epoch 31/50, Loss: 0.011688
Epoch 32/50, Loss: 0.011632
Epoch 33/50, Loss: 0.011587
Epoch 34/50, Loss: 0.011516
Epoch 35/50, Loss: 0.011476
Epoch 36/50, Loss: 0.011415
Epoch 37/50, Loss: 0.011384
Epoch 38/50, Loss: 0.011348
Epoch 39/50, Loss: 0.011309
Epoch 40/50, Loss: 0.011273
Epoch 41/50, Loss: 0.011231
Epoch 42/50, Loss: 0.011182
Epoch 43/50, Loss: 0.011166
Epoch 44/50, Loss: 0.011126
Epoch 45/50, Loss: 0.011095
Epoch 46/50, Loss: 0.011061
Epoch 47/50, Loss: 0.011028
Epoch 48/50, Loss: 0.010988
Epoch 49/50, Loss: 0.010971
Epoch 50/50, Loss: 0.010934
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=30
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.047214
Epoch 2/50, Loss: 0.026895
Epoch 3/50, Loss: 0.021694
Epoch 4/50, Loss: 0.019295
Epoch 5/50, Loss: 0.017626
Epoch 6/50, Loss: 0.016469
Epoch 7/50, Loss: 0.015627
Epoch 8/50, Loss: 0.014989
Epoch 9/50, Loss: 0.014541
Epoch 10/50, Loss: 0.014112
Epoch 11/50, Loss: 0.013770
Epoch 12/50, Loss: 0.013483
Epoch 13/50, Loss: 0.013231
Epoch 14/50, Loss: 0.012990
Epoch 15/50, Loss: 0.012785
Epoch 16/50, Loss: 0.012589
Epoch 17/50, Loss: 0.012440
Epoch 18/50, Loss: 0.012282
Epoch 19/50, Loss: 0.012156
Epoch 20/50, Loss: 0.012023
Epoch 21/50, Loss: 0.011911
Epoch 22/50, Loss: 0.011818
Epoch 23/50, Loss: 0.011734
Epoch 24/50, Loss: 0.011674
Epoch 25/50, Loss: 0.011583
Epoch 26/50, Loss: 0.011517
Epoch 27/50, Loss: 0.011450
Epoch 28/50, Loss: 0.011393
Epoch 29/50, Loss: 0.011328
Epoch 30/50, Loss: 0.011282
Epoch 31/50, Loss: 0.011239
Epoch 32/50, Loss: 0.011183
Epoch 33/50, Loss: 0.011120
Epoch 34/50, Loss: 0.011085
Epoch 35/50, Loss: 0.011045
Epoch 36/50, Loss: 0.011008
Epoch 37/50, Loss: 0.010957
Epoch 38/50, Loss: 0.010917
Epoch 39/50, Loss: 0.010896
Epoch 40/50, Loss: 0.010845
Epoch 41/50, Loss: 0.010826
Epoch 42/50, Loss: 0.010798
Epoch 43/50, Loss: 0.010749
Epoch 44/50, Loss: 0.010722
Epoch 45/50, Loss: 0.010698
Epoch 46/50, Loss: 0.010651
Epoch 47/50, Loss: 0.010631
Epoch 48/50, Loss: 0.010608
Epoch 49/50, Loss: 0.010568
Epoch 50/50, Loss: 0.010554
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=50
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.046362
Epoch 2/50, Loss: 0.026115
Epoch 3/50, Loss: 0.021351
Epoch 4/50, Loss: 0.019025
Epoch 5/50, Loss: 0.017531
Epoch 6/50, Loss: 0.016395
Epoch 7/50, Loss: 0.015599
Epoch 8/50, Loss: 0.015020
Epoch 9/50, Loss: 0.014552
Epoch 10/50, Loss: 0.014158
Epoch 11/50, Loss: 0.013844
Epoch 12/50, Loss: 0.013565
Epoch 13/50, Loss: 0.013347
Epoch 14/50, Loss: 0.013155
Epoch 15/50, Loss: 0.012970
Epoch 16/50, Loss: 0.012800
Epoch 17/50, Loss: 0.012673
Epoch 18/50, Loss: 0.012563
Epoch 19/50, Loss: 0.012436
Epoch 20/50, Loss: 0.012311
Epoch 21/50, Loss: 0.012219
Epoch 22/50, Loss: 0.012142
Epoch 23/50, Loss: 0.012047
Epoch 24/50, Loss: 0.011958
Epoch 25/50, Loss: 0.011907
Epoch 26/50, Loss: 0.011816
Epoch 27/50, Loss: 0.011749
Epoch 28/50, Loss: 0.011689
Epoch 29/50, Loss: 0.011625
Epoch 30/50, Loss: 0.011574
Epoch 31/50, Loss: 0.011500
Epoch 32/50, Loss: 0.011445
Epoch 33/50, Loss: 0.011406
Epoch 34/50, Loss: 0.011348
Epoch 35/50, Loss: 0.011293
Epoch 36/50, Loss: 0.011265
Epoch 37/50, Loss: 0.011221
Epoch 38/50, Loss: 0.011192
Epoch 39/50, Loss: 0.011147
Epoch 40/50, Loss: 0.011101
Epoch 41/50, Loss: 0.011085
Epoch 42/50, Loss: 0.011053
Epoch 43/50, Loss: 0.011014
Epoch 44/50, Loss: 0.011004
Epoch 45/50, Loss: 0.010982
Epoch 46/50, Loss: 0.010923
Epoch 47/50, Loss: 0.010915
Epoch 48/50, Loss: 0.010905
Epoch 49/50, Loss: 0.010877
Epoch 50/50, Loss: 0.010859

-> C.2: Explorando LLE (n_neighbors)
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=10
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.046576
Epoch 2/50, Loss: 0.026167
Epoch 3/50, Loss: 0.021601
Epoch 4/50, Loss: 0.019331
Epoch 5/50, Loss: 0.017861
Epoch 6/50, Loss: 0.016822
Epoch 7/50, Loss: 0.015949
Epoch 8/50, Loss: 0.015292
Epoch 9/50, Loss: 0.014813
Epoch 10/50, Loss: 0.014437
Epoch 11/50, Loss: 0.014158
Epoch 12/50, Loss: 0.013907
Epoch 13/50, Loss: 0.013700
Epoch 14/50, Loss: 0.013508
Epoch 15/50, Loss: 0.013346
Epoch 16/50, Loss: 0.013195
Epoch 17/50, Loss: 0.013024
Epoch 18/50, Loss: 0.012881
Epoch 19/50, Loss: 0.012737
Epoch 20/50, Loss: 0.012625
Epoch 21/50, Loss: 0.012534
Epoch 22/50, Loss: 0.012413
Epoch 23/50, Loss: 0.012335
Epoch 24/50, Loss: 0.012250
Epoch 25/50, Loss: 0.012148
Epoch 26/50, Loss: 0.012074
Epoch 27/50, Loss: 0.012047
Epoch 28/50, Loss: 0.011933
Epoch 29/50, Loss: 0.011875
Epoch 30/50, Loss: 0.011822
Epoch 31/50, Loss: 0.011771
Epoch 32/50, Loss: 0.011704
Epoch 33/50, Loss: 0.011652
Epoch 34/50, Loss: 0.011599
Epoch 35/50, Loss: 0.011547
Epoch 36/50, Loss: 0.011500
Epoch 37/50, Loss: 0.011459
Epoch 38/50, Loss: 0.011413
Epoch 39/50, Loss: 0.011372
Epoch 40/50, Loss: 0.011326
Epoch 41/50, Loss: 0.011283
Epoch 42/50, Loss: 0.011256
Epoch 43/50, Loss: 0.011210
Epoch 44/50, Loss: 0.011156
Epoch 45/50, Loss: 0.011147
Epoch 46/50, Loss: 0.011116
Epoch 47/50, Loss: 0.011087
Epoch 48/50, Loss: 0.011046
Epoch 49/50, Loss: 0.011023
Epoch 50/50, Loss: 0.010992
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=30
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.045843
Epoch 2/50, Loss: 0.026001
Epoch 3/50, Loss: 0.021565
Epoch 4/50, Loss: 0.019395
Epoch 5/50, Loss: 0.017899
Epoch 6/50, Loss: 0.016833
Epoch 7/50, Loss: 0.016040
Epoch 8/50, Loss: 0.015385
Epoch 9/50, Loss: 0.014810
Epoch 10/50, Loss: 0.014364
Epoch 11/50, Loss: 0.013986
Epoch 12/50, Loss: 0.013672
Epoch 13/50, Loss: 0.013378
Epoch 14/50, Loss: 0.013111
Epoch 15/50, Loss: 0.012894
Epoch 16/50, Loss: 0.012695
Epoch 17/50, Loss: 0.012530
Epoch 18/50, Loss: 0.012384
Epoch 19/50, Loss: 0.012232
Epoch 20/50, Loss: 0.012121
Epoch 21/50, Loss: 0.012016
Epoch 22/50, Loss: 0.011902
Epoch 23/50, Loss: 0.011820
Epoch 24/50, Loss: 0.011741
Epoch 25/50, Loss: 0.011654
Epoch 26/50, Loss: 0.011581
Epoch 27/50, Loss: 0.011511
Epoch 28/50, Loss: 0.011438
Epoch 29/50, Loss: 0.011358
Epoch 30/50, Loss: 0.011313
Epoch 31/50, Loss: 0.011270
Epoch 32/50, Loss: 0.011204
Epoch 33/50, Loss: 0.011137
Epoch 34/50, Loss: 0.011094
Epoch 35/50, Loss: 0.011053
Epoch 36/50, Loss: 0.010989
Epoch 37/50, Loss: 0.010949
Epoch 38/50, Loss: 0.010895
Epoch 39/50, Loss: 0.010849
Epoch 40/50, Loss: 0.010833
Epoch 41/50, Loss: 0.010776
Epoch 42/50, Loss: 0.010749
Epoch 43/50, Loss: 0.010720
Epoch 44/50, Loss: 0.010676
Epoch 45/50, Loss: 0.010645
Epoch 46/50, Loss: 0.010605
Epoch 47/50, Loss: 0.010580
Epoch 48/50, Loss: 0.010531
Epoch 49/50, Loss: 0.010519
Epoch 50/50, Loss: 0.010482
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=50
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.048420
Epoch 2/50, Loss: 0.027648
Epoch 3/50, Loss: 0.022417
Epoch 4/50, Loss: 0.019967
Epoch 5/50, Loss: 0.018392
Epoch 6/50, Loss: 0.017236
Epoch 7/50, Loss: 0.016354
Epoch 8/50, Loss: 0.015648
Epoch 9/50, Loss: 0.015130
Epoch 10/50, Loss: 0.014681
Epoch 11/50, Loss: 0.014289
Epoch 12/50, Loss: 0.013955
Epoch 13/50, Loss: 0.013652
Epoch 14/50, Loss: 0.013434
Epoch 15/50, Loss: 0.013199
Epoch 16/50, Loss: 0.013009
Epoch 17/50, Loss: 0.012841
Epoch 18/50, Loss: 0.012631
Epoch 19/50, Loss: 0.012487
Epoch 20/50, Loss: 0.012330
Epoch 21/50, Loss: 0.012218
Epoch 22/50, Loss: 0.012082
Epoch 23/50, Loss: 0.012003
Epoch 24/50, Loss: 0.011906
Epoch 25/50, Loss: 0.011829
Epoch 26/50, Loss: 0.011731
Epoch 27/50, Loss: 0.011677
Epoch 28/50, Loss: 0.011612
Epoch 29/50, Loss: 0.011556
Epoch 30/50, Loss: 0.011501
Epoch 31/50, Loss: 0.011414
Epoch 32/50, Loss: 0.011372
Epoch 33/50, Loss: 0.011313
Epoch 34/50, Loss: 0.011262
Epoch 35/50, Loss: 0.011217
Epoch 36/50, Loss: 0.011175
Epoch 37/50, Loss: 0.011142
Epoch 38/50, Loss: 0.011112
Epoch 39/50, Loss: 0.011043
Epoch 40/50, Loss: 0.011012
Epoch 41/50, Loss: 0.010973
Epoch 42/50, Loss: 0.010942
Epoch 43/50, Loss: 0.010909
Epoch 44/50, Loss: 0.010871
Epoch 45/50, Loss: 0.010846
Epoch 46/50, Loss: 0.010828
Epoch 47/50, Loss: 0.010781
Epoch 48/50, Loss: 0.010761
Epoch 49/50, Loss: 0.010729
Epoch 50/50, Loss: 0.010691

==================== DATASET: FashionMNIST (Sampling: 50.0%) ====================
   -> has_header: True. Saltando 1 fila(s).
Cargando train data desde: /home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/../data/FashionMNIST/fashion_mnist_train.csv...
Muestreo aplicado: 50.0%. Usando 30000 muestras.
Datos cargados (Shape: (30000, 784))
   -> has_header: True. Saltando 1 fila(s).
Cargando test data desde: /home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/../data/FashionMNIST/fashion_mnist_test.csv...
Muestreo aplicado: 50.0%. Usando 5000 muestras.
Datos cargados (Shape: (5000, 784))
INPUT_DIM detectado: 784

-> C.1: Explorando TSNE (Perplexity)
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=5
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.044240
Epoch 2/50, Loss: 0.023416
Epoch 3/50, Loss: 0.020599
Epoch 4/50, Loss: 0.018529
Epoch 5/50, Loss: 0.017250
Epoch 6/50, Loss: 0.016427
Epoch 7/50, Loss: 0.015752
Epoch 8/50, Loss: 0.015159
Epoch 9/50, Loss: 0.014727
Epoch 10/50, Loss: 0.014332
Epoch 11/50, Loss: 0.013978
Epoch 12/50, Loss: 0.013706
Epoch 13/50, Loss: 0.013460
Epoch 14/50, Loss: 0.013243
Epoch 15/50, Loss: 0.013059
Epoch 16/50, Loss: 0.012911
Epoch 17/50, Loss: 0.012722
Epoch 18/50, Loss: 0.012573
Epoch 19/50, Loss: 0.012436
Epoch 20/50, Loss: 0.012311
Epoch 21/50, Loss: 0.012203
Epoch 22/50, Loss: 0.012117
Epoch 23/50, Loss: 0.011988
Epoch 24/50, Loss: 0.011901
Epoch 25/50, Loss: 0.011843
Epoch 26/50, Loss: 0.011742
Epoch 27/50, Loss: 0.011646
Epoch 28/50, Loss: 0.011589
Epoch 29/50, Loss: 0.011531
Epoch 30/50, Loss: 0.011464
Epoch 31/50, Loss: 0.011410
Epoch 32/50, Loss: 0.011331
Epoch 33/50, Loss: 0.011290
Epoch 34/50, Loss: 0.011243
Epoch 35/50, Loss: 0.011188
Epoch 36/50, Loss: 0.011142
Epoch 37/50, Loss: 0.011105
Epoch 38/50, Loss: 0.011036
Epoch 39/50, Loss: 0.011037
Epoch 40/50, Loss: 0.010992
Epoch 41/50, Loss: 0.010949
Epoch 42/50, Loss: 0.010932
Epoch 43/50, Loss: 0.010879
Epoch 44/50, Loss: 0.010858
Epoch 45/50, Loss: 0.010848
Epoch 46/50, Loss: 0.010810
Epoch 47/50, Loss: 0.010767
Epoch 48/50, Loss: 0.010781
Epoch 49/50, Loss: 0.010740
Epoch 50/50, Loss: 0.010710
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=30
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.046964
Epoch 2/50, Loss: 0.024027
Epoch 3/50, Loss: 0.020696
Epoch 4/50, Loss: 0.018658
Epoch 5/50, Loss: 0.017468
Epoch 6/50, Loss: 0.016587
Epoch 7/50, Loss: 0.015913
Epoch 8/50, Loss: 0.015344
Epoch 9/50, Loss: 0.014872
Epoch 10/50, Loss: 0.014502
Epoch 11/50, Loss: 0.014156
Epoch 12/50, Loss: 0.013869
Epoch 13/50, Loss: 0.013598
Epoch 14/50, Loss: 0.013385
Epoch 15/50, Loss: 0.013192
Epoch 16/50, Loss: 0.012955
Epoch 17/50, Loss: 0.012814
Epoch 18/50, Loss: 0.012646
Epoch 19/50, Loss: 0.012517
Epoch 20/50, Loss: 0.012367
Epoch 21/50, Loss: 0.012250
Epoch 22/50, Loss: 0.012144
Epoch 23/50, Loss: 0.012049
Epoch 24/50, Loss: 0.011935
Epoch 25/50, Loss: 0.011825
Epoch 26/50, Loss: 0.011762
Epoch 27/50, Loss: 0.011673
Epoch 28/50, Loss: 0.011575
Epoch 29/50, Loss: 0.011520
Epoch 30/50, Loss: 0.011447
Epoch 31/50, Loss: 0.011376
Epoch 32/50, Loss: 0.011329
Epoch 33/50, Loss: 0.011272
Epoch 34/50, Loss: 0.011216
Epoch 35/50, Loss: 0.011173
Epoch 36/50, Loss: 0.011132
Epoch 37/50, Loss: 0.011084
Epoch 38/50, Loss: 0.011055
Epoch 39/50, Loss: 0.011024
Epoch 40/50, Loss: 0.010962
Epoch 41/50, Loss: 0.010951
Epoch 42/50, Loss: 0.010928
Epoch 43/50, Loss: 0.010893
Epoch 44/50, Loss: 0.010858
Epoch 45/50, Loss: 0.010843
Epoch 46/50, Loss: 0.010802
Epoch 47/50, Loss: 0.010787
Epoch 48/50, Loss: 0.010743
Epoch 49/50, Loss: 0.010723
Epoch 50/50, Loss: 0.010722
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=50
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.045200
Epoch 2/50, Loss: 0.023898
Epoch 3/50, Loss: 0.020380
Epoch 4/50, Loss: 0.018598
Epoch 5/50, Loss: 0.017379
Epoch 6/50, Loss: 0.016473
Epoch 7/50, Loss: 0.015774
Epoch 8/50, Loss: 0.015215
Epoch 9/50, Loss: 0.014773
Epoch 10/50, Loss: 0.014397
Epoch 11/50, Loss: 0.014069
Epoch 12/50, Loss: 0.013768
Epoch 13/50, Loss: 0.013534
Epoch 14/50, Loss: 0.013312
Epoch 15/50, Loss: 0.013140
Epoch 16/50, Loss: 0.012944
Epoch 17/50, Loss: 0.012776
Epoch 18/50, Loss: 0.012611
Epoch 19/50, Loss: 0.012475
Epoch 20/50, Loss: 0.012333
Epoch 21/50, Loss: 0.012229
Epoch 22/50, Loss: 0.012107
Epoch 23/50, Loss: 0.012007
Epoch 24/50, Loss: 0.011908
Epoch 25/50, Loss: 0.011804
Epoch 26/50, Loss: 0.011754
Epoch 27/50, Loss: 0.011661
Epoch 28/50, Loss: 0.011600
Epoch 29/50, Loss: 0.011523
Epoch 30/50, Loss: 0.011444
Epoch 31/50, Loss: 0.011385
Epoch 32/50, Loss: 0.011334
Epoch 33/50, Loss: 0.011273
Epoch 34/50, Loss: 0.011230
Epoch 35/50, Loss: 0.011182
Epoch 36/50, Loss: 0.011127
Epoch 37/50, Loss: 0.011107
Epoch 38/50, Loss: 0.011053
Epoch 39/50, Loss: 0.011017
Epoch 40/50, Loss: 0.010986
Epoch 41/50, Loss: 0.010950
Epoch 42/50, Loss: 0.010932
Epoch 43/50, Loss: 0.010880
Epoch 44/50, Loss: 0.010862
Epoch 45/50, Loss: 0.010857
Epoch 46/50, Loss: 0.010799
Epoch 47/50, Loss: 0.010785
Epoch 48/50, Loss: 0.010752
Epoch 49/50, Loss: 0.010740
Epoch 50/50, Loss: 0.010714

-> C.2: Explorando LLE (n_neighbors)
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=10
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.045344
Epoch 2/50, Loss: 0.023521
Epoch 3/50, Loss: 0.020728
Epoch 4/50, Loss: 0.018717
Epoch 5/50, Loss: 0.017385
Epoch 6/50, Loss: 0.016466
Epoch 7/50, Loss: 0.015833
Epoch 8/50, Loss: 0.015321
Epoch 9/50, Loss: 0.014866
Epoch 10/50, Loss: 0.014477
Epoch 11/50, Loss: 0.014142
Epoch 12/50, Loss: 0.013842
Epoch 13/50, Loss: 0.013619
Epoch 14/50, Loss: 0.013407
Epoch 15/50, Loss: 0.013177
Epoch 16/50, Loss: 0.013000
Epoch 17/50, Loss: 0.012815
Epoch 18/50, Loss: 0.012679
Epoch 19/50, Loss: 0.012526
Epoch 20/50, Loss: 0.012399
Epoch 21/50, Loss: 0.012265
Epoch 22/50, Loss: 0.012157
Epoch 23/50, Loss: 0.012053
Epoch 24/50, Loss: 0.011954
Epoch 25/50, Loss: 0.011855
Epoch 26/50, Loss: 0.011779
Epoch 27/50, Loss: 0.011696
Epoch 28/50, Loss: 0.011630
Epoch 29/50, Loss: 0.011555
Epoch 30/50, Loss: 0.011496
Epoch 31/50, Loss: 0.011451
Epoch 32/50, Loss: 0.011406
Epoch 33/50, Loss: 0.011349
Epoch 34/50, Loss: 0.011277
Epoch 35/50, Loss: 0.011245
Epoch 36/50, Loss: 0.011191
Epoch 37/50, Loss: 0.011153
Epoch 38/50, Loss: 0.011115
Epoch 39/50, Loss: 0.011098
Epoch 40/50, Loss: 0.011066
Epoch 41/50, Loss: 0.011023
Epoch 42/50, Loss: 0.010989
Epoch 43/50, Loss: 0.010964
Epoch 44/50, Loss: 0.010926
Epoch 45/50, Loss: 0.010910
Epoch 46/50, Loss: 0.010872
Epoch 47/50, Loss: 0.010845
Epoch 48/50, Loss: 0.010832
Epoch 49/50, Loss: 0.010786
Epoch 50/50, Loss: 0.010786
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=30
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.046957
Epoch 2/50, Loss: 0.023314
Epoch 3/50, Loss: 0.020368
Epoch 4/50, Loss: 0.018539
Epoch 5/50, Loss: 0.017288
Epoch 6/50, Loss: 0.016415
Epoch 7/50, Loss: 0.015712
Epoch 8/50, Loss: 0.015198
Epoch 9/50, Loss: 0.014760
Epoch 10/50, Loss: 0.014399
Epoch 11/50, Loss: 0.014039
Epoch 12/50, Loss: 0.013734
Epoch 13/50, Loss: 0.013449
Epoch 14/50, Loss: 0.013222
Epoch 15/50, Loss: 0.013024
Epoch 16/50, Loss: 0.012856
Epoch 17/50, Loss: 0.012727
Epoch 18/50, Loss: 0.012578
Epoch 19/50, Loss: 0.012464
Epoch 20/50, Loss: 0.012342
Epoch 21/50, Loss: 0.012237
Epoch 22/50, Loss: 0.012110
Epoch 23/50, Loss: 0.012040
Epoch 24/50, Loss: 0.011929
Epoch 25/50, Loss: 0.011852
Epoch 26/50, Loss: 0.011747
Epoch 27/50, Loss: 0.011660
Epoch 28/50, Loss: 0.011585
Epoch 29/50, Loss: 0.011509
Epoch 30/50, Loss: 0.011442
Epoch 31/50, Loss: 0.011396
Epoch 32/50, Loss: 0.011330
Epoch 33/50, Loss: 0.011259
Epoch 34/50, Loss: 0.011200
Epoch 35/50, Loss: 0.011153
Epoch 36/50, Loss: 0.011129
Epoch 37/50, Loss: 0.011070
Epoch 38/50, Loss: 0.011038
Epoch 39/50, Loss: 0.010995
Epoch 40/50, Loss: 0.010960
Epoch 41/50, Loss: 0.010943
Epoch 42/50, Loss: 0.010895
Epoch 43/50, Loss: 0.010861
Epoch 44/50, Loss: 0.010833
Epoch 45/50, Loss: 0.010839
Epoch 46/50, Loss: 0.010772
Epoch 47/50, Loss: 0.010769
Epoch 48/50, Loss: 0.010730
Epoch 49/50, Loss: 0.010710
Epoch 50/50, Loss: 0.010716
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=50
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.045862
Epoch 2/50, Loss: 0.023853
Epoch 3/50, Loss: 0.020936
Epoch 4/50, Loss: 0.019015
Epoch 5/50, Loss: 0.017722
Epoch 6/50, Loss: 0.016822
Epoch 7/50, Loss: 0.016099
Epoch 8/50, Loss: 0.015562
Epoch 9/50, Loss: 0.015074
Epoch 10/50, Loss: 0.014674
Epoch 11/50, Loss: 0.014315
Epoch 12/50, Loss: 0.014018
Epoch 13/50, Loss: 0.013758
Epoch 14/50, Loss: 0.013521
Epoch 15/50, Loss: 0.013313
Epoch 16/50, Loss: 0.013112
Epoch 17/50, Loss: 0.012958
Epoch 18/50, Loss: 0.012787
Epoch 19/50, Loss: 0.012616
Epoch 20/50, Loss: 0.012500
Epoch 21/50, Loss: 0.012367
Epoch 22/50, Loss: 0.012225
Epoch 23/50, Loss: 0.012124
Epoch 24/50, Loss: 0.012038
Epoch 25/50, Loss: 0.011928
Epoch 26/50, Loss: 0.011824
Epoch 27/50, Loss: 0.011778
Epoch 28/50, Loss: 0.011675
Epoch 29/50, Loss: 0.011605
Epoch 30/50, Loss: 0.011509
Epoch 31/50, Loss: 0.011457
Epoch 32/50, Loss: 0.011406
Epoch 33/50, Loss: 0.011342
Epoch 34/50, Loss: 0.011283
Epoch 35/50, Loss: 0.011238
Epoch 36/50, Loss: 0.011192
Epoch 37/50, Loss: 0.011152
Epoch 38/50, Loss: 0.011108
Epoch 39/50, Loss: 0.011063
Epoch 40/50, Loss: 0.011026
Epoch 41/50, Loss: 0.010999
Epoch 42/50, Loss: 0.010974
Epoch 43/50, Loss: 0.010941
Epoch 44/50, Loss: 0.010913
Epoch 45/50, Loss: 0.010878
Epoch 46/50, Loss: 0.010829
Epoch 47/50, Loss: 0.010814
Epoch 48/50, Loss: 0.010780
Epoch 49/50, Loss: 0.010752
Epoch 50/50, Loss: 0.010722

==================== DATASET: Cifar10 (Sampling: 50.0%) ====================
   -> has_header: False. Saltando 0 fila(s).
Cargando train data desde: /home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/../data/Cifar10/cifar10_train.csv...
Muestreo aplicado: 50.0%. Usando 25000 muestras.
Datos cargados (Shape: (25000, 3072))
   -> has_header: False. Saltando 0 fila(s).
Cargando test data desde: /home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/../data/Cifar10/cifar10_test.csv...
Muestreo aplicado: 50.0%. Usando 5000 muestras.
Datos cargados (Shape: (5000, 3072))
INPUT_DIM detectado: 3072

-> C.1: Explorando TSNE (Perplexity)
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=5
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.040609
Epoch 2/50, Loss: 0.024351
Epoch 3/50, Loss: 0.021349
Epoch 4/50, Loss: 0.019209
Epoch 5/50, Loss: 0.017700
Epoch 6/50, Loss: 0.016970
Epoch 7/50, Loss: 0.016302
Epoch 8/50, Loss: 0.015636
Epoch 9/50, Loss: 0.015202
Epoch 10/50, Loss: 0.014690
Epoch 11/50, Loss: 0.014197
Epoch 12/50, Loss: 0.014075
Epoch 13/50, Loss: 0.013956
Epoch 14/50, Loss: 0.013797
Epoch 15/50, Loss: 0.013628
Epoch 16/50, Loss: 0.013327
Epoch 17/50, Loss: 0.013258
Epoch 18/50, Loss: 0.013068
Epoch 19/50, Loss: 0.013024
Epoch 20/50, Loss: 0.012895
Epoch 21/50, Loss: 0.012866
Epoch 22/50, Loss: 0.012846
Epoch 23/50, Loss: 0.012884
Epoch 24/50, Loss: 0.012772
Epoch 25/50, Loss: 0.012762
Epoch 26/50, Loss: 0.012723
Epoch 27/50, Loss: 0.012738
Epoch 28/50, Loss: 0.012734
Epoch 29/50, Loss: 0.012632
Epoch 30/50, Loss: 0.012657
Epoch 31/50, Loss: 0.012571
Epoch 32/50, Loss: 0.012602
Epoch 33/50, Loss: 0.012572
Epoch 34/50, Loss: 0.012524
Epoch 35/50, Loss: 0.012531
Epoch 36/50, Loss: 0.012468
Epoch 37/50, Loss: 0.012475
Epoch 38/50, Loss: 0.012521
Epoch 39/50, Loss: 0.012390
Epoch 40/50, Loss: 0.012395
Epoch 41/50, Loss: 0.012402
Epoch 42/50, Loss: 0.012345
Epoch 43/50, Loss: 0.012364
Epoch 44/50, Loss: 0.012388
Epoch 45/50, Loss: 0.012233
Epoch 46/50, Loss: 0.012272
Epoch 47/50, Loss: 0.012256
Epoch 48/50, Loss: 0.012273
Epoch 49/50, Loss: 0.012199
Epoch 50/50, Loss: 0.012169
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=30
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.044073
Epoch 2/50, Loss: 0.027840
Epoch 3/50, Loss: 0.022205
Epoch 4/50, Loss: 0.020386
Epoch 5/50, Loss: 0.018619
Epoch 6/50, Loss: 0.017626
Epoch 7/50, Loss: 0.016973
Epoch 8/50, Loss: 0.016152
Epoch 9/50, Loss: 0.015810
Epoch 10/50, Loss: 0.015080
Epoch 11/50, Loss: 0.015006
Epoch 12/50, Loss: 0.014482
Epoch 13/50, Loss: 0.014326
Epoch 14/50, Loss: 0.013913
Epoch 15/50, Loss: 0.013776
Epoch 16/50, Loss: 0.013590
Epoch 17/50, Loss: 0.013455
Epoch 18/50, Loss: 0.013469
Epoch 19/50, Loss: 0.013247
Epoch 20/50, Loss: 0.013110
Epoch 21/50, Loss: 0.013050
Epoch 22/50, Loss: 0.013079
Epoch 23/50, Loss: 0.012904
Epoch 24/50, Loss: 0.012875
Epoch 25/50, Loss: 0.012862
Epoch 26/50, Loss: 0.012825
Epoch 27/50, Loss: 0.012764
Epoch 28/50, Loss: 0.012828
Epoch 29/50, Loss: 0.012765
Epoch 30/50, Loss: 0.012690
Epoch 31/50, Loss: 0.012704
Epoch 32/50, Loss: 0.012710
Epoch 33/50, Loss: 0.012795
Epoch 34/50, Loss: 0.012574
Epoch 35/50, Loss: 0.012569
Epoch 36/50, Loss: 0.012582
Epoch 37/50, Loss: 0.012564
Epoch 38/50, Loss: 0.012582
Epoch 39/50, Loss: 0.012550
Epoch 40/50, Loss: 0.012468
Epoch 41/50, Loss: 0.012735
Epoch 42/50, Loss: 0.012403
Epoch 43/50, Loss: 0.012421
Epoch 44/50, Loss: 0.012370
Epoch 45/50, Loss: 0.012394
Epoch 46/50, Loss: 0.012378
Epoch 47/50, Loss: 0.012355
Epoch 48/50, Loss: 0.012350
Epoch 49/50, Loss: 0.012327
Epoch 50/50, Loss: 0.012329
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=50
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.048101
Epoch 2/50, Loss: 0.029326
Epoch 3/50, Loss: 0.023271
Epoch 4/50, Loss: 0.021570
Epoch 5/50, Loss: 0.019773
Epoch 6/50, Loss: 0.018457
Epoch 7/50, Loss: 0.017547
Epoch 8/50, Loss: 0.016712
Epoch 9/50, Loss: 0.016136
Epoch 10/50, Loss: 0.015680
Epoch 11/50, Loss: 0.015243
Epoch 12/50, Loss: 0.014766
Epoch 13/50, Loss: 0.014361
Epoch 14/50, Loss: 0.014126
Epoch 15/50, Loss: 0.013766
Epoch 16/50, Loss: 0.013778
Epoch 17/50, Loss: 0.013522
Epoch 18/50, Loss: 0.013407
Epoch 19/50, Loss: 0.013254
Epoch 20/50, Loss: 0.013087
Epoch 21/50, Loss: 0.013067
Epoch 22/50, Loss: 0.013049
Epoch 23/50, Loss: 0.012988
Epoch 24/50, Loss: 0.013017
Epoch 25/50, Loss: 0.012916
Epoch 26/50, Loss: 0.013024
Epoch 27/50, Loss: 0.012895
Epoch 28/50, Loss: 0.013185
Epoch 29/50, Loss: 0.012756
Epoch 30/50, Loss: 0.012766
Epoch 31/50, Loss: 0.012788
Epoch 32/50, Loss: 0.012831
Epoch 33/50, Loss: 0.012720
Epoch 34/50, Loss: 0.012743
Epoch 35/50, Loss: 0.012746
Epoch 36/50, Loss: 0.012657
Epoch 37/50, Loss: 0.012623
Epoch 38/50, Loss: 0.012668
Epoch 39/50, Loss: 0.012608
Epoch 40/50, Loss: 0.012651
Epoch 41/50, Loss: 0.012560
Epoch 42/50, Loss: 0.012544
Epoch 43/50, Loss: 0.012501
Epoch 44/50, Loss: 0.012643
Epoch 45/50, Loss: 0.012416
Epoch 46/50, Loss: 0.012486
Epoch 47/50, Loss: 0.012457
Epoch 48/50, Loss: 0.012438
Epoch 49/50, Loss: 0.012408
Epoch 50/50, Loss: 0.012382

-> C.2: Explorando LLE (n_neighbors)
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=10
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.043676
Epoch 2/50, Loss: 0.026142
Epoch 3/50, Loss: 0.022162
Epoch 4/50, Loss: 0.020089
Epoch 5/50, Loss: 0.018358
Epoch 6/50, Loss: 0.017484
Epoch 7/50, Loss: 0.016703
Epoch 8/50, Loss: 0.016182
Epoch 9/50, Loss: 0.016065
Epoch 10/50, Loss: 0.015058
Epoch 11/50, Loss: 0.014509
Epoch 12/50, Loss: 0.014326
Epoch 13/50, Loss: 0.013982
Epoch 14/50, Loss: 0.013818
Epoch 15/50, Loss: 0.013580
Epoch 16/50, Loss: 0.013350
Epoch 17/50, Loss: 0.013318
Epoch 18/50, Loss: 0.013365
Epoch 19/50, Loss: 0.013180
Epoch 20/50, Loss: 0.013057
Epoch 21/50, Loss: 0.012977
Epoch 22/50, Loss: 0.013017
Epoch 23/50, Loss: 0.012941
Epoch 24/50, Loss: 0.012940
Epoch 25/50, Loss: 0.012985
Epoch 26/50, Loss: 0.012831
Epoch 27/50, Loss: 0.012825
Epoch 28/50, Loss: 0.012879
Epoch 29/50, Loss: 0.012814
Epoch 30/50, Loss: 0.012836
Epoch 31/50, Loss: 0.012707
Epoch 32/50, Loss: 0.012729
Epoch 33/50, Loss: 0.012682
Epoch 34/50, Loss: 0.012666
Epoch 35/50, Loss: 0.012699
Epoch 36/50, Loss: 0.012599
Epoch 37/50, Loss: 0.012606
Epoch 38/50, Loss: 0.012634
Epoch 39/50, Loss: 0.012543
Epoch 40/50, Loss: 0.012543
Epoch 41/50, Loss: 0.012564
Epoch 42/50, Loss: 0.012479
Epoch 43/50, Loss: 0.012469
Epoch 44/50, Loss: 0.012434
Epoch 45/50, Loss: 0.012480
Epoch 46/50, Loss: 0.012419
Epoch 47/50, Loss: 0.012371
Epoch 48/50, Loss: 0.012484
Epoch 49/50, Loss: 0.012347
Epoch 50/50, Loss: 0.012286
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=30
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.040552
Epoch 2/50, Loss: 0.025287
Epoch 3/50, Loss: 0.021827
Epoch 4/50, Loss: 0.019396
Epoch 5/50, Loss: 0.018133
Epoch 6/50, Loss: 0.017096
Epoch 7/50, Loss: 0.016177
Epoch 8/50, Loss: 0.015640
Epoch 9/50, Loss: 0.015244
Epoch 10/50, Loss: 0.014917
Epoch 11/50, Loss: 0.014451
Epoch 12/50, Loss: 0.014015
Epoch 13/50, Loss: 0.013764
Epoch 14/50, Loss: 0.013753
Epoch 15/50, Loss: 0.013463
Epoch 16/50, Loss: 0.013235
Epoch 17/50, Loss: 0.013171
Epoch 18/50, Loss: 0.013009
Epoch 19/50, Loss: 0.012909
Epoch 20/50, Loss: 0.013016
Epoch 21/50, Loss: 0.012877
Epoch 22/50, Loss: 0.012783
Epoch 23/50, Loss: 0.012820
Epoch 24/50, Loss: 0.012789
Epoch 25/50, Loss: 0.012751
Epoch 26/50, Loss: 0.012683
Epoch 27/50, Loss: 0.012713
Epoch 28/50, Loss: 0.012664
Epoch 29/50, Loss: 0.012760
Epoch 30/50, Loss: 0.012534
Epoch 31/50, Loss: 0.012628
Epoch 32/50, Loss: 0.012553
Epoch 33/50, Loss: 0.012550
Epoch 34/50, Loss: 0.012533
Epoch 35/50, Loss: 0.012465
Epoch 36/50, Loss: 0.012497
Epoch 37/50, Loss: 0.012475
Epoch 38/50, Loss: 0.012375
Epoch 39/50, Loss: 0.012354
Epoch 40/50, Loss: 0.012421
Epoch 41/50, Loss: 0.012300
Epoch 42/50, Loss: 0.012376
Epoch 43/50, Loss: 0.012259
Epoch 44/50, Loss: 0.012250
Epoch 45/50, Loss: 0.012260
Epoch 46/50, Loss: 0.012237
Epoch 47/50, Loss: 0.012263
Epoch 48/50, Loss: 0.012168
Epoch 49/50, Loss: 0.012134
Epoch 50/50, Loss: 0.012182
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=50
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.053083
Epoch 2/50, Loss: 0.032146
Epoch 3/50, Loss: 0.026669
Epoch 4/50, Loss: 0.023068
Epoch 5/50, Loss: 0.021576
Epoch 6/50, Loss: 0.020154
Epoch 7/50, Loss: 0.018814
Epoch 8/50, Loss: 0.017967
Epoch 9/50, Loss: 0.017349
Epoch 10/50, Loss: 0.016748
Epoch 11/50, Loss: 0.016080
Epoch 12/50, Loss: 0.015689
Epoch 13/50, Loss: 0.015472
Epoch 14/50, Loss: 0.015313
Epoch 15/50, Loss: 0.014841
Epoch 16/50, Loss: 0.014582
Epoch 17/50, Loss: 0.014205
Epoch 18/50, Loss: 0.014024
Epoch 19/50, Loss: 0.013962
Epoch 20/50, Loss: 0.013975
Epoch 21/50, Loss: 0.013753
Epoch 22/50, Loss: 0.013658
Epoch 23/50, Loss: 0.013549
Epoch 24/50, Loss: 0.013417
Epoch 25/50, Loss: 0.013463
Epoch 26/50, Loss: 0.013261
Epoch 27/50, Loss: 0.013253
Epoch 28/50, Loss: 0.013019
Epoch 29/50, Loss: 0.012962
Epoch 30/50, Loss: 0.012956
Epoch 31/50, Loss: 0.012996
Epoch 32/50, Loss: 0.012915
Epoch 33/50, Loss: 0.012887
Epoch 34/50, Loss: 0.012846
Epoch 35/50, Loss: 0.012823
Epoch 36/50, Loss: 0.012776
Epoch 37/50, Loss: 0.012761
Epoch 38/50, Loss: 0.012771
Epoch 39/50, Loss: 0.012748
Epoch 40/50, Loss: 0.012722
Epoch 41/50, Loss: 0.012926
Epoch 42/50, Loss: 0.012634
Epoch 43/50, Loss: 0.012601
Epoch 44/50, Loss: 0.012619
Epoch 45/50, Loss: 0.012627
Epoch 46/50, Loss: 0.012608
Epoch 47/50, Loss: 0.012594
Epoch 48/50, Loss: 0.012526
Epoch 49/50, Loss: 0.012556
Epoch 50/50, Loss: 0.012510

==================== DATASET: GlassIdentification (Sampling: 50.0%) ====================
   -> has_header: False. Saltando 0 fila(s).
Cargando train data desde: /home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/../data/GlassIdentification/glass_train.csv...
Muestreo aplicado: 50.0%. Usando 85 muestras.
Datos cargados (Shape: (85, 9))
   -> has_header: False. Saltando 0 fila(s).
Cargando test data desde: /home/walter/Uni/ML3/Practicas/P1ML3MixedManifold/experiments/../data/GlassIdentification/glass_test.csv...
Muestreo aplicado: 50.0%. Usando 21 muestras.
Datos cargados (Shape: (21, 9))
INPUT_DIM detectado: 9

-> C.1: Explorando TSNE (Perplexity)
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=5
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.009163
Epoch 2/50, Loss: 0.006983
Epoch 3/50, Loss: 0.005305
Epoch 4/50, Loss: 0.003994
Epoch 5/50, Loss: 0.002952
Epoch 6/50, Loss: 0.002110
Epoch 7/50, Loss: 0.001441
Epoch 8/50, Loss: 0.000925
Epoch 9/50, Loss: 0.000540
Epoch 10/50, Loss: 0.000273
Epoch 11/50, Loss: 0.000121
Epoch 12/50, Loss: 0.000073
Epoch 13/50, Loss: 0.000111
Epoch 14/50, Loss: 0.000207
Epoch 15/50, Loss: 0.000323
Epoch 16/50, Loss: 0.000423
Epoch 17/50, Loss: 0.000480
Epoch 18/50, Loss: 0.000484
Epoch 19/50, Loss: 0.000439
Epoch 20/50, Loss: 0.000357
Epoch 21/50, Loss: 0.000262
Epoch 22/50, Loss: 0.000171
Epoch 23/50, Loss: 0.000097
Epoch 24/50, Loss: 0.000046
Epoch 25/50, Loss: 0.000020
Epoch 26/50, Loss: 0.000017
Epoch 27/50, Loss: 0.000028
Epoch 28/50, Loss: 0.000045
Epoch 29/50, Loss: 0.000062
Epoch 30/50, Loss: 0.000074
Epoch 31/50, Loss: 0.000080
Epoch 32/50, Loss: 0.000078
Epoch 33/50, Loss: 0.000072
Epoch 34/50, Loss: 0.000062
Epoch 35/50, Loss: 0.000052
Epoch 36/50, Loss: 0.000042
Epoch 37/50, Loss: 0.000033
Epoch 38/50, Loss: 0.000026
Epoch 39/50, Loss: 0.000020
Epoch 40/50, Loss: 0.000017
Epoch 41/50, Loss: 0.000015
Epoch 42/50, Loss: 0.000014
Epoch 43/50, Loss: 0.000015
Epoch 44/50, Loss: 0.000016
Epoch 45/50, Loss: 0.000017
Epoch 46/50, Loss: 0.000018
Epoch 47/50, Loss: 0.000020
Epoch 48/50, Loss: 0.000020
Epoch 49/50, Loss: 0.000020
Epoch 50/50, Loss: 0.000019
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=30
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.009524
Epoch 2/50, Loss: 0.007421
Epoch 3/50, Loss: 0.005647
Epoch 4/50, Loss: 0.004153
Epoch 5/50, Loss: 0.002916
Epoch 6/50, Loss: 0.001930
Epoch 7/50, Loss: 0.001181
Epoch 8/50, Loss: 0.000669
Epoch 9/50, Loss: 0.000394
Epoch 10/50, Loss: 0.000332
Epoch 11/50, Loss: 0.000422
Epoch 12/50, Loss: 0.000576
Epoch 13/50, Loss: 0.000695
Epoch 14/50, Loss: 0.000728
Epoch 15/50, Loss: 0.000675
Epoch 16/50, Loss: 0.000560
Epoch 17/50, Loss: 0.000416
Epoch 18/50, Loss: 0.000282
Epoch 19/50, Loss: 0.000180
Epoch 20/50, Loss: 0.000119
Epoch 21/50, Loss: 0.000095
Epoch 22/50, Loss: 0.000098
Epoch 23/50, Loss: 0.000115
Epoch 24/50, Loss: 0.000136
Epoch 25/50, Loss: 0.000153
Epoch 26/50, Loss: 0.000161
Epoch 27/50, Loss: 0.000160
Epoch 28/50, Loss: 0.000149
Epoch 29/50, Loss: 0.000131
Epoch 30/50, Loss: 0.000108
Epoch 31/50, Loss: 0.000085
Epoch 32/50, Loss: 0.000064
Epoch 33/50, Loss: 0.000046
Epoch 34/50, Loss: 0.000034
Epoch 35/50, Loss: 0.000027
Epoch 36/50, Loss: 0.000025
Epoch 37/50, Loss: 0.000026
Epoch 38/50, Loss: 0.000030
Epoch 39/50, Loss: 0.000035
Epoch 40/50, Loss: 0.000039
Epoch 41/50, Loss: 0.000041
Epoch 42/50, Loss: 0.000041
Epoch 43/50, Loss: 0.000038
Epoch 44/50, Loss: 0.000033
Epoch 45/50, Loss: 0.000027
Epoch 46/50, Loss: 0.000022
Epoch 47/50, Loss: 0.000016
Epoch 48/50, Loss: 0.000013
Epoch 49/50, Loss: 0.000011
Epoch 50/50, Loss: 0.000010
-> Combinación: AE=LinearAutoencoder | Manifold=TSNE | Perplexity=50
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.005349
Epoch 2/50, Loss: 0.003874
Epoch 3/50, Loss: 0.002698
Epoch 4/50, Loss: 0.001798
Epoch 5/50, Loss: 0.001115
Epoch 6/50, Loss: 0.000636
Epoch 7/50, Loss: 0.000336
Epoch 8/50, Loss: 0.000192
Epoch 9/50, Loss: 0.000171
Epoch 10/50, Loss: 0.000224
Epoch 11/50, Loss: 0.000306
Epoch 12/50, Loss: 0.000373
Epoch 13/50, Loss: 0.000399
Epoch 14/50, Loss: 0.000388
Epoch 15/50, Loss: 0.000340
Epoch 16/50, Loss: 0.000276
Epoch 17/50, Loss: 0.000209
Epoch 18/50, Loss: 0.000149
Epoch 19/50, Loss: 0.000102
Epoch 20/50, Loss: 0.000072
Epoch 21/50, Loss: 0.000057
Epoch 22/50, Loss: 0.000053
Epoch 23/50, Loss: 0.000055
Epoch 24/50, Loss: 0.000059
Epoch 25/50, Loss: 0.000064
Epoch 26/50, Loss: 0.000067
Epoch 27/50, Loss: 0.000069
Epoch 28/50, Loss: 0.000070
Epoch 29/50, Loss: 0.000069
Epoch 30/50, Loss: 0.000066
Epoch 31/50, Loss: 0.000059
Epoch 32/50, Loss: 0.000051
Epoch 33/50, Loss: 0.000041
Epoch 34/50, Loss: 0.000032
Epoch 35/50, Loss: 0.000024
Epoch 36/50, Loss: 0.000018
Epoch 37/50, Loss: 0.000014
Epoch 38/50, Loss: 0.000013
Epoch 39/50, Loss: 0.000015
Epoch 40/50, Loss: 0.000017
Epoch 41/50, Loss: 0.000020
Epoch 42/50, Loss: 0.000022
Epoch 43/50, Loss: 0.000023
Epoch 44/50, Loss: 0.000022
Epoch 45/50, Loss: 0.000021
Epoch 46/50, Loss: 0.000019
Epoch 47/50, Loss: 0.000016
Epoch 48/50, Loss: 0.000014
Epoch 49/50, Loss: 0.000011
Epoch 50/50, Loss: 0.000010

-> C.2: Explorando LLE (n_neighbors)
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=10
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.014289
Epoch 2/50, Loss: 0.011791
Epoch 3/50, Loss: 0.009682
Epoch 4/50, Loss: 0.007896
Epoch 5/50, Loss: 0.006365
Epoch 6/50, Loss: 0.005033
Epoch 7/50, Loss: 0.003888
Epoch 8/50, Loss: 0.002926
Epoch 9/50, Loss: 0.002121
Epoch 10/50, Loss: 0.001474
Epoch 11/50, Loss: 0.000967
Epoch 12/50, Loss: 0.000594
Epoch 13/50, Loss: 0.000340
Epoch 14/50, Loss: 0.000188
Epoch 15/50, Loss: 0.000129
Epoch 16/50, Loss: 0.000147
Epoch 17/50, Loss: 0.000221
Epoch 18/50, Loss: 0.000318
Epoch 19/50, Loss: 0.000410
Epoch 20/50, Loss: 0.000475
Epoch 21/50, Loss: 0.000499
Epoch 22/50, Loss: 0.000476
Epoch 23/50, Loss: 0.000422
Epoch 24/50, Loss: 0.000347
Epoch 25/50, Loss: 0.000266
Epoch 26/50, Loss: 0.000190
Epoch 27/50, Loss: 0.000125
Epoch 28/50, Loss: 0.000075
Epoch 29/50, Loss: 0.000041
Epoch 30/50, Loss: 0.000023
Epoch 31/50, Loss: 0.000017
Epoch 32/50, Loss: 0.000021
Epoch 33/50, Loss: 0.000031
Epoch 34/50, Loss: 0.000043
Epoch 35/50, Loss: 0.000056
Epoch 36/50, Loss: 0.000066
Epoch 37/50, Loss: 0.000073
Epoch 38/50, Loss: 0.000076
Epoch 39/50, Loss: 0.000075
Epoch 40/50, Loss: 0.000070
Epoch 41/50, Loss: 0.000062
Epoch 42/50, Loss: 0.000053
Epoch 43/50, Loss: 0.000043
Epoch 44/50, Loss: 0.000033
Epoch 45/50, Loss: 0.000024
Epoch 46/50, Loss: 0.000017
Epoch 47/50, Loss: 0.000013
Epoch 48/50, Loss: 0.000010
Epoch 49/50, Loss: 0.000009
Epoch 50/50, Loss: 0.000010
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=30
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.013480
Epoch 2/50, Loss: 0.011701
Epoch 3/50, Loss: 0.010070
Epoch 4/50, Loss: 0.008623
Epoch 5/50, Loss: 0.007294
Epoch 6/50, Loss: 0.006077
Epoch 7/50, Loss: 0.004939
Epoch 8/50, Loss: 0.003891
Epoch 9/50, Loss: 0.002953
Epoch 10/50, Loss: 0.002134
Epoch 11/50, Loss: 0.001432
Epoch 12/50, Loss: 0.000894
Epoch 13/50, Loss: 0.000548
Epoch 14/50, Loss: 0.000369
Epoch 15/50, Loss: 0.000337
Epoch 16/50, Loss: 0.000397
Epoch 17/50, Loss: 0.000487
Epoch 18/50, Loss: 0.000559
Epoch 19/50, Loss: 0.000588
Epoch 20/50, Loss: 0.000568
Epoch 21/50, Loss: 0.000507
Epoch 22/50, Loss: 0.000427
Epoch 23/50, Loss: 0.000343
Epoch 24/50, Loss: 0.000265
Epoch 25/50, Loss: 0.000200
Epoch 26/50, Loss: 0.000151
Epoch 27/50, Loss: 0.000117
Epoch 28/50, Loss: 0.000096
Epoch 29/50, Loss: 0.000084
Epoch 30/50, Loss: 0.000081
Epoch 31/50, Loss: 0.000082
Epoch 32/50, Loss: 0.000086
Epoch 33/50, Loss: 0.000090
Epoch 34/50, Loss: 0.000093
Epoch 35/50, Loss: 0.000094
Epoch 36/50, Loss: 0.000093
Epoch 37/50, Loss: 0.000089
Epoch 38/50, Loss: 0.000082
Epoch 39/50, Loss: 0.000074
Epoch 40/50, Loss: 0.000064
Epoch 41/50, Loss: 0.000053
Epoch 42/50, Loss: 0.000043
Epoch 43/50, Loss: 0.000035
Epoch 44/50, Loss: 0.000028
Epoch 45/50, Loss: 0.000023
Epoch 46/50, Loss: 0.000021
Epoch 47/50, Loss: 0.000019
Epoch 48/50, Loss: 0.000019
Epoch 49/50, Loss: 0.000020
Epoch 50/50, Loss: 0.000021
-> Combinación: AE=LinearAutoencoder | Manifold=LLE | n_neighbors=50
Iniciando entrenamiento en cuda por 50 épocas...
Epoch 1/50, Loss: 0.014171
Epoch 2/50, Loss: 0.011818
Epoch 3/50, Loss: 0.009839
Epoch 4/50, Loss: 0.008114
Epoch 5/50, Loss: 0.006643
Epoch 6/50, Loss: 0.005370
Epoch 7/50, Loss: 0.004295
Epoch 8/50, Loss: 0.003413
Epoch 9/50, Loss: 0.002685
Epoch 10/50, Loss: 0.002080
Epoch 11/50, Loss: 0.001588
Epoch 12/50, Loss: 0.001200
Epoch 13/50, Loss: 0.000911
Epoch 14/50, Loss: 0.000691
Epoch 15/50, Loss: 0.000525
Epoch 16/50, Loss: 0.000396
Epoch 17/50, Loss: 0.000305
Epoch 18/50, Loss: 0.000242
Epoch 19/50, Loss: 0.000209
Epoch 20/50, Loss: 0.000203
Epoch 21/50, Loss: 0.000217
Epoch 22/50, Loss: 0.000238
Epoch 23/50, Loss: 0.000256
Epoch 24/50, Loss: 0.000267
Epoch 25/50, Loss: 0.000267
Epoch 26/50, Loss: 0.000255
Epoch 27/50, Loss: 0.000230
Epoch 28/50, Loss: 0.000196
Epoch 29/50, Loss: 0.000157
Epoch 30/50, Loss: 0.000118
Epoch 31/50, Loss: 0.000086
Epoch 32/50, Loss: 0.000062
Epoch 33/50, Loss: 0.000047
Epoch 34/50, Loss: 0.000040
Epoch 35/50, Loss: 0.000040
Epoch 36/50, Loss: 0.000043
Epoch 37/50, Loss: 0.000048
Epoch 38/50, Loss: 0.000052
Epoch 39/50, Loss: 0.000054
Epoch 40/50, Loss: 0.000054
Epoch 41/50, Loss: 0.000051
Epoch 42/50, Loss: 0.000047
Epoch 43/50, Loss: 0.000041
Epoch 44/50, Loss: 0.000035
Epoch 45/50, Loss: 0.000029
Epoch 46/50, Loss: 0.000024
Epoch 47/50, Loss: 0.000020
Epoch 48/50, Loss: 0.000017
Epoch 49/50, Loss: 0.000015
Epoch 50/50, Loss: 0.000014
✅ Resultados guardados en results/exp_C_hyperparams_manifold.csv

Process finished with exit code 0

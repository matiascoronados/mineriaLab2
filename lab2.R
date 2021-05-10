#install.packages("randomForest")
require(randomForest)
#install.packages("magrittr") # package installations are only needed the first time you use it
#install.packages("dplyr")    # alternative installation of the %>%
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)
#install.packages("Rtsne")
library(Rtsne)


#url.data.set <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff'
url.data.set <- "/Users/matiascoronado/Downloads/messidor_features.arff"

data.raw <- read.csv(url.data.set, header=FALSE, comment.char = "@")
df <- data.frame(data.raw)
colnames(df) <- c(
  "q",      #  0 Numero binario, relacionado con la calidad de la imagen. 0 = Mala calidad; 1 = Buena calidad.(remover)
  "ps",     #  1 Numero binario resultante de un pre-escaneo, que indica si la imagen presenta animalias graves en la retina.(remover)
  "nma.a",  #  2 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 0.5
  "nma.b",  #  3 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 0.6
  "nma.c",  #  4 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 0.7
  "nma.d",  #  5 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 0.8
  "nma.e",  #  6 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 0.9
  "nma.f",  #  7 Numero de MAs (microneurismas) encontrados en un nivel de confianza con alpha = 1.0
  "nex.a",  #  8 Numero de Exudates encontrados en un nivel de confianza con alpha = 0.5
  "nex.b",  #  9 Numero de Exudates encontrados en un nivel de confianza con alpha = 0.6
  "nex.c",  # 10 Numero de Exudates encontrados en un nivel de confianza con alpha = 0.7
  "nex.d",  # 11 Numero de Exudates encontrados en un nivel de confianza con alpha = 0.8
  "nex.e",  # 12 Numero de Exudates encontrados en un nivel de confianza con alpha = 0.9
  "nex.f",  # 13 Numero de Exudates encontrados en un nivel de confianza con alpha = 1.0
  "nex.g",  # 14 Numero de Exudates encontrados en un nivel de confianza con alpha = 1.0
  "nex.h",  # 15 Numero de Exudates encontrados en un nivel de confianza con alpha = 1.0
  "dd",     # 16 Distancia eucladiana entre el centro de la macula y el centro del disco optico
  "dm",     # 17 Diametro del disco optico
  "amfm",   # 18 Numeri binario, relacionado con la clasificacion AM/FM-based.(remover) (indirectamente es la clase)
  "class"   # 19 Clase del dato, en donde 1 = contiene signos de DR, 0 = no contiene signos de DR.(remover)
)


########### Se convierten en factor el valor binario de class.

df.final <- df
#Filtrando por q == 1
df.final <- filter(df.final, q==1)
df.final$class <- factor(df.final$class)
df.final$q <- factor(df.final$q)
df.final$ps <- factor(df.final$ps)
df.final$amfm <- factor(df.final$amfm)

datos.05 <- df.final[,-c(4,5,6,7,8,10,11,12,13,14,15,16)]
datos.06 <- df.final[,-c(3,5,6,7,8,9,11,12,13,14,15,16)]
datos.07 <- df.final[,-c(3,4,6,7,8,9,10,12,13,14,15,16)]
datos.08 <- df.final[,-c(3,4,5,7,8,9,10,11,13,14,15,16)]
datos.09 <- df.final[,-c(3,4,5,6,8,9,10,11,12,14,15,16)]
datos.10 <- df.final[,-c(3,4,5,6,7,9,10,11,12,13,15,16)]
df.final <- df.final

#Equilibrio entre la varianza y el sesgo (que tan bien se ajusta el modelo a nuevas obversaciones)
#Mientras mas grande el arbol, mayor es la varianza pero menor es el sesgo.
#arbol GRANDE: Aumenta varianza; pero baja sesgo
#arbol CHICO: alrevez

################################################################################
###########         INICIO ANALISIS Modelo A (con todas las variables)
################################################################################

set.seed(71)
modeloA <- randomForest(class ~ ., data=df.final, importance=TRUE, proximity=TRUE)
print(modeloA)
plot(modeloA)

#Importancia: este nos permite aplicar el principio de parcimonia; regularizacion.; este nos permite ver que tanto afecta la variable en el proceso de clasificainon
round(importance(modeloA),2)
#MeanDecreaseAccuracy;  Que tanto incide que se encuentre o no la variable de clasificacion.
#MeanCecreaseGini:      Tiene que ver con las impourezas de cada nodo al hacer las divicion de variables.
varImpPlot(modeloA)
#Este grafico nos indica cuales caracteristicas aportan menos
#Realizar 2 modelos:  Uno con todas las varialbes
#                     Otro con las mas importantes
#             Compara medidas de bondad y error

#Proximidad
data.mds <- cmdscale(1 - modeloA$proximity, eig=TRUE)
#escalamiento clasico multidimencional (la que explicao el profe en catedra)
op <- par(pty="s")
pairs(cbind(df.final[1:19],data.mds$points),cex=0.5,gap=0,
      col=c("red","green")[as.numeric(df.final$class)],
      main="Data: Predictos and MDS of Proximity Based on RandomForest")


par(op)
print(data.mds$GOF)
MDSplot(modeloA,df.final$class)

#Coordenadas paralelas
#install.packages("MASS")
require(MASS)

### OJO
#No se pueden utilizar variables tipo factor: 0 - 1
df.final.numeric <- df.final

df.final.numeric$q <- as.numeric(df.final.numeric$q)
df.final.numeric$ps <- as.numeric(df.final.numeric$ps)
df.final.numeric$amfm <- as.numeric(df.final.numeric$amfm)

parcoord(df.final.numeric[,1:19], var.label=TRUE,col=c("red","green")[as.numeric(df.final$class)])
#legend("bottomright",legend=c("Presenta ","No"),fill=2:4)


############################################# ROC
#############################################
#Medidas de bonbadad de ajuste ROC
#https://stats.stackexchange.com/questions/188616/how-can-we-calculate-roc-auc-for-classification-algorithm-such-as-random-forest
require(pROC)
rf.roc<-roc(df.final$class,modeloA$votes[,2])
plot(rf.roc)
auc(rf.roc)


#############################################TSNE
#############################################
require(Rtsne)
#No se pueden utilizar variables tipo factor: 0 - 1
df.final.numeric <- df.final

df.final.numeric$q <- as.numeric(df.final.numeric$q)
df.final.numeric$ps <- as.numeric(df.final.numeric$ps)
df.final.numeric$amfm <- as.numeric(df.final.numeric$amfm)

tsne <- Rtsne(as.matrix(df.final.numeric[,1:19]), check_duplicates = FALSE, pca = FALSE, perplexity=30, theta=0.5, dims=2)
cols <- rainbow(10)
plot(tsne$Y, t='n')
text(tsne$Y, labels=df.final.numeric[,20], col=cols[df.final.numeric[,20] +1])


metadata <- data.frame(sample_id = rownames(df.final.numeric),
                       colour = df.final.numeric$class)
library(ggplot2)
dadada <- data.frame(x = tsne$Y[,1],
                 y = tsne$Y[,2],
                 colour = metadata$colour)

ggplot(dadada, aes(x, y, colour = colour)) +
  geom_point()

################################################################################
###########         FIN ANALISIS Modelo A (con todas las variables)
################################################################################


################################################################################
###########         INICIO TESTEO DE MODELOS CON DIFERENTES ALPHAS
################################################################################

########CON LAS VARIABLES: q ps nma.* nex.* dd dm amfm clase
numeroArboles <- 550
numeroVecindades <- 2

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.10,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#38.66

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.09,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#40.75

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.08,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#41.79

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.07,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#40.31

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.06,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#39.36

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#34.58
plot(iris.rf100)

################################################################################
###########         FIN TESTEO DE MODELOS CON DIFERENTES ALPHAS
################################################################################

#Por lo anterior, se utilizara alpha = 0.5

################################################################################
###########         INICIO TESTEO DE MODELOS CON DIFERENTES VARIABLES 
################################################################################

datos.05.sinAMFM <- datos.05[,-c(7)]
datos.05.sinAMFM.sinQ <- datos.05[,-c(1,7)]
datos.05.sinAMFM.sinQ.sinPS <- datos.05[,-c(1,2,7)]

numeroArboles <- 550
numeroVecindades <- 2

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinAMFM,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 34.32

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinAMFM.sinQ,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 34.32

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinAMFM.sinQ.sinPS,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 35.1

################################################################################
###########         FIN TESTEO DE MODELOS CON DIFERENTES VARIABLES 
################################################################################

#Por parsimonia, se puede elimiar AMFM y Q, ya que dan el mismo error que el modelo con todas las varialbes presentes.



################### 
################### 
################### 
#Alpla definitivo es 0.5
#Las variables del modelo son: ps nma.a nex.a dd dm class
###################
###################
################### 



################################################################################
###########         INICIO TESTEO DE MODELOS, VARIANDO LA CANTIDAD DE ARBOLES Y VECINDADES
################################################################################

#### ERROR ENTREGADO ANTERIORMENTE.
# 34.32
#CON: 
#numeroArboles <- 550
#numeroVecindades <- 2

###############
numeroArboles <- 500
numeroVecindades <- 1
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinAMFM.sinQ,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 36.23

numeroArboles <- 1000
numeroVecindades <- 1
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinAMFM.sinQ,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 35.01

numeroArboles <- 1500
numeroVecindades <- 1
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinAMFM.sinQ,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 35.1
###############

numeroArboles <- 500
numeroVecindades <- 2
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinAMFM.sinQ,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 34.67

numeroArboles <- 1000
numeroVecindades <- 2
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinAMFM.sinQ,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 34.14

numeroArboles <- 1500
numeroVecindades <- 2
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinAMFM.sinQ,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 34.4
###############

#############
############# MODELO DEFINITIVO
#############

numeroArboles <- 550
numeroVecindades <- 2

set.seed(324)
modelo.definitivo <- randomForest(class ~ ., data=datos.05.sinAMFM.sinQ,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(modelo.definitivo)
# 34.4

################################################################################
###########         FIN TESTEO DE MODELOS, VARIANDO LA CANTIDAD DE ARBOLES Y VECINDADES
################################################################################










modelo.analisis <- modelo.definitivo
datos.analisis <- datos.05.sinAMFM.sinQ
cantidad.var <- dim(datos.05.sinAMFM.sinQ)[2]

################################################################################
###########         INICIO ANALISIS Modelo definitivo
################################################################################

#Error por arbol
plot(modelo.analisis)

#Importancia: este nos permite aplicar el principio de parcimonia; regularizacion.; este nos permite ver que tanto afecta la variable en el proceso de clasificainon
round(importance(modelo.analisis),2)
#MeanDecreaseAccuracy;  Que tanto incide que se encuentre o no la variable de clasificacion.
#MeanCecreaseGini:      Tiene que ver con las impourezas de cada nodo al hacer las divicion de variables.
varImpPlot(modelo.analisis)
#Este grafico nos indica cuales caracteristicas aportan menos
#Realizar 2 modelos:  Uno con todas las varialbes
#                     Otro con las mas importantes
#             Compara medidas de bondad y error

#Proximidad
data.mds <- cmdscale(1 - modelo.analisis$proximity, eig=TRUE)
#escalamiento clasico multidimencional (la que explicao el profe en catedra)
op <- par(pty="s")
pairs(cbind(datos.analisis[1:cantidad.var-1],data.mds$points),cex=0.5,gap=0,
      col=c("red","green")[as.numeric(datos.analisis$class)],
      main="Data: Predictos and MDS of Proximity Based on RandomForest")

par(op)
print(data.mds$GOF)


MDSplot(modelo.analisis,datos.analisis$class)

#Coordenadas paralelas
#install.packages("MASS")
require(MASS)

#No se pueden utilizar variables tipo factor: 0 - 1
df.final.numeric <- datos.analisis

df.final.numeric$q <- as.numeric(df.final.numeric$q)
df.final.numeric$ps <- as.numeric(df.final.numeric$ps)
df.final.numeric$amfm <- as.numeric(df.final.numeric$amfm)

parcoord(df.final.numeric[,1:cantidad.var-1], var.label=TRUE,col=c("red","green")[as.numeric(df.final.numeric$class)])
#legend("bottomright",legend=c("Presenta ","No"),fill=2:4)


############################################# ROC
#############################################
#Medidas de bonbadad de ajuste ROC
#https://stats.stackexchange.com/questions/188616/how-can-we-calculate-roc-auc-for-classification-algorithm-such-as-random-forest
require(pROC)
rf.roc<-roc(datos.analisis$class,modelo.analisis$votes[,2])
plot(rf.roc)
auc(rf.roc)


#############################################TSNE
#############################################
require(Rtsne)
#No se pueden utilizar variables tipo factor: 0 - 1
df.final.numeric <- datos.analisis
df.final.numeric$q <- as.numeric(df.final.numeric$q)
df.final.numeric$ps <- as.numeric(df.final.numeric$ps)
df.final.numeric$amfm <- as.numeric(df.final.numeric$amfm)

tsne <- Rtsne(as.matrix(df.final.numeric[,1:cantidad.var-1]), check_duplicates = FALSE, pca = FALSE, perplexity=30, theta=0.5, dims=2)
cols <- rainbow(10)
plot(tsne$Y, t='n')
text(tsne$Y, labels=df.final.numeric[,cantidad.var], col=cols[df.final.numeric[,cantidad.var] +1])


metadata <- data.frame(sample_id = rownames(df.final.numeric),
                       colour = df.final.numeric$class)
library(ggplot2)
dadada <- data.frame(x = tsne$Y[,1],
                     y = tsne$Y[,2],
                     colour = metadata$colour)

ggplot(dadada, aes(x, y, colour = colour)) +
  geom_point()

################################################################################
###########         FIN ANALISIS Modelo definitivo
################################################################################

















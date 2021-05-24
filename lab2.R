#install.packages("randomForest")
#install.packages("magrittr")
#install.packages("dplyr") 
#install.packages("Rtsne")
#install.packages("MASS")
#install.packages("pROC")
#install.packages("ggplot2")

require(randomForest)
library(magrittr)
library(dplyr)
library(Rtsne)
require(MASS)
require(pROC)
library(ggplot2)

url.data.set <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff'
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
df.final$class <- factor(df.final$class)
df.final$q <- factor(df.final$q)
df.final$ps <- factor(df.final$ps)
df.final$amfm <- factor(df.final$amfm)


#Se filtran los datos, para mantener los que son de buena calidad (q == 1)
df.final <- filter(df.final, q==1)

datos.05 <- df.final[,-c(4,5,6,7,8,10,11,12,13,14,15,16)]
datos.06 <- df.final[,-c(3,5,6,7,8,9,11,12,13,14,15,16)]
datos.07 <- df.final[,-c(3,4,6,7,8,9,10,12,13,14,15,16)]
datos.08 <- df.final[,-c(3,4,5,7,8,9,10,11,13,14,15,16)]
datos.09 <- df.final[,-c(3,4,5,6,8,9,10,11,12,14,15,16)]
datos.10 <- df.final[,-c(3,4,5,6,7,9,10,11,12,13,15,16)]
df.final <- df.final


################################################################################
###########         INICIO ANALISIS Modelo A (con todas las variables)
################################################################################
set.seed(71)
modeloA <- randomForest(class ~ ., data=df.final, importance=TRUE, proximity=TRUE)
print(modeloA)
plot(modeloA)

#Importancia
varImpPlot(modeloA)

#Proximidad
data.mds <- cmdscale(1 - modeloA$proximity, eig=TRUE)
#escalamiento clasico multidimencional
op <- par(pty="s")
pairs(cbind(df.final[1:19],data.mds$points),cex=0.5,gap=0,
      col=c("red","green")[as.numeric(df.final$class)],
      main="Data: Predictos and MDS of Proximity Based on RandomForest")

par(op)
MDSplot(modeloA,df.final$class)

#Coordenadas paralelas
df.final.numeric <- df.final
df.final.numeric$q <- as.numeric(df.final.numeric$q)
df.final.numeric$ps <- as.numeric(df.final.numeric$ps)
df.final.numeric$amfm <- as.numeric(df.final.numeric$amfm)
parcoord(df.final.numeric[,1:19], var.label=TRUE,col=c("red","green")[as.numeric(df.final$class)])


# Calculo de ROC
#Medidas de bonbadad de ajuste ROC
rf.roc<-roc(df.final$class,modeloA$votes[,2])
plot(rf.roc)
auc(rf.roc)

# Calculo de T-SNE
df.final.numeric <- df.final
df.final.numeric$q <- as.numeric(df.final.numeric$q)
df.final.numeric$ps <- as.numeric(df.final.numeric$ps)
df.final.numeric$amfm <- as.numeric(df.final.numeric$amfm)

tsne <- Rtsne(as.matrix(df.final.numeric[,1:19]), check_duplicates = FALSE, pca = FALSE, perplexity=30, theta=0.5, dims=2)

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
#38.62

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.09,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#38.27

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.08,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#38.19

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.07,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#38.27

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.06,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#37.49

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
#34.58
plot(iris.rf100)

################################################################################
###########         FIN TESTEO DE MODELOS CON DIFERENTES ALPHAS
################################################################################



################################################################################
###########         INICIO TESTEO DE MODELOS CON DIFERENTES VARIABLES 
################################################################################

datos.05.sinQ <- datos.05[,-c(1)]
datos.05.sinQ.sinPS <- datos.05[,-c(1,2)]
datos.05.sinQ.sinPS.sinAMFM <- datos.05[,-c(1,2,7)]
datos.05.sinQ.sinPS.sinAMFM.sinDD <- datos.05[,-c(1,2,5,7)]
datos.05.sinQ.sinPS.sinAMFM.sinDD.sinDS <- datos.05[,-c(1,2,5,7)]


numeroArboles <- 1000
numeroVecindades <- 2


set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
rf.roc<-roc(datos.05$class,iris.rf100$votes[,2])
auc(rf.roc)
# nvecindad = 2
# OBB 34.0
# Area under the curve: 0.7139


set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ,ntree= numeroArboles, mtry=2, importance=TRUE, proximity=TRUE)
print(iris.rf100)
plot(iris.rf100)
rf.roc<-roc(datos.05$class,iris.rf100$votes[,2])
auc(rf.roc)
# nvecindad = 2
# OBB 33.65
# Area under the curve: 0.7116


set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS,ntree= numeroArboles, mtry=3, importance=TRUE, proximity=TRUE)
print(iris.rf100)
rf.roc<-roc(datos.05$class,iris.rf100$votes[,2])
auc(rf.roc)
# nvecindad = 2
# OBB 34.61
# Area under the curve: 0.7086


set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
rf.roc<-roc(datos.05$class,iris.rf100$votes[,2])
auc(rf.roc)
# nvecindad = 2
# OBB 34.96
# Area under the curve: 0.6997

set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
plot(iris.rf100)
rf.roc<-roc(datos.05$class,iris.rf100$votes[,2])
auc(rf.roc)
# nvecindad = 2
# OBB 35.75
# Area under the curve: 0.6989


set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD.sinDS,ntree= numeroArboles, mtry=1, importance=TRUE, proximity=TRUE)
print(iris.rf100)
rf.roc<-roc(datos.05$class,iris.rf100$votes[,2])
plot(iris.rf100)
auc(rf.roc)
# nvecindad = 2
# OBB 36.7
# Area under the curve: 0.6882

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

###############
############### numeroVecinades = 1
###############
###############
numeroArboles <- 500
numeroVecindades <- 1
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 37.4

numeroArboles <- 1000
numeroVecindades <- 1
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 37.14

numeroArboles <- 2000
numeroVecindades <- 1
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 37.05


numeroArboles <- 5000
numeroVecindades <- 1
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 37.31


numeroArboles <- 10000
numeroVecindades <- 1
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 36.88


numeroArboles <- 20000
numeroVecindades <- 1
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 36.88

###############
###############
############### numeroVecinades = 2
###############
numeroArboles <- 500
numeroVecindades <- 2
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 35.83

numeroArboles <- 1000
numeroVecindades <- 2
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 35.75

numeroArboles <- 2000
numeroVecindades <- 2
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 35.92


numeroArboles <- 5000
numeroVecindades <- 2
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 36.62


numeroArboles <- 10000
numeroVecindades <- 2
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 36.36


numeroArboles <- 20000
numeroVecindades <- 2
set.seed(324)
iris.rf100 <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)
print(iris.rf100)
# 36.62

################################################################################
###########         FIN TESTEO DE MODELOS, VARIANDO LA CANTIDAD DE ARBOLES Y VECINDADES
################################################################################





############## modelo definitivo
############## modelo definitivo
############## modelo definitivo
############## modelo definitivo
############## modelo definitivo
############## modelo definitivo


numeroArboles <- 2500
numeroVecindades <- 2
set.seed(324)
datos.05.sinQ.sinPS.sinAMFM.sinDD <- datos.05[,-c(1,2,5,7)]
modelo.definitivo <- randomForest(class ~ ., data=datos.05.sinQ.sinPS.sinAMFM.sinDD,ntree= numeroArboles, mtry=numeroVecindades, importance=TRUE, proximity=TRUE)



############## modelo definitivo
############## modelo definitivo
############## modelo definitivo
############## modelo definitivo
############## modelo definitivo
############## modelo definitivo

modelo.analisis <- modelo.definitivo
datos.analisis <- datos.05.sinQ.sinPS.sinAMFM.sinDD
cantidad.var <- dim(datos.05.sinQ.sinPS.sinAMFM.sinDD)[2]


################################################################################
###########         INICIO ANALISIS Modelo definitivo
################################################################################

print(modelo.analisis)

#Error por arbol
plot(modelo.analisis)

#Importancia
varImpPlot(modelo.analisis)

#Proximidad
data.mds <- cmdscale(1 - modelo.analisis$proximity, eig=TRUE)
op <- par(pty="s")
pairs(cbind(datos.analisis[1:cantidad.var-1],data.mds$points),cex=0.5,gap=0,
      col=c("red","blue")[as.numeric(datos.analisis$class)],
      main="Data: Predictos and MDS of Proximity Based on RandomForest")

par(op)
print(data.mds$GOF)
MDSplot(modelo.analisis,datos.analisis$class)

#Coordenadas paralelas
df.final.numeric <- datos.analisis
df.final.numeric$q <- as.numeric(df.final.numeric$q)
df.final.numeric$ps <- as.numeric(df.final.numeric$ps)
df.final.numeric$amfm <- as.numeric(df.final.numeric$amfm)
parcoord(df.final.numeric[,1:cantidad.var-1], var.label=TRUE,col=c("red","green")[as.numeric(df.final.numeric$class)])


# Calculo de ROC
#Medidas de bonbadad de ajuste ROC
rf.roc<-roc(datos.analisis$class,modelo.analisis$votes[,2])
plot(rf.roc)
auc(rf.roc)


# Calculo de T-SNE
df.final.numeric <- datos.analisis
df.final.numeric$q <- as.numeric(df.final.numeric$q)
df.final.numeric$ps <- as.numeric(df.final.numeric$ps)
df.final.numeric$amfm <- as.numeric(df.final.numeric$amfm)

tsne <- Rtsne(as.matrix(df.final.numeric[,1:cantidad.var-1]), check_duplicates = FALSE, pca = FALSE, perplexity=30, theta=0.5, dims=2)
metadata <- data.frame(sample_id = rownames(df.final.numeric),
                       colour = df.final.numeric$class)

dadada <- data.frame(x = tsne$Y[,1],
                     y = tsne$Y[,2],
                     colour = metadata$colour)

ggplot(dadada, aes(x, y, colour = colour)) +
  geom_point()

################################################################################
###########         FIN ANALISIS Modelo definitivo
################################################################################












#install.packages("randomForest")
require(randomForest)
#install.packages("magrittr") # package installations are only needed the first time you use it
#install.packages("dplyr")    # alternative installation of the %>%
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)
#install.packages("M3C")
library(M3C)
install.packages("Rtsne")
library(Rtsne)


url.data.set <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff'

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
df$class <- factor(df$class)

########### Se pasan a numericos los valores q; ps; amfm
df$q <- as.numeric(df$q)
df$ps <- as.numeric(df$ps)
df$amfm <- as.numeric(df$amfm)
df$class <- as.numeric(df$class)


#Se realizara un analisis tomando en consideracion las duplas que contengan el mismo valor de alpha.
datos.05 <- df[,-c(1,4,5,6,7,8,10,11,12,13,14,15,16)]

datos.06 <- df[,-c(1,3,5,6,7,8,9,11,12,13,14,15,16)]

datos.07 <- df[,-c(1,3,4,6,7,8,9,10,12,13,14,15,16)]

datos.08 <- df[,-c(1,3,4,5,7,8,9,10,11,13,14,15,16)]

datos.09 <- df[,-c(1,3,4,5,6,8,9,10,11,12,14,15,16)]

datos.10 <- df[,-c(1,3,4,5,6,7,9,10,11,12,13,15,16)]

df.final <- df[,-c(1,19,20)]



#Equilibrio entre la varianza y el sesgo (que tan bien se ajusta el modelo a nuevas obversaciones)

#Mientras mas grande el arbol, mayor es la varianza pero menor es el sesgo.
#arbol GRANDE: Aumenta varianza; pero baja sesgo
#arbol CHICO: alrevez

set.seed(71)
data.rf <- randomForest(class ~ ., data=datos.10, importance=TRUE, proximity=TRUE)
print(data.rf)


#Importancia: este nos permite aplicar el principio de parcimonia; regularizacion.; este nos permite ver que tanto afecta la variable en el proceso de clasificainon
round(importance(data.rf),2)

#MeanDecreaseAccuracy;  Que tanto incide que se encuentre o no la variable de clasificacion.
#MeanCecreaseGini:      Tiene que ver con las impourezas de cada nodo al hacer las divicion de variables
varImpPlot(data.rf)

#Este grafico nos indica cuales caracteristicas aportan menos
#Realizar 2 modelos:  Uno con todas las varialbes
#                     Otro con las mas importantes
#             Compara medidas de bondad y error

#             Seguir principio de ***********parcimonia.*************



#Proximidad
data.mds <- cmdscale(1 - data.rf$proximity, eig=TRUE)
#escalamiento clasico multidimencional (la que explicao el profe en catedra)

op <- par(pty="s")
pairs(cbind(datos.10[1:4],data.mds$points),cex=0.5,gap=0,
      col=c("red","green")[as.numeric(datos.10$class)],
      main="Data: Predictos and MDS of Proximity Based on RandomForest")


par(op)
print(iris.mds$GOF)
MDSplot(data.rf,datos.10$class)
#Dim1 permite hacer un separamiento multidimencional entre clases
#Para dar solucion al problema de clasificaion

iris.rf100 <- randomForest(class ~ ., data=datos.10,ntree =100, importance=TRUE, proximity=TRUE)
print(iris.rf100)

iris.rf600 <- randomForest(class ~ ., data=datos.10,ntree =600, importance=TRUE, proximity=TRUE)
print(iris.rf600)

#Misma clasificacion; Menor error que 100 arboles.

#El metodo asegura que no va a existir sobreajuste

#Va a llegar a un punto en que no importara en numero de arboles ??

#Para probar diferentes numeros de vecindades: mtry

iris.rf100 <- randomForest(class ~ ., data=datos.10,ntree =100, mtry=3, importance=TRUE, proximity=TRUE)

print(iris.rf100)

#Siemore hay que visualizar
plot(iris.rf100)


plot(iris.rf600)

#Coordenadas paralelas
#install.packages("MASS")
require(MASS)
parcoord(datos.10[,1:4], var.label=TRUE,col=c("red","green")[as.numeric(datos.10$class)])
legend("bottomright",legend=c("Presenta ","No"),fill=2:4)






install.packages("randomForest")
require(randomForest)

set.seed(71)
iris.rf <- randomForest(Species ~ ., data=iris, importance=TRUE, proximity=TRUE)
print(iris.rf)

#Importancia: este nos permite aplicar el principio de parcimonia; regularizacion.; este nos permite ver que tanto afecta la variable en el proceso de clasificainon
round(importance(iris.rf),2)

#MeanDecreaseAccuracy;  Que tanto incide que se encuentre o no la variable de clasificacion.
#MeanCecreaseGini:      Tiene que ver con las impourezas de cada nodo al hacer las divicion de variables

varImpPlot(iris.rf) 

#Este grafico nos indica cuales caracteristicas aportan menos
#Realizar 2 modelos:  Uno con todas las varialbes
#                     Otro con las mas importantes
#             Compara medidas de bondad y error
#             Seguir principio de parcimonia.


#Proximidad
iris.mds <- cmdscale(1 - iris.rf$proximity, eig=TRUE)
#escalamiento clasico multidimencional (la que explicao el profe en catedra)

op <- par(pty="s")
pairs(cbind(iris[1:4],iris.mds$points),cex=0.5,gap=0,
      col=c("red","green","blue")[as.numeric(iris$Species)],
      main="Iris Data: Predictos and MDS of Proximity Based on RandomForest")

par(op)

print(iris.mds$GOF)
MDSplot(iris.rf,iris$Species)
#Dim1 permite hacer un separamiento multidimencional entre clases
#Para dar solucion al problema de clasificaion

iris.rf100 <- randomForest(Species ~ ., data=iris,ntree =100, importance=TRUE, proximity=TRUE)
print(iris.rf100)

iris.rf600 <- randomForest(Species ~ ., data=iris,ntree =600, importance=TRUE, proximity=TRUE)
print(iris.rf600)

#Misma clasificacion; Menor error que 100 arboles.

#El metodo asegura que no va a existir sobreajuste

#Va a llegar a un punto en que no importara en numero de arboles ??


#Para probar diferentes numeros de vecindades: mtry
iris.rf100 <- randomForest(Species ~ ., data=iris,ntree =100, mtry=3, importance=TRUE, proximity=TRUE)
print(iris.rf100)

#Siemore hay que visualizar
plot(iris.rf600)



#Coordenadas paralelas
install.packages("MASS")
require(MASS)
parcoord(iris[,1:4], var.label=TRUE,col=c("red","green","blue")[as.numeric(iris$Species)])
legend("bottomright",legend=c("setosa","versicolor","virginica"),fill=2:4)








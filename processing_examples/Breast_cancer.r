---
title: "Examen_ML - Mathieu EMILY"
author: "Ibrahima SOW"
date: "23/10/2020"
output: html_document
---

Ici, nous disposons d'un jeu de données sur el diagnostic du cancer du sein. Notre objectfi est ici d'essayer de voir quelles variables ont le plus d'importance déterministe pour le diagnostic final. Et si possible, nous essayerons de construire un prédicateur. 
### Libraries loading 
```{r message=FALSE, warning=FALSE, paged.print=FALSE}
library(caret)
library(tidyr)
library(fields)       ## For image.plot 
```

### Data importing
```{r}
mydata <- read.table("BreastTrain.csv",sep=",", header=T, dec=",")
str(mydata)
mydata$X
```

Le dataset est bien composé d'une trenteine de variables explicatives (nous les nommerons nos features) et d'une variables expliquée (notre target) qui est ici le diagnostic du cancer du sein (il est soit bénin B, ou malin M). 

### Pre-process 

#### Target assessment 
Étude de la variable target 

#### Class transforming
```{r}
mydata$diagnosis <- as.factor(mydata$diagnosis)
class(mydata$diagnosis)
```

#### Plotting
```{r}
plot(mydata$diagnosis,pch=15)
```
Il semblerait y avoir un certain déséquilibre dans notre target. Le niveau Bénin est prés de 40% plus important que le second niveau du target Malin. Ceci pourrait poser problème dans notre modélisation à venir. Mais nous chercherons à en tenir rigueur du mieux que possible.  

#### Correlations check
Nous cherchons ici à étudier maintenant nos features. Nous regardons s'il existe d'éventuelles corrélations pour assainir au mieux notre dataset même si nous ne sommes pas en grande dimension (ici n>>p).
```{r}
cols <- names(mydata)[3:32]
mydata[cols] <- lapply(mydata[cols], as.numeric) 

corrplots <- cor(as.matrix(mydata[,3:32],)) %>%
            corrplot::corrplot()
```
Il semble y avoir certaines corrélations dans notre dataset. Nous faisons cependant le choix de ne pas en tenir rigueur et de continuer notre analyse sans supprimer des informations. 

Afin de modéliser et donc chercher à voir quelles variables induisent le caractère bénin ou malin d'un cancer du sein. Nous allons utiliser le package Caret. Ce dernier nous permettra de tester une multitude d'algorithmes de classifications tout en réalisant de manière autonome le "tuning" des hyperparamètres de chacun d'eux. 

### Splitting : dans caret (afin d'avoir un jeu de donnée pour entrainer nos modèles et un autre pour les tester)

Vu que nous allons utiliser une multitude de modèles, nous pensons qu'il eest aussi nécessaire de maximiser la reproducibilité (au travers de set.seed qui contrôle la "randomness" de nos "splits"")
```{r}
# Randomness controling (Reproducibility)
set.seed(007)
```


Nous choisissons aussi de faire un 80/20 (ce qui est le plus recommandé dans les documents)
```{r}
trainIndex <- createDataPartition(mydata$diagnosis, p = .8,times = 1 ,list=FALSE)
mydata.train <- mydata[ trainIndex,]
mydata.test  <- mydata[-trainIndex,]
```

### Modeling

#### Using caret 
Sachant qu'aucun modèle n'est meilleur que les autres pour tout jeu de donnée (no-free-lunch theorom), nous avons donc pris le parti de choisir 5 modèles parmi les plus utilisés dans la classification en machine learning et nous les comparerons selon 2 critères principaux. 
```{r pressure, echo=FALSE, out.width = '40%', fig.align="center"}
knitr::include_graphics(as.character(paste0(getwd(),"/nfl.jpg")))
```

Afin de "controler l'entrainement"" de nos modèles, nous choisissons une méthode de rééchantillonnage adapté. En effet, au vu du nombre d'observation assez limité, nous nous tournons sur une méthode très utilisée et assez bien documentée qui est le bootstrap. Il s'agit d'un rééchantillonage avec remise (ici nous en faisons une centaine). Nous faisons aussi up_sampling avec l'argument sampling afin de tenir compte de ce déséquilibre entre classes.
```{r}
# train.control 
fitControl.boot <- trainControl(method = "boot",number=100, sampling="up" )
```

```{r}
# Randomness controling (Reproducibility) # Pour plus de sécurité 
set.seed(007)
```

Nous utilisons donc les algorithmes suivants pour leur performance sur ce type de données. De plus, certains d'entre eux sont plus ou moins interprétables (e.g nnet). Ce qui permettra d'avoir un critère de choix au delà des metrics par la suite. 
```{r message=FALSE, warning=TRUE, include=FALSE}
speed_nnet <- system.time(
  mod.nnet <- train(
               diagnosis~ .,
               data=mydata.train,
               method="nnet",
               trControl = fitControl.boot,
              metric="Accuracy" 
            ) 
)
```

```{r}
speed_knn <- system.time(
mod.knn <- train(
             diagnosis~ .,
             data=mydata.train,
             method="knn", 
             trControl = fitControl.boot,
            metric="Accuracy"
          ) 
)
```

```{r}
speed_lda <- system.time(
  mod.lda <- train(
               diagnosis~ .,
               data=mydata.train,
               method="lda", 
               trControl = fitControl.boot,
              metric="Accuracy" 
            ) 
)
```

```{r}
speed_svm <- system.time(
  mod.svm <- train(
               diagnosis~ .,
               data=mydata.train,
               method="svmLinear2", 
               trControl = fitControl.boot,
              metric="Accuracy" 
            ) 
)
```

```{r}
speed_rf <- system.time(
mod.rf <- train(
             diagnosis~ .,
             data=mydata.train,
             method="rf", 
             trControl = fitControl.boot,
            metric="Accuracy" 
          ) 
)
```


### Model comparisons

 - Critère : Performance de prédiction
```{r}
#################
## Model comparison
#################
results <- resamples(list(
  nnet = mod.nnet, 
	knn = mod.knn, 
	lda = mod.lda,
  Svm = mod.svm,
	RF = mod.rf))
# summarize the distributions
summary(results)

# dot plots of results
dotplot(results)
```

#### Interprétation des metrics :
Aprés train/tests et validation successifs sur données rééchantillonnées avec remise, nous constatons que le SVM s'est très bien débrouillé comparé aux autres méthodes.  Nous constatons de plus que le KNN ne sont pas très performant sur nos données. Le Knn étant une méthode non-paramétrique qui plus est assez peu interprétatble est ici donc à proscrire par rapport aux autres méthodes. Les réseaux de neurone restent aussi assez peu performants. Au delà de l'accucaracy, le Kappa permet d'avoir une confirmation de nos interprétations ci-dessus car ce critère nous permet de prendre en considération le déséquilibre existant notre notre target. 

 - Critère : computational cost
Pour comparer les temps mis 
```{r}
# CPU time comparison 
time_df <- do.call(rbind, Map(data.frame, x= list("speed_nnet", "speed_knn", "speed_lda", "speed_svm", "speed_rf"), y=list(speed_nnet[1], speed_knn[1], speed_lda[1], speed_svm[1], speed_rf[1])))
time_df
```
Concernant toujours la comparaison de nos modèles, nous constatons dees différences assez importances. Comme montré ci-dessus, l'utilisation d'algorithmes tels les réseaux de neurone ou les forêts aléatories prennent beaucoup de temps au processeur. Ce qui est assez discriminant pour notre but final qui est de trouver des algorithmes à la fois explicatifs mais aussi assez performant par rapport à la gestion de la mémoire et du processeur. Le svm que nous avions identifié comme étant assez performant est ici assez peu gourmand niveau temps. Nous continuerons donc notre analyse qu'avec le svm, la lda malgré qu'ils soient très peu interprétables, ils très flexibles et donc réduisent au mieux notre biais. 

### Variables selection : quelle variable a le plus d'importance

```{r}
# Using varImps 
plot(varImp(mod.svm), 7)
plot(varImp(mod.lda), 7)

```
Nous constatons qu'avec nos deux modèles chosis, nous constatons une stabilité dans le choix de l'importance des variables. De fait, ce qui distingue le mieux un cancer du sein bénin ou malin c'est : concave.points_mean // concave.points_worst // perimeter_worst // radius_worst // area_worst ... 
Notre expertise limitée en la domaine ne nous permet cependant pas de pousser ces explications. 

### Prédire si un cancer est bénin ou malin





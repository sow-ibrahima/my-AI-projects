---
title: "<center>TD Parcours **Data Science** - Machine learning<center/>"
author: "<center>Groupe 1 - AVIT // BEELMEON // SOW</center>"
date: "`r format(Sys.time(), '%B %d, %Y')`"
output:
  pdf_document:
    toc: yes
  html_document:
    number_sections: yes
    toc: yes
---

Dans ce travail pratique destiné à nous familiariser aux méthodes de Machine Learning, nous allons présenter les Random Forests, une technique qui a connu un gros essort depuis sa mise en place à la fois pour des problématiques de regression et de classification. 

Après avoir brièvement étudié les données à disposition, nous allons travailler avec trois algorithmes de RandomForest de caret : rf, Rborist et RRF, et comparer leurs performances respectives.

# Importation des données

Nous commençons par importer la librairie **caret** qui contient le jeu de donnée **BloodBrain** sur lequel nous allons étudier l'efficacité des différents algorithmes de RandomForest, ainsi que des fonctions et méthodes utiles. Nous importons de plus la librairie tidyverse, qui est une collection de plusieurs librairies dont readr, dplyr, tidyr, ggplot2 pour pouvoir traiter et visualiser avec la philosophie "tidy" notre jeu de données. 

```{r message=FALSE, warning=FALSE}
library(caret)
library(tidyverse)
```

L'importation du jeu de données BloodBrain à partir de caret se fait facilement avec la fonction utils::data(). 

```{r}
data(BloodBrain)
```

**BloodBrain** contient un dataframe **bbbDescr** de 208 observations et 134 variables, ainsi qu'un vecteur réponse **logBBB** de taille 208. Il est issu des travaux de Mente et Lombardo des laboratoires de R&D du géant américain Pfizer. Ces dernières dans leur  [article](https://link.springer.com/article/10.1007/s10822-005-9001-7){target="_blank"} paru en 2005 dans le Journal of Computer-Aided Molecular Design ont développé des modèles qui prédisent le log ratio de la concentration de composés chimiques dans le cerveau et dans le sang.   
On étudie ici le rôle de l'interface sang/encéphale, d'où le nom du jeu de données : Blood Brain Barrier (=BBB). Pour chaque composé chimique sont calculés 3 ensembles de descripteurs moléculaires : **MOE 2D**, **rule-of-five** et **Charge Polar Surface Area**. Le dataframe bbbDescr est donc constitué par les variables explicatives X (il s'agit des features) et le vecteur réponse logBBB est notre variable expliquée (notre target), que l'on souhaite tenter de prédire. Nous sommes donc dans le cadre d'un apprentissage supervisé.  


Afin de mieux appréhender cette problématique et avant toute tentative de prédiction avec les RandomForest, nous allons étudier nos features et target.  
Nous avons décidé par souci de praticité de lier le vecteur réponse à notre dataframe de variables explicatives. La majorité des jeu de données pour le machine learning adoptant a priori cette configuration. 

```{r}
mydata <- cbind(bbbDescr, data.frame(logBBB))
```

**Étude des variables explicatives : Features**
Nous pouvons tout d'abord regarder la structure du jeu de données.

```{r}
# str(mydata[,-135]) # -135 pour ne pas considérer la colonne target (logBBB)
#Comme il y a 134 variables, on n'affiche pas celles-ci dans le Rmd pour des raisons de clareté.
```

L'ensemble des variables paraissent continues (integer/numeric). 

Nous cherchons aussi la présence de valeurs manquantes qui peuvent perturber les prédictions de nos modèles. 

```{r}
table(is.na(mydata[,-135]))
```

Nous constatons l'absence de données manquantes, le jeu de données est parfaitement complet.

Nous pouvons aussi étudier les potentielles corrélations entre variables. L'étude des corrélations entre variables explicatives est très importante dans une démarche de modélisation et de prédiction. Si une multitude de variables sont corrélées cela induit une redondance dans l'information, ce qui perturbe certains algorithmes prédictifs de base très utilisés comme les régressions.  
Nous utilisons ici la fonction corrplot du package du même nom.  
```{r,fig.align='center', echo=FALSE}
cm <- cor(mydata[,-135])
corrplot::corrplot(cm, order="hclust", tl.pos="n")
```

Le heatmap ci-dessus révèle la présence d'une multitude de corrélations (positives et négatives) dans nos variables explicatives, ce qui était attendu compte tenu du nombre important de variables et du contexte biologique. Selon les méthodes que l'on souhaite utiliser, certains auteurs recommandent d'éliminer les variables beaucoup trop corrélées.  
Dans notre cas, les Random Forest sont une famille de méthodes qui gèrent très efficacement la redondance de l'information, et nous pouvons nous permettre de garder toutes les variables (ce sont les algorithmes de Random Forest eux-mêmes qui vont gérer la redondance).

  
  
  
**Étude de la variable réponse : Target**

Nous pouvons d'abord sortir les première statistiques qui lui sont associées.
```{r}
summary(mydata$logBBB)
```

...Et la représenter graphiquement.

```{r,fig.align='center', echo=FALSE}
mydata$logBBB %>% 
  data.frame() %>% 
    ggplot(aes(x=logBBB)) +
      geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8) 
```

Nous pouvons observer que notre variable réponse suit une loi de densité proche de celle d'une loi normale.


```{r}
table(is.na(mydata$logBBB))
```

Là encore, il n'y a pas de données manquantes.   
  
  
Nous souhaitons donc effectuer une prédiction d'une valeur quantitative à partir de données quantitatives. Le jeu de données est complet (il n'y a pas de valeurs manquantes). Il y a beaucoup de variables prédictives (presque autant que d'observations !), et il y a une forte redondance de l'information entre celles-ci.  
L'utilisation de Random Forest pour notre prédiction de logBBB paraît tout à fait justifiée.


# Famille de méthodes : Random Forest

Après notre brève présentation du jeu de données, nous allons maintenant appliquer les Random Forest à celui-ci, et étudier leur efficacité.

## Description de la famille de méthodes

Les méthodes Random Forest découlent directement des arbres de décision.  

Le principe d'un arbre de décision est plutôt simple :  
Celui-ci est constitué de plusieurs noeuds. A chaque noeud, on sélectionne une des variables d'entrées, et selon la valeur de celle-ci, l'individu est envoyé vers un des noeuds suivants. A la fin de la ramification de l'arbre, une classe (s'il s'agit d'une classification) ou une valeur (s'il s'agit d'une régression) est associée à chaque "feuille" de l'arbre.  
L'arbre est créé par apprentissage, c'est à dire qu'un jeu de données d'entraînement permet de construire un arbre efficace sur celui-ci, en prévision d'une utilisation sur un jeu de données à prédire.  
Contrairement à la plupart des autres méthodes de Machine learning, les arbres de décision ont la vertue de ne pas être des boîtes noires : on sait exactement quels sont les critères utilisés pour la classification/régression.  
Malheureusement, en pratique, les arbres de décision sont peu efficaces car peu robustes, fortement soumis au surapprentissage (c'est à dire qu'un arbre pourra donner l'impression de bien fonctionner sur le jeu de données d'entraînement, mais aura des performances médiocres sur le jeu de données à prédire).  

Les Random Forest sont un développement visant à contrer ce problèmes :  
Au lieu d'utiliser un seul arbre de décision, on génère de très nombreux arbres de décision, et la prédiction finale est une moyenne des prédictions de tous ces arbres. Pour que les différents arbres de la Forêt soient différents les un des autres (s'ils étaient tous semblables, cela reviendrait à ne pas faire de Random Forest mais juste un arbre de décision), on utilise un bootstrap (rééchantillonnage avec remise) différent pour chacun des arbres, et des contraintes sur les prédicteurs à utiliser sont ajoutées aléatoirement.  
Cette stratégie est payante, et les Random Forest sont des méthodes de prédiction très efficaces et utilisées. On peut en revanche regretter qu'elles perdent du coup l'avantage des arbres de décision de ne pas être des boîtes noires.  
  
  
Dans notre approche de modélisation et de prédiction, nous utilisons comme indiqué précédemment le package caret. Ce dernier propose une multitude de familles de méthodes, et de nombreuses sous-méthodes dans chacune de ces familles de méthodes.  

Nous allons comparer ici trois Random Forest proposées par caret : rf, Rborist et RRFglobal. Ceux-ci ne sont que des exemples : caret met à disposition un total de 238 modèles de RandomForest différents.  
La différence entre ces nombreuses variations de Random Forest semble résider dans les paramètres optimisés lors de la création de la forêt.



## Outils dans R

# Mise en oeuvre sous R

On sépare le jeu de données en un jeu de données d'entraînement et un jeu de données de validation. Le premier va nous servir à entraîner nos modèles et générer les forêts aléatoires, le second nous permettra d'évaluer l'efficacité de nos régressions.

```{r}
set.seed(007)
trainIndex <- createDataPartition(y=mydata$logBBB, times=1, p=0.8, list=F)
# Train 
Train_set <- mydata[trainIndex,]
Test_set <- mydata[-trainIndex,]
```

Après avoir importé les librairies complémentaires de chacune des méthodes, on génère les 3 Random Forest différentes. Les modèles sont entraînés sur les données centrées et réduites, afin d'éviter que l'importance de certaines des variables ne soit surestimée par rapport à d'autres.  
De plus, on ajoute system.time afin de pouvoir comparer les temps d'exécution des 3 fonctions.
```{r message=FALSE, warning=FALSE}
library(randomForest)
library(Rborist)
library(RRF)
set.seed(007)
fit_Control_rcv <- trainControl(method = "repeatedcv",number=5) #pour chacun des modèles l'entraînement se fait par crossvalidations, et on effectue 5 cycles de crossvalidations.

speed_rf <- system.time(
mod_rf <- train(
             logBBB ~.,
             Train_set,
             method="rf", 
             trControl = fit_Control_rcv, 
             importance=T, 
             preProcess = c("center","scale") # Centrage réduction.
          )
)

speed_Rborist <- system.time(
mod_Rborist <- train(
             logBBB ~.,
             Train_set,
             method="Rborist", 
             trControl = fit_Control_rcv,
             importance=T, 
             preProcess = c("center","scale")
          )
)

speed_RRFglobal <- system.time(
mod_RRFglobal<- train(
             logBBB ~.,
             Train_set,
             method="RRFglobal", 
             trControl = fit_Control_rcv, 
             importance=T,
             preProcess = c("center","scale")
          )
)

```


On compare les temps de calcul pour chacune des méthodes, qui diffèrent fortement. Cependant, nous avons pu constater que sur nos différentes machines, les vitesses relatives n'étaient pas les mêmes : sur deux de nos ordinateurs, Rborist était (de loin) la méthode la plus lente, tandis qu'elle était la plus rapide sur le troisième.

```{r,fig.align='center', echo=FALSE}
CPUtime_df <- do.call(rbind, Map(data.frame, algorithms= list("rf", "Rborist", "RRFglobal"), speed=list(speed_rf[1], 
                                                                                       speed_Rborist[1], speed_RRFglobal[1])))

CPUtime_df %>% 
  ggplot(aes(x=algorithms, y=speed)) + 
  geom_bar(stat = "identity", width = 0.5, fill = c("#E04E62", "#4FA1FF", "#000000")) +  theme_minimal() +
  ggtitle("CPU time for each of the random forest algorithm") +
  xlab("algorithms") + ylab("speed (sec)") +
  theme(plot.title = element_text(hjust = 0.5))

```

Nous pouvons également observer quelles sont les variables utilisées majoritairement par chacun des trois modèles prédictifs. On constate que ce ne sont pas les mêmes : cela découle de la forte corrélation entre les variables précédemment évoquée (d'ailleurs, si on relançait les algorithmes, ce ne serait sans doute pas les mêmes variables pour un algorithme donné, il y a une part d'aléa).  

L'intérêt des méthodes de forêts aléatoires réside aussi dans la notion d'importance de variables. En effet, on peut, en regardant quelles sont les variables utilisées dans la décision de chaque arbre de la forêt aléatoire, déterminer quelles sont les variables le splus utilisées par la forêt.  

Nous pouvons constater avec le classement (top 10) des variables les plus importantes que mêmes si les 3 méthodes ont des variables les plus importantes communes, il y a tout de même des différences. On peut aussi souligner que rf accorde une importance prépondérante à une variable en particulier (fnsa3).  
Comme évoqué précédemment, ces différences sont dues au Feature selection, qui a notamment des effets sur l'ordre d'importance des variables explicatives.

```{r}
plot(varImp(mod_rf),10)
plot(varImp(mod_Rborist), 10)
plot(varImp(mod_RRFglobal), 10)
```

On constate que les 3 modèles ont des RMSE (écarts quadratiques moyens) semblables sur les jeux d'entraînements, RRFglobal étant légèrement moins bon (RMSE plus élevé).  
Il est néanmoins bon de rappeler que la capacité d'un modèle à coller aux données d'entraînement n'est pas forcément représentative de sa capacité prédictive (problématique du surapprentissage).
```{r}
max(mod_rf$results$RMSE)
max(mod_Rborist$results$RMSE)
max(mod_RRFglobal$results$RMSE)
```

Les critères de performance présentés ici sont le MAE, le RMSE et le Rsquared (qui sont trois mesures de l'erreur).  
Caret fournit en effet une fonction caret::resamples qui permet de collecter et d'analyser les résultats d'entraînement de modèles. La visualisation peut par la suite se faire facilement avec un dotplot également disponible dans le package. 
```{r,fig.align='center', echo=FALSE}
#################
## Model comparison
#################
results <- resamples(list(
	RF = mod_rf,
	Rborist = mod_Rborist, 
	RRFglobal = mod_RRFglobal))

# dot plots of results
dotplot(results)
```

Nous constatons avec les trois critères présentés ci-dessus (le Mean Absolute Error - le Root Mean Square Error et le R^2) que les trois méthodes ont des performances très semblables. Ces trois critères sont les plus utilisés dans les problématiques de regression.  

Au delà de ces critères fournis directement par caret::train(), il est nécessaire d'étudier la performance de nos modèles sur des données autres que celles sur lesquelles ils ont été entrainés. Ce qui est un écueil à éviter dans une démarche cohérente et rigoureuse de machine learning.  

Afin de réaliser l'étude de la performance nous utilisons la fonction caret::predict() en lui fournissant le modèle sur lequel il va se baser et le Test_set (données de test).
```{r}
pred_rf <- predict(mod_rf, newdata = Test_set)
pred_Rborist <- predict(mod_Rborist, newdata = Test_set)
pred_RFglobal <- predict(mod_RRFglobal, newdata = Test_set)
```

En régression, contrairement à une situation de classification, il n'est pas possible d'utiliser une matrice de confusion pour évaluer la performance de prédiction de nos modèles de Random Forest. <br> Nous nous proposons ici d'utiliser une technique assez équivalente où l'on calcule la corrélation entre les valeurs prédites de logBBB et le logBBB dans le Test_set. 
<center>**Plus la corrélation entre ces valeurs est forte, plus notre modèle sera performant**.</center> <br>

Nous réprésentons graphiquement la performance de prédiction de chacune de nos méthodes. <br>

```{r,fig.align='center', echo=FALSE}
pred_perf<- data.frame(algorithms = c("rf", "Rborist", "RRFglobal"),
                            performance   = c(cor(Test_set$logBBB, pred_rf), 
                                         cor(Test_set$logBBB, pred_Rborist), 
                                        cor(Test_set$logBBB, pred_RFglobal))) 

pred_perf %>%
    ggplot(aes(x=algorithms, y=performance)) + 
    geom_bar(stat = "identity", width = 0.5, fill = c("#E04E62", "#4FA1FF", "#000000")) +  theme_minimal() +
    ggtitle("Prediction performance for each of the random forest algorithm") +
    xlab("algorithms") + ylab("Performance (%)") + ylim(0, 1) +
    theme(plot.title = element_text(hjust = 0.5))

```

<center>**Il ne semble pas y avoir de différence significative en termes de performance de prédiction entre nos trois méthodes.**</center>  <br>

Nos trois modèles fournissent des prédictions sur le jeu de test correlées à 75-80% avec la variable à prédire (il faudrait voir, en fonction du contexte, s'il s'agit d'un taux de corrélation acceptable ou non).  

En guise de rappel, nous pouvons distinguer les trois méthodes que nous avons utilisées par les [hyperparamètres](https://fr.wikipedia.org/wiki/Hyperparam%C3%A8tre){target="_blank"} optimisés.  

Pour rf caret n'optimise que mtry, pour Rborist predFixed et minNode sont optimisés et pour RRFglobal c'est mtry et coefReg qui sont optimisés. 

```{r}
mod_rf$modelInfo$parameters[]
mod_Rborist$modelInfo$parameters[]
mod_RRFglobal$modelInfo$parameters[]
```



**Pour finir** :  
- Même si elles peuvent être considérées comme des boîtes noires, les méthodes de Random Forest ont connu une ascension très importante depuis leur initiation par Breiman et al. (2001) pour répondre d'abord aux insuffisances des arbres de décison puis à celles du [bagging](https://dataanalyticspost.com/Lexique/bagging/){target="_blank"} (la transition entre arbres de décision et random forests).  
- il en existe plusieurs déclinaisons disponibles dans caret. Nous en avons présenté ici 3 : rf, Rborist et RRFglobal.  
- au delà de leur efficacité démontrée en classification comme en regression elles nous ont paru intéressantes, permettant notamment de bien voir quelles étaient les variables les plus utilisées par le modèle de prédiction.  


# Session Info
```{r, echo=FALSE}
sessionInfo()
```

# Références

1. R Core Team (2020). R: A language and environment for statistical computing. R Foundation for Statistical
  Computing, Vienna, Austria. URL https://www.R-project.org/.
2. Max Kuhn (2020). caret: Classification and Regression Training. R package version 6.0-86.
  https://CRAN.R-project.org/package=caret
3. https://topepo.github.io/caret/

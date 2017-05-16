# programme de demarrage - atelier "Introduction a la data science" - Ladies Of Code - 16 mai 2017
# source des donnees: dataset du challenge MDSF du 10-05-2016

library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)
library(Hmisc)

# chargement des donnees

dossier_data = "FRENCHDATA/ecole42_10052016"
dossier_data = "E:/ecole42_10052016"
chemin_train = paste0(dossier_data, "/boites_medicaments_train.csv")
raw = read.table(chemin_train, sep = ";", header = T, fill = T, quote = "", encoding = 'UTF-8')

# fonction d'erreur entre prediction et realite

mape_error = function(y, ypred){mean(abs((y - ypred)/y))*100}

# exploration

str(raw)
ggplot(raw, aes(prix)) + geom_histogram(binwidth = 1) # => prendre le log du prix pour le modele !

# definition train/test

# partition equilibree sur log(prix)
train_test = createDataPartition(log(raw$prix), times = 2, p = 0.7, list = F) 
train = raw[train_test, ]
test = raw[-train_test, ]
# Cree une variable categorie pour pouvoir separer les datasets apres la preparation des donnees
train$categorie = 'train' 
test$categorie  = 'test'
# Combine train et test pour avoir un seul dataset a manipuler pour le feature engineering
full = as.data.frame(bind_rows(train, test)) 

# feature engineering 

# aggréger les niveaux rare en 1 seul niveau ("OTHERS")
full$libelle_combined      = combine.levels(full$libelle,      minlev=.01)
full$titulaires_combined   = combine.levels(full$titulaires,   minlev=.01) 
full$substances_combined   = combine.levels(full$substances,   minlev=.01)
full$forme.pharma_combined = combine.levels(full$forme.pharma, minlev=.01)
full$voies.admin_combined  = combine.levels(full$voies.admin,  minlev=.01)

# separation train/test apres le feature engineering

train = full[full$categorie == "train", ]
test = full[full$categorie == "test", ]

# definition du modele

doModel <- function(y, train, test) {

  rf = randomForest(data  = train,
                    log(prix) ~ nb_plaquette + nb_ampoule + nb_flacon + nb_tube + nb_stylo + nb_seringue + 
                      nb_pilulier + nb_sachet + nb_comprime + nb_gelule + nb_film + nb_poche + 
                      nb_capsule + nb_ml + statut + etat.commerc + date.declar.annee + date.amm.annee + 
                      agrement.col + tx.rembours + forme.pharma_combined + voies.admin_combined + 
                      statut.admin + substances_combined + libelle_combined + titulaires_combined, 
                    ntree = 10)

  p = exp(predict(rf, test))

}

# construction des ensembles de cross-validation

K = 3 # on partitionne l'echantillon en K pour avoir K estimations de la performance du modele
set.seed(42) # 
train$cv_id = sample(1:K, nrow(train), replace = T)

# entrainer un modele sur chaque ensemble de cv 
# et estimer l'erreur sur la partie des donnees non utilisee pour l'entrainement

for(i in 1:K){
  train_cv = train[train$cv_id != i, ]
  test_cv  = train[train$cv_id == i, ]
  
  p = doModel(log(train_cv$prix),
             subset(train_cv, select=-c(cv_id)),
             subset(test_cv, select=-c(cv_id, prix)))
  
  mape = mape_error(y = test_cv$prix, ypred = p)
  print(paste("*** estimation de l'erreur cv:", mape))
}

# calculer l'erreur sur le test set, en utilisant tout le train set pour l'entrainement, avec les parametres trouves ci-dessus:
# (ne pas utiliser pour l'optimisation des parametres !)
p = doModel(log(train$prix),
            subset(train, select=-c(cv_id)),
            subset(test, select=-c(prix)))
mape = mape_error(y = test$prix, ypred = p)
print(paste("*** estimation de l'erreur sur le test set:", mape))

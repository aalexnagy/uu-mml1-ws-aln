################################################################################
######################## MML1 - WorkShop:VYSTUPNI PRACE ########################
################################################################################
#### Jmeno: 		Alexander
#### Prijmeni:	Nagy
################################################################################
################################################################################

library(tensorflow)
library(keras)
library(ANN2)


################################################################################
################################### Cast - 1 ###################################
################################################################################

############ PRIPRAVA DAT ################
# nactu data
	mnist <- dataset_mnist()
	str(mnist)

# pro praktickou práci vytáhám objekty z listu- abych mel jen daná pole:
	c(train_images, train_labels) %<-% mnist$train
	c(test_images, test_labels) %<-% mnist$test
	d_train <- train_images # trenovaci data
	d_test <- test_images # testovaci data
	
	## Uprava dimenzi ##
	# trenovaci data :
	dim(d_train)
	dim(d_train)<-c(60000, 28*28)
	dim(d_train) # kontrola, ze se prevedlo
	# testovaci data :
	dim(d_test)
	dim(d_test)<-c(10000, 28*28)
	dim(d_test) # kontrola, ze se prevedlo

### KONTROLA SPRAVNEHO PREVEDENI DIMENZI - Podivam se, co mam za obrazky ###
	# pripravim si plot
		plot(c(0,28), c(0,28))
	# zobrazim obr 1 z d_train (5) v plotu:
		#rasterImage(d_train[1,,]/255,0,0,28,28) ## OBSO! : nepouzivam po zmene dimenzi
		rasterImage((matrix(d_train[1,],28,28)/255),
            		xleft=0,
		            xright=28,
            		ybottom=0,
		            ytop=28)
	# - jake cislo je na obr 1 d_train (5)???
		train_labels[1]
	# zobrazim obr 10 v plotu
		rasterImage((matrix(d_test[10,],28,28)/255),
            		xleft=0,
		            xright=28,
            		ybottom=0,
		            ytop=28)
	# jake cislo jen a obr 10 ???
		test_labels[10]


############ BUDOVANI NN v ANN2 ################
### Natrenuji NN
fit<-neuralnetwork(X=d_train, y=train_labels,hidden.layers = 140,
                   activ.functions = "relu", optim.type = "adam",
                   n.epochs = 50, batch.size=3000)
### Zobrazim vyvoj trenovani NN v grafu
plot(fit)
f<-predict(fit, d_test)
ft<-table(f$predictions,test_labels) # porovnaní skutecnyh a predikovanych hodnot
ft
sum(diag(ft))/sum(ft) # kolik bylo spravne identifikovanych hodnoty 

# ## pokud se mi nezda, tak dotrenuji
# train(fit,X=d_train,y=train_labels,n.epochs=5,batch.size = 8000)
# plot(fit)
# f<-predict(fit, d_test)
# ft<-table(f$predictions,test_labels)
# ft
# sum(diag(ft))/sum(ft)

# vytvoreni slovniku nazvu trid
class_names=c(0,1,2,3,4,5,6,7,8,9)


# Zobrazeni prvnich 100 cisel pro predikci na "ft" a zda byla urcena (ano = modra; ne = cervena)
# uprava okna pro zobrazovani vice vysledku za raz (vybral jsem 10x10)
par(mfcol=c(10,10),mar=c(0,0,1,0),xaxs="i",yaxs="i") 

for(i in 1:100){
  pic<-matrix(d_test[i,],ncol=28)
  pic<-t(apply(pic, 2, rev)) # musim transponovat, aby se vykreslilo; jde o vlastnost fce "image"
	# barevne odliseni spatnych zarazeni
	if(as.numeric(f$predictions[i])==test_labels[i]){
    barva="blue"
  }else{
    barva="red"
  }
  
  image(1:28,1:28,pic,col=gray((0:255)/255), xaxt="n",yaxt="n",
        main=paste(class_names[as.numeric(f$predictions[i])+1],
                   " (", class_names[test_labels[i]+1],")"),
        cex.main=.9, col.main=barva)
}


################################################################################
################################### Cast - 2 ###################################
################################################################################
################################################################################
######################## Convultion Neural Netvork(CNN) ########################
################################################################################


############ DEFINICE FILTRU ################
## Definive FCE "myfilter"
#### Co dela: Aplikuje filtr na obrazek
	myfilter<-function(M,Fi){
    M2<-matrix(0,nrow=(nrow(M)-2),ncol=(ncol(M)-2))
    for(j in 2:(ncol(M)-1)){
      for(i in 2:(nrow(M)-1)){
        M2[i-1,j-1]<- sum(M[(i-1):(i+1),(j-1):(j+1)]*Fi)
      }
    }
    (M2-min(M2))/(max(M2)-min(M2))
  }

############ DEFINICE PARAMETRU FILTRU ############
### Filtr: Detekce hran
Fi <-matrix(c(rep(1,3),rep(0,3),rep(-1,3)),3) # vertikalni hrany
Fi2 <- t(Fi) # horizontalni hrany
### Filtr: Zaostreni
Fi3 <- matrix(c(rep(-1,3),c(-1,8,-1),rep(-1,3)),3) #zaostreni
### Filtr: Rozostreni
Fi4 <- matrix(c(rep(1,3),c(0,2,0),rep(0.5,3)),3) # rozostruje


################ DEFINICE POOL ################
## definice fce "mean_pool"
	mean_pool<-function(M,k=c(2,2)){
		j=floor(ncol(M)/k[1])
		i=floor(nrow(M)/k[2])
		M2<-matrix(0,ncol=j,nrow=i)
		
		for(kj in 1:j){
			for(ki in 1:i){
				M2[ki,kj]=mean(M[1:k[1]+k[1]*(ki-1),1:k[2]+k[2]*(kj-1)])
			}
		}
			M2
	}
## definice fce "max_pool"
	max_pool<-function(M,k=c(2,2)){
		j=floor(ncol(M)/k[1])
		i=floor(nrow(M)/k[2])
		M2<-matrix(0,ncol=j,nrow=i)
		
		for(kj in 1:j){
			for(ki in 1:i){
				M2[ki,kj]=max(M[1:k[1]+k[1]*(ki-1),1:k[2]+k[2]*(kj-1)])
			}
		}
		M2
	}

## definice fce "min_pool"
	min_pool<-function(M,k=c(2,2)){
	  j=floor(ncol(M)/k[1])
	  i=floor(nrow(M)/k[2])
	  M2<-matrix(0,ncol=j,nrow=i)
	  
	  for(kj in 1:j){
	    for(ki in 1:i){
	      M2[ki,kj]=min(M[1:k[1]+k[1]*(ki-1),1:k[2]+k[2]*(kj-1)])
	    }
	  }
	  M2
	}

################ APLIKACE FILTER + POOL NA DATA ################
	## Pouzito: 1x filtr na ostrost; 2x max_pool();
	d_train2 <- matrix(nrow = nrow(d_train), ncol = 36) # col = 36, protoze 6x6=36 hodnot
	for (i in 1:nrow(d_train)) {
	  obr<-matrix(d_train[i,],ncol=28)
	  d_train2[i,] = c(max_pool(max_pool(myfilter(obr/255, Fi3))))
	}
	## Pouzito: 1x filtr na ostrost;  2x max_pool();
	d_test2 <- matrix(nrow = nrow(d_test), ncol = 36) # col = 36, protoze 6x6=36 hodnot
	for (i in 1:nrow(d_test)) {
	  obr<-matrix(d_test[i,],ncol=28)
	  d_test2[i,] = c(max_pool(max_pool(myfilter(obr/255, Fi3))))
	}

	
  # # test jak se aplikoval filtr a pooly
  # pic<-matrix(d_train2[1,],ncol=6)
  # plot(c(0,6),c(0,6))
  # rasterImage(pic,0,0,6,6)
	
################################################################################
################ Natrenuji NN na upravenych treningovich datech ################
	
	fit2<-neuralnetwork(X=d_train2, y=train_labels,hidden.layers = 140,
	                   activ.functions = "relu", optim.type = "adam",
	                   n.epochs = 50, batch.size=3000)
	### Zobrazim vyvoj trenovani NN v grafu
	plot(fit2)
	f2<-predict(fit2, d_test2)
	ft2<-table(f2$predictions,test_labels) # porovnaní skutecnyh a predikovanych hodnot
	ft2
	sum(diag(ft2))/sum(ft2) # kolik bylo spravne identifikovanych hodnoty 
	# ## pokud se mi nezda, tak dotrenuji
	# train(fit2,X=d_train2,y=train_labels,n.epochs=30,batch.size = 8000)
	# plot(fit2)
	# f2<-predict(fit2, d_test2)
	# ft2<-table(f2$predictions,test_labels)
	# ft2
	# sum(diag(ft2))/sum(ft2)

	################ vytvoreni slovniku nazvu trid; jiz vytvoren vyse
	# class_names=c(0,1,2,3,4,5,6,7,8,9)
	
	################ Zobraz. prvnich 100 cisel predikce na "ft2" ################
	##################### Urceni: ano = modra; ne = cervena #####################
	par(mfcol=c(10,10),mar=c(0,0,1,0),xaxs="i",yaxs="i") 
	
	for(i in 1:100){
	  pic<-matrix(d_test2[i,],ncol=6)
	  pic<-t(apply(pic, 2, rev)) # musim transponovat, aby se vykreslilo; jde o vlastnost fce "image"
	  # barevne odliseni spatnych zarazeni
	  if(as.numeric(f2$predictions[i])==test_labels[i]){
	    barva="blue"
	  }else{
	    barva="red"
	  }
	  
	  image(1:6,1:6,pic,col=gray((0:255)/255), xaxt="n",yaxt="n",
	        main=paste(class_names[as.numeric(f2$predictions[i])+1],
	                   " (", class_names[test_labels[i]+1],")"),
	        cex.main=.9, col.main=barva)
	}
	

	
# neuronovou sit pomoci ANN (kdo chce muze pak vysledek porovat S TF)
# kdyby jste zkusili vlastni typ filtru(CNN) a poolingu!!!
# detekce hran (horizontalni i vertilakni )+ pooling

# obyč detekce čisel pomocí jen ANN
# detekce čisel s CNN (vámi definované fitlry!!!) + pooling
# porovat výsledky!! presnost na test setu a na train setu!
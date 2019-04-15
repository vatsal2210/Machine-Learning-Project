# import libraries
library(cluster)
library(ggplot2)
library(factoextra)

#start execution time for whole code
start.time <- Sys.time()

#import dataset
data <- read.csv("C:/train.csv")
head(data)

# removing index values
data <- data[, -1]
head(data)

# renaming columns

colnames(data)[1] <- "Churches"
colnames(data)[2] <- "Resorts"
colnames(data)[3] <- "Beaches"
colnames(data)[4] <- "Parks"
colnames(data)[5] <- "Theatres"
colnames(data)[6] <- "Museums"
colnames(data)[7] <- "Malls"
colnames(data)[8] <- "Zoo"
colnames(data)[9] <- "Restaurants"
colnames(data)[10] <- "Pubs/Bars"
colnames(data)[11] <- "Local Services"
colnames(data)[12] <- "Burger/Pizza Shops"
colnames(data)[13] <- "Hotels/Other Lodgings"
colnames(data)[14] <- "Juice Bars"
colnames(data)[15] <- "Art Galeries"
colnames(data)[16] <- "Dance Clubs"
colnames(data)[17] <- "Swimming Pools"
colnames(data)[18] <- "Gyms"
colnames(data)[19] <- "Bakeries"
colnames(data)[20] <- "Beauty & Spas"
colnames(data)[21] <- "Cafes"
colnames(data)[22] <- "View Points"
colnames(data)[23] <- "Monuments"
colnames(data)[24] <- "Gardens"

#print the dataset values
head(data)
summary(data)

#compute k-means
#start time for k-means
start.time1 <- Sys.time()
set.seed(123)

#determine the no: of clusters
km.res <- kmeans(data, 3, nstart = 25)

# Cluster number for each of the observations
km.res$cluster

# Cluster size
km.res$size

# Cluster means
km.res$centers

# print result
print(km.res)

#end the timer
end.time1 <- Sys.time()
time.taken1 <- end.time1 - start.time1
sprintf("The time taken for k-means %.4s sec", time.taken1)


# Compute k-medoids
# start timer
start.time2 <- Sys.time()

#Plot k-medoids
fviz_cluster(pam.res)
pam.res <- pam(scale(data), 3)

#print the result of k-medoids
pam.res$medoids
head(pam.res$cluster)

#Display the cluster plot for k-medoids
clusplot(pam.res, main = "Cluster plot, k = 3", 
         color = TRUE)

#end the timer
end.time2 <- Sys.time()
time.taken2 <- end.time2 - start.time2
sprintf("The time taken for k-medoids %.4s sec", time.taken2)


#compute CLARA

#start timer
start.time3 <- Sys.time()

#determine no: of clusters
clara(data, 3, samples = 15)
set.seed(1234)

#compute clara
clarax <- clara(data, 3, samples=50)

# Cluster plot
fviz_cluster(clarax, stand = FALSE, geom = "point",
             pointsize = 1)

# Silhouette plot
plot(silhouette(clarax),  col = 1:24, main = "Silhouette plot")  
pam(data, 3)
pam.res <- pam(scale(data), 3)
head(pam.res$cluster)
clusplot(pam.res, main = "Cluster plot, k = 3", 
         color = TRUE)

#end the timer
end.time3 <- Sys.time()
time.taken3 <- end.time3 - start.time3
sprintf("The time taken for CLARA %.4s sec", time.taken3)

#end the timer
end.time <- Sys.time()
time.taken <- end.time - start.time
sprintf("Code execution time %.4s sec", time.taken)


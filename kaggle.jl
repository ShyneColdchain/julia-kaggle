# Load packages
Pkg.add("Images")
Pkg.add("DataFrames")

using Images
using DataFrames

pwd()
homedir()
cd("$(homedir())/Documents/Coding/Julia/trial")
pwd()
path = pwd()

# Kaggle Julia tutorial: identifying characters from Google Street View images
# showcasing Julia: intuitive syntax and design

###################################################
############ Preprocessing Data ###################
###################################################


# typeData = train or test
# labelsInfo = IDs of each image to be read
# image = trainResized or testResized = 20x20
# so, imageSize = 400
# path = location of data files

function read_data(typeData, labelsInfo, imageSize, path)
  # initilize x matrix
  x = zeros(size(labelsInfo, 1), imageSize)

  #@printf("This is labelsInfoTrain: %s", labelsInfoTrain)
  #@printf("This is labelsInfoTrain[ID]: %s", labelsInfoTrain[:ID])

  for (index, idImage) in enumerate(labelsInfo[:ID])

    nameFile = "$(path)/data/$(typeData)Resized/$(idImage).Bmp"
    #read image file
    #if typeData == "test"
    #  @printf("typeData: %s", typeData)
    #  nameFile = "$(path)/data/testResized/$(idImage).Bmp"
    #end

    img = imread(nameFile)

    # check if correct size
    assert(size(img) == (20,20))

    # convert img to float values
    temp = float32(img)

    # convert color images to gray images
    # by taking average of the color scales
    #if ndims(temp) == 3
    #  temp = mean(temp.data, 1)
    #end
    # or simply convert all to grayscale
    temp = convert(Image{Gray}, temp)
    #@printf("This is temp-gray: %s", temp)

    temp_img = reinterpret(Float32, float32(temp))

    img_vector = reshape(temp_img, 1, imageSize)

    # transform image matrix to vector and store in data matrix
    #@printf("x: %s %s\n", index, idImage)            # print out status
    x[index, :] = img_vector
  end

  return x

end

imageSize = 400            # 20 x 20 pixel
path = pwd()
@printf("The path: %s", path)
@printf("Image size: %s", imageSize)

# read info about train data (IDs)
labelsInfoTrain = readtable("$(path)/data/trainLabels.csv")
# read info about test data (IDs)
labelsInfoTest = readtable("$(path)/data/sampleSubmission.csv")

num_images_train = size(labelsInfoTrain)[1]
num_images_test = size(labelsInfoTest)[1]

# read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

# read information about test data (IDs)
labelsInfoTest = readtable("$(path)/data/sampleSubmission.csv")

# read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)

# now xTrain and xTest are training and testing matrices, respectively

# get only first character of string - convert from string to character
yTrain = map(x -> x[1], labelsInfoTrain[:Class])
yTrain = int(yTrain)         # convert from character to integer

###################################################
############### Training Data #####################
###################################################

# need ml algorithm that learns patterns in images
# that identify the character in the label
# thus = random forest

Pkg.add("DecisionTree")
using DecisionTree

# 1. number of features to choose at each split - sq-root(number of features)
# 2. number of trees                            - bigger = better
# 3. ratio of subsampling

# number of features to tree at each split = split_features
# split_features = sq-root(number of features) = sq-root(400) = 20
split_features = 20
num_trees = 50
ratio_sub = 1.0       # ratio of subsampling

# trained model
model = build_forest(yTrain, xTrain, split_features, num_trees, ratio_sub)
@printf("The trained model: %s", model)

## Trees: 50
## Avg Leaves: 2188.0
## Avg Depth: 19.28

# apply trained model to test data
predict_test = apply_forest(model, xTest)

# check if wrong
wrong = find(predict_test.!=yTrain)
@printf("Wrong: %s", wrong)          # wrong = 3055
char(predict_test[3055])
char(yTrain[3055])

# so, our random forest thought that the '1' at 3055
# looked more like an 'E' (it predicted it as an E)

# Convert integer predictions to character
labelsInfoTest[:Class] = char(predict_test)

# save predictions
writetable("$(path)/juliaSubmission.csv", labelsInfoTest, separator=',',header=true)

# check accuracy: n-fold cross validation - used to test performance of model
folds = 4
accuracy = nfoldCV_forest(yTrain, xTrain, split_features, num_trees, folds, ratio_sub)
@printf("4 fold accuracy: $(mean(accuracy))")


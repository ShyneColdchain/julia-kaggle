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

# typeData = train or test
# labelsInfo = IDs of each image to be read
# image = trainResized or testResized = 20x20
# so, imageSize = 400
# path = location of data files

function read_data(typeData, labelsInfo, imageSize, path)
  # initilize x matrix
  x = zeros(size(labelsInfoTrain, 1), imageSize)

  #@printf("This is labelsInfoTrain: %s", labelsInfoTrain)
  #@printf("This is labelsInfoTrain[ID]: %s", labelsInfoTrain[:ID])

  for (index, idImage) in enumerate(labelsInfoTrain[:ID])
      #read image file
      nameFile = "$(path)/data/trainResized/$(idImage).Bmp"
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

# neural networks in Julia: tutorial
# data: winequality-red.csv
Pkg.add("DataFrames")
Pkg.add("Gadfly")
Pkg.add("PyPlot")
using DataFrames

pwd()
cd("./Documents/Coding/Julia/trial")
pwd()

versioninfo()

# read in df to train
train_df = readtable("data/winequality-red.csv", separator=';')
size(train_df)
typeof(train_df)


###############################################################
#################  {  Draw Histogram  }  ######################
###############################################################

# get overview of data (stat)
describe(train_df)

# Gadfly
using Gadfly

# get value counts of quality col
_, count = hist(train_df[:quality])
class = sort(unique(train_df[:quality]))
value_counts = DataFrame(count=count, class=class)
println(value_counts)
p = plot(value_counts, x="class", y="count",
         Geom.bar(), Guide.title("Class distributions for quality")
         )
draw(PNG(14cm, 10cm), p)

###############################################################
##############  {  train and test files }  ####################
###############################################################

# create train and test data splits
y = train_df[:quality]
x = train_df[:, 1:11] # matrix of all except quality
# vector() and matrix() from blog post
y = array(y)
x = array(x)

n = length(y)
is_train = shuffle([1:n] .> floor(n * .25))

x_train,x_test = x[is_train,:],x[!is_train,:]
y_train,y_test = y[is_train],y[!is_train]

# print to user
println("Total # observations: ", n)
println("Training set size: ", sum(is_train))
println("Test set size: ", sum(!is_train))

###############################################################
#################  {  Standard scalar  }  #####################
###############################################################

# neural networks are scale sensitive
# weights and biases are initialized before backpropagation

# similar to C structs - make scalars
# type = composition type
# = struct (c)
# a composition type with functions = object
type StandardScalar
  mean::Vector{Float64}
  std::Vector{Float64}
end

# initialize empty scalar
function StandardScalar()
  StandardScalar(Array(Float64, 0), Array(Float64, 0))
end

# compute mean and std of each col
function fit_std_scalar!(std_scalar::StandardScalar, x::Matrix{Float64})
  n_rows, n_cols = size(x_test)
  std_scalar.std = zeros(n_cols)
  std_scalar.mean = zeros(n_cols)

  for i = 1:n_cols
    std_scalar.mean[i] = mean(x[:,i])
    std_scalar.std[i] = std(x[:,i])
  end
end


###############################################################
#####################  {  Transform  }  #######################
###############################################################

# IMPORTANT: unlike NumPy (Python) or R
# array x array = array multiplication (not element-wise multiplication)
# this is like Matlab (but unlike R and Python)
array1 = [1, 2, 3]
array2 = [4, 5, 6]
array1 * array2           # ERROR

# for element wise multiplication:
# array1 .* array2
array1 .* array2           # vector
array1 .* array2'          # matrix

# further vectorize the transformation
function transform(std_scalar::StandardScalar, x::Matrix{Float64})
  # element wise subtraction of mean and division of std
  (x .- std_scalar.mean') ./ std_scalar.std'
end

# fit and transform
function fit_transform!(std_scalar::StandardScalar, x::Matrix{Float64})
  fit_std_scalar!(std_scalar, x)
  transform(std_scalar, x)
end

# fit scalar on training data and then transform the test
std_scalar = StandardScalar()

n_rows, n_cols = size(x_test)

# cols before scaling
println("Col means before scaling: ")
for i = 1:n_cols
  # C printf function
  @printf("%0.3f ", (mean(x_test[:, i])))
end

println("std_scalar: ", std_scalar)
println("x_train: ", x_train)
x_train = fit_transform!(std_scalar, x_train)
x_test = transform(std_scalar, x_test)

# after transforming
println("\n Col means after scaling:")
for i = 1:n_cols
  @printf("%0.3f ", (mean(x_test[:,i])))
end

###############################################################
##############  {  Artificial Neural Networks }  ##############
###############################################################

# artificial neural network package
# strong ability to model non-linear relationships
Pkg.clone("https://github.com/EricChiang/ANN.jl.git")
using ANN

# neural networks: layers of linear models
# initialize by specifying the size of one layer,
# 'hidden layer'
# (larger hidden layer = likely to overfit
# smaller hidden layer = overgeneralize)
ann = ArtificialNeuralNetwork(12)

# epochs = length of training
# alpha = rate of training
# lambda = scales penalty to control over-fitting
fit!(ann, x_train, y_train, epochs=30, alpha=0.1, lambda=1e-5)

y_proba = predict(ann, x_test)
y_pred = Array(Int64, length(y_test))

# translate class index to label
for i in 1:length(y_test)
  y_pred[i] = ann.classes[indmax(y_proba[i, :])]
end

println("Prediction accuracy: ", mean(y_pred .== y_test))

# confusion matrix
function confusion_matrix(y_true::Array{Int64,1}, y_pred::Array{Int64,1})
    classes = sort(unique([unique(y_true), unique(y_pred)]))
    cm = zeros(Int64, length(classes), length(classes))

    for i in 1:length(y_test)
        # translate label to index
        true_class = findfirst(classes, y_test[i])
        pred_class = findfirst(classes, y_pred[i])
        # pred_class = row, true_class = column
        cm[pred_class, true_class] += 1
    end
    cm
end

# cols = true classes
# rows = predicted classes
confusion_matrix(y_test, y_pred)

###############################################################
##################  {  Artificial Neural }  ###################
###############################################################

type ArtificialNeuron
  weights::Vector{Float64}
  bias::Float64
  activation_func::Function
end

function activate(an::ArtificialNeuron, x::Vector{Float64})
  # sum of weights * input + bias
  pre_activation = an.weights' * x + an.bias
  an.activation_func(pre_activation)
end

# Artificial neruon looks like linear model:
# input * weight + bias
function sigm(a)
  1. / (1. + exp(-a))
end

weights = [1., -3., 1.]
bias = 0.5
an = ArtificialNeuron(weights, bias, sigm)

# trial ->
x = [-2., 1., 0.]
activate(an, x)

type ArtificalNeuron
    weights::Vector{Float64}
    bias::Float64
    activation_func::Function
end

function activate(an::ArtificalNeuron, x::Vector{Float64})
    # sum of weights times input, then add bias
    pre_activation = an.weights' * x + an.bias
    an.activation_func(pre_activation)
end

function sigm(a)
    1. / (1. + exp(-a))
end

# hard code in some weights and biases
weights = [1.,-3.,1.]
bias = 0.5
an = ArtificalNeuron(weights,bias,sigm)

# What are the outputs of these values?
x = [-2.,1.,0.]
activate(an,x)

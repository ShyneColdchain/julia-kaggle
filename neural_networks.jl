# neural networks in Julia: tutorial
# data: winequality-red.csv
Pkg.add("DataFrames")
Pkg.add("Gadfly")
using DataFrames
using Gadfly

# read in df to train
train_df = readtable("data/winequality-red.csv", separator=';')

# get value counts of quality col
count = hist(train_df[:quality])
class = sort(unique(train_df[:quality]))
value_counts = DataFrame(count=count, class=class)
println(value_counts)

# draw histogram
p = plot(value_counts, x="class", y="count",
         Geom.bar(), Guide.title("Class distributions for quality")
         )
draw(PNG(14cm, 10cm), p)

# create train and test data splits
x = vector(train_df[:quality])
x = matrix(train_df[[colnames(train_df) .!=:quality]])
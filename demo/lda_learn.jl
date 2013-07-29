# Demo the learning of Latent Dirichlet Allocation (LDA)

using ProbTopicModels

## ground-truth configuration

alpha = [1.5, 2.5, 1.8]
ndocs = 200

toc1 = [0.6, 0.3, 0.05, 0.02, 0.03]
toc2 = [0.1, 0.3, 0.4, 0.1, 0.1]
toc3 = [0.02, 0.05, 0.03, 0.5, 0.4]

topics = hcat(toc1, toc2, toc3)
model = LDAModel(alpha, topics)

println("Underlying Model (dump): ")
println("---------------------------")
dump(model)
println()

# generate data

ndocs = 200
doclen = 1000
println("Generating $(ndocs) documents (each of length $(doclen)) ...")

corpus = Array(SDocument, ndocs)

for i = 1 : ndocs
	doc = randdoc(model, doclen)
	corpus[i] = doc
end

# initialize model






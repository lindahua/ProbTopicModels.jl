# Demo the learning of Latent Dirichlet Allocation (LDA)

using NumericExtensions
using ProbTopicModels

## ground-truth configuration

alpha = [1.5, 2.5, 1.8]
ndocs = 200

toc1 = [0.6, 0.3, 0.05, 0.02, 0.03]
toc2 = [0.1, 0.3, 0.4, 0.1, 0.1]
toc3 = [0.02, 0.05, 0.03, 0.5, 0.4]

topics = hcat(toc1, toc2, toc3)
model = LDAModel(alpha, topics)
K = ntopics(model)
V = nterms(model)

# generate data

ndocs = 500
doclen = 1000
println("Generating $(ndocs) documents (each of length $(doclen)) ...")

corpus = Array(SDocument, ndocs)

for i = 1 : ndocs
	doc = randdoc(model, doclen)
	corpus[i] = doc
end

# initialize model

alpha_init = [1.0, 1.0, 1.0]
topics_init = normalize(topics + 0.2, 1, 1)
model_init = LDAModel(alpha_init, topics_init)

results, info = learn(corpus, model_init, 
	LDAVarLearn(tol=ndocs * 1.0e-3, maxiter=500, display=:iter))

println("Underlying Model (dump): ")
println("---------------------------")
dump(model)
println()

println("Learned Model (dump):")
println("---------------------------")
dump(results.model)
println()


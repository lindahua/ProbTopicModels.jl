# Example of variational inference based on LDA model

using Distributions
using NumericExtensions
using ProbTopicModels

topics = [ 0.70 0.20 0.04 0.03 0.03; 
		   0.10 0.40 0.45 0.02 0.03;
		   0.01 0.02 0.02 0.45 0.50 ]'

V = size(topics, 1)
K = size(topics, 2)

lda_model = LDAModel(5.0, topics)
println("Demo of LDA variational inference ...")

function printvec(title, x::Vector{Float64})
	print(title)
	print('[')
	for i in 1 : length(x)
		@printf("%7.4f ", x[i])
	end
	println(']')
end

const ndocs = 3
const doc_len = 500

for i in 1 : ndocs
	println("Document $i:")

	theta = rand(lda_model.dird)
	doc = randdoc(lda_model, theta, doc_len)

	println("  doc     = $(int(doc.counts))")
	printvec("  theta    = ", theta)

	res = lda_varinfer(lda_model, doc)

	rtheta = mean_theta(res)
	printvec("  E(theta) = ", rtheta)
	@printf("  difference with truth = %.4f\n", asum(rtheta - theta))
	println()
end


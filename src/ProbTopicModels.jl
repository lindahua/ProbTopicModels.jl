module ProbTopicModels
	using NumericExtensions
	using Distributions
	using MLBase

	import Base.length, Base.show
	import MLBase.objective, MLBase.update!, MLBase.initialize

	export

	# types
	Vocabulary, vocabulary, encode, SDocument, count_words,

	# common names
	infer, learn, initialize, ntopics, nterms,

	# LDA
	LDAModel, randdoc, 
	LDAVarInfer, LDAVarInferResults, mean_theta,
    LDAVarLearn, LDAVarLearnResults

	include("types.jl")
	include("utils.jl")
	include("lda.jl")
end

module ProbTopicModels
	using NumericExtensions
	using Distributions
	using MLBase

	import Base.length, Base.show

	export

	# types
	Vocabulary, vocabulary, encode, SDocument, count_words,

	# LDA
	LDAModel, randdoc, LDAVarInferOptions, LDAVarInferResults, mean_theta,
    da_varinfer_objv, lda_varinfer_init!, lda_varinfer_update!, lda_varinfer


	include("types.jl")
	include("lda.jl")
end

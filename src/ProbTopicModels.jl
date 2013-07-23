module ProbTopicModels
	using NumericExtensions
	using Distributions
	using MLBase

	import Base.length, Base.show

	export

	# types
	Vocabulary, vocabulary, encode, SDocument, count_words,

	# common names
	infer, learn,

	# LDA
	LDAModel, randdoc, LDAVarInfer, LDAVarInferResults, mean_theta,
    lda_varinfer_objv, lda_varinfer_init!, lda_varinfer_update!


	include("types.jl")
	include("utils.jl")
	include("lda.jl")
end

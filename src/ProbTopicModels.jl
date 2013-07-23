module ProbTopicModels
	using NumericExtensions
	using Distributions

	import Base.length, Base.show

	export

	# types
	Vocabulary, vocabulary, encode, SDocument, count_words


	include("types.jl")
end

# Types to represent topics and documents

#################################################
#
#  Vocabulary
#
#  Each vocabulary maintains a list of terms
#  and a map from terms to integers
#
#################################################

immutable Vocabulary{Term}
	terms::Vector{Term}
	imap::Dict{Term,Int}

	function Vocabulary(terms::Vector{Term})
		imap = (Term=>Int)[e[2] => e[1] for e in enumerate(terms)]
		new(terms, imap)
	end
end

vocabulary{Term}(terms::Vector{Term}) = Vocabulary{Term}(terms)
length(v::Vocabulary) = length(v.terms)

function encode{T}(v::Vocabulary{T}, t::T)
	get(v.imap, t, nothing)
end

function encode{T}(v::Vocabulary{T}, tseq::AbstractVector{T})
	imap = v.imap
	r = Array(Int, 0)
	sizehint(r, length(tseq))
	for t in tseq
		v = get(imap, t, nothing)
		if !(v == nothing)
			push!(r, v::Int)
		end
	end
	r
end

#################################################
#
#  Simple Document
#
#  Each document is characterized by word counts
#
#################################################

immutable SDocument
	terms::Vector{Int}          # the list of terms in the document
	counts::Vector{Float64}     # the number of times each word occurs
	sum_counts::Float64

	SDocument(terms::Vector{Int}, counts::Vector{Int}) = new(terms, float64(counts), float64(sum(counts)))
	SDocument(terms::Vector{Int}, counts::Vector{Float64}) = new(terms, counts, sum(counts))

	# create a document from a sequence of (possibly repeated) terms
	function SDocument(seq::Vector{Int})
		terms = Int[]
		counts = Int[]
		len = length(seq)

		s = sort(seq)
		pre_w = -1
		n = 0

		for w in s
			if w == pre_w  # continue with previous word
				counts[n] += 1.
			else           # new word
				n += 1
				push!(terms, w)
				push!(counts, 1.)
				pre_w = w
			end
		end

		new(terms, counts, float64(len))
	end
end

histlength(d::SDocument) = length(d.terms)

function count_words(vsiz::Int, docs::AbstractVector{SDocument})
	# count total number of words over a corpus

	r = zeros(vsiz)
	for d in docs
		terms = d.terms
		cnts = d.counts
		for i in 1 : length(terms)
			r[terms[i]] += cnts[i]
		end
	end
	r
end

function max_histlength(docs::AbstractVector{SDocument})
	maxlen = 0
	for d in docs
		l = histlength(d)
		if l > maxlen
			maxlen = l
		end
	end
	maxlen
end


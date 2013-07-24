# Utilities to support the main computation

function soft_topic_assign!(
	tlogB::Matrix{Float64},       # tranposed log(B): size (K, V)
	bias::Vector{Float64},        # bias to topics: size (K,)
	terms::AbstractVector{Int},   # a list of terms: size (n,)
	h::Vector{Float64},               # word counts (n,)
	φ::Matrix{Float64},           # soft assignment: size(K, n)
	τ::Vector{Float64})           # topic weights            

	K = size(tlogB, 1)
	V = size(tlogB, 2)
	n = length(terms)

	@assert size(φ, 1) == K && size(φ, 2) >= n

	fill!(τ, 0.)
	o::Int = 0  # base offset for φ, φ[o + k] -- φ[k, i]

	for i in 1 : n
		@inbounds w = terms[i]
		if w < 1 || w > V
			throw(ArgumentError("soft_topic_assign!: term index out of bound."))
		end
		ow = K * (w - 1)

		# combine evidence: prior bias + log likelihood
		mp = -Inf			
		for k in 1 : K
			@inbounds φ[o + k] = v = tlogB[ow + k] + bias[k]
			if v > mp
				mp = v
			end
		end

		# softmax (normalize φ)
		# accumulate to γ at the same time

		s = 0.
		for k in 1 : K
			ki = o + k
			@inbounds s += (φ[ki] = exp(φ[ki] - mp))				
		end

		inv_s = 1.0 / s
		for k in 1 : K
			@inbounds pk = (φ[o + k] *= inv_s)
			@inbounds τ[k] += pk * h[i]
		end

		o += K
	end	
end


function add_word_counts!(
	W::Matrix{Float64},       # accumulator, size (K, V) 
	terms::Vector{Int},       # list of terms, size (n,)
	cnts::Vector{Float64},    # counts of terms, size (n,)
	φ::Matrix{Float64})       # soft assignment table, size (K, n+)

	K = size(W, 1)
	n = length(terms)

	for i = 1:n
		w = terms[i]
		c = cnts[i]

		for k = 1:K
			W[k,w] += c * φ[k,i]
		end
	end
end

